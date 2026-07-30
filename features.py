"""
features.py — the richer point-in-time feature set the agents reason over.

`data_loader.compute_features` gives each agent two numbers (63-day momentum and
a 5-day reversal proxy). That is enough to test the plumbing but thin for an
analyst prompt. This module computes a fuller cross-section, borrowing the
technical vocabulary FAgent's DataAnalyst uses (SMA distance, RSI, MACD,
Bollinger, multi-horizon momentum, risk metrics) and adding the things that only
became possible once the price cache stored full OHLCV + dividends:

  * high/low based volatility (Parkinson, ATR) — a far more efficient estimator
    than close-to-close, and simply not computable from adjusted close alone
  * volume: activity shock and Amihud illiquidity
  * the overnight/intraday return split, which have different dynamics
  * trailing dividend yield, a real income signal rather than a price proxy

Two design points that matter, and that make this different from FAgent:

  1. FAgent trades ONE stock, so absolute levels ("RSI = 68") are meaningful on
     their own. Here 498 names compete for the same book, so what matters is
     each name's standing IN THE CROSS-SECTION. Every ratio feature therefore
     also gets a `_z` (tanh-z-scored, in (-1,1)) and a `_pct` (0-100 percentile)
     twin. Percentiles are what the prompt shows: "momentum in the 92nd
     percentile" is far more legible to a small model than "+0.78".

  2. Everything is POINT-IN-TIME. A feature at date `asof` reads only rows dated
     <= asof. Forward returns stay in evaluation.py and never enter a prompt.

The rule-based score is deliberately NOT changed: `momentum` and `value` are
returned with exactly their legacy definitions and values, so
StockAgent._score_from_data keeps producing identical numbers and previous
rule-based results stay comparable. The new features reach the model through the
prompt only.

LOOKBACK BUDGET: the backtest window starts 2024-10-01 and the first rebalance
is 2024-12-31, which is only 64 trading days in. Anything needing 126 or 252
days (12-1 momentum, 52-week-high proximity) silently degrades to the longest
window available, so those are NOT included here — they need the cache
backfilled to ~2023 first. Every lookback below is <= 63 days for that reason.
"""

import bisect
import math
from typing import Dict, List, Optional

# --------------------------------------------------------------------------
# Which features get cross-sectionally normalised (_z and _pct twins added).
# Features whose raw units are already interpretable (RSI, percentages) are
# still normalised, because the *prompt* may want either form.
# --------------------------------------------------------------------------
RATIO_FEATURES = [
    "momentum", "value", "mom_21", "dist_sma20", "dist_sma50", "rsi14",
    "bb_pctb", "macd_hist", "vol_20", "park_vol_20", "atr", "drawdown_60",
    "skew_20", "vol_shock", "dollar_vol", "amihud", "gap_5", "intraday_5",
    "close_loc_5", "div_yield", "sector_rel_mom", "beta", "idio_vol",
    "resid_mom", "sector_corr",
]

# What the agent prompt actually shows. Kept deliberately short: the prompt is
# sent once per stock per round, so at 498 stocks x 3 rounds every extra line is
# ~1,500 extra lines of generation pressure. Each entry is
# (feature, label, how-to-render).
PROMPT_FEATURES = [
    ("momentum", "3-month momentum", "pct"),
    ("mom_21", "1-month momentum", "pct"),
    ("value", "1-week reversal", "pct"),
    ("sector_rel_mom", "momentum vs its sector", "pct"),
    ("dist_sma50", "price vs 50-day average", "signed_pct_raw"),
    ("rsi14", "RSI(14)", "raw1"),
    ("vol_20", "annualised volatility", "pct_and_raw"),
    ("beta", "beta to the market", "raw2"),
    ("vol_shock", "volume vs normal", "pct"),
    ("div_yield", "dividend yield", "pct_raw"),
]


# ==========================================================================
# Indexed price history — build once, slice cheaply for every rebalance date
# ==========================================================================
class PriceHistory:
    """
    OHLCV rows indexed for fast point-in-time windows.

    `data_loader.compute_features` rescans a ticker's whole series for every
    feature on every date. That is fine for 2 features x 30 names; at ~20
    features x 498 names x 373 dates it is not. Here each ticker's dates are
    sorted once and located with bisect, so a window is a list slice.

    Also precomputes, once for the whole run:
      * `market[date]`  equal-weight mean daily return across the universe, the
                        market proxy beta and residual momentum are measured
                        against (no index data needed, and it matches the
                        dollar-neutral book's own universe)
      * `sector[sec][date]` the same within each sector
    """

    def __init__(self, ohlcv: Dict[str, List[dict]],
                 sectors: Optional[Dict[str, str]] = None):
        self.rows = {t: rows for t, rows in ohlcv.items() if rows}
        self.dates = {t: [r["date"] for r in rows] for t, rows in self.rows.items()}
        self.sectors = sectors or {}
        self._returns = {t: self._daily_returns(rows)
                         for t, rows in self.rows.items()}
        self.market, self.sector_returns = self._aggregate_returns()

    @staticmethod
    def _daily_returns(rows):
        """{date: simple adjclose return} for one ticker."""
        out, prev = {}, None
        for r in rows:
            px = r.get("adjclose")
            if px is not None and prev:
                out[r["date"]] = px / prev - 1.0
            if px:
                prev = px
        return out

    def _aggregate_returns(self):
        """Equal-weight market and per-sector mean return, by date."""
        market_sums, market_n = {}, {}
        sector_sums, sector_n = {}, {}
        for t, rets in self._returns.items():
            sec = self.sectors.get(t)
            for date, r in rets.items():
                market_sums[date] = market_sums.get(date, 0.0) + r
                market_n[date] = market_n.get(date, 0) + 1
                if sec:
                    sector_sums.setdefault(sec, {})
                    sector_n.setdefault(sec, {})
                    sector_sums[sec][date] = sector_sums[sec].get(date, 0.0) + r
                    sector_n[sec][date] = sector_n[sec].get(date, 0) + 1
        market = {d: market_sums[d] / market_n[d] for d in market_sums}
        sector = {sec: {d: sector_sums[sec][d] / sector_n[sec][d]
                        for d in sector_sums[sec]} for sec in sector_sums}
        return market, sector

    def upto(self, ticker: str, asof: str) -> int:
        """Number of rows dated <= asof (0 if the ticker has no such row)."""
        dates = self.dates.get(ticker)
        if not dates:
            return 0
        return bisect.bisect_right(dates, asof)

    def window(self, ticker: str, asof: str, n: int) -> List[dict]:
        """The last `n` rows dated <= asof (fewer if history is short)."""
        end = self.upto(ticker, asof)
        return self.rows[ticker][max(0, end - n):end]

    def return_window(self, ticker: str, asof: str, n: int):
        """(dates, returns) for the last `n` daily returns dated <= asof."""
        rows = self.window(ticker, asof, n + 1)
        rets = self._returns.get(ticker, {})
        dates = [r["date"] for r in rows if r["date"] in rets]
        return dates, [rets[d] for d in dates]


# ==========================================================================
# Small statistics helpers (plain Python by project convention)
# ==========================================================================
def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _stdev(xs):
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _ema(values, span):
    """Exponential moving average, seeded on the first value."""
    if not values:
        return None
    alpha = 2.0 / (span + 1.0)
    out = values[0]
    for v in values[1:]:
        out = alpha * v + (1 - alpha) * out
    return out


def _safe_div(a, b):
    if a is None or b in (None, 0):
        return None
    return a / b


# ==========================================================================
# Per-stock features
# ==========================================================================
def _trailing_return(rows, lookback):
    """Adjclose return over the last `lookback` bars of `rows`."""
    if len(rows) <= lookback:
        return None
    old, now = rows[-lookback - 1].get("adjclose"), rows[-1].get("adjclose")
    if not old or now is None:
        return None
    return now / old - 1.0


def _rsi(closes, period=14):
    """
    RSI on simple averages of gains/losses — the same formulation FAgent uses
    (data_utils.py:287-291), so the two projects' numbers are comparable.
    """
    if len(closes) <= period:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))][-period:]
    gain = _mean([max(0.0, d) for d in deltas])
    loss = _mean([max(0.0, -d) for d in deltas])
    if not loss:
        return 100.0 if gain else 50.0
    rs = gain / loss
    return 100.0 - (100.0 / (1.0 + rs))


def _parkinson_vol(rows):
    """
    High-low range volatility, annualised. ~5x more efficient than a
    close-to-close estimate for the same window, which is the whole reason for
    storing high/low in the cache.
    """
    terms = []
    for r in rows:
        hi, lo = r.get("high"), r.get("low")
        if hi and lo and lo > 0 and hi >= lo:
            terms.append(math.log(hi / lo) ** 2)
    if not terms:
        return None
    var = sum(terms) / (4.0 * math.log(2.0) * len(terms))
    return math.sqrt(var) * math.sqrt(252.0)


def _atr_pct(rows, period=14):
    """Average true range over `period` bars, as a fraction of the last close."""
    if len(rows) < 2:
        return None
    trs = []
    for prev, cur in zip(rows[-period - 1:-1], rows[-period:]):
        hi, lo, pc = cur.get("high"), cur.get("low"), prev.get("close")
        if hi is None or lo is None or pc is None:
            continue
        trs.append(max(hi - lo, abs(hi - pc), abs(lo - pc)))
    last_close = rows[-1].get("close")
    return _safe_div(_mean(trs), last_close)


def _max_drawdown(rets):
    """Worst peak-to-trough decline of the compounded return path."""
    if not rets:
        return None
    cum, peak, worst = 1.0, 1.0, 0.0
    for r in rets:
        cum *= (1.0 + r)
        peak = max(peak, cum)
        worst = min(worst, cum / peak - 1.0)
    return worst


def _skew(xs):
    if len(xs) < 3:
        return None
    m, s = _mean(xs), _stdev(xs)
    if not s:
        return None
    return sum(((x - m) / s) ** 3 for x in xs) / len(xs)


def _beta_and_residual(stock_rets, mkt_rets):
    """
    OLS beta of the stock on the equal-weight market, plus the volatility of the
    residual (idiosyncratic risk) — the part a dollar-neutral book is actually
    exposed to.
    """
    n = min(len(stock_rets), len(mkt_rets))
    if n < 20:
        return None, None
    s, m = stock_rets[-n:], mkt_rets[-n:]
    ms, mm = _mean(s), _mean(m)
    var_m = sum((x - mm) ** 2 for x in m)
    if var_m == 0:
        return None, None
    beta = sum((a - ms) * (b - mm) for a, b in zip(s, m)) / var_m
    alpha = ms - beta * mm
    resid = [a - (alpha + beta * b) for a, b in zip(s, m)]
    idio = _stdev(resid)
    return beta, (idio * math.sqrt(252.0) if idio is not None else None)


def _correlation(xs, ys):
    n = min(len(xs), len(ys))
    if n < 20:
        return None
    xs, ys = xs[-n:], ys[-n:]
    mx, my = _mean(xs), _mean(ys)
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    vx = sum((a - mx) ** 2 for a in xs)
    vy = sum((b - my) ** 2 for b in ys)
    if vx <= 0 or vy <= 0:
        return None
    return cov / math.sqrt(vx * vy)


def _one_stock(hist: PriceHistory, ticker: str, asof: str,
               mom_lookback: int, rev_lookback: int) -> dict:
    """Every per-stock (Tier A) feature for one name at one date."""
    rows = hist.window(ticker, asof, 70)          # enough for all lookbacks
    if not rows:
        return {}
    closes = [r["adjclose"] for r in rows if r.get("adjclose") is not None]
    last = rows[-1]
    px = last.get("adjclose")
    f = {}

    # --- momentum / reversal (legacy definitions preserved exactly) -------
    f["momentum_raw"] = _trailing_return(rows, mom_lookback)
    rev = _trailing_return(rows, rev_lookback)
    f["value_raw"] = -rev if rev is not None else None
    f["mom_21_raw"] = _trailing_return(rows, 21)

    # --- trend: distance from moving averages ----------------------------
    for span in (20, 50):
        sma = _mean(closes[-span:]) if len(closes) >= span else None
        f[f"dist_sma{span}"] = (px / sma - 1.0) if (sma and px) else None

    # --- oscillators -----------------------------------------------------
    f["rsi14"] = _rsi(closes)
    if len(closes) >= 26:
        macd = _ema(closes[-60:], 12) - _ema(closes[-60:], 26)
        # Histogram = MACD minus its own 9-period signal. Scaled by price so it
        # is comparable across a $20 stock and a $900 one.
        f["macd_hist"] = _safe_div(macd, px)
    else:
        f["macd_hist"] = None
    if len(closes) >= 20:
        mid, sd = _mean(closes[-20:]), _stdev(closes[-20:])
        # %B: 0 = at the lower band, 1 = at the upper band.
        f["bb_pctb"] = ((px - (mid - 2 * sd)) / (4 * sd)) if sd else None
    else:
        f["bb_pctb"] = None

    # --- risk ------------------------------------------------------------
    _, rets = hist.return_window(ticker, asof, 60)
    sd20 = _stdev(rets[-20:]) if len(rets) >= 20 else None
    f["vol_20"] = sd20 * math.sqrt(252.0) if sd20 is not None else None
    f["park_vol_20"] = _parkinson_vol(rows[-20:])
    f["atr"] = _atr_pct(rows)          # ATR as a fraction of close
    f["drawdown_60"] = _max_drawdown(rets)
    f["skew_20"] = _skew(rets[-20:]) if len(rets) >= 20 else None

    # --- volume / liquidity ----------------------------------------------
    vols = [r.get("volume") for r in rows if r.get("volume")]
    if len(vols) >= 25:
        recent, normal = _mean(vols[-5:]), _mean(vols[-60:])
        f["vol_shock"] = _safe_div(recent, normal)
    else:
        f["vol_shock"] = None
    dollar = [r["volume"] * r["close"] for r in rows[-20:]
              if r.get("volume") and r.get("close")]
    f["dollar_vol"] = math.log(_mean(dollar)) if dollar else None
    # Amihud illiquidity: price impact per dollar traded, scaled for readability.
    if dollar and rets:
        pairs = list(zip(rets[-len(dollar):], dollar))
        f["amihud"] = _mean([abs(r) / d * 1e9 for r, d in pairs if d > 0])
    else:
        f["amihud"] = None

    # --- overnight vs intraday -------------------------------------------
    gaps, intra, locs = [], [], []
    for prev, cur in zip(rows[-6:-1], rows[-5:]):
        o, c, pc = cur.get("open"), cur.get("close"), prev.get("close")
        hi, lo = cur.get("high"), cur.get("low")
        if o and pc:
            gaps.append(o / pc - 1.0)
        if o and c:
            intra.append(c / o - 1.0)
        if hi is not None and lo is not None and hi > lo and c is not None:
            locs.append((c - lo) / (hi - lo))
    f["gap_5"] = sum(gaps) if gaps else None
    f["intraday_5"] = sum(intra) if intra else None
    f["close_loc_5"] = _mean(locs)

    # --- income ----------------------------------------------------------
    # Trailing dividends over the available window, scaled to a year. Early in
    # the backtest only ~64 bars exist, so this is an annualised estimate from a
    # partial year rather than a true trailing-12-month yield.
    win = hist.window(ticker, asof, 252)
    divs = sum(r.get("dividend") or 0.0 for r in win)
    if px and win:
        f["div_yield"] = (divs / px) * (252.0 / len(win))
    else:
        f["div_yield"] = None

    # --- market-relative (Tier B) ---------------------------------------
    dates, srets = hist.return_window(ticker, asof, 60)
    mkt = [hist.market.get(d) for d in dates]
    mkt = [m for m in mkt if m is not None]
    beta, idio = _beta_and_residual(srets, mkt)
    f["beta"], f["idio_vol"] = beta, idio
    sec = hist.sectors.get(ticker)
    sec_series = hist.sector_returns.get(sec, {}) if sec else {}
    f["sector_corr"] = _correlation(srets, [sec_series.get(d, 0.0) for d in dates])
    return f


# ==========================================================================
# Cross-sectional normalisation
# ==========================================================================
def _tanh_zscore(values):
    """Standardise across the universe, then squash to (-1, 1) — as in
    data_loader.compute_features, so the legacy features keep identical values."""
    present = [v for v in values if v is not None]
    if len(present) < 2:
        return [0.0 for _ in values]
    mean = sum(present) / len(present)
    var = sum((v - mean) ** 2 for v in present) / len(present)
    std = math.sqrt(var) or 1.0
    return [0.0 if v is None else math.tanh((v - mean) / std) for v in values]


def _percentiles(values):
    """0-100 percentile rank within the cross-section (ties share a rank)."""
    present = sorted(v for v in values if v is not None)
    if not present:
        return [None for _ in values]
    out = []
    for v in values:
        if v is None:
            out.append(None)
            continue
        lo = bisect.bisect_left(present, v)
        hi = bisect.bisect_right(present, v)
        out.append(100.0 * ((lo + hi) / 2.0) / len(present))
    return out


def compute_extended_features(ohlcv: Dict[str, List[dict]], asof: str,
                              sectors: Optional[Dict[str, str]] = None,
                              mom_lookback: int = 63, rev_lookback: int = 5,
                              history: Optional[PriceHistory] = None) -> dict:
    """
    Point-in-time feature cross-section: {ticker: {feature: value}}.

    A drop-in superset of `data_loader.compute_features`: `momentum` and `value`
    keep their legacy tanh-z-scored definitions and values, so
    StockAgent.rule_initial produces identical scores and old rule-based results
    stay comparable. Everything else is additive.

    Each ratio feature appears three ways: `X` (raw, natural units), `X_z`
    (tanh-z across the universe, in (-1,1)) and `X_pct` (0-100 percentile).

    Pass `history=PriceHistory(ohlcv, sectors)` when looping over many dates —
    building it once is what keeps a 373-date walk-forward affordable.
    """
    hist = history or PriceHistory(ohlcv, sectors)
    tickers = [t for t in ohlcv if t in hist.rows]
    raw = {t: _one_stock(hist, t, asof, mom_lookback, rev_lookback)
           for t in tickers}

    # Sector-relative momentum needs the cross-section, so it comes after.
    sector_mom = {}
    for t in tickers:
        sec = (sectors or {}).get(t)
        m = raw[t].get("momentum_raw")
        if sec and m is not None:
            sector_mom.setdefault(sec, []).append(m)
    sector_mean = {s: sum(v) / len(v) for s, v in sector_mom.items()}
    for t in tickers:
        sec = (sectors or {}).get(t)
        m = raw[t].get("momentum_raw")
        raw[t]["sector_rel_mom"] = (
            m - sector_mean[sec] if (sec in sector_mean and m is not None) else None)
        beta, mom = raw[t].get("beta"), raw[t].get("momentum_raw")
        mkt_mom = _mean([raw[o].get("momentum_raw") for o in tickers
                         if raw[o].get("momentum_raw") is not None])
        raw[t]["resid_mom"] = (mom - beta * mkt_mom
                               if None not in (beta, mom, mkt_mom) else None)

    # Legacy names, legacy values: tanh-z of the same raw inputs.
    out = {t: {"date": asof} for t in tickers}
    for legacy, source in (("momentum", "momentum_raw"), ("value", "value_raw"),
                           ("mom_21", "mom_21_raw")):
        zs = _tanh_zscore([raw[t].get(source) for t in tickers])
        for t, z in zip(tickers, zs):
            out[t][legacy] = z
            out[t][legacy + "_raw"] = raw[t].get(source)

    for name in RATIO_FEATURES:
        if name in ("momentum", "value", "mom_21"):
            continue                       # already handled above
        vals = [raw[t].get(name) for t in tickers]
        zs, pcts = _tanh_zscore(vals), _percentiles(vals)
        for t, v, z, p in zip(tickers, vals, zs, pcts):
            out[t][name] = v
            out[t][name + "_z"] = z
            out[t][name + "_pct"] = p
    # Percentiles for the legacy trio too, since the prompt renders those.
    for name in ("momentum", "value", "mom_21"):
        pcts = _percentiles([raw[t].get(name + "_raw") for t in tickers])
        for t, p in zip(tickers, pcts):
            out[t][name + "_pct"] = p
    return out


# ==========================================================================
# Rendering for the prompt
# ==========================================================================
def make_feature_fn(prices, tickers, start: str, end: str,
                    sectors: Optional[Dict[str, str]] = None,
                    mom_lookback: int = 63, rev_lookback: int = 5,
                    feature_set: Optional[str] = None):
    """
    Return `fn(asof) -> {ticker: features}` for whichever feature set is active.

    A run script only has to build this once and call it per rebalance date; the
    expensive PriceHistory indexing (and the OHLCV read) happens here, not inside
    the date loop. `feature_set` defaults to `config.FEATURE_SET`:

      "basic"     data_loader.compute_features — momentum + reversal only
      "extended"  this module's full cross-section

    Both return the same `momentum`/`value` values, so switching sets never
    changes a rule-based score — only what the LLM prompt can see.
    """
    from data_loader import compute_features, load_ohlcv
    if feature_set is None:
        import config
        feature_set = getattr(config, "FEATURE_SET", "basic")

    if feature_set == "basic":
        def basic(asof):
            return compute_features(prices, asof, mom_lookback, rev_lookback)
        return basic
    if feature_set != "extended":
        raise ValueError(f"FEATURE_SET must be 'basic' or 'extended', got "
                         f"{feature_set!r}")

    ohlcv = load_ohlcv(list(tickers), start, end)
    history = PriceHistory(ohlcv, sectors)

    def extended(asof):
        return compute_extended_features(ohlcv, asof, sectors, mom_lookback,
                                         rev_lookback, history=history)
    return extended


def _ordinal(n: float) -> str:
    """1 -> '1st', 92 -> '92nd'. The prompt is prose the model reads; '92th'
    reads as a typo and costs nothing to get right."""
    i = int(round(n))
    if 10 <= i % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(i % 10, "th")
    return f"{i}{suffix}"


def describe(feats: dict) -> str:
    """
    Render one stock's features as compact prompt lines.

    Percentiles carry the cross-sectional meaning ("87th pct of 498") which is
    what a relative score needs; a few features are also shown in natural units
    where those are independently meaningful (RSI, volatility, yield, beta).
    Missing features are skipped rather than shown as zero, so the model is not
    told a fabricated value.
    """
    lines = []
    for key, label, how in PROMPT_FEATURES:
        pct = feats.get(key + "_pct")
        raw = feats.get(key if key not in ("momentum", "value", "mom_21")
                        else key + "_raw")
        if how == "pct":
            if pct is None:
                continue
            lines.append(f"  {label}: {_ordinal(pct)} percentile")
        elif how == "signed_pct_raw":
            if raw is None:
                continue
            lines.append(f"  {label}: {raw * 100:+.1f}%")
        elif how == "raw1":
            if raw is None:
                continue
            lines.append(f"  {label}: {raw:.0f}")
        elif how == "raw2":
            if raw is None:
                continue
            lines.append(f"  {label}: {raw:.2f}")
        elif how == "pct_raw":
            if raw is None:
                continue
            lines.append(f"  {label}: {raw * 100:.1f}%")
        elif how == "pct_and_raw":
            if raw is None or pct is None:
                continue
            lines.append(f"  {label}: {raw * 100:.0f}% ({_ordinal(pct)} percentile)")
    return "\n".join(lines)
