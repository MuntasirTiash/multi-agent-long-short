"""
data_loader.py — real, point-in-time price data for the DJIA-30.

Phase 0 fed the agents random numbers. Phase 1 replaces that with real daily
adjusted-close prices and computes two price-based features per stock.

Design choices that matter for research validity:

  * CACHE-FIRST: prices live in `data/price_cache/{TICKER}_{START}_{END}.csv` and
    are read from disk, so runs are fast, reproducible, and work offline on HPC
    compute nodes with no internet. `download_prices.py` populates the whole
    universe up front and owns the download + file format; a cache miss here
    calls into it as a convenience, which is the fragile path on a compute node.

  * ZERO-COST source: Yahoo Finance's public chart endpoint. No API key, no
    `yfinance` install, no paid data vendor.

  * COLUMNS: the cache stores date/open/high/low/close/adjclose/volume/dividend/
    split_ratio. `load_prices` returns just the adjusted-close series (what every
    feature and forward return is built from); `load_ohlcv` returns everything.

  * POINT-IN-TIME: features at a rebalance date `asof` are computed using ONLY
    prices dated <= asof. The forward return used to *grade* those features
    (in evaluation.py) uses future prices and is NEVER shown to the agents.

  * LEAKAGE-SAFE WINDOW: we work after Oct-2024, past Qwen2.5's training cutoff,
    so a future LLM agent cannot have memorised these outcomes.

Note on the universe: index membership itself is only *approximately* point-in-
time here (we use a fixed current member list; NVDA/DOW changed in Nov-2024).
A fully rigorous study needs the historical membership on each date — flagged
as a Phase-3 refinement, not needed for this preliminary pipeline check.
"""

import csv
import math
import os
import time

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "price_cache")


# --------------------------------------------------------------------------
# 1. Read the cache (populated by download_prices.py)
# --------------------------------------------------------------------------
def _cache_path(ticker: str, start: str, end: str) -> str:
    return os.path.join(CACHE_DIR, f"{ticker}_{start}_{end}.csv")


def _download(ticker: str, start: str, end: str, path: str):
    """
    Fill one cache miss, writing the full OHLCV+events schema.

    Delegates to download_prices so there is exactly ONE place that knows the
    cache file format. Use `python download_prices.py` directly to populate a
    whole universe up front — that is the supported path for large universes and
    for compute nodes, where a lazy per-ticker download is fragile.
    """
    from download_prices import fetch, write_csv
    write_csv(path, fetch(ticker, start, end))
    time.sleep(0.3)           # be polite to the free endpoint


def _read_cached(path: str):
    """
    Read one cached CSV -> list of row dicts, chronological.

    Handles both the current wide schema (date,open,high,low,close,adjclose,
    volume,dividend,split_ratio) and the legacy two-column date,adjclose files,
    so an old cache directory still loads. Numeric fields become floats; blanks
    become None.
    """
    with open(path, newline="") as f:
        rows = []
        for raw in csv.DictReader(f):
            row = {"date": raw["date"]}
            for key, value in raw.items():
                if key == "date":
                    continue
                try:
                    row[key] = float(value) if value not in ("", None) else None
                except ValueError:
                    row[key] = None
            rows.append(row)
    return rows


def load_prices(tickers, start: str, end: str):
    """
    Load {ticker: [(date, adjclose), ...]} for all tickers, cache-first.

    The adjusted-close series is what every feature and forward return is built
    from, so this stays the primary loader and its return shape is unchanged.
    Call `load_ohlcv` when you need the other columns.
    """
    return {t: [(r["date"], r["adjclose"]) for r in series
                if r.get("adjclose") is not None]
            for t, series in load_ohlcv(tickers, start, end).items()}


def load_ohlcv(tickers, start: str, end: str):
    """
    Load {ticker: [row_dict, ...]} with every cached column, cache-first.

    Each row has date/open/high/low/close/adjclose/volume/dividend/split_ratio
    (missing keys on a legacy two-column cache file). This is the hook for
    features the adjusted close alone cannot express — dollar volume, true
    range, overnight gaps — none of which are wired into compute_features yet.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    out = {}
    for ticker in tickers:
        path = _cache_path(ticker, start, end)
        if not os.path.exists(path):
            _download(ticker, start, end, path)
        out[ticker] = _read_cached(path)
    return out


# --------------------------------------------------------------------------
# 2. Trading-date calendar and point-in-time slicing
# --------------------------------------------------------------------------
def common_dates(prices):
    """Sorted list of dates present for EVERY ticker (a shared calendar)."""
    date_sets = [set(d for d, _ in series) for series in prices.values()]
    shared = set.intersection(*date_sets)
    return sorted(shared)


def _price_on_or_before(series, asof: str):
    """Most recent (date, price) at or before `asof`; None if none exists."""
    chosen = None
    for date, price in series:      # series is chronological
        if date <= asof:
            chosen = (date, price)
        else:
            break
    return chosen


def _return_between(series, start_date: str, end_date: str):
    """Simple return from the close nearest-before start to nearest-before end."""
    a = _price_on_or_before(series, start_date)
    b = _price_on_or_before(series, end_date)
    if not a or not b or a[1] == 0:
        return None
    return b[1] / a[1] - 1.0


# --------------------------------------------------------------------------
# 3. Point-in-time features (what the agents actually see)
# --------------------------------------------------------------------------
def _trailing_return(series, asof: str, lookback_days: int):
    """Return over the `lookback_days` trading days ending at `asof`."""
    hist = [(d, p) for d, p in series if d <= asof]
    if len(hist) <= lookback_days:
        return None
    old = hist[-lookback_days - 1][1]
    now = hist[-1][1]
    if old == 0:
        return None
    return now / old - 1.0


def _tanh_zscore(values):
    """Cross-sectionally standardise, then squash to (-1, 1) with tanh."""
    present = [v for v in values if v is not None]
    if len(present) < 2:
        return [0.0 for _ in values]
    mean = sum(present) / len(present)
    var = sum((v - mean) ** 2 for v in present) / len(present)
    std = math.sqrt(var) or 1.0
    return [0.0 if v is None else math.tanh((v - mean) / std) for v in values]


def compute_features(prices, asof: str, mom_lookback=63, rev_lookback=5):
    """
    Build the per-stock data dict the StockAgent consumes, at date `asof`.

    Two price-only factors, each cross-sectionally normalised to ~[-1, 1] so
    the 30 names are directly comparable:

      momentum : trailing `mom_lookback`-day return (~3 months). Classic
                 cross-sectional momentum factor.
      value    : short-term REVERSAL proxy = negative of the trailing
                 `rev_lookback`-day return. (A true value factor needs
                 fundamentals; the free price API has none, so we use the
                 well-known short-term reversal effect as a stand-in. Swap in a
                 real book/price ratio here once fundamentals are wired.)

    Returns {ticker: {"momentum": .., "value": .., "date": asof}} — exactly the
    shape Phase 0's StockAgent already expects, so nothing downstream changes.
    """
    tickers = list(prices.keys())
    raw_mom = [_trailing_return(prices[t], asof, mom_lookback) for t in tickers]
    raw_rev = [_trailing_return(prices[t], asof, rev_lookback) for t in tickers]

    mom = _tanh_zscore(raw_mom)
    value = _tanh_zscore([-(r) if r is not None else None for r in raw_rev])

    return {t: {"momentum": mom[i], "value": value[i], "date": asof}
            for i, t in enumerate(tickers)}


def forward_returns(prices, asof: str, horizon_days: int):
    """
    {ticker: return} over the `horizon_days` AFTER `asof`.

    Evaluation-only: this looks into the future to grade a prediction and must
    never be passed to an agent. Returns None for a ticker without enough data.
    """
    result = {}
    for t, series in prices.items():
        hist = [(d, p) for d, p in series if d <= asof]
        future = [(d, p) for d, p in series if d > asof]
        if not hist or len(future) < horizon_days:
            result[t] = None
            continue
        start_p = hist[-1][1]
        end_p = future[horizon_days - 1][1]
        result[t] = (end_p / start_p - 1.0) if start_p else None
    return result


def rebalance_dates(prices, step_days=5, min_history=63, horizon_days=5):
    """
    Pick rebalance dates from the shared calendar, leaving room for a `min_history`
    warm-up before the first and a `horizon_days` tail after the last.
    """
    cal = common_dates(prices)
    if len(cal) <= min_history + horizon_days:
        return []
    usable = cal[min_history:len(cal) - horizon_days]
    return usable[::step_days]
