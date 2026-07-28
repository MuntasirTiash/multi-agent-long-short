"""
evaluation.py — the small set of metrics we grade a ranking with.

Given the agents' final scores and the (future) forward returns, we ask:

  rank_ic          Did higher-scored stocks actually do better? Spearman rank
                   correlation between score and forward return, in [-1, 1].
                   This is the standard cross-sectional signal-quality metric.

  long_short_spread The return of an equal-weight (top-N long) minus
                   (bottom-N short) book — the thing the portfolio actually
                   earns before costs.

Everything is plain arithmetic on Python lists (numpy optional). Deliberately
tiny so it is easy to trust.
"""

from typing import Dict, List


def _rank(values: List[float]) -> List[float]:
    """Average ranks (ties share the mean rank), 1-based."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0            # mean of the tied positions
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx == 0 or vy == 0:
        return 0.0
    return cov / (vx ** 0.5 * vy ** 0.5)


def rank_ic(scores: Dict[str, float], fwd: Dict[str, float]) -> float:
    """Spearman correlation of scores vs forward returns over shared tickers."""
    tickers = [t for t in scores if fwd.get(t) is not None]
    if len(tickers) < 3:
        return 0.0
    s = _rank([scores[t] for t in tickers])
    r = _rank([fwd[t] for t in tickers])
    return _pearson(s, r)


def long_short_spread(longs: List[str], shorts: List[str],
                      fwd: Dict[str, float]) -> float:
    """Equal-weight long-book return minus short-book return (before costs)."""
    def book_return(names):
        vals = [fwd[t] for t in names if fwd.get(t) is not None]
        return sum(vals) / len(vals) if vals else 0.0
    return book_return(longs) - book_return(shorts)


def summarize(series: List[float], periods_per_year: int = 52) -> Dict[str, float]:
    """
    Mean, std, and an annualised Sharpe for a list of per-period returns.

    `periods_per_year` sets the annualisation: 52 for weekly rebalancing (the
    default, so older callers are unchanged), 252 for a daily return series.
    """
    n = len(series)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "sharpe": 0.0, "n": 0}
    mean = sum(series) / n
    var = sum((x - mean) ** 2 for x in series) / n
    std = var ** 0.5
    sharpe = (mean / std * (periods_per_year ** 0.5)) if std > 0 else 0.0
    return {"mean": mean, "std": std, "sharpe": sharpe, "n": n}


# --------------------------------------------------------------------------
# Weighted portfolio return + turnover (used by the manager-agent backtest)
# --------------------------------------------------------------------------
def portfolio_return(weights: Dict[str, float], fwd: Dict[str, float]) -> float:
    """
    Return of a weighted book: sum of signed weight * forward return.

    `weights` are signed (longs positive, shorts negative). For a dollar-neutral
    book — long weights summing to +1, short weights summing to -1 — this equals
    the long-book return minus the short-book return, i.e. the same quantity
    `long_short_spread` computes for the equal-weight case. Names whose forward
    return is missing are dropped and the remaining weights renormalised on each
    side so the book stays dollar-neutral rather than silently taking a net tilt.
    """
    longs = {t: w for t, w in weights.items() if w > 0 and fwd.get(t) is not None}
    shorts = {t: w for t, w in weights.items() if w < 0 and fwd.get(t) is not None}
    long_ret = short_ret = 0.0
    lw = sum(longs.values())
    if lw > 0:
        long_ret = sum(w / lw * fwd[t] for t, w in longs.items())
    sw = sum(-w for w in shorts.values())
    if sw > 0:
        short_ret = sum(-w / sw * fwd[t] for t, w in shorts.items())
    return long_ret - short_ret


def weight_turnover(prev_weights, weights) -> float:
    """
    One-sided turnover between two weight vectors, in [0, 1].

    Standard 0.5 * sum|w - w_prev| over the union of names. A brand-new book
    (no previous) is turnover 1.0. This is what daily trading costs are charged
    on in the manager backtest.
    """
    if prev_weights is None:
        return 1.0
    names = set(weights) | set(prev_weights)
    return 0.5 * sum(abs(weights.get(t, 0.0) - prev_weights.get(t, 0.0))
                     for t in names)


# --------------------------------------------------------------------------
# Harness upgrade 1: transaction costs
# --------------------------------------------------------------------------
def turnover(prev_longs, prev_shorts, longs, shorts) -> float:
    """
    Fraction of the book that CHANGED since last rebalance, in [0, 1].

    Counts names newly entering the long book and the short book, divided by
    the total number of positions. A brand-new book (no previous) is turnover
    1.0. This is what we charge trading costs on.
    """
    total = len(longs) + len(shorts)
    if total == 0:
        return 0.0
    if prev_longs is None:
        return 1.0
    entered = len(set(longs) - set(prev_longs)) + len(set(shorts) - set(prev_shorts))
    return entered / total


def apply_cost(gross_spread: float, turn: float, roundtrip_bps: float) -> float:
    """
    Net return after trading costs.

    `roundtrip_bps` is the cost of fully turning the book once (10-20 bps is
    the standard large-cap assumption). We charge it in proportion to turnover.
    """
    return gross_spread - turn * (roundtrip_bps / 10_000.0)


# --------------------------------------------------------------------------
# Harness upgrade 2: Monte-Carlo null test (is the strategy better than luck?)
# --------------------------------------------------------------------------
def monte_carlo_null(fwd_series: List[Dict[str, float]], n_long: int,
                     n_short: int, n_trials: int = 2000, seed: int = 42):
    """
    Build the null distribution of cumulative long/short return under RANDOM
    stock picking, following MarketSenseAI's significance test.

    For each of `n_trials` trials we, at every rebalance date, pick `n_long`
    random longs and `n_short` random shorts, take the spread, and compound
    across dates into one cumulative return. The returned sorted list is the
    null distribution a real strategy must beat to be more than luck.

    `fwd_series` is one {ticker: forward_return} dict per rebalance date.
    """
    import random as _random
    rng = _random.Random(seed)
    null_cumulative = []
    for _ in range(n_trials):
        cum = 1.0
        for fwd in fwd_series:
            names = [t for t in fwd if fwd[t] is not None]
            if len(names) < n_long + n_short:
                continue
            picks = rng.sample(names, n_long + n_short)
            longs, shorts = picks[:n_long], picks[n_long:]
            spread = (sum(fwd[t] for t in longs) / n_long
                      - sum(fwd[t] for t in shorts) / n_short)
            cum *= (1.0 + spread)
        null_cumulative.append(cum - 1.0)
    null_cumulative.sort()
    return null_cumulative


def percentile_and_p(value: float, null_sorted: List[float]) -> Dict[str, float]:
    """
    Where does `value` sit in the null distribution?

    Returns the percentile (fraction of null draws below `value`) and a
    one-sided p-value (fraction of null draws >= `value`) — small p means the
    result is unlikely under random picking.
    """
    n = len(null_sorted)
    if n == 0:
        return {"percentile": 0.0, "p_value": 1.0}
    below = sum(1 for x in null_sorted if x < value)
    at_or_above = n - below
    return {"percentile": below / n, "p_value": at_or_above / n}
