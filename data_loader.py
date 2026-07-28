"""
data_loader.py — real, point-in-time price data for the DJIA-30.

Phase 0 fed the agents random numbers. Phase 1 replaces that with real daily
adjusted-close prices and computes two price-based features per stock.

Design choices that matter for research validity:

  * CACHE-FIRST (like FAgent/data): we download once to `price_cache/` and read
    from disk afterwards, so runs are fast, reproducible, and work offline on
    HPC compute nodes with no internet.

  * ZERO-COST source: Yahoo Finance's public chart endpoint, hit directly with
    `requests`. No API key, no `yfinance` install, no paid data vendor.

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
import datetime as dt
import json
import math
import os
import time

import requests

CACHE_DIR = os.path.join(os.path.dirname(__file__), "price_cache")
_YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"


# --------------------------------------------------------------------------
# 1. Download + cache daily adjusted-close prices
# --------------------------------------------------------------------------
def _to_epoch(date_str: str) -> int:
    d = dt.datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    return int(d.timestamp())


def _cache_path(ticker: str, start: str, end: str) -> str:
    return os.path.join(CACHE_DIR, f"{ticker}_{start}_{end}.csv")


def _fetch_from_yahoo(ticker: str, start: str, end: str):
    """Return list of (date_str, adjclose) for one ticker from Yahoo."""
    url = _YAHOO.format(ticker=ticker)
    params = {"period1": _to_epoch(start), "period2": _to_epoch(end),
              "interval": "1d"}
    r = requests.get(url, params=params,
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    result = r.json()["chart"]["result"][0]
    timestamps = result["timestamp"]
    adj = result["indicators"]["adjclose"][0]["adjclose"]
    rows = []
    for ts, price in zip(timestamps, adj):
        if price is None:
            continue
        date = dt.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
        rows.append((date, float(price)))
    return rows


def load_prices(tickers, start: str, end: str):
    """
    Load {ticker: [(date, adjclose), ...]} for all tickers, cache-first.

    Downloads any ticker not already cached, then reads everything from disk.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    prices = {}
    for ticker in tickers:
        path = _cache_path(ticker, start, end)
        if not os.path.exists(path):
            rows = _fetch_from_yahoo(ticker, start, end)
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["date", "adjclose"])
                writer.writerows(rows)
            time.sleep(0.3)   # be polite to the free endpoint
        with open(path) as f:
            reader = csv.reader(f)
            next(reader)      # skip header
            prices[ticker] = [(d, float(p)) for d, p in reader]
    return prices


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
