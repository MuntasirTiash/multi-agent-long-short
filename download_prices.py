"""
download_prices.py — populate the price cache for a whole universe, in one go.

`data_loader.load_prices` downloads lazily, one ticker at a time, and stores only
the adjusted close. That is fine for the Dow-30 but two things break when moving
to the S&P 500:

  * a lazy download from a compute node is fragile (500 sequential fetches inside
    a job that is meant to be doing research), and
  * adjusted close alone rules out volume-, range- and gap-based features.

So this script front-loads the whole download from a node that has internet, and
stores every per-bar field Yahoo's chart endpoint returns:

    date,open,high,low,close,adjclose,volume,dividend,split_ratio

`open/high/low/close/volume` are the raw (unadjusted) bar; `adjclose` is the
split- and dividend-adjusted close and is the only column `data_loader` reads
today, so old callers are unaffected. `dividend` and `split_ratio` come from the
endpoint's `events` block and are 0.0 on days with no corporate action — enough
to reconstruct an adjustment factor if a feature ever needs raw prices.

Usage (run on a node with internet — the NJIT login node, or a compute node that
still has egress):

    python download_prices.py                      # S&P 500, default window
    python download_prices.py --purge              # wipe the cache dir first
    python download_prices.py --tickers AAPL,MSFT  # ad-hoc list
    python download_prices.py --start 2020-01-01 --end 2026-07-01

The cache filename embeds the window (`{TICKER}_{START}_{END}.csv`), which is
what `data_loader._cache_path` looks for, so the --start/--end you download with
must match the START/END constants in the run_* scripts.
"""

import argparse
import csv
import datetime as dt
import logging
import os
import time

import requests

# Same window as the run_* scripts (post-Qwen2.5-cutoff, leakage-safe).
DEFAULT_START, DEFAULT_END = "2024-10-01", "2026-07-01"
DEFAULT_TICKER_FILE = os.path.join("data", "sp500", "sp500_ticker.csv")
DEFAULT_CACHE_DIR = os.path.join("data", "price_cache")

_YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
_HEADERS = {"User-Agent": "Mozilla/5.0"}      # a bare UA gets HTTP 429
FIELDS = ["date", "open", "high", "low", "close", "adjclose", "volume",
          "dividend", "split_ratio"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("download")


# --------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------
def read_tickers(path: str):
    """Read the Symbol column of a ticker CSV (Symbol,Name,Sector)."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    key = "Symbol" if rows and "Symbol" in rows[0] else list(rows[0].keys())[0]
    return [r[key].strip() for r in rows if r[key].strip()]


def read_sectors(path: str):
    """{symbol: sector} from the same file, for downstream topology work."""
    with open(path, newline="") as f:
        return {r["Symbol"].strip(): r.get("Sector", "").strip()
                for r in csv.DictReader(f) if r["Symbol"].strip()}


# --------------------------------------------------------------------------
# Download one ticker
# --------------------------------------------------------------------------
def _to_epoch(date_str: str) -> int:
    d = dt.datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    return int(d.timestamp())


def _day(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%d")


def _events_by_day(events: dict, kind: str, field: str):
    """{date: amount} for dividends / splits, keyed by UTC day."""
    out = {}
    for ev in (events or {}).get(kind, {}).values():
        try:
            out[_day(int(ev["date"]))] = float(ev[field])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def fetch(ticker: str, start: str, end: str, tries: int = 4):
    """
    Return a list of row dicts (one per trading day) for `ticker`.

    Retries on 429/5xx with exponential backoff; raises on final failure so the
    caller can record the ticker as missing and keep going.
    """
    params = {"period1": _to_epoch(start), "period2": _to_epoch(end),
              "interval": "1d", "events": "div,splits"}
    last = None
    for attempt in range(tries):
        try:
            r = requests.get(_YAHOO.format(ticker=ticker), params=params,
                             headers=_HEADERS, timeout=30)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            if 400 <= r.status_code < 500:
                # A delisted/renamed symbol 404s forever — don't burn backoff.
                raise RuntimeError(f"{ticker}: HTTP {r.status_code} "
                                   f"(symbol not on Yahoo?)")
            r.raise_for_status()
            return _parse(r.json())
        except RuntimeError:
            raise
        except Exception as e:                 # network, HTTP, or shape problem
            last = e
            if attempt < tries - 1:
                wait = 2.0 * (2 ** attempt)    # 2s, 4s, 8s
                log.warning("  %s: %s — retrying in %.0fs", ticker, e, wait)
                time.sleep(wait)
    raise RuntimeError(f"{ticker}: gave up after {tries} tries ({last})")


def _parse(payload: dict):
    """Chart JSON -> list of row dicts. Bars with no close are dropped."""
    result = payload["chart"]["result"][0]
    stamps = result.get("timestamp") or []
    quote = (result["indicators"].get("quote") or [{}])[0]
    adj_block = (result["indicators"].get("adjclose") or [{}])[0]
    adj = adj_block.get("adjclose") or [None] * len(stamps)

    divs = _events_by_day(result.get("events"), "dividends", "amount")
    splits = _events_by_day(result.get("events"), "splits", "numerator")
    # Splits report numerator/denominator; the ratio is what matters.
    for ev in (result.get("events") or {}).get("splits", {}).values():
        try:
            day = _day(int(ev["date"]))
            splits[day] = float(ev["numerator"]) / float(ev["denominator"])
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue

    def col(name):
        return quote.get(name) or [None] * len(stamps)

    o, h, l, c, v = (col(k) for k in ("open", "high", "low", "close", "volume"))
    rows = []
    for i, ts in enumerate(stamps):
        close = c[i] if i < len(c) else None
        adjclose = adj[i] if i < len(adj) else None
        if close is None and adjclose is None:
            continue                      # a genuinely empty bar (halt/holiday)
        day = _day(ts)
        rows.append({
            "date": day,
            "open": o[i], "high": h[i], "low": l[i], "close": close,
            "adjclose": adjclose if adjclose is not None else close,
            "volume": v[i],
            "dividend": divs.get(day, 0.0),
            "split_ratio": splits.get(day, 0.0),
        })
    return rows


def write_csv(path: str, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if row.get(k) is None else row.get(k))
                        for k in FIELDS})


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def purge(cache_dir: str):
    """Delete every cached price CSV (they are all re-downloadable)."""
    if not os.path.isdir(cache_dir):
        return 0
    n = 0
    for name in sorted(os.listdir(cache_dir)):
        if name.endswith(".csv"):
            os.remove(os.path.join(cache_dir, name))
            n += 1
    log.info("purged %d existing CSV file(s) from %s", n, cache_dir)
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--tickers", default=DEFAULT_TICKER_FILE,
                    help="ticker CSV path, or a comma-separated list of symbols")
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--purge", action="store_true",
                    help="delete existing CSVs in the cache dir first")
    ap.add_argument("--sleep", type=float, default=0.25,
                    help="pause between tickers (be polite to the free endpoint)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="leave already-cached tickers alone (resume a run)")
    args = ap.parse_args()

    if os.path.exists(args.tickers):
        tickers = read_tickers(args.tickers)
        source = args.tickers
    else:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        source = "command line"
    os.makedirs(args.cache_dir, exist_ok=True)
    if args.purge:
        purge(args.cache_dir)

    log.info("%d tickers from %s -> %s (%s .. %s)", len(tickers), source,
             args.cache_dir, args.start, args.end)

    ok, failed, skipped = [], [], []
    for i, ticker in enumerate(tickers, 1):
        path = os.path.join(args.cache_dir,
                            f"{ticker}_{args.start}_{args.end}.csv")
        if args.skip_existing and os.path.exists(path):
            # Still describe it, so the report is a manifest of the whole
            # universe rather than only of this invocation's downloads.
            with open(path, newline="") as f:
                cached = [r["date"] for r in csv.DictReader(f)]
            skipped.append((ticker, len(cached),
                            cached[0] if cached else "",
                            cached[-1] if cached else ""))
            continue
        try:
            rows = fetch(ticker, args.start, args.end)
        except Exception as e:
            log.error("[%d/%d] %-6s FAILED: %s", i, len(tickers), ticker, e)
            failed.append((ticker, str(e)))
            continue
        if not rows:
            log.error("[%d/%d] %-6s FAILED: no bars returned", i, len(tickers),
                      ticker)
            failed.append((ticker, "no bars returned"))
            continue
        write_csv(path, rows)
        ok.append((ticker, len(rows), rows[0]["date"], rows[-1]["date"]))
        if i % 25 == 0 or i == len(tickers):
            log.info("[%d/%d] %-6s %d bars %s .. %s", i, len(tickers), ticker,
                     len(rows), rows[0]["date"], rows[-1]["date"])
        time.sleep(args.sleep)

    # ---- report ---------------------------------------------------------
    present = ok + skipped                 # everything that now has a CSV
    bar_counts = sorted(n for _, n, _, _ in present)
    log.info("done: %d downloaded, %d failed, %d already cached",
             len(ok), len(failed), len(skipped))
    if bar_counts:
        short = [(t, n) for t, n, _, _ in present if n < bar_counts[-1]]
        log.info("bars per ticker: max=%d, min=%d; %d ticker(s) have a SHORT "
                 "history (listed mid-window)", bar_counts[-1], bar_counts[0],
                 len(short))
        if short:
            log.info("shortest: %s", ", ".join(
                f"{t}({n})" for t, n in sorted(short, key=lambda x: x[1])[:12]))
            log.warning("data_loader.common_dates intersects dates across ALL "
                        "tickers, so these truncate the shared calendar — drop "
                        "them or widen the window before using the full universe.")
    if failed:
        log.warning("%d failed (delisted/renamed symbols 404): %s", len(failed),
                    ", ".join(t for t, _ in failed))

    report = os.path.join(os.path.dirname(args.cache_dir) or ".",
                          "download_report.csv")
    with open(report, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "status", "n_bars", "first_date", "last_date",
                    "error"])
        for t, n, d0, d1 in ok:
            w.writerow([t, "downloaded", n, d0, d1, ""])
        for t, n, d0, d1 in skipped:
            w.writerow([t, "cached", n, d0, d1, ""])
        for t, err in failed:
            w.writerow([t, "failed", 0, "", "", err])
    log.info("per-ticker report -> %s", report)


if __name__ == "__main__":
    main()
