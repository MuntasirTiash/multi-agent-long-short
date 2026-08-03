"""
fetch_market_cap.py — point-in-time shares outstanding from SEC XBRL.

The hierarchical setting defines an industry leader as its largest member by
market cap, which needs shares outstanding. Yahoo's quote endpoints now return
401 without a crumb, so the source here is the SEC's free XBRL API, keyed by the
CIK already stored in data/sp500/sp500_constituents.csv:

    https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/dei/
        EntityCommonStockSharesOutstanding.json

Why this source is the right one: every datapoint carries both the period it
describes (`end`) and the date it became public (`filed`). Leader selection uses
the newest count FILED on or before the rebalance date, so the leader is chosen
from information that existed then — no look-ahead on the most structurally
important variable in the setting. A current-snapshot market cap would silently
leak the future.

Output: data/shares_outstanding/{TICKER}.json
    {"ticker": "AAPL", "cik": "0000320193",
     "points": [["filed_date", shares], ...]}   # sorted by filed date

Usage (needs internet — login node, or a compute node with egress):
    python fetch_market_cap.py                      # all universe tickers
    python fetch_market_cap.py --tickers AAPL,MSFT
    python fetch_market_cap.py --skip-existing      # resume

SEC asks for a descriptive User-Agent and rate-limits to ~10 requests/second;
this stays well under that.
"""

import argparse
import csv
import json
import logging
import os
import time

import requests

OUT_DIR = os.path.join("data", "shares_outstanding")
CONSTITUENTS = os.path.join("data", "sp500", "sp500_constituents.csv")
# SEC requires a real contact address in the User-Agent for API access.
HEADERS = {"User-Agent": os.getenv("SEC_USER_AGENT",
                                   "NJIT research ms3235@njit.edu")}
CONCEPTS = [
    ("dei", "EntityCommonStockSharesOutstanding"),
    # Fallbacks: some filers tag only the us-gaap concepts.
    ("us-gaap", "CommonStockSharesOutstanding"),
    ("us-gaap", "CommonStockSharesIssued"),
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("marketcap")


def read_ciks(path=CONSTITUENTS):
    """{ticker: zero-padded 10-digit CIK} from the constituents file."""
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found — run update_sp500_list.py first.")
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            cik = (row.get("CIK") or "").strip()
            sym = (row.get("Symbol") or "").strip()
            if sym and cik.isdigit():
                out[sym] = cik.zfill(10)
    return out


def fetch_facts(cik: str, tries: int = 3):
    """
    All XBRL facts for one filer, or None.

    Uses `companyfacts`, not `companyconcept`. The per-concept endpoint returns
    HTTP 200 with an EMPTY units payload for some filers — Coca-Cola is one — so
    it silently looks like "no data" when 71 datapoints exist. companyfacts is
    reliable and needs one request instead of up to three.
    """
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    for attempt in range(tries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            if r.status_code == 404:
                return None
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.json().get("facts", {})
        except Exception as e:
            if attempt == tries - 1:
                log.warning("  CIK%s failed: %s", cik, e)
                return None
            time.sleep(2.0 * (2 ** attempt))
    return None


def points_for(cik: str):
    """
    [(filed_date, shares)] sorted by filed date, newest last.

    Keyed on `filed`, not on the period end: what matters is when the number
    became knowable, since leader selection may only use what was public on the
    rebalance date. Where several datapoints share a filed date we keep the
    largest, because a 10-K/10-Q cover page reports total shares outstanding
    while other tags may cover a single class.
    """
    facts = fetch_facts(cik)
    if not facts:
        return []
    best = {}
    for taxonomy, tag in CONCEPTS:
        concept = (facts.get(taxonomy) or {}).get(tag)
        if not concept:
            continue
        for rows in (concept.get("units") or {}).values():
            for row in rows or []:
                filed, val = row.get("filed"), row.get("val")
                if not filed or val in (None, 0):
                    continue
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    continue
                if filed not in best or val > best[filed]:
                    best[filed] = val
        if best:
            break                       # first concept that yields data wins
    return sorted(best.items())


def load_shares(out_dir=OUT_DIR):
    """
    {ticker: [(filed_date, shares)]} from the cache — what LeaderSelector wants.

    Returns {} if nothing has been fetched, so callers can fall back to the
    dollar-volume proxy rather than crash.
    """
    if not os.path.isdir(out_dir):
        return {}
    out = {}
    for name in sorted(os.listdir(out_dir)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(out_dir, name)) as f:
                blob = json.load(f)
            pts = [(str(d), float(v)) for d, v in blob.get("points", [])]
            if pts:
                out[blob.get("ticker") or name[:-5]] = sorted(pts)
        except Exception as e:
            log.warning("skipping %s: %s", name, e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=None,
                    help="comma list; default = every ticker in the universe")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--sleep", type=float, default=0.15,
                    help="pause between requests (SEC allows ~10/s)")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    ciks = read_ciks()
    if args.tickers:
        wanted = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        import config
        wanted = list(config.UNIVERSE)
    os.makedirs(args.out_dir, exist_ok=True)

    ok, missing, skipped = [], [], 0
    log.info("fetching shares outstanding for %d tickers from SEC XBRL",
             len(wanted))
    for i, ticker in enumerate(wanted, 1):
        path = os.path.join(args.out_dir, f"{ticker}.json")
        if args.skip_existing and os.path.exists(path):
            skipped += 1
            continue
        cik = ciks.get(ticker)
        if not cik:
            log.warning("[%d/%d] %-6s no CIK in constituents file", i,
                        len(wanted), ticker)
            missing.append(ticker)
            continue
        pts = points_for(cik)
        if not pts:
            log.warning("[%d/%d] %-6s no shares-outstanding data", i,
                        len(wanted), ticker)
            missing.append(ticker)
            continue
        with open(path, "w") as f:
            json.dump({"ticker": ticker, "cik": cik,
                       "points": [[d, v] for d, v in pts]}, f)
        ok.append(ticker)
        if i % 50 == 0 or i == len(wanted):
            log.info("[%d/%d] %-6s %d datapoints, latest %s = %.0fM shares",
                     i, len(wanted), ticker, len(pts), pts[-1][0],
                     pts[-1][1] / 1e6)
        time.sleep(args.sleep)

    log.info("done: %d fetched, %d without data, %d already cached",
             len(ok), len(missing), skipped)
    if missing:
        log.warning("no data for: %s", ", ".join(missing))
        log.warning("those tickers fall back to the dollar-volume proxy when "
                    "LEADER_METRIC=market_cap (LeaderSelector.size -> None).")


if __name__ == "__main__":
    main()
