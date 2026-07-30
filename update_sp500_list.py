"""
update_sp500_list.py — refresh data/sp500/sp500_ticker.csv from Wikipedia.

The checked-in ticker list goes stale: it still carried ANTM (renamed ELV in
2022), CDAY (renamed DAY), CTLT (acquired 2025) and other symbols that simply
404 at Yahoo. This script re-scrapes the current index membership from the
Wikipedia "List of S&P 500 companies" constituents table and rewrites the file,
printing a diff of what changed.

Output files (both under data/sp500/):

  sp500_ticker.csv       Symbol,Name,Sector — the schema download_prices.py and
                         any topology code reads. Unchanged contract.
  sp500_constituents.csv Symbol,Name,Sector,SubIndustry,Headquarters,DateAdded,
                         CIK — everything the table offers, for the fundamentals
                         and filings work that needs a CIK.

Symbols are normalised to Yahoo's convention (BRK.B -> BRK-B) so they can be
passed straight to the chart endpoint.

IMPORTANT — this is CURRENT membership, not point-in-time. Using it to backtest a
window that starts in 2024 introduces survivorship bias: today's list only
contains companies that survived and stayed in the index. Fixing that needs the
historical member list per rebalance date (a known Phase-3 gap).

Usage:
    python update_sp500_list.py                 # fetch, diff, rewrite
    python update_sp500_list.py --dry-run       # show the diff, write nothing
    python update_sp500_list.py --html page.html  # parse a saved page instead
"""

import argparse
import csv
import os
import re
import shutil
from html.parser import HTMLParser

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
OUT_DIR = os.path.join("data", "sp500")
TICKER_FILE = os.path.join(OUT_DIR, "sp500_ticker.csv")
FULL_FILE = os.path.join(OUT_DIR, "sp500_constituents.csv")


# --------------------------------------------------------------------------
# Parse the constituents table
# --------------------------------------------------------------------------
class _ConstituentsParser(HTMLParser):
    """
    Pull rows out of the first table whose id is "constituents".

    Written against the stdlib rather than pandas/bs4 to keep this project's
    zero-third-party-dependency-in-the-core convention (requests aside).
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self._in_table = False
        self._depth = 0
        self._row = None
        self._cell = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "table":
            if not self._in_table and attrs.get("id") == "constituents":
                self._in_table = True
                self._depth = 1
            elif self._in_table:
                self._depth += 1          # a nested table inside a cell
        if not self._in_table:
            return
        if tag == "tr" and self._depth == 1:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if not self._in_table:
            return
        if tag == "table":
            self._depth -= 1
            if self._depth == 0:
                self._in_table = False
        elif tag in ("td", "th") and self._cell is not None:
            text = re.sub(r"\s+", " ", "".join(self._cell)).strip()
            self._row.append(text)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def yahoo_symbol(symbol: str) -> str:
    """Yahoo writes share classes with a dash: BRK.B -> BRK-B."""
    return symbol.strip().upper().replace(".", "-")


def parse_constituents(html: str):
    """Return a list of dicts, one per index member."""
    parser = _ConstituentsParser()
    parser.feed(html)
    rows = parser.rows
    if not rows:
        raise SystemExit("Could not find the 'constituents' table — did the "
                         "page layout change?")
    header = [h.lower() for h in rows[0]]

    def idx(*names, default=None):
        for name in names:
            for i, h in enumerate(header):
                if h.startswith(name):
                    return i
        return default

    i_sym = idx("symbol")
    i_name = idx("security", "company")
    i_sec = idx("gics sector", "sector")
    i_sub = idx("gics sub", "sub-industry")
    i_hq = idx("headquarters")
    i_added = idx("date added", "date first added")
    i_cik = idx("cik")
    if i_sym is None or i_name is None or i_sec is None:
        raise SystemExit(f"Unexpected table header: {rows[0]}")

    out = []
    for row in rows[1:]:
        if len(row) <= max(i_sym, i_name, i_sec):
            continue                      # a malformed/spanning row
        symbol = yahoo_symbol(row[i_sym])
        if not re.fullmatch(r"[A-Z][A-Z0-9-]*", symbol):
            continue
        def cell(i):
            return row[i].strip() if i is not None and i < len(row) else ""
        out.append({"Symbol": symbol, "Name": cell(i_name),
                    "Sector": cell(i_sec), "SubIndustry": cell(i_sub),
                    "Headquarters": cell(i_hq), "DateAdded": cell(i_added),
                    "CIK": cell(i_cik)})
    # De-duplicate, keeping first occurrence, then sort for a stable diff.
    seen, unique = set(), []
    for row in out:
        if row["Symbol"] in seen:
            continue
        seen.add(row["Symbol"])
        unique.append(row)
    return sorted(unique, key=lambda r: r["Symbol"])


# --------------------------------------------------------------------------
# Diff + write
# --------------------------------------------------------------------------
def read_existing(path):
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {r["Symbol"].strip(): r for r in csv.DictReader(f)
                if r.get("Symbol", "").strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", help="parse a saved HTML file instead of fetching")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.html:
        with open(args.html, encoding="utf-8") as f:
            html = f.read()
    else:
        import requests
        r = requests.get(WIKI_URL, headers={"User-Agent": "Mozilla/5.0"},
                         timeout=60)
        r.raise_for_status()
        html = r.text

    members = parse_constituents(html)
    print(f"parsed {len(members)} constituents from "
          f"{args.html or 'Wikipedia'}")

    old = read_existing(TICKER_FILE)
    new_syms = {m["Symbol"] for m in members}
    added = sorted(new_syms - set(old))
    removed = sorted(set(old) - new_syms)
    resector = sorted(
        (s, old[s].get("Sector", ""), m["Sector"])
        for s in (new_syms & set(old))
        for m in [next(x for x in members if x["Symbol"] == s)]
        if old[s].get("Sector", "").strip() != m["Sector"].strip())

    print(f"\nwas {len(old)} symbols -> now {len(new_syms)}")
    print(f"  added   ({len(added)}): {', '.join(added) or '-'}")
    print(f"  removed ({len(removed)}): {', '.join(removed) or '-'}")
    if resector:
        print(f"  sector renamed/changed ({len(resector)}):")
        for sym, was, now in resector:
            print(f"      {sym}: {was!r} -> {now!r}")

    if args.dry_run:
        print("\n--dry-run: no files written")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(TICKER_FILE):
        backup = TICKER_FILE + ".bak"
        shutil.copy2(TICKER_FILE, backup)
        print(f"\nbacked up previous list -> {backup}")

    with open(TICKER_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Symbol", "Name", "Sector"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(members)
    with open(FULL_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Symbol", "Name", "Sector",
                                          "SubIndustry", "Headquarters",
                                          "DateAdded", "CIK"])
        w.writeheader()
        w.writerows(members)
    print(f"wrote {TICKER_FILE} ({len(members)} rows)")
    print(f"wrote {FULL_FILE} (full metadata incl. CIK)")
    if added:
        print(f"\nNext: fetch prices for the new names —\n"
              f"  python download_prices.py --skip-existing")


if __name__ == "__main__":
    main()
