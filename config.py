"""
Configuration for the orchestration harness.

Everything you might want to tweak lives here so the other modules stay clean.
No secrets, no API keys — this project is zero-cost by design (see README).

The universe is now READ FROM DISK rather than hardcoded: `data/sp500/`
`sp500_ticker.csv` holds the 498 S&P-500 firms that have complete price history
over the backtest window (see `update_sp500_list.py` for how that file is built
and `download_prices.py` for the price cache). Keeping one file as the single
source of truth means the universe, the cached prices and the sector map cannot
drift apart.
"""

import csv
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------
# Universe: one StockAgent is created per ticker.
#
# UNIVERSE maps ticker -> GICS sector, because the "sector" communication
# topology wires agents to their true economic peers (topology T3 in the
# research plan).
#
# NOTE this is CURRENT index membership, not point-in-time membership, so a
# backtest over it carries survivorship bias (a known Phase-3 gap): the firms
# in the file are the ones that survived and stayed in the index.
# --------------------------------------------------------------------------
UNIVERSE_FILE = os.getenv(
    "UNIVERSE_FILE", os.path.join(_HERE, "data", "sp500", "sp500_ticker.csv"))

# The 30 DJIA names the scaffold originally used. Kept so the earlier
# experiments stay reproducible — select them with UNIVERSE=dow30. Sectors come
# from the same CSV either way, so the labels are consistent across both modes.
DOW30 = ["AAPL", "MSFT", "IBM", "CSCO", "CRM", "NVDA", "V", "JPM", "GS", "AXP",
         "TRV", "JNJ", "UNH", "MRK", "AMGN", "HD", "MCD", "NKE", "AMZN", "DIS",
         "PG", "KO", "WMT", "CVX", "CAT", "BA", "HON", "MMM", "VZ", "DOW"]


def load_universe(path: str = None, subset: str = None) -> dict:
    """
    Read {ticker: sector} from a Symbol,Name,Sector CSV.

    `subset` selects a named slice: "dow30" for the original 30 DJIA names,
    anything else (or None) for the whole file. Raises rather than returning an
    empty universe, because a silent empty dict would make every downstream
    metric quietly meaningless.
    """
    path = path or UNIVERSE_FILE
    if not os.path.exists(path):
        raise SystemExit(
            f"Universe file not found: {path}\n"
            f"Build it with:  python update_sp500_list.py\n"
            f"then fetch prices with:  python download_prices.py")
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("Symbol", "").strip()]
    universe = {r["Symbol"].strip(): r.get("Sector", "").strip() for r in rows}
    if not universe:
        raise SystemExit(f"Universe file has no rows: {path}")

    if (subset or "").lower() == "dow30":
        missing = [t for t in DOW30 if t not in universe]
        if missing:
            raise SystemExit(f"dow30 subset requested but {missing} are not in "
                             f"{path}")
        return {t: universe[t] for t in DOW30}
    return universe


# UNIVERSE=dow30 falls back to the original 30-name book; default is all 498.
UNIVERSE = load_universe(subset=os.getenv("UNIVERSE"))

# --------------------------------------------------------------------------
# Portfolio construction
#
# BOOK_SIZING decides who owns the number of positions:
#
#   "fixed"    always exactly N_LONG / N_SHORT names (the original behaviour).
#   "flexible" the ManagerAgent decides, within [MIN_POSITIONS, MAX_POSITIONS]
#              per side. Its rule brain takes every name whose score clears
#              LONG_SCORE_THRESHOLD / SHORT_SCORE_THRESHOLD, so the book widens
#              when many names look attractive and narrows when few do; its LLM
#              brain is handed the same bounds and picks its own count.
#
# This only affects the ManagerAgent, i.e. only run_daily_backtest.py.
# run_analysis.py and run_llm_analysis.py always grade fixed top-N/bottom-N
# books (and draw their Monte-Carlo null with the same fixed N), so they keep
# using N_LONG / N_SHORT directly.
# --------------------------------------------------------------------------
BOOK_SIZING = "flexible"

# Used when BOOK_SIZING == "fixed", and by the two walk-forward scripts. These
# are absolute counts, not fractions, so widening the universe makes a fixed book
# MORE concentrated: 5-and-5 out of 498 names is the top and bottom ~1%, versus
# the top/bottom sixth of the Dow-30.
N_LONG = 5      # how many top-ranked names go into the long book
N_SHORT = 5     # how many bottom-ranked names go into the short book

# Bounds for the flexible book, per side. MIN keeps the book two-sided (a
# one-sided book is a directional bet, not the market-neutral book we grade);
# MAX ~= a decile of the 498-name universe.
MIN_POSITIONS = 5
MAX_POSITIONS = 50

# Conviction gates for the flexible rule brain. These have to be set RELATIVE TO
# THE UNIVERSE SIZE to mean anything: 60/40 (the cut-offs
# StockAgent._direction_from_score uses for its LONG/SHORT labels) admit 86-197
# of the 498 names on a typical date, which is always above MAX_POSITIONS, so the
# cap would bind every day and the book would be a fixed 50-and-50. At 75/25 the
# qualifying count runs ~11-91 (mean ~37), so the gate is what usually decides
# the size and the book genuinely breathes with the cross-section.
LONG_SCORE_THRESHOLD = 75.0
SHORT_SCORE_THRESHOLD = 25.0

# How many candidates per side the LLM manager is shown. The full universe does
# not fit a small model's context (498 analyst rows is ~12k tokens against
# vLLM's 4096-token default), so the manager reviews the strongest names from
# each end of the ranking.
MANAGER_SHORTLIST = 30

# --------------------------------------------------------------------------
# Features the agents reason over
#
#   "basic"    data_loader.compute_features: 63-day momentum + 5-day reversal,
#              tanh-z-scored. The original two-signal prompt.
#   "extended" features.py: the same two PLUS trend (SMA distance, MACD,
#              Bollinger), RSI, high/low volatility (Parkinson, ATR), realised
#              vol, drawdown, skew, volume shock, Amihud illiquidity, the
#              overnight/intraday split, dividend yield, and the cross-sectional
#              set (sector-relative momentum, beta, idiosyncratic vol, residual
#              momentum, sector correlation).
#
# `momentum` and `value` are bit-identical in both, so the rule-based score
# (StockAgent._score_from_data, still 0.6*momentum + 0.4*value) does not change
# and earlier rule-based results stay comparable. The extra features reach the
# model through the prompt only — which is also why this costs tokens: the brief
# grows from ~60 to ~135 tokens per stock per round.
# --------------------------------------------------------------------------
FEATURE_SET = "extended"

# --------------------------------------------------------------------------
# Communication
# --------------------------------------------------------------------------
# Which wiring pattern connects the agents. One of the names understood by
# topology.build_topology(): "full", "sparse", or "sector".
#
# Universe size changes what these cost. At 498 firms "full" gives every agent
# 497 peer messages to read in one prompt, which blows past a 4k context window
# (serve_vllm_gpu.sh sets --max-model-len 4096) and costs O(N^2) messages;
# "sector" gives 20-78 peers depending on the sector. "sparse" is the only one
# whose prompt size is independent of the universe.
TOPOLOGY = "sparse"

# For the "sparse" topology: how many peers each agent talks to.
SPARSE_DEGREE = 3

# How many revision rounds happen after the first independent assessment.
# The literature suggests 2-3 rounds is the sweet spot (see gap analysis).
N_ROUNDS = 2

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
# A fixed seed so the sparse topology (and the rule-based demo) is deterministic.
SEED = 42
