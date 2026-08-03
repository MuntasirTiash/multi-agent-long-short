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

# ---- sparse: how many peers, absolutely or as a share of the cross-section ----
# "absolute" uses SPARSE_DEGREE (the legacy behaviour, so old runs reproduce);
# "pct" uses SPARSE_PCT * (N-1), which keeps the share of the universe constant
# as the universe grows. A fixed degree silently changes the experiment: 3 peers
# is 10% of the Dow-30 but 0.6% of 498 names. Percentage mode is also how a
# DEGREE-MATCHED random control is built — see TOPOLOGY_PLAN.md section 3, since
# comparing full(497 peers) vs sector(54) vs sparse(3) otherwise confounds
# structure with degree.
SPARSE_MODE = "pct"
SPARSE_PCT = 0.10                 # 10% of the cross-section => 50 peers at N=498
SPARSE_DEGREE = 3                 # used when SPARSE_MODE == "absolute"

# Symmetrise the graph (if A hears B, B hears A). `sparse` and `corr_topk` are
# naturally DIRECTED — out-degree is exactly k but in-degree varies, so some
# agents are read by nobody — while `sector`/`full`/`corr_threshold` are
# symmetric. Set True to remove that confound; note it roughly doubles mean
# degree, and hence prompt cost.
SPARSE_SYMMETRIC = False

# ---- correlation-based topologies (corr_topk | corr_threshold | corr_anti) ----
# Agents are wired by return co-movement rather than by sector: the market's own
# notion of "peer" instead of the classification agency's.
CORR_WINDOW = 60                  # trailing trading days of returns
CORR_REFRESH = 21                 # rebuild the graph every N rebalance dates
                                  #   1 = daily rewiring, 21 = ~monthly.
                                  #   BOTH are experimental arms (see
                                  #   TOPOLOGY_PLAN.md section 8), not a
                                  #   cost compromise: measured ~8 min vs ~25 s
                                  #   of correlation over a 373-date run.
CORR_ABS = False                  # False: signed (co-movers only).
                                  # True: |corr|, so strongly NEGATIVELY
                                  # correlated peers connect too — arguably more
                                  # informative for a long/short book. Both are
                                  # experimental arms; run each.
CORR_TOP_K = 10                   # corr_topk / corr_anti: peers per agent.
                                  # Fixed degree, so it can be degree-matched.
CORR_THRESHOLD = 0.5              # corr_threshold: minimum correlation for an
                                  # edge. UNIVERSE-DEPENDENT — calibrate against
                                  # the measured degree distribution before
                                  # trusting it (`python topology_report.py`),
                                  # exactly like LONG_SCORE_THRESHOLD.
CORR_MAX_DEGREE = 50              # cap for corr_threshold, whose degree is
                                  # otherwise unbounded and uneven

# Drop edges between share classes of the SAME company (GOOG/GOOGL, FOX/FOXA,
# NWS/NWSA correlate at 0.99+ precisely because they are one issuer). Without
# this a correlation graph always pairs them, and the "peer opinion" is that
# company's own view echoed back rather than external information.
CORR_EXCLUDE_SAME_ISSUER = True

# How many revision rounds happen after the first independent assessment.
# The literature suggests 2-3 rounds is the sweet spot (see gap analysis).
N_ROUNDS = 2

# --------------------------------------------------------------------------
# Hierarchical industry-leader setting (see hierarchy.py, TOPOLOGY_PLAN.md T6)
#
# Tier 1 = all stock agents; tier 2 = one leader per industry, which reads its
# whole industry's opinions (INCLUDING ITS OWN) and nominates longs/shorts; the
# leaders then talk to each other (the council) before tier 3, the
# IndustryManagerAgent, builds the book.
# --------------------------------------------------------------------------
HIERARCHY_GROUPING = "sector"      # "sector" (11 groups, 21-79 firms) is the
                                   # usable level. "sub_industry" has 127 groups
                                   # averaging 3.9 firms with 26 SINGLETONS that
                                   # would each be their own leaderless leader.

LEADER_METRIC = "dollar_volume"    # how the leader is chosen, all point-in-time:
                                   #   "market_cap"    the real definition;
                                   #                   needs fetch_market_cap.py
                                   #   "dollar_volume" proxy, no extra data
                                   #   "random"        the CONTROL that separates
                                   #                   "the biggest firm knows
                                   #                   something" from "any
                                   #                   aggregator helps"
LEADER_SIZE_WINDOW = 20            # trailing days for the dollar-volume metric
LEADER_PICKS_PER_SIDE = 3          # longs/shorts each leader nominates
LEADER_TABLE_CAP = 40              # max industry members shown in one leader
                                   # prompt (the largest sector has 79; at ~35
                                   # tokens a row that crowds a 4k window, so
                                   # over the cap the extremes are shown)

LEADER_COUNCIL_ROUNDS = 1          # 0 disables the council entirely
LEADER_COUNCIL_TOPOLOGY = "full"   # graph BETWEEN leaders. 11 agents, so "full"
                                   # is cheap; any topology name works.

INDUSTRY_NEUTRAL = False           # equalise each industry's contribution to the
                                   # book. Only expressible in this setting,
                                   # since positions arrive grouped by industry.

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
# A fixed seed so the sparse topology (and the rule-based demo) is deterministic.
SEED = 42
