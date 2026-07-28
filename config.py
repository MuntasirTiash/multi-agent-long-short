"""
Configuration for the Phase 0 orchestration scaffold.

Everything you might want to tweak lives here so the other modules stay clean.
No secrets, no API keys — this project is zero-cost by design (see README).
"""

# --------------------------------------------------------------------------
# Universe: the DJIA 30. One StockAgent is created per ticker.
#
# We keep a (ticker -> GICS-style sector) map because one of our planned
# communication topologies wires agents together *by sector* (topology T3 in
# the research plan). In Phase 1 this list must become POINT-IN-TIME (the exact
# index members on each historical date) to avoid survivorship/look-ahead bias.
# For Phase 0 it is just a fixed, illustrative list.
# --------------------------------------------------------------------------
UNIVERSE = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "IBM": "Technology",
    "CSCO": "Technology",
    "CRM": "Technology",
    "NVDA": "Technology",
    "V": "Financials",
    "JPM": "Financials",
    "GS": "Financials",
    "AXP": "Financials",
    "TRV": "Financials",
    "JNJ": "Healthcare",
    "UNH": "Healthcare",
    "MRK": "Healthcare",
    "AMGN": "Healthcare",
    "HD": "Consumer",
    "MCD": "Consumer",
    "NKE": "Consumer",
    "AMZN": "Consumer",
    "DIS": "Consumer",
    "PG": "Staples",
    "KO": "Staples",
    "WMT": "Staples",
    "CVX": "Energy",
    "CAT": "Industrials",
    "BA": "Industrials",
    "HON": "Industrials",
    "MMM": "Industrials",
    "VZ": "Communications",
    "DOW": "Materials",
}

# --------------------------------------------------------------------------
# Portfolio construction
# --------------------------------------------------------------------------
N_LONG = 5      # how many top-ranked names go into the long book
N_SHORT = 5     # how many bottom-ranked names go into the short book

# --------------------------------------------------------------------------
# Communication
# --------------------------------------------------------------------------
# Which wiring pattern connects the agents. One of the names understood by
# topology.build_topology(): "full", "sparse", or "sector".
TOPOLOGY = "sparse"

# For the "sparse" topology: how many peers each agent talks to.
SPARSE_DEGREE = 3

# How many revision rounds happen after the first independent assessment.
# The literature suggests 2-3 rounds is the sweet spot (see gap analysis).
N_ROUNDS = 2

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
# A fixed seed so the (currently rule-based, randomised) demo is deterministic.
SEED = 42
