"""
Communication topology = who talks to whom.

A topology is just a dictionary mapping each ticker to the list of tickers it
is allowed to hear from:

    {"AAPL": ["MSFT", "IBM"], "MSFT": ["AAPL", "NVDA"], ...}

Why this matters (and why it is a research variable, not a detail): the papers
show that connecting *every* agent to every other agent is both expensive
(O(N^2) messages) and often *worse* than a sparse graph, because one agent's
mistake spreads to everyone. So we make the wiring pluggable and will compare
several patterns later. Phase 0 ships three simple ones.
"""

import random
from typing import Dict, List


def build_topology(name: str, tickers: List[str], sectors: Dict[str, str],
                   degree: int = 3, seed: int = 42) -> Dict[str, List[str]]:
    """Dispatch to the requested topology builder by name."""
    if name == "full":
        return full_topology(tickers)
    if name == "sparse":
        return sparse_topology(tickers, degree=degree, seed=seed)
    if name == "sector":
        return sector_topology(tickers, sectors)
    raise ValueError(f"Unknown topology: {name!r} "
                     f"(expected 'full', 'sparse', or 'sector')")


def full_topology(tickers: List[str]) -> Dict[str, List[str]]:
    """Everyone hears everyone else. Simple, expensive, our upper-cost baseline."""
    return {t: [o for o in tickers if o != t] for t in tickers}


def sparse_topology(tickers: List[str], degree: int = 3,
                    seed: int = 42) -> Dict[str, List[str]]:
    """
    Each agent hears from `degree` randomly chosen peers.

    Uses Python's standard `random` module, seeded for reproducibility. This is
    the pattern the topology papers found to be a strong, cheap default.
    """
    rng = random.Random(seed)
    graph: Dict[str, List[str]] = {}
    for t in tickers:
        others = [o for o in tickers if o != t]
        k = min(degree, len(others))
        graph[t] = rng.sample(others, k)
    return graph


def sector_topology(tickers: List[str],
                    sectors: Dict[str, str]) -> Dict[str, List[str]]:
    """
    Each agent hears only from peers in the SAME sector.

    This is the finance-native idea: a stock is best judged against its true
    economic peers (compare AAPL to MSFT, not to KO).
    """
    graph: Dict[str, List[str]] = {}
    for t in tickers:
        graph[t] = [o for o in tickers
                    if o != t and sectors[o] == sectors[t]]
    return graph
