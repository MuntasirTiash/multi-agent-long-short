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


NAMES = ("full", "sparse", "sector", "corr_topk", "corr_threshold", "corr_anti")

# The correlation-based topologies need a per-date correlation ranking, which
# lives in correlation.py (see correlation.make_topology_fn — that is what the
# run scripts call, and it handles the window, the refresh cadence and caching).
CORR_NAMES = ("corr_topk", "corr_threshold", "corr_anti")


def build_topology(name: str, tickers: List[str], sectors: Dict[str, str],
                   degree: int = 3, seed: int = 42, pct: float = None,
                   symmetric: bool = False,
                   ranking: Dict[str, List] = None, threshold: float = 0.5,
                   top_k: int = 10, max_degree: int = None
                   ) -> Dict[str, List[str]]:
    """
    Dispatch to the requested topology builder by name.

    `pct`/`symmetric` apply to "sparse"; `ranking`/`threshold`/`top_k`/
    `max_degree` to the corr_* family. `ranking` is
    {ticker: [(peer, corr), ...]} sorted most-correlated first, as produced by
    correlation.RollingCorrelation.ranking().
    """
    if name == "full":
        return full_topology(tickers)
    if name == "sparse":
        return sparse_topology(tickers, degree=degree, seed=seed, pct=pct,
                               symmetric=symmetric)
    if name == "sector":
        return sector_topology(tickers, sectors)
    if name in CORR_NAMES:
        if ranking is None:
            raise ValueError(
                f"topology {name!r} needs a correlation ranking — build it with "
                f"correlation.make_topology_fn(...) rather than calling "
                f"build_topology directly.")
        return correlation_topology(tickers, ranking, mode=name,
                                    threshold=threshold, top_k=top_k,
                                    max_degree=max_degree, symmetric=symmetric)
    raise ValueError(f"Unknown topology: {name!r} (expected one of {NAMES})")


def full_topology(tickers: List[str]) -> Dict[str, List[str]]:
    """Everyone hears everyone else. Simple, expensive, our upper-cost baseline."""
    return {t: [o for o in tickers if o != t] for t in tickers}


def resolve_degree(n_tickers: int, degree: int = None, pct: float = None,
                   min_degree: int = 1) -> int:
    """
    Peers per agent, either as an absolute count or a FRACTION of the universe.

    A fixed degree means a different experiment as the universe grows: 3 peers is
    10% of the Dow-30 but 0.6% of the 498-name cross-section. `pct` keeps the
    share of the cross-section constant instead, which is what makes a
    degree-matched comparison against a structural topology possible (a random
    control with the same mean degree as `sector`, say).
    """
    if pct is not None:
        return max(min_degree, min(n_tickers - 1, round(pct * (n_tickers - 1))))
    return max(0, min(n_tickers - 1, degree if degree is not None else 3))


def symmetrise(graph: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Make the graph undirected by union: if A hears B, B also hears A.

    Worth knowing why this exists. `sparse` samples each agent's peers
    independently, so it is DIRECTED — out-degree is exactly k, but in-degree is
    ~Poisson(k), meaning some agents are read by nobody and others by many.
    `sector` and `full` are symmetric. That difference is a confound in any
    sparse-vs-sector comparison, so this lets the graphs be matched on it.
    Symmetrising raises the mean degree (roughly doubling it for small k).
    """
    out = {t: set(peers) for t, peers in graph.items()}
    for t, peers in graph.items():
        for p in peers:
            if p in out:
                out[p].add(t)
    return {t: sorted(peers) for t, peers in out.items()}


def sparse_topology(tickers: List[str], degree: int = 3,
                    seed: int = 42, pct: float = None,
                    symmetric: bool = False) -> Dict[str, List[str]]:
    """
    Each agent hears from `degree` randomly chosen peers (or `pct` of the
    universe — see resolve_degree).

    Uses Python's standard `random` module, seeded for reproducibility. This is
    the pattern the topology papers found to be a strong, cheap default, and with
    `pct` it doubles as the degree-matched random control for the structural
    topologies.
    """
    rng = random.Random(seed)
    k = resolve_degree(len(tickers), degree=degree, pct=pct)
    graph: Dict[str, List[str]] = {}
    for t in tickers:
        others = [o for o in tickers if o != t]
        graph[t] = rng.sample(others, min(k, len(others)))
    return symmetrise(graph) if symmetric else graph


def correlation_topology(tickers: List[str], ranking: Dict[str, List],
                         mode: str = "corr_topk", threshold: float = 0.5,
                         top_k: int = 10, max_degree: int = None,
                         symmetric: bool = False) -> Dict[str, List[str]]:
    """
    Wire agents by return co-movement. Three modes off the same ranking:

      corr_topk       the `top_k` most correlated peers. Degree is FIXED, which
                      is why this is the primary correlation experiment: it can
                      be degree-matched against sparse/sector, so a difference is
                      attributable to *which* peers rather than how many.
      corr_threshold  every peer with correlation >= `threshold`. Degree is
                      uncontrolled and varies per agent and over time — a
                      tightly co-moving utility gets dozens of neighbours while an
                      idiosyncratic name gets none. That unevenness is itself a
                      finding, but it confounds structure with degree and makes
                      prompt length (hence cost) vary per agent, so `max_degree`
                      caps it.
      corr_anti       the `top_k` LEAST correlated peers. The rival hypothesis:
                      correlated peers largely tell you what you already know,
                      so diverse peers may carry more information for a
                      long/short book.

    `ranking` is {ticker: [(peer, corr), ...]} sorted most-correlated first;
    whether that correlation is signed or absolute is decided upstream in
    correlation.py (both are experimental arms).

    A ticker with no usable return history gets an empty peer list; the
    schedulers already leave such an agent on its current view.
    """
    graph: Dict[str, List[str]] = {}
    for t in tickers:
        ranked = ranking.get(t) or []
        if mode == "corr_topk":
            peers = [p for p, _ in ranked[:top_k]]
        elif mode == "corr_anti":
            peers = [p for p, _ in ranked[-top_k:]] if ranked else []
        elif mode == "corr_threshold":
            peers = [p for p, c in ranked if c >= threshold]
            if max_degree is not None:
                peers = peers[:max_degree]     # ranked, so this keeps the best
        else:
            raise ValueError(f"unknown correlation mode {mode!r}")
        graph[t] = peers
    return symmetrise(graph) if symmetric else graph


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
