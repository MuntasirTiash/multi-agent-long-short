"""
correlation.py — rolling return correlations, and the topologies built from them.

The research question is whether *who talks to whom* matters. Sector membership is
one answer to "who are this stock's peers"; return co-movement is another, and it
is the market's own answer rather than a classification agency's. This module
computes the correlation cross-section point-in-time and hands `topology.py` a
ranking it can wire into a graph.

Design notes that matter:

  * POINT-IN-TIME. Correlations at date `asof` use only returns dated <= asof.
    Nothing here ever touches a forward return.

  * PURE PYTHON, and fast enough. The full 498-name matrix is 123,753 pairs;
    measured at ~1.2 s per date by normalising each return vector once and
    reducing each correlation to a dot product. A 373-date daily-rewired run
    therefore costs ~8 minutes of correlation, and a monthly-rewired one ~25 s.
    No numpy, per the project convention that the numeric core stays auditable.

  * BOTH SIGNED AND ABSOLUTE are supported (`use_abs`), because they ask
    different questions: signed connects co-movers, absolute also connects
    strongly *negatively* correlated names, whose views arguably matter more to a
    long/short book ("my hedge is rallying"). Both are experimental arms.

  * REFRESH CADENCE is a variable, not an optimisation. `refresh=1` rewires every
    rebalance date, `refresh=21` roughly monthly; both are arms of the
    experiment, so the cache exists to make the choice free rather than to force
    the cheap one.

Entry point for the run scripts:

    topo_fn = correlation.make_topology_fn(tickers, sectors, START, END)
    graph   = topo_fn("corr_topk", asof)        # or any topology name

which handles correlation setup, caching and the config knobs in one place.
"""

import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import topology as topo

log = logging.getLogger("correlation")


class RollingCorrelation:
    """
    Rolling pairwise return correlation over a trailing window.

    `returns` is {ticker: {date: daily_return}}. Build once for a whole run; call
    `ranking(asof)` per rebalance date.
    """

    def __init__(self, returns: Dict[str, Dict[str, float]], window: int = 60,
                 min_overlap: int = 20, use_abs: bool = False,
                 issuers: Optional[Dict[str, str]] = None):
        self.returns = returns
        self.window = window
        self.min_overlap = min_overlap
        self.use_abs = use_abs
        # Optional {ticker: issuer} map. Dual-class listings (GOOG/GOOGL,
        # FOX/FOXA, NWS/NWSA) correlate at 0.99+ because they ARE the same
        # company, so a correlation graph always pairs them and the "peer view"
        # is that company's own opinion echoed back. Passing issuers excludes
        # same-issuer edges so peer information is genuinely external.
        self.issuers = issuers or {}
        # One shared calendar: every ticker in the cleaned universe has the same
        # trading days, and a missing day just drops that ticker for the date.
        dates = set()
        for series in returns.values():
            dates.update(series)
        self.calendar = sorted(dates)
        self._cache: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}

    # ------------------------------------------------------------------
    def window_dates(self, asof: str) -> List[str]:
        """The last `window` trading days at or before `asof`."""
        import bisect
        end = bisect.bisect_right(self.calendar, asof)
        return self.calendar[max(0, end - self.window):end]

    def _normalised(self, tickers, asof):
        """
        Per ticker, the mean-centred return vector divided by its norm.

        With that, corr(i,j) is just dot(i,j) — which is what makes 123k pairs
        affordable in plain Python. Tickers without a full window are dropped and
        simply get no peers.
        """
        dates = self.window_dates(asof)
        if len(dates) < self.min_overlap:
            return [], {}
        vecs = {}
        for t in tickers:
            series = self.returns.get(t)
            if not series:
                continue
            vec = [series.get(d) for d in dates]
            if any(v is None for v in vec):
                continue                   # incomplete history in this window
            mean = sum(vec) / len(vec)
            centred = [v - mean for v in vec]
            norm = math.sqrt(sum(c * c for c in centred))
            if norm <= 0:                  # a flat series has no correlation
                continue
            vecs[t] = [c / norm for c in centred]
        return dates, vecs

    def ranking(self, tickers: List[str], asof: str
                ) -> Dict[str, List[Tuple[str, float]]]:
        """
        {ticker: [(peer, corr), ...]} sorted most-correlated FIRST.

        Sorted once here so every topology mode is a slice: top-k off the front,
        anti-correlation off the back, threshold a filter.
        """
        key = (f"{asof}|{len(tickers)}|{self.window}|{self.use_abs}"
               f"|{bool(self.issuers)}")
        if key in self._cache:
            return self._cache[key]

        dates, vecs = self._normalised(tickers, asof)
        names = [t for t in tickers if t in vecs]
        pairs: Dict[str, List[Tuple[str, float]]] = {t: [] for t in tickers}
        for i, a in enumerate(names):
            va = vecs[a]
            issuer_a = self.issuers.get(a)
            for b in names[i + 1:]:
                if issuer_a is not None and self.issuers.get(b) == issuer_a:
                    continue               # same company, different share class
                vb = vecs[b]
                c = sum(x * y for x, y in zip(va, vb))
                if self.use_abs:
                    c = abs(c)
                pairs[a].append((b, c))
                pairs[b].append((a, c))
        for t in pairs:
            pairs[t].sort(key=lambda x: x[1], reverse=True)
        self._cache[key] = pairs
        if len(self._cache) > 8:           # keep memory bounded on long runs
            self._cache.pop(next(iter(self._cache)))
        log.debug("correlation ranking %s: %d tickers, %d-day window",
                  asof, len(names), len(dates))
        return pairs


# ==========================================================================
# Returns source
# ==========================================================================
def issuer_map(path: str = None) -> Dict[str, str]:
    """
    {ticker: issuer} from sp500_constituents.csv, collapsing share classes.

    The constituents table names dual-class listings "Alphabet Inc. (Class A)"
    and "(Class C)", so stripping the parenthetical gives one issuer per company
    and lets same-issuer edges be excluded. Returns {} if the file is absent, in
    which case nothing is excluded.
    """
    import csv as _csv
    import re as _re
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "data", "sp500", "sp500_constituents.csv")
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, newline="") as f:
        for row in _csv.DictReader(f):
            name = _re.sub(r"\s*\(Class [^)]*\)\s*$", "",
                           (row.get("Name") or "").strip())
            sym = (row.get("Symbol") or "").strip()
            if sym and name:
                out[sym] = name
    return out


def returns_from_ohlcv(ohlcv: Dict[str, List[dict]]
                       ) -> Dict[str, Dict[str, float]]:
    """{ticker: {date: adjclose return}} — the input RollingCorrelation wants."""
    out = {}
    for ticker, rows in ohlcv.items():
        series, prev = {}, None
        for r in rows:
            px = r.get("adjclose")
            if px is not None and prev:
                series[r["date"]] = px / prev - 1.0
            if px:
                prev = px
        out[ticker] = series
    return out


# ==========================================================================
# The factory the run scripts call
# ==========================================================================
def make_topology_fn(tickers: List[str], sectors: Dict[str, str],
                     start: str, end: str, seed: int = None,
                     window: int = None, refresh: int = None,
                     use_abs: bool = None, top_k: int = None,
                     threshold: float = None, max_degree: int = None,
                     sparse_pct: float = None, sparse_degree: int = None,
                     symmetric: bool = None):
    """
    Return `fn(name, asof) -> {ticker: [peers]}` for any topology name.

    Non-correlation topologies (full/sparse/sector) are built directly. The
    corr_* family gets a `RollingCorrelation` built once here, with the graph
    rebuilt every `refresh` rebalance dates: `refresh=1` rewires daily,
    `refresh=21` roughly monthly. Both are experimental arms, so the caching is
    about making the choice cheap, not about forcing the cheap one.

    Defaults come from config, so a run script needs no knowledge of the knobs.
    """
    import config

    def cfg(value, name, default):
        return value if value is not None else getattr(config, name, default)

    seed = cfg(seed, "SEED", 42)
    window = cfg(window, "CORR_WINDOW", 60)
    refresh = max(1, cfg(refresh, "CORR_REFRESH", 21))
    use_abs = cfg(use_abs, "CORR_ABS", False)
    top_k = cfg(top_k, "CORR_TOP_K", 10)
    threshold = cfg(threshold, "CORR_THRESHOLD", 0.5)
    max_degree = cfg(max_degree, "CORR_MAX_DEGREE", 50)
    symmetric = cfg(symmetric, "SPARSE_SYMMETRIC", False)
    exclude_same_issuer = getattr(config, "CORR_EXCLUDE_SAME_ISSUER", True)
    sparse_degree = cfg(sparse_degree, "SPARSE_DEGREE", 3)
    mode = getattr(config, "SPARSE_MODE", "absolute")
    sparse_pct = sparse_pct if sparse_pct is not None else (
        getattr(config, "SPARSE_PCT", None) if mode == "pct" else None)

    state = {"corr": None, "n_built": 0, "anchor": None,
             "ranking": None, "graphs": {}}

    def _ranking(asof):
        """Correlation ranking, recomputed only every `refresh` dates."""
        if state["corr"] is None:
            from data_loader import load_ohlcv
            log.info("correlation: %d-day window, refresh every %d date(s), "
                     "%s correlation", window, refresh,
                     "absolute" if use_abs else "signed")
            issuers = issuer_map() if exclude_same_issuer else {}
            if issuers:
                dual = sum(1 for t in tickers
                           if sum(1 for o in tickers
                                  if issuers.get(o) == issuers.get(t)) > 1)
                if dual:
                    log.info("correlation: excluding same-issuer edges "
                             "(%d dual-class tickers)", dual)
            state["corr"] = RollingCorrelation(
                returns_from_ohlcv(load_ohlcv(list(tickers), start, end)),
                window=window, use_abs=use_abs, issuers=issuers)
        if (state["anchor"] is None
                or state["n_built"] % refresh == 0):
            state["ranking"] = state["corr"].ranking(list(tickers), asof)
            state["anchor"] = asof
            state["graphs"] = {}           # graphs depend on the ranking
        return state["ranking"]

    def fn(name: str, asof: str) -> Dict[str, List[str]]:
        if name not in topo.CORR_NAMES:
            return topo.build_topology(name, list(tickers), sectors,
                                       degree=sparse_degree, seed=seed,
                                       pct=sparse_pct, symmetric=symmetric)
        ranking = _ranking(asof)
        if name not in state["graphs"]:
            state["graphs"][name] = topo.build_topology(
                name, list(tickers), sectors, ranking=ranking,
                threshold=threshold, top_k=top_k, max_degree=max_degree,
                symmetric=symmetric)
        return state["graphs"][name]

    def advance():
        """Call once per rebalance date, after building that date's graphs."""
        state["n_built"] += 1

    fn.advance = advance
    fn.anchor = lambda: state["anchor"]
    return fn


# ==========================================================================
# Characterising a topology before spending GPU time on it
# ==========================================================================
def degree_report(graph: Dict[str, List[str]]) -> dict:
    """
    Degree statistics for a graph, plus the prompt cost they imply.

    Out-degree is how many peers an agent READS (this drives its prompt length);
    in-degree is how many agents read it (this drives its influence). They differ
    for any directed topology, which `sparse` and `corr_topk` both are.
    """
    out_deg = sorted(len(v) for v in graph.values())
    in_count = {t: 0 for t in graph}
    for peers in graph.values():
        for p in peers:
            if p in in_count:
                in_count[p] += 1
    in_deg = sorted(in_count.values())

    def stats(xs):
        if not xs:
            return {"min": 0, "max": 0, "mean": 0.0, "median": 0}
        return {"min": xs[0], "max": xs[-1],
                "mean": round(sum(xs) / len(xs), 1),
                "median": xs[len(xs) // 2]}

    mean_out = sum(out_deg) / len(out_deg) if out_deg else 0
    return {
        "n_agents": len(graph),
        "n_edges": sum(out_deg),
        "out_degree": stats(out_deg),
        "in_degree": stats(in_deg),
        "isolated": sum(1 for d in out_deg if d == 0),
        "unheard": sum(1 for d in in_deg if d == 0),
        # ~6.2 tokens per peer line plus ~50 of scaffolding, measured against
        # StockAgent.revise_prompt on this universe.
        "est_revise_tokens": int(round(50 + 6.2 * mean_out)),
    }


def format_degree_report(name: str, rep: dict) -> str:
    o, i = rep["out_degree"], rep["in_degree"]
    return (f"{name:<16} out-deg {o['min']:>3}-{o['max']:<3} "
            f"(mean {o['mean']:>5})  in-deg {i['min']:>3}-{i['max']:<3} "
            f"(mean {i['mean']:>5})  isolated {rep['isolated']:>3}  "
            f"unheard {rep['unheard']:>3}  ~{rep['est_revise_tokens']:>4} tok/prompt")
