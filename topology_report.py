"""
topology_report.py — characterise a topology BEFORE spending GPU time on it.

Two questions this answers, both of which have bitten this project before:

  1. What degree does a threshold actually produce? `CORR_THRESHOLD` is
     universe-dependent in exactly the way the manager's score gate was: a value
     that looks selective can connect every agent to dozens of peers, so the
     `CORR_MAX_DEGREE` cap binds every date and "threshold topology" silently
     becomes "top-50 topology". Measure, then pick.

  2. What does each topology cost per prompt, and which random degree matches a
     structural topology? Comparing full(497 peers) vs sector(54) vs sparse(3)
     confounds structure with degree; the degree-matched control is the fix
     (TOPOLOGY_PLAN.md section 3), and this prints the matching percentage.

Usage:
    python topology_report.py                     # all topologies, 3 dates
    python topology_report.py --dates 6
    python topology_report.py --thresholds 0.3,0.4,0.5,0.6,0.7,0.8
"""

import argparse
import logging
import statistics

import config
import correlation as corr
import data_loader as dl
import topology as topo

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("report")

START, END = "2024-10-01", "2026-07-01"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", type=int, default=3)
    ap.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.7,0.8")
    ap.add_argument("--topk", default="5,10,25,50")
    args = ap.parse_args()

    tickers = list(config.UNIVERSE)
    sectors = config.UNIVERSE
    prices = dl.load_prices(tickers, START, END)
    all_dates = dl.rebalance_dates(prices, 1, 63, 1)
    step = max(1, len(all_dates) // args.dates)
    dates = all_dates[::step][:args.dates]
    n = len(tickers)
    print(f"universe {n} firms | dates sampled: {', '.join(dates)}\n")

    ohlcv = dl.load_ohlcv(tickers, START, END)
    returns = corr.returns_from_ohlcv(ohlcv)

    # ---- 1. the static (non-correlation) topologies ----------------------
    print("=" * 100)
    print("STATIC TOPOLOGIES")
    print("=" * 100)
    static = {
        "full": dict(),
        "sector": dict(),
        "sparse(3)": dict(degree=3),
        "sparse(pct=1%)": dict(pct=0.01),
        "sparse(pct=5%)": dict(pct=0.05),
        "sparse(pct=10%)": dict(pct=0.10),
        "sparse(pct=20%)": dict(pct=0.20),
        "sparse(10%,sym)": dict(pct=0.10, symmetric=True),
    }
    sector_mean = None
    for label, kw in static.items():
        name = "sparse" if label.startswith("sparse") else label
        g = topo.build_topology(name, tickers, sectors, seed=config.SEED, **kw)
        rep = corr.degree_report(g)
        print(corr.format_degree_report(label, rep))
        if label == "sector":
            sector_mean = rep["out_degree"]["mean"]
    if sector_mean:
        print(f"\n  degree-matched control for `sector` (mean {sector_mean} peers): "
              f"SPARSE_PCT = {sector_mean / (n - 1):.3f}")

    # ---- 2. correlation levels ------------------------------------------
    print("\n" + "=" * 100)
    print(f"CORRELATION DISTRIBUTION  (window={config.CORR_WINDOW}d, "
          f"{'absolute' if config.CORR_ABS else 'signed'})")
    print("=" * 100)
    issuers = (corr.issuer_map()
               if getattr(config, "CORR_EXCLUDE_SAME_ISSUER", True) else {})
    rc = corr.RollingCorrelation(returns, window=config.CORR_WINDOW,
                                 use_abs=config.CORR_ABS, issuers=issuers)
    for asof in dates:
        ranking = rc.ranking(tickers, asof)
        vals = [c for peers in ranking.values() for _, c in peers]
        vals.sort()
        if not vals:
            print(f"{asof}: no correlations (insufficient history)")
            continue
        q = [vals[int(p * (len(vals) - 1))] for p in (0.05, 0.25, 0.5, 0.75, 0.95)]
        print(f"{asof}: pairwise corr  p5={q[0]:+.2f} p25={q[1]:+.2f} "
              f"median={q[2]:+.2f} p75={q[3]:+.2f} p95={q[4]:+.2f}  "
              f"max={vals[-1]:+.2f}")

    # ---- 3. threshold calibration ---------------------------------------
    print("\n" + "=" * 100)
    print("THRESHOLD CALIBRATION — does the gate bind before CORR_MAX_DEGREE "
          f"({config.CORR_MAX_DEGREE})?")
    print("=" * 100)
    print(f"{'threshold':>10}{'mean out-deg':>14}{'min':>6}{'max':>7}"
          f"{'isolated':>10}{'capped':>8}   verdict")
    for thr in [float(x) for x in args.thresholds.split(",")]:
        means, mins, maxs, isolated, capped = [], [], [], [], []
        for asof in dates:
            ranking = rc.ranking(tickers, asof)
            g = topo.correlation_topology(tickers, ranking,
                                          mode="corr_threshold", threshold=thr,
                                          max_degree=None)
            rep = corr.degree_report(g)
            means.append(rep["out_degree"]["mean"])
            mins.append(rep["out_degree"]["min"])
            maxs.append(rep["out_degree"]["max"])
            isolated.append(rep["isolated"])
            capped.append(sum(1 for v in g.values()
                              if len(v) > config.CORR_MAX_DEGREE))
        mean = statistics.mean(means)
        verdict = ("cap binds for most agents" if statistics.mean(capped) > 0.5 * len(tickers)
                   else "cap binds sometimes" if statistics.mean(capped) > 1
                   else "gate decides" if mean >= 1
                   else "TOO STRICT - graph nearly empty")
        print(f"{thr:>10.2f}{mean:>14.1f}{min(mins):>6}{max(maxs):>7}"
              f"{statistics.mean(isolated):>10.0f}"
              f"{statistics.mean(capped):>8.0f}   {verdict}")

    # ---- 4. the correlation topologies as configured ---------------------
    print("\n" + "=" * 100)
    print("CORRELATION TOPOLOGIES (as configured)")
    print("=" * 100)
    for asof in dates:
        ranking = rc.ranking(tickers, asof)
        print(f"-- {asof}")
        for label, kw in (
                (f"corr_topk({config.CORR_TOP_K})",
                 dict(mode="corr_topk", top_k=config.CORR_TOP_K)),
                (f"corr_anti({config.CORR_TOP_K})",
                 dict(mode="corr_anti", top_k=config.CORR_TOP_K)),
                (f"corr_threshold({config.CORR_THRESHOLD})",
                 dict(mode="corr_threshold", threshold=config.CORR_THRESHOLD,
                      max_degree=config.CORR_MAX_DEGREE))):
            g = topo.correlation_topology(tickers, ranking, **kw)
            print("   " + corr.format_degree_report(label,
                                                    corr.degree_report(g)))

    # ---- 5. how much do the graphs move between dates? ------------------
    print("\n" + "=" * 100)
    print("GRAPH STABILITY — Jaccard edge overlap vs an anchor date, by lag")
    print("This is what decides whether CORR_REFRESH=1 (daily) and =21 (monthly)")
    print("are different experiments: if the lag-21 graph still overlaps the")
    print("anchor heavily, monthly rewiring loses almost nothing.")
    print("=" * 100)
    anchor = all_dates[len(all_dates) // 2]
    i0 = all_dates.index(anchor)
    lags = [(lag, all_dates[i0 + lag]) for lag in (1, 5, 21, 63)
            if i0 + lag < len(all_dates)]
    print(f"anchor {anchor}; lags: "
          f"{', '.join(f'+{l}d={d}' for l, d in lags)}\n")
    for label, kw in ((f"corr_topk({config.CORR_TOP_K})",
                       dict(mode="corr_topk", top_k=config.CORR_TOP_K)),
                      (f"corr_anti({config.CORR_TOP_K})",
                       dict(mode="corr_anti", top_k=config.CORR_TOP_K)),
                      (f"corr_threshold({config.CORR_THRESHOLD})",
                       dict(mode="corr_threshold",
                            threshold=config.CORR_THRESHOLD,
                            max_degree=config.CORR_MAX_DEGREE))):
        base = {(t, p) for t, peers in
                topo.correlation_topology(tickers, rc.ranking(tickers, anchor),
                                          **kw).items() for p in peers}
        cells = []
        for lag, d in lags:
            e = {(t, p) for t, peers in
                 topo.correlation_topology(tickers, rc.ranking(tickers, d),
                                           **kw).items() for p in peers}
            union = base | e
            cells.append(f"+{lag}d {len(base & e) / len(union):.0%}" if union
                         else f"+{lag}d   -")
        print(f"  {label:<24} " + "   ".join(cells))


if __name__ == "__main__":
    main()
