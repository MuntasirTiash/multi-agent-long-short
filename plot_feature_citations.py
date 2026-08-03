"""
plot_feature_citations.py — which prompt features do the agents actually cite?

Each StockAgent is shown ten features (features.PROMPT_FEATURES) and asked for a
<=50-word thesis. With that budget it can only mention a few, so *which* few is a
real finding: it says which signals the model is reasoning from, and which are
being ignored despite costing tokens on every call.

This reads one rebalance date out of a transcript and plots the share of theses
citing each feature.

    python plot_feature_citations.py --date 2024-12-31
    python plot_feature_citations.py --date 2024-12-31 --round 1 --topology sparse
    python plot_feature_citations.py --transcript results/transcript_*.jsonl \
        --date 2025-01-02 --theme dark

Reading the result — two cautions:
  * Citing a feature is not the same as USING it. The score may lean on something
    the thesis never mentions, and vice versa; a 7B model's stated reasoning is
    only loosely coupled to its output. For attribution, regress scores on
    features rather than trusting this chart.
  * The keyword patterns below are a proxy. They are deliberately broad (e.g.
    "sector|peers|industry" all count as the sector feature), so treat the bars
    as relative emphasis, not exact counts.

Colours come from the data-viz reference palette (categorical slots 1 and 2),
used unchanged, so they carry their documented CVD clearance.
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")                       # headless: cluster nodes have no display
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, BoxStyle

# --------------------------------------------------------------------------
# Palette (reference instance; both modes are selected, not auto-flipped)
# --------------------------------------------------------------------------
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
              "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
              "series1": "#2a78d6", "series2": "#eb6834"},
    "dark":  {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
              "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
              "series1": "#3987e5", "series2": "#d95926"},
}

# How each prompt feature shows up in prose. Broad on purpose — see the caution
# in the module docstring.
PATTERNS = {
    "momentum":       r"3-month|three-month|quarterly momentum|longer-term momentum",
    "mom_21":         r"1-month|one-month|monthly momentum|short-term momentum",
    "value":          r"revers|1-week|one-week|weekly pullback",
    "sector_rel_mom": r"sector|peers|industry|relative strength within",
    "dist_sma50":     r"50-day|moving average|above its average|below its average",
    "rsi14":          r"\bRSI\b|overbought|oversold",
    "vol_20":         r"volatil|\bvol\b",
    "beta":           r"\bbeta\b|market sensitivity",
    "vol_shock":      r"volume|trading activity|turnover",
    "div_yield":      r"dividend|yield",
}


def load_features():
    """(key, label, is_percentile) per prompt feature, straight from features.py
    so this chart cannot drift from what the agents were actually shown."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from features import PROMPT_FEATURES
    return [(k, label, how == "pct") for k, label, how in PROMPT_FEATURES]


def read_theses(path, date, round_num=None, topology=None):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                    # a half-flushed line in a live run
            if r.get("date") != date:
                continue
            if round_num is not None and r.get("round") != round_num:
                continue
            if topology and r.get("topology") != topology:
                continue
            rows.append(r)
    return rows


def count_citations(rows, feats):
    counts = Counter()
    per_thesis = []
    for r in rows:
        thesis = (r.get("thesis") or "").lower()
        hits = 0
        for key, _, _ in feats:
            pat = PATTERNS.get(key)
            if pat and re.search(pat, thesis):
                counts[key] += 1
                hits += 1
        per_thesis.append(hits)
    return counts, per_thesis


# --------------------------------------------------------------------------
# Chart
# --------------------------------------------------------------------------
def rounded_barh(ax, y, width, height, color, radius_px=4):
    """
    One bar: 4px rounded data-end, square at the baseline.

    FancyBboxPatch rounds all four corners, so a plain rectangle is laid over the
    baseline end to square it — the spec wants the growth end rounded and the
    origin flush against the axis.
    """
    if width <= 0:
        return
    ax.figure.canvas.draw()                 # need a real bbox to convert px->data
    bbox = ax.get_window_extent()
    x0, x1 = ax.get_xlim()
    r = radius_px * (x1 - x0) / max(bbox.width, 1)
    r = min(r, width / 2)
    ax.add_patch(FancyBboxPatch(
        (0, y - height / 2), width, height,
        boxstyle=BoxStyle("Round", pad=0, rounding_size=r),
        linewidth=0, facecolor=color, mutation_aspect=1, zorder=3))
    ax.add_patch(Rectangle((0, y - height / 2), min(r, width), height,
                           linewidth=0, facecolor=color, zorder=3))


def plot(counts, feats, n, date, theme, out, subtitle_extra=""):
    t = THEMES[theme]
    data = sorted(((k, lab, pct, counts.get(k, 0) / n if n else 0)
                   for k, lab, pct in feats),
                  key=lambda x: x[3])       # ascending: biggest ends up on top

    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    ys = range(len(data))
    ax.set_xlim(0, max(0.05, max(d[3] for d in data) * 1.18))
    ax.set_ylim(-0.7, len(data) - 0.3)

    # Bars are capped at 24px of RENDERED thickness. A fixed fraction of the row
    # cannot do that — it scales with figure height — so convert the cap through
    # the axes' actual pixel height and let the band's leftover be air.
    fig.canvas.draw()
    row_px = ax.get_window_extent().height / (len(data) + 0.4)
    height = min(0.62, 24.0 / max(row_px, 1))

    for y, (_, _, is_pct, share) in zip(ys, data):
        rounded_barh(ax, y, share, height,
                     t["series1"] if is_pct else t["series2"])

    # Bars -> value at the tip. Every bar is labelled, so the x-axis is dropped
    # rather than repeating the same numbers as ticks.
    for y, (_, _, _, share) in zip(ys, data):
        ax.text(share + ax.get_xlim()[1] * 0.012, y, f"{share:.0%}",
                va="center", ha="left", fontsize=10, color=t["ink2"])

    ax.set_yticks(list(ys))
    ax.set_yticklabels([d[1] for d in data], fontsize=10.5, color=t["ink"])
    ax.set_xticks([])
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(t["axis"])
    ax.spines["left"].set_linewidth(1)
    ax.tick_params(axis="y", length=0, pad=6)

    sub = f"share of {n} analyst theses mentioning each feature{subtitle_extra}"
    ax.text(0, 1.085, f"Which prompt features the agents cite — {date}",
            transform=ax.transAxes, fontsize=13.5, color=t["ink"], va="bottom")
    ax.text(0, 1.030, sub, transform=ax.transAxes, fontsize=10,
            color=t["ink2"], va="bottom")

    # Two categories -> a legend is required; identity never rests on colour
    # alone, and the y labels name every bar anyway.
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=9,
                          color=t["series1"], label="shown as a percentile"),
               plt.Line2D([], [], marker="s", linestyle="", markersize=9,
                          color=t["series2"], label="shown in raw units")]
    leg = ax.legend(handles=handles, loc="lower right", frameon=False,
                    fontsize=10, labelcolor=t["ink2"], handletextpad=0.6)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])            # text wears ink, never the series hue

    fig.tight_layout(rect=(0, 0, 1, 0.90))   # reserve the title band
    fig.savefig(out, facecolor=t["surface"], bbox_inches="tight")
    print(f"chart -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="rebalance date, YYYY-MM-DD")
    ap.add_argument("--transcript", default=None,
                    help="transcript.jsonl (default: newest daily run dir's)")
    ap.add_argument("--round", type=int, default=0,
                    help="round to analyse (0 = independent assessment)")
    ap.add_argument("--topology", default=None)
    ap.add_argument("--theme", default="light", choices=("light", "dark"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = args.transcript
    if path and "*" in path:
        matches = sorted(glob.glob(path), key=os.path.getmtime)
        path = matches[-1] if matches else None
    if not path:
        dirs = [d for d in glob.glob("results/daily_2*") if os.path.isdir(d)]
        if not dirs:
            raise SystemExit("no results/daily_* run found; pass --transcript")
        path = os.path.join(max(dirs, key=os.path.getmtime), "transcript.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"transcript not found: {path}")

    feats = load_features()
    rows = read_theses(path, args.date, args.round, args.topology)
    if not rows:
        raise SystemExit(
            f"no messages for date={args.date} round={args.round}"
            f"{' topology=' + args.topology if args.topology else ''} in {path}")
    counts, per_thesis = count_citations(rows, feats)
    n = len(rows)

    # Table view: every value in the chart is readable without the chart.
    print(f"transcript: {path}")
    print(f"date {args.date}, round {args.round}"
          f"{', topology ' + args.topology if args.topology else ''}: "
          f"{n} theses\n")
    print(f"{'feature':<26}{'rendered':<12}{'cited':>7}{'share':>8}")
    for key, label, is_pct in sorted(feats, key=lambda f: -counts.get(f[0], 0)):
        c = counts.get(key, 0)
        print(f"{label:<26}{'percentile' if is_pct else 'raw units':<12}"
              f"{c:>7}{c / n:>7.0%}")
    mean_hits = sum(per_thesis) / len(per_thesis)
    print(f"\nfeatures cited per thesis: mean {mean_hits:.1f}, "
          f"max {max(per_thesis)}; {per_thesis.count(0)} theses cited none")

    out = args.out or f"feature_citations_{args.date}_r{args.round}_{args.theme}.png"
    plot(counts, feats, n, args.date, args.theme, out,
         subtitle_extra=(f", round {args.round}"
                         + (f", {args.topology}" if args.topology else "")))

    csv_path = out.rsplit(".", 1)[0] + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "rendered_as", "cited", "share", "n_theses"])
        for key, label, is_pct in feats:
            c = counts.get(key, 0)
            w.writerow([label, "percentile" if is_pct else "raw", c,
                        round(c / n, 4), n])
    print(f"table -> {csv_path}")


if __name__ == "__main__":
    main()
