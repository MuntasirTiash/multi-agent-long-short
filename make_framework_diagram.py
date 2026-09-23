"""
make_framework_diagram.py — render docs/framework.svg, the one-page picture of
how this system works.

Written as a generator rather than a hand-authored SVG for two reasons: the
topology gallery needs 45 computed edge coordinates for `full` alone, and the
numbers in the boxes (498 firms, N_ROUNDS, thresholds) should be editable in one
place when the framework moves.

    python make_framework_diagram.py                 # -> docs/framework.svg
    python make_framework_diagram.py --out talk.svg

Stdlib only, matching the rest of the core. The output is a self-contained SVG:
no external fonts, no script, no images, so it drops into Keynote, PowerPoint,
LaTeX (\\includegraphics with svg support) or a browser unchanged. A light card
is painted behind everything so it stays readable on a dark slide too.

Colour carries meaning and repeats, so it earns the legend:
    blue   = the LLM brain and anything it produced
    orange = the rule brain / the fallback path
    grey   = data, plumbing, structure
"""

import argparse
import math
import os

# --------------------------------------------------------------------------
# Palette — the project's chart palette (plot_feature_citations.THEMES["light"])
# --------------------------------------------------------------------------
SURFACE = "#fcfcfb"
CARD    = "#f5f5f1"
WHITE   = "#ffffff"
INK     = "#0b0b0b"
INK2    = "#52514e"
MUTED   = "#898781"
GRID    = "#e1e0d9"
AXIS    = "#c3c2b7"
BLUE    = "#2a78d6"
BLUE_BG = "#eaf2fc"
ORANGE  = "#eb6834"
ORG_BG  = "#fdeee7"

SANS = ("ui-sans-serif,-apple-system,'Segoe UI',Roboto,'Helvetica Neue',"
        "Arial,sans-serif")
MONO = "ui-monospace,'SF Mono',Menlo,Consolas,'Liberation Mono',monospace"

W, H = 1480, 1270
M = 30                      # page margin
RIGHT = W - M               # 1450


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------
def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def rect(x, y, w, h, fill="none", stroke=None, sw=1.0, rx=8, dash=None):
    a = [f'x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}"',
         f'rx="{rx}" fill="{fill}"']
    if stroke:
        a.append(f'stroke="{stroke}" stroke-width="{sw}"')
    if dash:
        a.append(f'stroke-dasharray="{dash}"')
    return f'<rect {" ".join(a)}/>'


def txt(x, y, s, size=10.4, fill=INK2, anchor="start", weight=None,
        family=None, ls=None, opacity=None):
    a = [f'x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}"']
    if anchor != "start":
        a.append(f'text-anchor="{anchor}"')
    if weight:
        a.append(f'font-weight="{weight}"')
    if family:
        a.append(f'font-family="{family}"')
    if ls:
        a.append(f'letter-spacing="{ls}"')
    if opacity:
        a.append(f'opacity="{opacity}"')
    return f'<text {" ".join(a)}>{esc(s)}</text>'


def line(x1, y1, x2, y2, stroke=INK2, sw=1.2, dash=None, marker=None,
         opacity=None):
    a = [f'x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"',
         f'stroke="{stroke}" stroke-width="{sw}"']
    if dash:
        a.append(f'stroke-dasharray="{dash}"')
    if marker:
        a.append(f'marker-end="url(#{marker})"')
    if opacity:
        a.append(f'opacity="{opacity}"')
    return f'<line {" ".join(a)}/>'


def polyline(pts, stroke=INK2, sw=1.2, dash=None, marker=None):
    d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    a = [f'points="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}"']
    if dash:
        a.append(f'stroke-dasharray="{dash}"')
    if marker:
        a.append(f'marker-end="url(#{marker})"')
    return f'<polyline {" ".join(a)}/>'


def box(out, x, y, w, title=None, lines=(), fill=WHITE, stroke=GRID, sw=1.0,
        dash=None, rx=7, title_size=11.4, line_size=10.2, pad=10,
        title_fill=INK, line_fill=INK2, anchor="start", accent=None,
        line_family=None):
    """
    A titled box that sizes itself to its content and returns its bottom edge.

    Auto-height is the point: stacking boxes by reading back the previous
    bottom keeps the panels aligned when a label is reworded, which hand-typed
    y coordinates do not.
    """
    lead = line_size * 1.34
    h = pad * 2
    if title:
        h += title_size * 1.3
    h += len(lines) * lead
    if title and lines:
        h += 2

    out.append(rect(x, y, w, h, fill=fill, stroke=stroke, sw=sw, rx=rx,
                    dash=dash))
    if accent:                                   # 3px meaning-bearing left edge
        out.append(f'<path d="M{x+3:.1f},{y+1:.1f} L{x+3:.1f},{y+h-1:.1f}" '
                   f'stroke="{accent}" stroke-width="3" '
                   f'stroke-linecap="round"/>')

    tx = x + pad if anchor == "start" else x + w / 2
    cy = y + pad
    if title:
        cy += title_size * 0.95
        out.append(txt(tx, cy, title, size=title_size, fill=title_fill,
                       weight="600", anchor=anchor))
        cy += title_size * 0.35 + 2
    for i, s in enumerate(lines):
        cy += lead if (title or i) else line_size * 0.95
        out.append(txt(tx, cy, s, size=line_size, fill=line_fill,
                       anchor=anchor, family=line_family))
    return y + h


def section(out, y, num, label):
    """Numbered band heading with a hairline running to the right margin."""
    s = f"{num} — {label}"
    out.append(txt(M, y, s, size=11.2, fill=MUTED, weight="700", ls="1.7"))
    wpx = len(s) * (11.2 * 0.60 + 1.7)
    out.append(line(M + wpx + 14, y - 4, RIGHT, y - 4, stroke=GRID, sw=1))


# --------------------------------------------------------------------------
# Band 3 helper: one topology drawn on a fixed 10-node ring
# --------------------------------------------------------------------------
def ring(n=10, r=44.0):
    return [(r * math.cos(math.radians(-90 + 360 * i / n)),
             r * math.sin(math.radians(-90 + 360 * i / n))) for i in range(n)]


def mini_graph(out, cx, cy, edges, n=10, r=44.0, edge_op=0.62, sw=1.15):
    """Same ring, same node order in every panel — only the edges differ, so
    the six panels are directly comparable by eye."""
    pts = ring(n, r)
    deg = [0] * n
    out.append(f'<g transform="translate({cx:.1f},{cy:.1f})">')
    for i, j in edges:
        deg[i] += 1
        deg[j] += 1
        out.append(line(pts[i][0], pts[i][1], pts[j][0], pts[j][1],
                        stroke=BLUE, sw=sw, opacity=edge_op))
    for i, (px, py) in enumerate(pts):
        if deg[i]:
            out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.6" '
                       f'fill="{INK2}" stroke="{SURFACE}" stroke-width="1.4"/>')
        else:                                    # isolated agent: hollow node
            out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.4" '
                       f'fill="{SURFACE}" stroke="{MUTED}" '
                       f'stroke-width="1.5" stroke-dasharray="2 1.6"/>')
    out.append('</g>')


ALL_PAIRS = [(i, j) for i in range(10) for j in range(i + 1, 10)]

TOPOLOGIES = [
    ("full", ALL_PAIRS, 0.22,
     ["one clique — every agent", "reads all 497 others"], "degree 497"),
    ("sector", [(0, 1), (0, 2), (1, 2),
                (3, 4), (3, 5), (3, 6), (4, 5), (4, 6), (5, 6),
                (7, 8), (7, 9), (8, 9)], 0.62,
     ["cliques of one GICS sector;", "11 groups over 498 names"], "degree ≈ 54"),
    ("sparse (pct)", [(0, 4), (0, 7), (1, 7), (1, 3), (2, 5), (2, 9),
                      (3, 9), (4, 6), (5, 8), (6, 8)], 0.62,
     ["a fixed SHARE of the", "cross-section, not a count"], "degree = 10% ≈ 50"),
    # Deliberately INTERLEAVED, not the contiguous arcs `sector` uses: the two
    # panels sit next to each other, and the caption's claim that correlation
    # clusters are not GICS sectors should be visible, not just asserted.
    ("corr_topk", [(0, 3), (3, 6), (6, 0), (1, 4), (4, 7), (7, 1),
                   (2, 5), (5, 8), (8, 2), (9, 0), (9, 5)], 0.62,
     ["the k most-correlated peers;", "clusters cut across sectors"],
     "degree = k = 10"),
    ("corr_threshold", [(0, 1), (0, 2), (0, 8), (0, 9), (1, 2), (1, 8),
                        (1, 9), (2, 8), (2, 9), (8, 9), (3, 4), (4, 5)], 0.62,
     ["everyone with ρ > τ — degree", "floats, some agents isolate"],
     "degree 23–46 at τ = 0.5"),
    ("corr_anti", [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)], 0.62,
     ["the LEAST-correlated peers,", "a deliberate contrarian feed"],
     "degree = k = 10"),
]


# ==========================================================================
# The drawing
# ==========================================================================
def build():
    o = []
    o.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
             f'width="{W}" height="{H}" role="img" '
             f'aria-label="How the agent-orchestration framework works: price '
             f'data becomes point-in-time features, 498 stock agents each form '
             f'an independent opinion and then revise it after reading peers '
             f'routed through a configurable communication graph, a manager '
             f'agent turns the final opinions into a dollar-neutral long-short '
             f'book, and evaluation grades it on out-of-sample forward returns '
             f'that never enter a prompt.">')
    o.append('<title>Agent Orchestration — framework overview</title>')
    o.append('<defs>')
    for name, col in (("aInk", INK2), ("aBlue", BLUE), ("aOrange", ORANGE),
                      ("aMuted", MUTED)):
        o.append(f'<marker id="{name}" viewBox="0 0 10 10" refX="9.2" refY="5" '
                 f'markerWidth="5.6" markerHeight="5.6" '
                 f'orient="auto-start-reverse">'
                 f'<path d="M0.5,0.8 L9.4,5 L0.5,9.2 z" fill="{col}"/></marker>')
    o.append('</defs>')
    o.append(rect(0, 0, W, H, fill=SURFACE, rx=0))
    o.append(f'<g font-family="{SANS}">')

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    o.append(txt(M, 52, "Agent Orchestration", size=27, fill=INK, weight="700"))
    o.append(txt(M, 80,
                 "One LLM agent per S&P-500 firm forms an independent view, "
                 "revises it after reading peers routed over a configurable "
                 "graph, and a manager turns the result into a long/short book.",
                 size=13.2, fill=INK2))
    o.append(txt(M, 100,
                 "Research question — does letting the agents talk produce a "
                 "better book than the same agents working alone?",
                 size=13.2, fill=BLUE, weight="600"))

    lx = 905
    for col, label in ((BLUE, "LLM brain (local Qwen2.5-7B)"),
                       (ORANGE, "rule brain / fallback"),
                       (MUTED, "data & plumbing")):
        o.append(rect(lx, 43, 11, 11, fill=col, rx=3))
        o.append(txt(lx + 17, 52, label, size=10.6, fill=INK2))
        lx += 17 + len(label) * 6.0 + 26
    o.append(line(M, 112, RIGHT, 112, stroke=AXIS, sw=1))

    # ------------------------------------------------------------------
    # Band 1 — the pipeline
    # ------------------------------------------------------------------
    section(o, 144, "01", "THE PIPELINE, END TO END")

    bw, gap, by = 190, 56, 176
    stages = [
        ("Price data", ["data/price_cache/", "498 tickers · OHLCV,",
                        "dividends, splits", "437 bars per name"]),
        ("Features", ["features.py", "26 point-in-time signals",
                      "10 of them shown to the", "agent, as percentiles"]),
        ("Tier 1 · StockAgents", ["498 agents, one per",
                                  "ticker. Round 0 = an",
                                  "independent opinion", "with no peer input"]),
        ("Debate", ["MessageBus + topology", "N_ROUNDS = 2 revisions",
                    "each agent re-scores", "after reading its peers"]),
        ("Manager", ["ManagerAgent", "reads all 498 opinions",
                     "picks and sizes both", "sides of the book"]),
        ("Evaluation", ["evaluation.py", "rank IC vs forward ret.",
                        "net of 20 bps round-trip", "Monte-Carlo null → p"]),
    ]
    accents = [MUTED, MUTED, BLUE, BLUE, BLUE, MUTED]
    labels = ["prices", "brief", "opinions", "scores", "weights"]
    bottoms = []
    for i, (title, lines) in enumerate(stages):
        x = M + i * (bw + gap)
        bottoms.append(box(o, x, by, bw, title, lines, fill=WHITE,
                           stroke=GRID, accent=accents[i], title_size=12.6,
                           line_size=10.2, pad=11))
    bb = max(bottoms)
    mid = by + (bb - by) / 2
    for i, lab in enumerate(labels):
        x0 = M + i * (bw + gap) + bw
        o.append(line(x0 + 3, mid, x0 + gap - 4, mid, stroke=INK2, sw=1.3,
                      marker="aInk"))
        o.append(txt(x0 + gap / 2, mid - 8, lab, size=9.8, fill=MUTED,
                     anchor="middle"))

    # The leakage bypass: forward returns physically route AROUND the agents.
    fy = bb + 46
    o.append(polyline([(M + bw / 2, bb), (M + bw / 2, fy),
                       (M + 5 * (bw + gap) + bw / 2, fy),
                       (M + 5 * (bw + gap) + bw / 2, bb + 4)],
                      stroke=MUTED, sw=1.4, dash="6 4", marker="aMuted"))
    o.append(rect(W / 2 - 306, fy - 10, 612, 20, fill=SURFACE, rx=4))
    o.append(txt(W / 2, fy + 4,
                 "forward returns t → t+H  ·  reach evaluation.py directly and "
                 "never appear in any prompt",
                 size=11, fill=MUTED, anchor="middle"))

    # ------------------------------------------------------------------
    # Band 2 — inside the components
    # ------------------------------------------------------------------
    section(o, 400, "02", "INSIDE THE COMPONENTS")

    PY, PH, PW = 418, 380, 446
    PX = [M, M + PW + 41, M + 2 * (PW + 41)]
    for px in PX:
        o.append(rect(px, PY, PW, PH, fill=CARD, stroke=GRID, sw=1, rx=12))

    heads = [("Inside one StockAgent",
              "the dual-brain pattern — an LLM path with a rule twin behind it"),
             ("One revision round",
              "who reads whom, and when the shared world is allowed to move"),
             ("The ManagerAgent",
              "all 498 final opinions → one dollar-neutral book")]
    for px, (t, s) in zip(PX, heads):
        o.append(txt(px + 20, PY + 28, t, size=14.6, fill=INK, weight="700"))
        o.append(txt(px + 20, PY + 46, s, size=10.4, fill=MUTED))

    # ---------------- Panel A: StockAgent -----------------------------
    ax = PX[0] + 20
    aw = PW - 40
    lcx, rcx = ax + 118, ax + 328           # left (LLM) / right (rule) rails

    y = box(o, ax, PY + 60, aw, None,
            ["one stock's brief — 10 features as cross-sectional percentiles",
             "+ inbox: the peer messages this topology lets it read"],
            fill=WHITE, stroke=GRID, line_size=10.4)
    o.append(line(lcx, y, lcx, y + 18, stroke=INK2, sw=1.2, marker="aInk"))
    o.append(line(rcx, y, rcx, y + 18, stroke=ORANGE, sw=1.2, marker="aOrange"))

    y2 = box(o, ax, y + 18, 236, "prompt → Qwen2.5-7B",
             ["local & open-weight; returns", "JSON: score, direction,",
              "confidence, thesis"],
             fill=BLUE_BG, stroke=BLUE, sw=1.1, line_size=9.9, pad=9,
             title_fill=BLUE)
    y2r = box(o, ax + 250, y + 18, 156, "rule twin",
              ["0.6 · momentum", "+ 0.4 · reversal", "— always computed"],
              fill=ORG_BG, stroke=ORANGE, sw=1.1, line_size=9.9, pad=9,
              title_fill=ORANGE)

    o.append(line(lcx, y2, lcx, y2 + 12, stroke=BLUE, sw=1.2, marker="aBlue"))
    gy = box(o, ax + 18, y2 + 12, 200, "message_from_data()",
             ["is `score` a real number?"], fill=WHITE, stroke=BLUE, sw=1.2,
             dash="4 3", line_size=9.8, pad=8, title_size=10.8,
             title_fill=BLUE)

    merge = max(gy, y2r) + 30
    o.append(line(lcx, gy, lcx, merge, stroke=BLUE, sw=1.3, marker="aBlue"))
    o.append(txt(lcx + 8, gy + 19, "yes — keep the model's answer", size=9.6,
                 fill=BLUE, weight="600"))
    o.append(line(rcx, y2r, rcx, merge, stroke=ORANGE, sw=1.3, marker="aOrange"))
    o.append(line(ax + 222, gy - 14, rcx - 6, gy - 14, stroke=ORANGE, sw=1.2,
                  marker="aOrange"))
    o.append(txt((ax + 222 + rcx) / 2, gy - 19, "no", size=9.6, fill=ORANGE,
                 anchor="middle", weight="600"))

    y3 = box(o, ax, merge, aw, "StockMessage — the one wire format",
             ["ticker · score 0–100 · direction · confidence · "
              "thesis ≤ 50 words · round"],
             fill=WHITE, stroke=AXIS, sw=1.2, line_size=9.6)
    o.append(txt(ax, y3 + 18,
                 "The rule twin is built before the model is ever called, at "
                 "every call site.", size=10.2, fill=INK2))
    o.append(txt(ax, y3 + 32,
                 "Measured on the 32-day run: 76 fallbacks in 111,552 calls "
                 "(0.07%).", size=10.2, fill=INK2))

    # ---------------- Panel B: one round ------------------------------
    bx = PX[1] + 20
    o.append(txt(bx + 63, PY + 66, "bus.latest · round r−1", size=9.8,
                 fill=MUTED, anchor="middle"))
    o.append(txt(bx + 349, PY + 66, "bus.latest · round r", size=9.8,
                 fill=MUTED, anchor="middle"))

    rows = [("AAPL", "72", "68"), ("MSFT", "55", "59"),
            ("XOM", "38", "44"), ("JPM", "61", "57")]
    top, ph, pgap = PY + 76, 28, 6
    for i, (tk, s0, s1) in enumerate(rows):
        yy = top + i * (ph + pgap)
        cyy = yy + ph / 2 + 3.6
        o.append(rect(bx, yy, 126, ph, fill=WHITE, stroke=GRID, rx=6))
        o.append(txt(bx + 11, cyy, tk, size=10.6, fill=INK, weight="600",
                     family=MONO))
        o.append(txt(bx + 115, cyy, s0, size=10.6, fill=INK2, anchor="end",
                     family=MONO))
        o.append(rect(bx + 289, yy, 117, ph, fill=WHITE, stroke=GRID, rx=6))
        o.append(txt(bx + 300, cyy, tk, size=10.6, fill=INK, weight="600",
                     family=MONO))
        o.append(txt(bx + 395, cyy, s1, size=10.6, fill=BLUE, anchor="end",
                     weight="600", family=MONO))
        o.append(line(bx + 129, cyy - 3.6, bx + 146, cyy - 3.6, stroke=INK2,
                      sw=1.1, marker="aInk"))
        o.append(line(bx + 269, cyy - 3.6, bx + 286, cyy - 3.6, stroke=BLUE,
                      sw=1.1, marker="aBlue"))

    bot = top + 4 * (ph + pgap) - pgap
    o.append(rect(bx + 149, top, 120, bot - top, fill=BLUE_BG, stroke=BLUE,
                  sw=1.1, rx=7))
    mcx = bx + 209
    o.append(txt(mcx, top + 52, "revise()", size=11.8, fill=BLUE,
                 weight="700", anchor="middle"))
    for i, s in enumerate(["reads inbox_for(t)", "= only the peers",
                           "the topology allows"]):
        o.append(txt(mcx, top + 70 + i * 12.5, s, size=9.6, fill=INK2,
                     anchor="middle"))

    o.append(line(bx + 279, top - 8, bx + 279, bot + 8, stroke=ORANGE,
                  sw=1.4, dash="5 4"))
    o.append(txt(bx + 279, bot + 22, "post barrier", size=9.6, fill=ORANGE,
                 anchor="middle", weight="600"))

    cy = box(o, bx, bot + 34, 406, None,
             ["All 498 revisions are computed off the SAME snapshot, then",
              "posted together. Posting as you go would let the last agent in",
              "dict order react to a world the first one never saw.",
              "N_ROUNDS = 2 revision rounds run per rebalance date."],
             fill=WHITE, stroke=BLUE, sw=1.1, line_size=10.2, pad=9)
    o.append(txt(bx, cy + 20,
                 "BatchScheduler fires a whole round's prompts at once — inside "
                 "a round", size=10.2, fill=INK2))
    o.append(txt(bx, cy + 34,
                 "the agents are independent, so batching changes no result.",
                 size=10.2, fill=INK2))

    # ---------------- Panel C: manager --------------------------------
    cx0 = PX[2] + 20
    cw = PW - 40
    mlcx, mrcx = cx0 + 117, cx0 + 329

    y = box(o, cx0, PY + 60, cw, "candidates() — a shortlist, not the table",
            ["Rank all 498 by score, then show the manager only the",
             "top 30 and bottom 30 — the full table runs ≈12k tokens",
             "against a 4,096 context, so every call would fail."],
            fill=WHITE, stroke=GRID, line_size=9.9, pad=9, title_size=11)
    o.append(line(mlcx, y, mlcx, y + 12, stroke=BLUE, sw=1.2, marker="aBlue"))

    y2 = box(o, cx0, y + 12, 234, "LLM manager → JSON",
             ["longs, shorts, weights, rationale"],
             fill=BLUE_BG, stroke=BLUE, sw=1.1, line_size=9.9, pad=9,
             title_fill=BLUE)
    o.append(line(mlcx, y2, mlcx, y2 + 12, stroke=BLUE, sw=1.2, marker="aBlue"))
    gy = box(o, cx0 + 8, y2 + 12, 218, "reject the book if…",
             ["· a pick is off the shortlist",
              "· one side is empty — that is a",
              "   directional bet, not this book"],
             fill=WHITE, stroke=BLUE, sw=1.2, dash="4 3", line_size=9.6,
             pad=8, title_size=10.8, title_fill=BLUE)
    # The rule book sits on the LLM box's row, not the gate's, so the gate can
    # divert into its output rail below — the same shape as Panel A.
    gyr = box(o, cx0 + 252, y + 12, 154, "rule brain",
              ["conviction =", "confidence ×", "|score − 50|"],
              fill=ORG_BG, stroke=ORANGE, sw=1.1, line_size=9.9, pad=9,
              title_fill=ORANGE)
    o.append(line(mrcx, y, mrcx, y + 12, stroke=ORANGE, sw=1.2,
                  marker="aOrange"))
    o.append(line(cx0 + 230, gy - 22, mrcx - 8, gy - 22, stroke=ORANGE,
                  sw=1.2, marker="aOrange"))
    o.append(txt((cx0 + 230 + mrcx) / 2, gy - 27, "reject", size=9.6,
                 fill=ORANGE, anchor="middle", weight="600"))

    merge = max(gy, gyr) + 14
    o.append(line(mlcx, gy, mlcx, merge, stroke=BLUE, sw=1.3, marker="aBlue"))
    o.append(line(mrcx, gyr, mrcx, merge, stroke=ORANGE, sw=1.3,
                  marker="aOrange"))
    y3 = box(o, cx0, merge, cw, "dollar-neutral book",
             ["long weights sum to +1 · short weights to −1 · "
              "gross 2 · net 0"],
             fill=WHITE, stroke=AXIS, sw=1.2, line_size=9.6)
    o.append(txt(cx0, y3 + 18,
                 "BOOK_SIZING = \"flexible\": the manager chooses its own count "
                 "per side,", size=10.2, fill=INK2))
    o.append(txt(cx0, y3 + 32,
                 "inside [5, 50], instead of a hard-wired top-5 / bottom-5.",
                 size=10.2, fill=INK2))

    # ------------------------------------------------------------------
    # Band 3 — topologies
    # ------------------------------------------------------------------
    section(o, 838, "03", "THE EXPERIMENTAL VARIABLE: WHO HEARS WHOM")

    tw, tg, ty, th = 220, 20, 856, 200
    for i, (name, edges, op, caption, degree) in enumerate(TOPOLOGIES):
        x = M + i * (tw + tg)
        c = x + tw / 2
        o.append(rect(x, ty, tw, th, fill=CARD, stroke=GRID, sw=1, rx=10))
        o.append(txt(c, ty + 22, name, size=13, fill=INK, weight="700",
                     anchor="middle", family=MONO))
        mini_graph(o, c, ty + 76, edges, edge_op=op)
        for k, s in enumerate(caption):
            o.append(txt(c, ty + 142 + k * 13, s, size=9.7, fill=INK2,
                         anchor="middle"))
        o.append(txt(c, ty + 180, degree, size=10.4, fill=BLUE,
                     anchor="middle", weight="700"))

    o.append(txt(M, ty + th + 24,
                 "Same ten agents, same ring, in every panel — only the edges "
                 "change. Each agent still scores only its own stock; the graph "
                 "decides whose score it may read before revising.",
                 size=10.8, fill=INK2))
    o.append(txt(M, ty + th + 40,
                 "Degree and structure are confounded today (full = 497 peers, "
                 "sector ≈ 54, sparse ≈ 50, corr_topk = 10), so any structural "
                 "claim needs the two controls in TOPOLOGY_PLAN.md §3: a "
                 "degree-matched random graph, and a peer-shuffling placebo.",
                 size=10.8, fill=INK2))
    o.append(txt(M, ty + th + 56,
                 "A second tier exists in hierarchy.py — one industry leader "
                 "per sector aggregates its members and the leaders then form "
                 "their own council — but it has not been run against a model "
                 "yet, so it is left off this picture.",
                 size=10.8, fill=MUTED))

    # ------------------------------------------------------------------
    # Band 4 — invariants
    # ------------------------------------------------------------------
    section(o, 1138, "04", "INVARIANTS THE HARNESS ENFORCES")

    notes = [
        ("Leakage safety", MUTED, [
            "Features at date t read prices ≤ t only.",
            "Forward returns go straight to evaluation.py — no",
            "prompt has ever seen one. The backtest window opens",
            "2024-10-01, past Qwen2.5's training cutoff."]),
        ("A run never dies on a bad generation", ORANGE, [
            "Every LLM call site precomputes its rule twin.",
            "Unparseable JSON, a non-numeric score, a ticker off",
            "the shortlist, a one-sided book — each falls back and",
            "continues; _count_fallbacks() logs the rate per round."]),
        ("Every run states its own cost", BLUE, [
            "instrumentation.py records calls, prompt/completion",
            "tokens, wall time, latency percentiles and fallbacks",
            "per (date, topology, round), then projects the full",
            "date range. Exact vs estimated is always labelled."]),
    ]
    for px, (t, col, lines) in zip(PX, notes):
        box(o, px, 1152, PW, t, lines, fill=WHITE, stroke=GRID,
            accent=col, title_size=11.6, line_size=10.2, pad=11)

    o.append('</g></svg>')
    return "\n".join(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("docs", "framework.svg"))
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    svg = build()
    with open(out, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"{out}  ({len(svg) / 1024:.0f} KB, viewBox {W}x{H})")


if __name__ == "__main__":
    main()
