"""
inspect_run.py — see what the agents are doing WHILE a job is still running.

The metrics files are rewritten after every date now (checkpoint_outputs), but a
job that started before that change only leaves two incremental artifacts behind:

    <run_dir>/run.log          every round, ranking, fallback and GRADE line
    <run_dir>/transcript.jsonl one JSON line per agent message, per round

That is enough to reconstruct everything that matters. This script reads both and
reports the agent behaviour a running job would otherwise only reveal at the end:

  * progress: dates done, rounds per date, how long each date took
  * FALLBACKS: how many agents' LLM output was unusable (a "LLM" run that is
    quietly mostly rule-based is the failure this project refuses to hide)
  * per-round score distribution and DISPERSION — whether communication is
    making agents converge (herding) or diverge
  * revision behaviour: how far agents move, and whether they move TOWARD their
    peers' mean, which is the actual mechanism under test
  * per-date/topology returns parsed from the GRADE lines, written to a CSV so a
    partial run is still analysable

Usage:
    python inspect_run.py                       # newest run dir
    python inspect_run.py --run-dir results/daily_2026...
    python inspect_run.py --transcript results/transcript_sparse..._498x32.jsonl
    python inspect_run.py --csv                 # also write interim_returns.csv
"""

import argparse
import glob
import json
import os
import re
import statistics
from collections import defaultdict


def newest_run_dir():
    dirs = [d for d in glob.glob("results/daily_2*") if os.path.isdir(d)]
    if not dirs:
        raise SystemExit("no results/daily_* directories found")
    return max(dirs, key=os.path.getmtime)


# ==========================================================================
# transcript.jsonl — the per-agent record
# ==========================================================================
def read_transcript(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue          # a partially-flushed final line while running
    return rows


def report_transcript(rows):
    if not rows:
        print("  (transcript empty — the first round has not finished yet; every "
              "message of a round is written together when its batch returns)")
        return
    print(f"  {len(rows):,} agent messages recorded")
    dates = sorted({r["date"] for r in rows})
    print(f"  dates seen: {len(dates)}  ({dates[0]} .. {dates[-1]})")

    groups = defaultdict(list)
    for r in rows:
        groups[(r["date"], r["topology"], r["round"])].append(r)

    print(f"\n  {'date':<12}{'topology':<16}{'rnd':>4}{'n':>6}"
          f"{'mean':>8}{'stdev':>8}{'min':>7}{'max':>7}  score dispersion")
    for key in sorted(groups):
        date, topo, rnd = key
        scores = [r["score"] for r in groups[key]]
        sd = statistics.stdev(scores) if len(scores) > 1 else 0.0
        bar = "#" * min(40, int(sd))
        print(f"  {date:<12}{topo:<16}{rnd:>4}{len(scores):>6}"
              f"{statistics.mean(scores):>8.1f}{sd:>8.1f}"
              f"{min(scores):>7.1f}{max(scores):>7.1f}  {bar}")

    # Did revisions move agents toward their peers? That is the mechanism the
    # whole project is testing, and peers_seen makes it measurable directly.
    print("\n  revision behaviour (from peers_seen):")
    for (date, topo, rnd), rs in sorted(groups.items()):
        moved = [r for r in rs if r.get("peers_seen")]
        if not moved:
            continue
        toward = away = 0
        deltas = []
        for r in moved:
            peer_mean = statistics.mean(p["score"] for p in r["peers_seen"])
            # The agent's own prior score is not in the row, so use the distance
            # between its post-revision score and the peer mean as the signal:
            # closer than half the score range means it accommodated its peers.
            deltas.append(abs(r["score"] - peer_mean))
            if abs(r["score"] - peer_mean) < 25:
                toward += 1
            else:
                away += 1
        print(f"    {date} {topo:<14} r{rnd}: {len(moved):>4} agents read peers, "
              f"mean |score - peer_mean| = {statistics.mean(deltas):5.1f}, "
              f"{toward} within 25pts / {away} beyond")

    llm_like = sum(1 for r in rows if len(r.get("thesis", "")) > 90)
    print(f"\n  theses longer than 90 chars: {llm_like:,}/{len(rows):,} "
          f"({llm_like / len(rows):.0%}) — rule-based theses are short and "
          f"templated, so a high share here means the model really answered")
    for r in rows[:2] + rows[-2:]:
        print(f"    [{r['date']}|{r['topology']}|r{r['round']}] {r['ticker']:<6}"
              f"{r['score']:>6.1f} {r['direction']:<8}{r['thesis'][:88]}")


# ==========================================================================
# run.log — grades, fallbacks, timing
# ==========================================================================
GRADE = re.compile(r"GRADE \[([^\]]+)\]: gross=([-+][\d.]+)% net=([-+][\d.]+)% "
                   r"turnover=([\d.]+) IC\(r0->final\)=([-+][\d.]+)->([-+][\d.]+)")
REBAL = re.compile(r"# \[(\d+)/(\d+)\] REBALANCE (\d{4}-\d{2}-\d{2})")
STAMP = re.compile(r"^(\d\d:\d\d:\d\d)")
USAGE = re.compile(r"usage \[([^|]+)\|([^\]]+)\] (\d+) calls, ([\d,]+) tok")


def report_log(path, want_csv=False, run_dir=None):
    if not os.path.exists(path):
        print("  (no run.log)")
        return
    grades, rebals, usages, fallbacks = [], [], [], 0
    first_stamp = last_stamp = None
    with open(path, errors="replace") as f:
        for line in f:
            m = STAMP.match(line.split(" INFO")[0].strip()) or STAMP.match(line)
            if m:
                first_stamp = first_stamp or m.group(1)
                last_stamp = m.group(1)
            r = REBAL.search(line)
            if r:
                rebals.append((int(r.group(1)), int(r.group(2)), r.group(3),
                               last_stamp))
            g = GRADE.search(line)
            if g:
                grades.append({"topology": g.group(1),
                               "gross_ret_pct": float(g.group(2)),
                               "net_ret_pct": float(g.group(3)),
                               "turnover": float(g.group(4)),
                               "rank_ic_round0": float(g.group(5)),
                               "rank_ic_final": float(g.group(6))})
            u = USAGE.search(line)
            if u:
                usages.append((u.group(1), u.group(2), int(u.group(3)),
                               int(u.group(4).replace(",", ""))))
            if "FALLBACK" in line:
                fallbacks += 1

    if rebals:
        print(f"  progress: date {rebals[-1][0]} of {rebals[-1][1]} "
              f"({rebals[-1][2]})  |  log spans {first_stamp} -> {last_stamp}")
        if len(rebals) > 1:
            print(f"  dates started so far: "
                  f"{', '.join(r[2] for r in rebals[-4:])}")
    print(f"  FALLBACK lines: {fallbacks}"
          + ("  <-- agents whose LLM output was unusable" if fallbacks else
             "  (none: every answer parsed)"))
    if usages:
        tot_calls = sum(u[2] for u in usages)
        tot_tok = sum(u[3] for u in usages)
        print(f"  rounds completed: {len(usages)}  |  {tot_calls:,} calls, "
              f"{tot_tok:,} tokens so far")
        for stage, rnd, calls, tok in usages[-3:]:
            print(f"    {stage:<16}{rnd:<10}{calls:>5} calls {tok:>10,} tok")

    if grades:
        print(f"\n  graded books: {len(grades)}")
        by_topo = defaultdict(list)
        for g in grades:
            by_topo[g["topology"]].append(g)
        print(f"    {'topology':<16}{'n':>4}{'meanGross%':>12}{'meanNet%':>10}"
              f"{'cumNet%':>9}{'meanIC_r0':>11}{'meanIC_fin':>11}")
        for topo, gs in sorted(by_topo.items()):
            cum = 1.0
            for g in gs:
                cum *= (1 + g["net_ret_pct"] / 100)
            print(f"    {topo:<16}{len(gs):>4}"
                  f"{statistics.mean(g['gross_ret_pct'] for g in gs):>12.3f}"
                  f"{statistics.mean(g['net_ret_pct'] for g in gs):>10.3f}"
                  f"{(cum - 1) * 100:>9.2f}"
                  f"{statistics.mean(g['rank_ic_round0'] for g in gs):>11.3f}"
                  f"{statistics.mean(g['rank_ic_final'] for g in gs):>11.3f}")
        print("    (IC_r0 vs IC_fin is the no-comm baseline vs post-communication "
              "signal quality — the comparison the hypothesis rests on)")
        if want_csv and run_dir:
            out = os.path.join(run_dir, "interim_returns.csv")
            import csv as _csv
            with open(out, "w", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=list(grades[0].keys()))
                w.writeheader()
                w.writerows(grades)
            print(f"    wrote {out} ({len(grades)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--transcript", default=None,
                    help="standalone transcript (e.g. the weekly job's)")
    ap.add_argument("--csv", action="store_true",
                    help="write interim_returns.csv from the GRADE lines")
    args = ap.parse_args()

    if args.transcript:
        print(f"=== transcript: {args.transcript}")
        report_transcript(read_transcript(args.transcript))
        return

    run_dir = args.run_dir or newest_run_dir()
    print("=" * 78)
    print(f"RUN DIR: {run_dir}")
    print("=" * 78)
    for name in sorted(os.listdir(run_dir)):
        size = os.path.getsize(os.path.join(run_dir, name))
        note = "  <-- written only at the end of the run" if (
            size == 0 and name != "transcript.jsonl") else ""
        print(f"  {name:<24}{size:>12,} bytes{note}")

    print("\n--- run.log ---------------------------------------------------")
    report_log(os.path.join(run_dir, "run.log"), args.csv, run_dir)
    print("\n--- transcript.jsonl ------------------------------------------")
    report_transcript(read_transcript(os.path.join(run_dir, "transcript.jsonl")))


if __name__ == "__main__":
    main()
