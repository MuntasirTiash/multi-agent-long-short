"""
run_llm_analysis.py — Phase 2 walk-forward with BATCHED local-LLM agents,
comparing NO-communication against ALL communication topologies in one job.

For each weekly rebalance date it:
  1. runs the independent round-0 assessment ONCE (topology-agnostic) -> the
     "no-comm" B2 baseline, and
  2. for each topology in TOPOLOGIES, replays the peer-revision rounds starting
     from that same round-0 -> the "comm: <topology>" treatments.

Every portfolio is graded with the full harness (rank-IC, transaction costs,
Monte-Carlo null significance). Sharing round-0 across topologies means we pay
for the expensive independent assessment only once per date.

Verbose step-by-step logging is printed as it goes (date header, per-round LLM
vs fallback counts, and each configuration's IC / spread / long+short books) so
the SLURM log reads like a narrative of what happened.

Point it at an LLM (zero API cost):

    # production: a vLLM server on a GPU node (see serve_vllm_gpu.sh)
    export LLM_BACKEND=openai-compatible LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
    export LLM_BASE_URL=http://localhost:8000/v1
    python run_llm_analysis.py

    # smoke test on a CPU login node (slow, tiny):
    export LLM_BACKEND=transformers LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct
    DEMO_N=10 MAX_DATES=1 python run_llm_analysis.py

Environment knobs: TOPOLOGIES (default "full,sparse,sector"), DEMO_N (universe
size), MAX_DATES (limit rebalance dates).
"""

import json
import logging
import os

import config
from data_loader import (load_prices, rebalance_dates, compute_features,
                         forward_returns)
from stock_agent import StockAgent
from topology import build_topology
from message_bus import MessageBus
from batch_orchestration import BatchScheduler
from batch_llm import HttpBatchClient, SequentialBatchClient
from transcript import TranscriptRecorder
import evaluation as ev

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("analysis")

START, END = "2024-10-01", "2026-07-01"
STEP_DAYS, HORIZON = 5, 5
MOM_LOOKBACK, REV_LOOKBACK = 63, 5
ROUNDTRIP_BPS = 20.0
MC_TRIALS = 5000


def build_batch_client():
    """Pick the batch client from LLM_BACKEND."""
    backend = os.getenv("LLM_BACKEND", "none")
    if backend == "openai-compatible":
        base_url = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
        model = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        log.info("Batch client: HTTP -> %s (%s)", base_url, model)
        return HttpBatchClient(base_url, model, StockAgent.SYSTEM,
                               StockAgent.SCHEMA)
    if backend == "transformers":
        from llm import make_llm_client
        llm = make_llm_client()
        log.info("Batch client: in-process transformers (%s), sequential "
                 "(CPU smoke test only)", llm.model_name)
        return SequentialBatchClient(llm, StockAgent.SYSTEM, StockAgent.SCHEMA)
    raise SystemExit("Set LLM_BACKEND=openai-compatible (vLLM) or transformers. "
                     "Use run_analysis.py for the rule-based harness.")


def books_from_scores(scores):
    ranked = sorted(scores, key=lambda t: scores[t], reverse=True)
    return ranked[:config.N_LONG], ranked[-config.N_SHORT:]


def grade_and_log(label, messages, fwd):
    """Grade a {ticker: message} snapshot, log a readable line, return metrics."""
    scores = {t: m.score for t, m in messages.items()}
    longs, shorts = books_from_scores(scores)
    ic = ev.rank_ic(scores, fwd)
    spread = ev.long_short_spread(longs, shorts, fwd)
    log.info("  %-18s IC=%+.3f spread=%+.2f%%  L=%s  S=%s",
             label, ic, spread * 100, longs, shorts)
    return ic, spread, longs, shorts


def main():
    batch_client = build_batch_client()
    topologies = os.getenv("TOPOLOGIES", "full,sparse,sector").split(",")
    tickers = list(config.UNIVERSE.keys())[:int(os.getenv("DEMO_N", 30))]
    sectors = config.UNIVERSE

    log.info("Loading prices for %d stocks (%s -> %s), cache-first ...",
             len(tickers), START, END)
    prices = load_prices(tickers, START, END)
    dates = rebalance_dates(prices, STEP_DAYS, MOM_LOOKBACK, HORIZON)
    dates = dates[:int(os.getenv("MAX_DATES", len(dates)))]
    log.info("%d rebalance dates (%s -> %s); topologies=%s; rounds=%d",
             len(dates), dates[0], dates[-1], topologies, config.N_ROUNDS)

    labels = ["no-comm (round 0)"] + [f"comm: {t}" for t in topologies]
    ics = {k: [] for k in labels}
    gross = {k: [] for k in labels}
    net = {k: [] for k in labels}
    prev = {k: (None, None) for k in labels}
    fwd_series = []

    # Full per-agent transcript: every message, every round, to a JSONL file.
    transcript_path = (f"results/transcript_{'-'.join(topologies)}"
                       f"_{len(tickers)}x{len(dates)}.jsonl")
    recorder = TranscriptRecorder(transcript_path, log_each=True)
    log.info("Recording full agent transcript -> %s", transcript_path)

    for i, asof in enumerate(dates, 1):
        log.info("==== [%d/%d] rebalance %s ==============================",
                 i, len(dates), asof)
        feats = compute_features(prices, asof, MOM_LOOKBACK, REV_LOOKBACK)
        fwd = forward_returns(prices, asof, HORIZON)
        fwd_series.append(fwd)

        # Fresh agents for this date; run the shared round-0 once (recorded
        # under topology "round0-shared" since it precedes any communication).
        agents = {t: StockAgent(t, sectors[t]) for t in tickers}
        seed_bus = MessageBus(build_topology("full", tickers, sectors))
        round0 = BatchScheduler(agents, seed_bus, batch_client, config.N_ROUNDS,
                                recorder=recorder, date=asof,
                                topology="round0-shared").initial_round(feats)

        def record(label, msgs):
            ic, spread, longs, shorts = grade_and_log(label, msgs, fwd)
            pl, ps = prev[label]
            turn = ev.turnover(pl, ps, longs, shorts)
            ics[label].append(ic)
            gross[label].append(spread)
            net[label].append(ev.apply_cost(spread, turn, ROUNDTRIP_BPS))
            prev[label] = (longs, shorts)

        record("no-comm (round 0)", round0)

        # Replay revision rounds from the same round-0 for each topology.
        for topo in topologies:
            bus = MessageBus(build_topology(topo, tickers, sectors,
                                            degree=config.SPARSE_DEGREE,
                                            seed=config.SEED))
            for msg in round0.values():
                bus.post(msg)
            final = BatchScheduler(agents, bus, batch_client, config.N_ROUNDS,
                                   recorder=recorder, date=asof,
                                   topology=topo).revision_rounds()
            record(f"comm: {topo}", final)

    log.info("Transcript complete: %d agent messages -> %s",
             recorder.count(), transcript_path)

    # -- significance + summary -------------------------------------------
    log.info("Building Monte-Carlo null (%d random portfolios) ...", MC_TRIALS)
    null = ev.monte_carlo_null(fwd_series, config.N_LONG, config.N_SHORT,
                               MC_TRIALS, config.SEED)

    def cum(series):
        c = 1.0
        for r in series:
            c *= (1.0 + r)
        return c - 1.0

    print(f"\n{'configuration':<22}{'meanIC':>8}{'grossCum%':>10}"
          f"{'netCum%':>9}{'Sharpe':>8}{'MCpct':>7}{'p':>7}")
    print("-" * 71)
    results = {}
    for label in labels:
        mean_ic = sum(ics[label]) / len(ics[label])
        g, n = cum(gross[label]), cum(net[label])
        s = ev.summarize(net[label])
        sig = ev.percentile_and_p(g, null)
        results[label] = {"mean_ic": mean_ic, "gross_cum": g, "net_cum": n,
                          "sharpe": s["sharpe"], **sig}
        print(f"{label:<22}{mean_ic:>8.3f}{g * 100:>10.1f}{n * 100:>9.1f}"
              f"{s['sharpe']:>8.2f}{sig['percentile']:>7.0%}{sig['p_value']:>7.3f}")

    os.makedirs("results", exist_ok=True)
    out = (f"results/llm_analysis_{'-'.join(topologies)}"
           f"_{len(tickers)}x{len(dates)}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out}")
    print("Read: does any 'comm' row beat 'no-comm' on netCum% AND clear the "
          "null (low p)?")


if __name__ == "__main__":
    main()
