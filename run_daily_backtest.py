"""
run_daily_backtest.py — DAILY manager-agent backtest, end to end.

Every trading day in a leakage-safe window we run the full pipeline:

    for each daily rebalance date:
        features  = point-in-time momentum + reversal for the 30 stocks
        agents    = the 30 StockAgents debate (round 0 + N revision rounds)
        book      = the ManagerAgent reads all 30 final opinions (score,
                    direction, confidence, thesis) and builds a dollar-neutral
                    long/short portfolio with conviction weights
        ret       = the book's NEXT-day return (used only to grade, never shown)

We collect the daily return series and, at the end, report the annualised
Sharpe ratio (mean/std * sqrt(252)), the cumulative return, and the hit rate,
both gross and net of trading costs. A per-day CSV is written to results/ so you
can plot the equity curve or inspect any single day's book and the manager's
rationale.

Two modes, chosen by the LLM_BACKEND environment variable (zero API cost):

    # rule-based, runs anywhere with no model (validates the whole pipeline):
    python run_daily_backtest.py

    # local-LLM agents + local-LLM manager (a vLLM server on a GPU node):
    export LLM_BACKEND=openai-compatible LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
    export LLM_BASE_URL=http://localhost:8000/v1
    python run_daily_backtest.py

Env knobs: DEMO_N (universe size), MAX_DATES (limit days, e.g. for a smoke
test), TOPOLOGY (override config.TOPOLOGY).
"""

import csv
import json
import logging
import os

import config
from data_loader import (load_prices, rebalance_dates, compute_features,
                         forward_returns)
from stock_agent import StockAgent
from topology import build_topology
from message_bus import MessageBus, RoundScheduler
from batch_orchestration import BatchScheduler
from manager_agent import ManagerAgent
import evaluation as ev

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("daily")

START, END = "2024-10-01", "2026-07-01"
STEP_DAYS, HORIZON = 1, 1          # DAILY rebalance, graded on next-day return
MOM_LOOKBACK, REV_LOOKBACK = 63, 5
ROUNDTRIP_BPS = 20.0               # round-trip trading cost (large-cap assumption)
PERIODS_PER_YEAR = 252            # daily -> annualise the Sharpe with sqrt(252)


# --------------------------------------------------------------------------
# LLM wiring (all optional; None everywhere -> pure rule-based, zero cost)
# --------------------------------------------------------------------------
def build_llm_clients():
    """
    Return (batch_client_for_agents, llm_client_for_manager).

    Both are None unless LLM_BACKEND is set, in which case the 30 agents use the
    concurrent batch client (one call per round) and the manager uses a single
    per-day LLMClient call. See run_llm_analysis.py for the same backend choice.
    """
    backend = os.getenv("LLM_BACKEND", "none")
    if backend == "none":
        return None, None
    from llm import make_llm_client
    if backend == "openai-compatible":
        from batch_llm import HttpBatchClient
        base_url = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
        model = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        log.info("LLM agents: HTTP batch -> %s (%s)", base_url, model)
        agent_batch = HttpBatchClient(base_url, model, StockAgent.SYSTEM,
                                      StockAgent.SCHEMA)
    elif backend == "transformers":
        from batch_llm import SequentialBatchClient
        agent_batch = SequentialBatchClient(make_llm_client(), StockAgent.SYSTEM,
                                            StockAgent.SCHEMA)
    else:
        raise SystemExit(f"Unknown LLM_BACKEND={backend!r}")
    manager_llm = make_llm_client()       # one JSON call per day for the manager
    log.info("LLM manager: single-call client per day")
    return agent_batch, manager_llm


# --------------------------------------------------------------------------
# One day's agent debate -> {ticker: final StockMessage}
# --------------------------------------------------------------------------
def run_agents(tickers, sectors, feats, agent_batch, topology_name, n_rounds):
    agents = {t: StockAgent(t, sectors[t],
                            llm_client=None) for t in tickers}
    bus = MessageBus(build_topology(topology_name, tickers, sectors,
                                    degree=config.SPARSE_DEGREE, seed=config.SEED))
    if agent_batch is None:
        # Rule-based: the simple one-at-a-time scheduler is plenty fast.
        return RoundScheduler(agents, bus, n_rounds=n_rounds).run(feats)
    # LLM: batched round 0 + batched revision rounds on the same bus.
    scheduler = BatchScheduler(agents, bus, agent_batch, n_rounds)
    scheduler.initial_round(feats)
    return scheduler.revision_rounds()


def main():
    topology = os.getenv("TOPOLOGY", config.TOPOLOGY)
    tickers = list(config.UNIVERSE.keys())[:int(os.getenv("DEMO_N", 30))]
    sectors = config.UNIVERSE
    agent_batch, manager_llm = build_llm_clients()
    manager = ManagerAgent(config.N_LONG, config.N_SHORT, llm_client=manager_llm)

    log.info("Loading prices for %d stocks (%s -> %s), cache-first ...",
             len(tickers), START, END)
    prices = load_prices(tickers, START, END)
    dates = rebalance_dates(prices, STEP_DAYS, MOM_LOOKBACK, HORIZON)
    dates = dates[:int(os.getenv("MAX_DATES", len(dates)))]
    mode = "LLM" if manager_llm is not None else "rule-based"
    log.info("%d DAILY rebalance dates (%s -> %s); topology=%s; rounds=%d; "
             "manager=%s", len(dates), dates[0], dates[-1], topology,
             config.N_ROUNDS, mode)

    gross, net = [], []                    # per-day book returns
    rows = []                              # per-day detail for the CSV
    prev_w = None

    for i, asof in enumerate(dates, 1):
        feats = compute_features(prices, asof, MOM_LOOKBACK, REV_LOOKBACK)
        fwd = forward_returns(prices, asof, HORIZON)      # next-day returns

        final = run_agents(tickers, sectors, feats, agent_batch, topology,
                           config.N_ROUNDS)
        book = manager.build(final, sectors)

        g = ev.portfolio_return(book.weights, fwd)
        turn = ev.weight_turnover(prev_w, book.weights)
        n = ev.apply_cost(g, turn, ROUNDTRIP_BPS)
        gross.append(g)
        net.append(n)
        prev_w = book.weights

        rows.append({"date": asof, "gross_ret": g, "net_ret": n,
                     "turnover": turn, "longs": "|".join(book.longs),
                     "shorts": "|".join(book.shorts), "source": book.source,
                     "rationale": book.rationale})
        if i % 20 == 0 or i == len(dates):
            log.info("[%d/%d] %s  gross=%+.3f%% net=%+.3f%%  L=%s S=%s",
                     i, len(dates), asof, g * 100, n * 100, book.longs,
                     book.shorts)

    report(dates, gross, net, rows, tickers, topology, mode)


# --------------------------------------------------------------------------
# Summary: the Sharpe ratio is the headline number the user asked for.
# --------------------------------------------------------------------------
def cumulative(series):
    c = 1.0
    for r in series:
        c *= (1.0 + r)
    return c - 1.0


def report(dates, gross, net, rows, tickers, topology, mode):
    gs = ev.summarize(gross, periods_per_year=PERIODS_PER_YEAR)
    ns = ev.summarize(net, periods_per_year=PERIODS_PER_YEAR)
    gross_cum, net_cum = cumulative(gross), cumulative(net)
    hit = sum(1 for r in net if r > 0) / len(net) if net else 0.0

    print("\n" + "=" * 62)
    print(f"DAILY MANAGER-AGENT BACKTEST  ({mode}, topology={topology})")
    print(f"{len(dates)} trading days: {dates[0]} -> {dates[-1]}")
    print("=" * 62)
    print(f"{'':<16}{'gross':>12}{'net (20bps)':>14}")
    print(f"{'mean daily %':<16}{gs['mean'] * 100:>12.4f}{ns['mean'] * 100:>14.4f}")
    print(f"{'daily vol %':<16}{gs['std'] * 100:>12.4f}{ns['std'] * 100:>14.4f}")
    print(f"{'cumulative %':<16}{gross_cum * 100:>12.2f}{net_cum * 100:>14.2f}")
    print(f"{'Sharpe (ann.)':<16}{gs['sharpe']:>12.2f}{ns['sharpe']:>14.2f}")
    print(f"{'hit rate':<16}{'':>12}{hit:>14.1%}")
    print("=" * 62)
    print(f"Sharpe annualised with sqrt({PERIODS_PER_YEAR}); dollar-neutral "
          f"long/short book,\ngraded on next-day returns; costs = "
          f"{ROUNDTRIP_BPS:.0f}bps round-trip on daily turnover.")

    os.makedirs("results", exist_ok=True)
    tag = f"{mode.replace('-', '')}_{topology}_{len(tickers)}x{len(dates)}"
    csv_path = f"results/daily_backtest_{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {"mode": mode, "topology": topology, "n_days": len(dates),
               "start": dates[0], "end": dates[-1],
               "gross": {"mean": gs["mean"], "vol": gs["std"],
                         "cum": gross_cum, "sharpe": gs["sharpe"]},
               "net": {"mean": ns["mean"], "vol": ns["std"],
                       "cum": net_cum, "sharpe": ns["sharpe"], "hit_rate": hit}}
    json_path = f"results/daily_backtest_{tag}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nPer-day detail -> {csv_path}\nSummary      -> {json_path}")


if __name__ == "__main__":
    main()
