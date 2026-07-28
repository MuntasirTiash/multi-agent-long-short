"""
run_llm_demo.py — Phase 2 demo: real data + real (local, open-source) LLM agents.

Same pipeline as before, but each StockAgent now reasons with a local Qwen2.5
model instead of the arithmetic rule. Nothing else changes — that is the whole
point of the Phase-0 design.

Turn the LLM on with environment variables (zero API cost):

    # in-process HuggingFace transformers (downloads the model once):
    export LLM_BACKEND=transformers LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct
    python run_llm_demo.py

    # or a local server you started (Ollama / vLLM), better on a GPU node:
    export LLM_BACKEND=openai-compatible LLM_MODEL=qwen2.5:7b
    python run_llm_demo.py

If LLM_BACKEND is unset the agents fall back to the rule-based logic, so this
script always runs. The 0.5B model is tiny enough for a CPU login node; use 7B+
on a GPU compute node for real quality.

To keep a CPU demo quick you can shrink the universe:
    export DEMO_N=6         # use only the first 6 Dow names
"""

import os

import config
from data_loader import load_prices, rebalance_dates, compute_features
from stock_agent import StockAgent
from topology import build_topology
from message_bus import MessageBus, RoundScheduler
from aggregator import rank_and_split
from llm import make_llm_client

START, END = "2024-10-01", "2026-07-01"


def main():
    tickers = list(config.UNIVERSE.keys())
    demo_n = int(os.getenv("DEMO_N", len(tickers)))
    tickers = tickers[:demo_n]
    sectors = config.UNIVERSE

    # Build the (shared) LLM client once; None => rule-based fallback.
    llm = make_llm_client()
    mode = "rule-based (no LLM configured)" if llm is None \
        else f"LLM: {llm.backend} / {llm.model_name}"
    print(f"Reasoning mode: {mode}")
    print(f"Universe: {len(tickers)} stocks\n")

    # Real data at the most recent leakage-safe rebalance date.
    prices = load_prices(tickers, START, END)
    asof = rebalance_dates(prices, step_days=5, min_history=63, horizon_days=5)[-1]
    feats = compute_features(prices, asof)
    print(f"As-of date: {asof}\n")

    # One agent per stock, all sharing the same LLM client.
    agents = {t: StockAgent(t, sectors[t], llm_client=llm) for t in tickers}
    topology = build_topology(config.TOPOLOGY, tickers, sectors,
                              degree=config.SPARSE_DEGREE, seed=config.SEED)
    bus = MessageBus(topology)
    scheduler = RoundScheduler(agents, bus, n_rounds=config.N_ROUNDS)
    final = scheduler.run(feats)

    portfolio = rank_and_split(final, config.N_LONG, config.N_SHORT)
    print("\nFinal ranking (best to worst):")
    print("-" * 60)
    for rank, msg in enumerate(portfolio.ranking, start=1):
        print(f"{rank:2d}. {msg.summary()}  | {msg.thesis[:38]}")
    print("\n" + "=" * 40)
    print(f"LONG  book: {portfolio.longs}")
    print(f"SHORT book: {portfolio.shorts}")
    print("=" * 40)


if __name__ == "__main__":
    main()
