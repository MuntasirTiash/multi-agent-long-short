"""
run_demo.py — Phase 0 end-to-end demo.

Run it with:   python run_demo.py

It wires together every module and executes one full rebalance:

    fake data -> agents form views -> agents talk (topology) -> rank -> book

Everything here is zero-cost and offline: the "market data" is generated with a
seeded random number generator, and the agents reason with a simple rule (no
LLM). This proves the plumbing works. Phase 1 swaps the fake data for real
point-in-time DJIA prices; Phase 2 swaps the rule for a local open-source LLM.
"""

import logging
import random

import config
from stock_agent import StockAgent
from topology import build_topology
from message_bus import MessageBus, RoundScheduler
from aggregator import rank_and_split

logging.basicConfig(level=logging.INFO, format="%(message)s")


def make_fake_market_data(tickers, seed):
    """
    Build a placeholder data dict per ticker.

    Each stock gets a random `momentum` and `value` in [-1, 1]. This stands in
    for the real features Phase 1 will compute from prices/fundamentals. Seeded,
    so the demo is fully reproducible.
    """
    rng = random.Random(seed)
    data = {}
    for t in tickers:
        data[t] = {
            "momentum": rng.uniform(-1, 1),
            "value": rng.uniform(-1, 1),
            "date": "2024-11-01",   # a post-training-cutoff date (leakage-safe)
        }
    return data


def main():
    tickers = list(config.UNIVERSE.keys())

    # 1. Create one agent per stock.
    agents = {t: StockAgent(ticker=t, sector=config.UNIVERSE[t])
              for t in tickers}

    # 2. Decide who talks to whom.
    topology = build_topology(
        config.TOPOLOGY, tickers, config.UNIVERSE,
        degree=config.SPARSE_DEGREE, seed=config.SEED,
    )
    print(f"\nTopology: {config.TOPOLOGY} "
          f"(e.g. AAPL hears from {topology['AAPL']})")

    # 3. Generate placeholder market data.
    market_data = make_fake_market_data(tickers, config.SEED)

    # 4. Run the multi-round conversation.
    bus = MessageBus(topology)
    scheduler = RoundScheduler(agents, bus, n_rounds=config.N_ROUNDS)
    final_messages = scheduler.run(market_data)

    # 5. Turn final scores into a long/short book.
    portfolio = rank_and_split(final_messages, config.N_LONG, config.N_SHORT)

    # 6. Show the result.
    print("\nFinal ranking (best to worst):")
    print("-" * 40)
    for rank, msg in enumerate(portfolio.ranking, start=1):
        print(f"{rank:2d}. {msg.summary()}")

    print("\n" + "=" * 40)
    print(f"LONG  book: {portfolio.longs}")
    print(f"SHORT book: {portfolio.shorts}")
    print("=" * 40)


if __name__ == "__main__":
    main()
