"""
run_analysis.py — Phase 1 preliminary walk-forward analysis on REAL data.

What it does, week by week, over a post-Oct-2024 (leakage-safe) window:

    for each weekly rebalance date:
        features   = point-in-time momentum + reversal for the 30 stocks
        fwd_return = the NEXT week's return (used only to grade, never shown)
        for each configuration (baselines + comm topologies):
            scores     = run that configuration
            record rank-IC(scores, fwd_return) and long/short spread

Then it prints, per configuration, the mean rank-IC, how often IC is positive,
the mean weekly long-short spread, an annualised Sharpe, and the cumulative
long-short return.

IMPORTANT — what this does and does NOT show. The agents are still the Phase-0
RULE-BASED placeholders (no LLM yet). So this validates the *pipeline and the
evaluation harness on real data*, and shows what the mechanical communication
rule does. It is NOT yet a test of the scientific hypothesis (that needs the
Phase-2 LLM agents). Read the numbers as "the plumbing produces sane, gradeable
signals," not "communication works."

Run:  python run_analysis.py
"""

import random

import config
from data_loader import (load_prices, rebalance_dates, compute_features,
                         forward_returns)
from stock_agent import StockAgent
from topology import build_topology
from message_bus import MessageBus, RoundScheduler
import evaluation as ev

# --- analysis window (after Qwen2.5 training cutoff) ----------------------
START = "2024-10-01"
END = "2026-07-01"
STEP_DAYS = 5        # weekly rebalance
HORIZON = 5          # grade against next week's return
MOM_LOOKBACK = 63    # ~3-month momentum
REV_LOOKBACK = 5     # 1-week reversal proxy


def run_orchestration(tickers, sectors, feats, topology_name, n_rounds):
    """Run one configuration and return {ticker: final_score}."""
    agents = {t: StockAgent(t, sectors[t]) for t in tickers}
    topology = build_topology(topology_name, tickers, sectors,
                              degree=config.SPARSE_DEGREE, seed=config.SEED)
    bus = MessageBus(topology)
    scheduler = RoundScheduler(agents, bus, n_rounds=n_rounds)
    final = scheduler.run(feats)
    return {t: m.score for t, m in final.items()}


ROUNDTRIP_BPS = 20.0     # round-trip trading cost assumption (large-cap)
MC_TRIALS = 5000         # Monte-Carlo null portfolios


def grade(scores, fwd):
    """Return (rank_ic, gross_spread, longs, shorts) for one date's scores."""
    ranked = sorted(scores, key=lambda t: scores[t], reverse=True)
    longs = ranked[:config.N_LONG]
    shorts = ranked[-config.N_SHORT:]
    ic = ev.rank_ic(scores, fwd)
    spread = ev.long_short_spread(longs, shorts, fwd)
    return ic, spread, longs, shorts


def main():
    tickers = list(config.UNIVERSE.keys())
    sectors = config.UNIVERSE

    print(f"Loading real prices for {len(tickers)} DJIA names "
          f"({START} -> {END}), cache-first ...")
    prices = load_prices(tickers, START, END)
    dates = rebalance_dates(prices, step_days=STEP_DAYS,
                            min_history=MOM_LOOKBACK, horizon_days=HORIZON)
    print(f"{len(dates)} weekly rebalance dates "
          f"from {dates[0]} to {dates[-1]}\n")

    # Configurations to compare. Each maps a name -> a scoring function(feats).
    def momentum_only(feats):
        return {t: feats[t]["momentum"] for t in feats}       # no agents at all

    def random_scores(feats, _rng=random.Random(config.SEED)):
        return {t: _rng.random() for t in feats}

    configs = {
        "random (floor)":        lambda f: random_scores(f),
        "momentum-only":         momentum_only,
        "B2 no-comm (round 0)":  lambda f: run_orchestration(tickers, sectors, f, "sparse", 0),
        "comm: full":            lambda f: run_orchestration(tickers, sectors, f, "full", config.N_ROUNDS),
        "comm: sparse":          lambda f: run_orchestration(tickers, sectors, f, "sparse", config.N_ROUNDS),
        "comm: sector":          lambda f: run_orchestration(tickers, sectors, f, "sector", config.N_ROUNDS),
    }

    ics = {name: [] for name in configs}
    gross = {name: [] for name in configs}      # per-date gross spreads
    net = {name: [] for name in configs}        # per-date spreads after costs
    prev_book = {name: (None, None) for name in configs}
    fwd_series = []                             # for the Monte-Carlo null

    for asof in dates:
        feats = compute_features(prices, asof, MOM_LOOKBACK, REV_LOOKBACK)
        fwd = forward_returns(prices, asof, HORIZON)
        fwd_series.append(fwd)
        for name, score_fn in configs.items():
            ic, spread, longs, shorts = grade(score_fn(feats), fwd)
            prev_l, prev_s = prev_book[name]
            turn = ev.turnover(prev_l, prev_s, longs, shorts)
            ics[name].append(ic)
            gross[name].append(spread)
            net[name].append(ev.apply_cost(spread, turn, ROUNDTRIP_BPS))
            prev_book[name] = (longs, shorts)

    # Monte-Carlo null: what does RANDOM stock-picking earn on these same weeks?
    print(f"Building Monte-Carlo null ({MC_TRIALS} random portfolios) ...\n")
    null = ev.monte_carlo_null(fwd_series, config.N_LONG, config.N_SHORT,
                               n_trials=MC_TRIALS, seed=config.SEED)

    def cumulative(series):
        c = 1.0
        for r in series:
            c *= (1.0 + r)
        return c - 1.0

    # --- report ----------------------------------------------------------
    print(f"{'configuration':<24}{'meanIC':>8}{'grossCum%':>10}"
          f"{'netCum%':>9}{'Sharpe':>8}{'MCpct':>7}{'p':>7}")
    print("-" * 73)
    for name in configs:
        mean_ic = sum(ics[name]) / len(ics[name])
        gross_cum = cumulative(gross[name])
        net_cum = cumulative(net[name])
        s = ev.summarize(net[name])
        sig = ev.percentile_and_p(gross_cum, null)     # vs the random null
        print(f"{name:<24}{mean_ic:>8.3f}{gross_cum * 100:>10.1f}"
              f"{net_cum * 100:>9.1f}{s['sharpe']:>8.2f}"
              f"{sig['percentile']:>7.0%}{sig['p_value']:>7.3f}")

    print(f"\nCosts: {ROUNDTRIP_BPS:.0f}bps round-trip charged on turnover. "
          f"MCpct = percentile of gross cumulative return vs {MC_TRIALS} random\n"
          f"long/short books; p = fraction of random books that did as well or "
          f"better.\nReminder: agents are still rule-based placeholders — this "
          f"validates the\nupgraded harness on real data, not the communication "
          f"hypothesis (Phase 2).")


if __name__ == "__main__":
    main()
