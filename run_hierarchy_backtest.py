"""
run_hierarchy_backtest.py — daily backtest of the THREE-TIER industry-leader
setting, graded side by side with the flat manager on identical inputs.

Per rebalance date:

    ROUND 0    all stock agents form independent opinions (shared, so every
               arm below is graded on exactly the same analyst views)
    ---- arm A: HIERARCHY -------------------------------------------------
    tier 2     each industry's leader (largest member, point-in-time) reads its
               whole industry INCLUDING ITSELF -> IndustryReport
    council    leaders read each other's reports and recentre their industries
    tier 3     IndustryManagerAgent turns the reports into a dollar-neutral book
    ---- arm B: FLAT (matched baseline) ----------------------------------
    manager    the ordinary ManagerAgent reads all 498 opinions directly
    ---------------------------------------------------------------------
    grade      each book on the NEXT day's return (never shown to an agent)

Arm B exists because "the hierarchy worked" is uninterpretable on its own: the
hierarchy changes both the routing AND the aggregator, so it has to be compared
against the flat manager on the same round-0 opinions. Add the random-leader
control (LEADER_METRIC=random) to separate "the largest firm's view carries
information" from "any designated aggregator helps".

Zero API cost either way, chosen by LLM_BACKEND:
    python run_hierarchy_backtest.py                    # rule-based, runs anywhere
    export LLM_BACKEND=openai-compatible \
           LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
           LLM_BASE_URL=http://localhost:8000/v1
    python run_hierarchy_backtest.py                    # LLM agents + leaders + manager

Env knobs: MAX_DATES, DEMO_N, LEADER_METRIC, COUNCIL_ROUNDS, ARMS
(comma list from hierarchy,flat).
"""

import csv
import datetime
import json
import logging
import os

import config
import evaluation as ev
from data_loader import load_prices, load_ohlcv, rebalance_dates, forward_returns
from hierarchy import IndustryManagerAgent, LeaderSelector, run_hierarchy
from manager_agent import ManagerAgent
from stock_agent import StockAgent

log = logging.getLogger("hierarchy_bt")

START, END = "2024-10-01", "2026-07-01"
STEP_DAYS, HORIZON = 1, 1
MOM_LOOKBACK, REV_LOOKBACK = 63, 5
ROUNDTRIP_BPS = 20.0
PERIODS_PER_YEAR = 252


# ==========================================================================
# Setup
# ==========================================================================
def setup_run_dir(mode, arms):
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("results",
                           f"hier_{stamp}_{mode.replace('-', '')}_"
                           f"{'-'.join(arms)}_{config.LEADER_METRIC}")
    os.makedirs(run_dir, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(),
              logging.FileHandler(os.path.join(run_dir, "run.log"))):
        h.setFormatter(fmt)
        root.addHandler(h)
    return run_dir


def build_llm_clients(usage=None):
    """(agent_batch, leader_llm, manager_llm, mode) — all None => rule-based."""
    backend = os.getenv("LLM_BACKEND", "none")
    if backend == "none":
        return None, None, None, "rule-based"
    from batch_llm import HttpBatchClient
    if backend == "openai-compatible":
        base = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
        model = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        agent_batch = HttpBatchClient(base, model, StockAgent.SYSTEM,
                                      StockAgent.SCHEMA)

        def single(system, schema, max_tokens=512):
            """One-shot JSON client over the same proven HTTP path."""
            client = HttpBatchClient(base, model, system, schema,
                                     max_tokens=max_tokens)

            class _Adapter:
                date, topology = "?", "-"

                def generate_json(self, prompt, schema=None,
                                  system_message=None):
                    out = client.generate_json_batch([prompt])[0]
                    if usage is not None:
                        from instrumentation import RoundUsage
                        pt, ct, lat, exact = client.pop_usage()
                        if not exact:
                            pt, ct = usage.counter.count(prompt), 0
                        usage.record(RoundUsage(
                            date=self.date, topology=self.topology,
                            stage=self.stage, n_calls=1, prompt_tokens=pt,
                            completion_tokens=ct,
                            wall_s=sum(lat) if lat else 0.0, latencies=lat,
                            source="exact" if exact else "estimated"))
                    return out

            return _Adapter()

        from hierarchy import IndustryLeader
        leader_llm = single(IndustryLeader.SYSTEM, IndustryLeader.SCHEMA)
        leader_llm.stage = "leader"
        manager_llm = single(IndustryManagerAgent.SYSTEM,
                             IndustryManagerAgent.SCHEMA)
        manager_llm.stage = "manager"
    elif backend == "transformers":
        from batch_llm import SequentialBatchClient
        from llm import make_llm_client
        shared = make_llm_client()
        agent_batch = SequentialBatchClient(shared, StockAgent.SYSTEM,
                                            StockAgent.SCHEMA)
        leader_llm = manager_llm = shared
        if usage is not None:
            from instrumentation import CountingLLM
            leader_llm = CountingLLM(shared, usage, stage="leader")
            manager_llm = CountingLLM(shared, usage, stage="manager")
    else:
        raise SystemExit(f"Unknown LLM_BACKEND={backend!r}")
    return agent_batch, leader_llm, manager_llm, f"LLM({os.getenv('LLM_MODEL','?')})"


# ==========================================================================
# One date
# ==========================================================================
def round0(agents, tickers, feats, agent_batch, usage, asof):
    """Shared independent assessment — identical inputs for every arm."""
    prompts = [agents[t].initial_prompt(feats[t]) for t in tickers]
    rules = [agents[t].rule_initial(feats[t]) for t in tickers]
    if agent_batch is None:
        datas = [None] * len(prompts)
    else:
        datas = agent_batch.generate_json_batch(prompts)
    if usage is not None:
        from instrumentation import RoundUsage
        pt = ct = 0
        lat, exact = [], False
        if agent_batch is not None and hasattr(agent_batch, "pop_usage"):
            pt, ct, lat, exact = agent_batch.pop_usage()
        if not exact:
            pt = sum(usage.counter.count(p) for p in prompts)
            ct = 256 * len(prompts) if agent_batch is None else 0
        usage.record(RoundUsage(date=asof, topology="tier1", stage="round0",
                                n_calls=len(prompts), prompt_tokens=pt,
                                completion_tokens=ct, latencies=lat,
                                wall_s=sum(lat) if lat else 0.0,
                                source="exact" if exact else "estimated",
                                llm_called=agent_batch is not None))
    out = {}
    for t, data, rule in zip(tickers, datas, rules):
        out[t] = rule if data is None else agents[t].message_from_data(data, 0,
                                                                       rule)
    return out


def log_reports(reports, leaders):
    log.info("  tier 2 — %d industry reports", len(reports))
    for r in sorted(reports, key=lambda x: x.mean_score(), reverse=True):
        log.info("    %-24s leader=%-6s [%s] mean=%5.1f  L=%s  S=%s",
                 r.industry, r.leader, r.source, r.mean_score(),
                 [f"{p.ticker}:{p.score:.0f}" for p in r.longs],
                 [f"{p.ticker}:{p.score:.0f}" for p in r.shorts])
        log.info("      outlook: %s", r.outlook)


# ==========================================================================
# Main
# ==========================================================================
def main():
    arms = [a.strip() for a in os.getenv("ARMS", "hierarchy,flat").split(",")
            if a.strip()]
    council_rounds = int(os.getenv("COUNCIL_ROUNDS",
                                   config.LEADER_COUNCIL_ROUNDS))
    metric = os.getenv("LEADER_METRIC", config.LEADER_METRIC)
    config.LEADER_METRIC = metric              # so the run dir name matches

    all_tickers = list(config.UNIVERSE)
    tickers = all_tickers[:int(os.getenv("DEMO_N", len(all_tickers)))]
    industries = {t: config.UNIVERSE[t] for t in tickers}

    from instrumentation import TokenCounter, UsageLog
    usage = UsageLog(TokenCounter(os.getenv("LLM_MODEL")))
    agent_batch, leader_llm, manager_llm, mode = build_llm_clients(usage)

    run_dir = setup_run_dir(mode, arms)
    log.info("=" * 78)
    log.info("RESULTS DIR: %s", os.path.abspath(run_dir))
    log.info("=" * 78)

    prices = load_prices(tickers, START, END)
    ohlcv = load_ohlcv(tickers, START, END)
    dates = rebalance_dates(prices, STEP_DAYS, MOM_LOOKBACK, HORIZON)
    total_dates = len(dates)
    dates = dates[:int(os.getenv("MAX_DATES", len(dates)))]

    from features import make_feature_fn
    feature_fn = make_feature_fn(prices, tickers, START, END, industries,
                                 MOM_LOOKBACK, REV_LOOKBACK)

    shares = {}
    if metric == "market_cap":
        from fetch_market_cap import load_shares
        shares = load_shares()
        if not shares:
            raise SystemExit(
                "LEADER_METRIC=market_cap but data/shares_outstanding/ is empty."
                "\nRun:  python fetch_market_cap.py")
    selector = LeaderSelector(ohlcv, industries, metric=metric,
                              window=config.LEADER_SIZE_WINDOW,
                              seed=config.SEED, shares=shares)
    cov = selector.coverage(dates[0])
    log.info("MODE=%s | %d dates (%s -> %s) | arms=%s | leader metric=%s "
             "(usable for %d/%d firms, %.0f%%) | council=%d round(s) on %s | "
             "industry-neutral=%s", mode, len(dates), dates[0], dates[-1], arms,
             metric, cov["usable"], cov["total"], cov["pct"], council_rounds,
             config.LEADER_COUNCIL_TOPOLOGY, config.INDUSTRY_NEUTRAL)
    if cov["pct"] < 100:
        log.warning("leader metric unavailable for %d firms — they cannot be "
                    "elected leader, which biases the setting toward firms with "
                    "clean data", cov["total"] - cov["usable"])

    hier_manager = IndustryManagerAgent(
        config.N_LONG, config.N_SHORT, llm_client=manager_llm,
        sizing=config.BOOK_SIZING, min_positions=config.MIN_POSITIONS,
        max_positions=config.MAX_POSITIONS,
        long_threshold=config.LONG_SCORE_THRESHOLD,
        short_threshold=config.SHORT_SCORE_THRESHOLD,
        industry_neutral=config.INDUSTRY_NEUTRAL)
    flat_manager = ManagerAgent(
        config.N_LONG, config.N_SHORT, llm_client=manager_llm,
        sizing=config.BOOK_SIZING, min_positions=config.MIN_POSITIONS,
        max_positions=config.MAX_POSITIONS,
        long_threshold=config.LONG_SCORE_THRESHOLD,
        short_threshold=config.SHORT_SCORE_THRESHOLD,
        shortlist=config.MANAGER_SHORTLIST)

    series = {a: {"gross": [], "net": [], "ic": []} for a in arms}
    prev_w = {a: None for a in arms}
    rows, leader_rows = [], []

    for i, asof in enumerate(dates, 1):
        log.info("\n" + "#" * 78)
        log.info("# [%d/%d] %s", i, len(dates), asof)
        log.info("#" * 78)
        for client in (leader_llm, manager_llm):
            if hasattr(client, "date"):
                client.date = asof
        feats = feature_fn(asof)
        fwd = forward_returns(prices, asof, HORIZON)

        agents = {t: StockAgent(t, industries[t]) for t in tickers}
        msgs = round0(agents, tickers, feats, agent_batch, usage, asof)
        ic = ev.rank_ic({t: m.score for t, m in msgs.items()}, fwd)
        log.info("  tier 1 — %d opinions, round-0 rank-IC %+.3f", len(msgs), ic)

        books = {}
        if "hierarchy" in arms:
            leaders = selector.leaders(asof)
            council_graph = None
            if config.LEADER_COUNCIL_TOPOLOGY != "full" and council_rounds:
                import topology as topo
                lead_tickers = sorted(leaders.values())
                council_graph = topo.build_topology(
                    config.LEADER_COUNCIL_TOPOLOGY, lead_tickers,
                    {t: industries[t] for t in lead_tickers},
                    degree=config.SPARSE_DEGREE, seed=config.SEED)
            reports, stats = run_hierarchy(
                msgs, industries, leaders, llm_client=leader_llm,
                picks_per_side=config.LEADER_PICKS_PER_SIDE,
                council_rounds=council_rounds, council_graph=council_graph,
                table_cap=config.LEADER_TABLE_CAP)
            log_reports(reports, leaders)
            if stats["rule_reports"] and mode != "rule-based":
                log.warning("  ! %d/%d leader reports FELL BACK to rule",
                            stats["rule_reports"], stats["n_industries"])
            books["hierarchy"] = hier_manager.build(reports)
            for r in reports:
                leader_rows.append({
                    "date": asof, "industry": r.industry, "leader": r.leader,
                    "source": r.source, "n_members": r.n_members,
                    "mean_score": round(r.mean_score(), 2),
                    "longs": "|".join(p.ticker for p in r.longs),
                    "shorts": "|".join(p.ticker for p in r.shorts),
                    "outlook": r.outlook})
        if "flat" in arms:
            books["flat"] = flat_manager.build(msgs, industries)

        for arm, book in books.items():
            g = ev.portfolio_return(book.weights, fwd)
            turn = ev.weight_turnover(prev_w[arm], book.weights)
            n = ev.apply_cost(g, turn, ROUNDTRIP_BPS)
            prev_w[arm] = book.weights
            series[arm]["gross"].append(g)
            series[arm]["net"].append(n)
            series[arm]["ic"].append(ic)
            log.info("  %-10s [%s] %2d long / %2d short  gross=%+.3f%% "
                     "net=%+.3f%% turnover=%.2f  gross_exp=%.2f net_exp=%+.0e",
                     arm.upper(), book.source, len(book.longs),
                     len(book.shorts), g * 100, n * 100, turn,
                     book.gross_exposure(), book.net_exposure())
            log.info("    rationale: %s", book.rationale)
            rows.append({"date": asof, "arm": arm, "gross_ret": g, "net_ret": n,
                         "turnover": turn, "manager_source": book.source,
                         "n_long": len(book.longs), "n_short": len(book.shorts),
                         "longs": "|".join(book.longs),
                         "shorts": "|".join(book.shorts),
                         "rationale": book.rationale})

    report(run_dir, mode, arms, dates, series, rows, leader_rows, usage,
           total_dates, metric, council_rounds)


def _cum(xs):
    c = 1.0
    for r in xs:
        c *= (1.0 + r)
    return c - 1.0


def report(run_dir, mode, arms, dates, series, rows, leader_rows, usage,
           total_dates, metric, council_rounds):
    metrics = {}
    log.info("\n" + "=" * 78)
    log.info("SUMMARY  mode=%s  %d days  %s -> %s  leader=%s  council=%d",
             mode, len(dates), dates[0], dates[-1], metric, council_rounds)
    log.info("-" * 78)
    log.info("%-12s %9s %9s %9s %9s %7s", "arm", "grSharpe", "grCum%",
             "ntSharpe", "ntCum%", "hit%")
    for arm in arms:
        s = series[arm]
        if not s["gross"]:
            continue
        gs = ev.summarize(s["gross"], periods_per_year=PERIODS_PER_YEAR)
        ns = ev.summarize(s["net"], periods_per_year=PERIODS_PER_YEAR)
        hit = sum(1 for r in s["net"] if r > 0) / len(s["net"])
        metrics[arm] = {
            "n_days": len(s["gross"]),
            "gross": {"sharpe": gs["sharpe"], "cum": _cum(s["gross"]),
                      "mean": gs["mean"], "vol": gs["std"]},
            "net": {"sharpe": ns["sharpe"], "cum": _cum(s["net"]),
                    "mean": ns["mean"], "vol": ns["std"], "hit_rate": hit},
            "mean_rank_ic_round0": sum(s["ic"]) / len(s["ic"])}
        log.info("%-12s %9.2f %9.2f %9.2f %9.2f %7.0f", arm,
                 gs["sharpe"], _cum(s["gross"]) * 100, ns["sharpe"],
                 _cum(s["net"]) * 100, hit * 100)
    log.info("=" * 78)
    if "hierarchy" in metrics and "flat" in metrics:
        d = (metrics["hierarchy"]["net"]["cum"] - metrics["flat"]["net"]["cum"])
        log.info("hierarchy minus flat, net cumulative: %+.2f%% — both graded on "
                 "IDENTICAL round-0 opinions, so this isolates routing +"
                 " aggregation.", d * 100)
        log.info("Not yet evidence: needs the random-leader control "
                 "(LEADER_METRIC=random) and a matched Monte-Carlo null.")

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump({"mode": mode, "leader_metric": metric,
                   "council_rounds": council_rounds,
                   "industry_neutral": config.INDUSTRY_NEUTRAL,
                   "grouping": config.HIERARCHY_GROUPING,
                   "arms": metrics}, f, indent=2)
    for name, data in (("daily_returns.csv", rows),
                       ("industry_reports.csv", leader_rows)):
        if data:
            with open(os.path.join(run_dir, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
                w.writeheader()
                w.writerows(data)
    log.info("=" * 78)
    usage.report(total_dates=total_dates, n_topologies=len(arms))
    usage.write_csv(os.path.join(run_dir, "token_usage.csv"))
    log.info("\nRESULTS DIR: %s", os.path.abspath(run_dir))
    log.info("  run.log / metrics.json / daily_returns.csv")
    log.info("  industry_reports.csv  one row per (date, industry): leader, "
             "picks, outlook")
    log.info("  token_usage.csv       tokens + seconds per stage")


if __name__ == "__main__":
    main()
