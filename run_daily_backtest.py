"""
run_daily_backtest.py — DAILY manager-agent backtest with FULL observability.

Every trading day in a leakage-safe window we run the whole pipeline and log it
in exhaustive detail (no blind spots), for one or several communication
topologies at once so their behaviour can be compared side by side:

    for each daily rebalance date:
        features = point-in-time momentum + reversal for the 30 stocks
        ROUND 0  = every StockAgent's INDEPENDENT opinion (topology-agnostic;
                   computed once and shared as the matched no-comm baseline)
        for each topology:
            ROUND 1..N = agents revise after reading peers on that topology
            MANAGER    = reads all 30 final opinions -> dollar-neutral book
        return   = each book's NEXT-day return (grading only; never shown)

WHAT GETS LOGGED (to <run_dir>/run.log AND the console):
  * RESULTS DIR path, printed at the start and again at the end.
  * FALLBACKS: every time an agent's (or the manager's) LLM output is unusable
    and the rule-based twin is used instead — with the ticker, round, topology,
    and the RAW model text + parse error. A "LLM" run that is silently mostly
    rule-based is the main thing we refuse to hide (see CLAUDE.md).
  * RANKING after round 0 and after every revision round: the 30 stocks sorted
    by score, each shown with its NEXT-DAY return and the rank-IC of that stage.
  * MEMORY of every agent after each round (its own score/direction history).
  * MANAGER DECISION: the full 30-row table it saw, its rationale, and the
    weighted long/short book it produced.
  * Full theses everywhere — never truncated.

METRICS are written as machine-readable files, not just printed:
  * metrics.json / metrics.csv  — per-topology summary (Sharpe, cum, hit, etc.)
  * daily_returns.csv           — one row per (date, topology)
  * fallbacks.csv               — one row per fallback event
  * transcript.jsonl            — every agent message, every round (grep/pandas)

Modes (zero API cost either way), chosen by LLM_BACKEND:
    python run_daily_backtest.py                      # rule-based, runs anywhere
    export LLM_BACKEND=openai-compatible \
           LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
           LLM_BASE_URL=http://localhost:8000/v1
    python run_daily_backtest.py                      # local-LLM agents + manager

Env knobs: TOPOLOGIES (comma list, default full,sparse,sector), DEMO_N,
MAX_DATES, ROUNDS (override config.N_ROUNDS).
"""

import csv
import datetime
import json
import logging
import os

import config
from data_loader import (load_prices, rebalance_dates, compute_features,
                         forward_returns)
from stock_agent import StockAgent
from topology import build_topology
from message_bus import MessageBus
from manager_agent import ManagerAgent
from transcript import TranscriptRecorder
import evaluation as ev

log = logging.getLogger("daily")

START, END = "2024-10-01", "2026-07-01"
STEP_DAYS, HORIZON = 1, 1          # DAILY rebalance, graded on next-day return
MOM_LOOKBACK, REV_LOOKBACK = 63, 5
ROUNDTRIP_BPS = 20.0              # round-trip trading cost (large-cap assumption)
PERIODS_PER_YEAR = 252           # daily -> annualise Sharpe with sqrt(252)


# ==========================================================================
# LLM wiring (all optional; None everywhere -> pure rule-based, zero cost)
# ==========================================================================
def _http_manager_llm(base_url, model):
    """Adapter so the manager reaches vLLM through the same proven HTTP path."""
    from batch_llm import HttpBatchClient
    client = HttpBatchClient(base_url, model, ManagerAgent.SYSTEM,
                             ManagerAgent.SCHEMA, max_tokens=512)

    class _Adapter:
        def generate_json(self, prompt, schema=None, system_message=None):
            return client.generate_json_batch([prompt])[0]

    return _Adapter()


def build_llm_clients():
    """
    Return (agent_batch_client, manager_llm, mode_str).

    Both clients are None (=> rule-based) unless LLM_BACKEND is set. The 30
    agents use the concurrent batch client (one call per round); the manager
    uses a single per-day JSON call.
    """
    backend = os.getenv("LLM_BACKEND", "none")
    if backend == "none":
        return None, None, "rule-based"
    if backend == "openai-compatible":
        from batch_llm import HttpBatchClient
        base_url = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
        model = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        log.info("LLM agents : HTTP batch -> %s (%s)", base_url, model)
        agent_batch = HttpBatchClient(base_url, model, StockAgent.SYSTEM,
                                      StockAgent.SCHEMA)
        manager_llm = _http_manager_llm(base_url, model)
    elif backend == "transformers":
        from batch_llm import SequentialBatchClient
        from llm import make_llm_client
        shared = make_llm_client()
        agent_batch = SequentialBatchClient(shared, StockAgent.SYSTEM,
                                            StockAgent.SCHEMA)
        manager_llm = shared              # same in-process model, single call
    else:
        raise SystemExit(f"Unknown LLM_BACKEND={backend!r}")
    log.info("LLM manager: single JSON call per day")
    return agent_batch, manager_llm, f"LLM({os.getenv('LLM_MODEL', '?')})"


# ==========================================================================
# One round of the debate, batched, with per-agent fallback capture
# ==========================================================================
def _usable(data) -> bool:
    """True iff `data` is a parseable LLM answer (mirrors message_from_data)."""
    if not isinstance(data, dict):
        return False
    try:
        float(data["score"])
        return True
    except (KeyError, TypeError, ValueError):
        return False


def _generate(agent_batch, prompts):
    """LLM batch, or a list of None (=> use the rule, not a fallback)."""
    if agent_batch is None:
        return [None] * len(prompts)
    return agent_batch.generate_json_batch(prompts)


def run_round0(agents, tickers, feats, agent_batch, recorder, date, topo_label):
    """Round 0: independent assessment. Returns (messages, fallback_events)."""
    prompts = [agents[t].initial_prompt(feats[t]) for t in tickers]
    rules = [agents[t].rule_initial(feats[t]) for t in tickers]
    datas = _generate(agent_batch, prompts)

    messages, fallbacks = {}, []
    for t, data, rule in zip(tickers, datas, rules):
        if data is None:                         # rule mode: no LLM in play
            msg = rule
        else:
            msg = agents[t].message_from_data(data, 0, rule)
            if not _usable(data):
                fallbacks.append(_fb(date, topo_label, 0, t, data))
        messages[t] = msg
        agents[t].memory.remember(date, msg.score, msg.direction.value)
        recorder.record(date, topo_label, msg)
    return messages, fallbacks


def run_revision(agents, tickers, bus, r, agent_batch, recorder, date, topo):
    """One revision round on `bus`. Returns (messages, fallback_events)."""
    active, prompts, rules, seen = [], [], [], []
    for t in tickers:
        peers = bus.inbox_for(t)
        if not peers:                            # isolated agent keeps its view
            continue
        active.append(t)
        seen.append(peers)
        prompts.append(agents[t].revise_prompt(bus.latest[t], peers))
        rules.append(agents[t].rule_revise(bus.latest[t], peers))
    datas = _generate(agent_batch, prompts)

    revised, fallbacks = {}, []
    for t, data, rule, peers in zip(active, datas, rules, seen):
        if data is None:
            msg = rule
        else:
            msg = agents[t].message_from_data(data, r, rule)
            if not _usable(data):
                fallbacks.append(_fb(date, topo, r, t, data))
        revised[t] = msg
        recorder.record(date, topo, msg, peers_seen=peers)
    for t, msg in revised.items():               # post the whole snapshot at once
        bus.post(msg)
        agents[t].memory.remember(date, msg.score, msg.direction.value)
    return dict(bus.latest), fallbacks


def _fb(date, topo, r, ticker, data):
    """Build one fallback record and log it loudly with the raw model text."""
    raw = str(data.get("_raw", ""))[:800]
    err = str(data.get("_error", "no 'score' field"))
    log.warning("  ! FALLBACK  %-5s round %d [%s]  -> used RULE.  error=%s",
                ticker, r, topo, err)
    if raw:
        log.warning("      raw model output: %s", raw)
    return {"date": date, "topology": topo, "round": r, "ticker": ticker,
            "error": err, "raw": raw}


# ==========================================================================
# Logging helpers (rankings, memory, manager decision) — full detail
# ==========================================================================
def log_ranking(tag, messages, fwd):
    """Log the 30 stocks best->worst with next-day return; return the rank-IC."""
    scores = {t: m.score for t, m in messages.items()}
    ic = ev.rank_ic(scores, fwd)
    ranked = sorted(messages.values(), key=lambda m: m.score, reverse=True)
    log.info("  RANKING after %s  (rank-IC vs next-day return = %+.3f):", tag, ic)
    for i, m in enumerate(ranked, 1):
        rv = fwd.get(m.ticker)
        nxt = f"{rv * 100:+6.2f}%" if rv is not None else "   n/a"
        log.info("    #%2d %-5s score=%6.2f %-7s conf=%.2f  next_day=%s | %s",
                 i, m.ticker, m.score, m.direction.value, m.confidence, nxt,
                 m.thesis)
    return ic


def log_memory(tag, agents, tickers):
    """Dump each agent's own score/direction history after `tag`."""
    log.info("  MEMORY after %s (each agent's own call history, round order):",
             tag)
    for t in tickers:
        calls = agents[t].memory.calls
        trail = "  ".join(f"r{i}:{c.score:.1f}/{c.direction[:4]}"
                          for i, c in enumerate(calls))
        log.info("    %-5s %s", t, trail)


def log_manager(book, messages, fwd):
    """Log exactly how the manager built the book, with full theses."""
    tag = "LLM" if book.source == "llm" else "RULE (fell back / no model)"
    log.info("  MANAGER DECISION  [%s]", tag)
    log.info("    rationale: %s", book.rationale or "(none)")
    log.info("    gross exposure=%.2f  net exposure=%+.3f (0 = dollar-neutral)",
             book.gross_exposure(), book.net_exposure())
    for side, names in (("LONG ", book.longs), ("SHORT", book.shorts)):
        log.info("    %s book:", side)
        for t in names:
            m = messages[t]
            rv = fwd.get(t)
            nxt = f"{rv * 100:+6.2f}%" if rv is not None else "   n/a"
            log.info("      %-5s weight=%+.3f  score=%6.2f conf=%.2f "
                     "next_day=%s | %s", t, book.weights.get(t, 0.0), m.score,
                     m.confidence, nxt, m.thesis)


# ==========================================================================
# Run directory + logging setup
# ==========================================================================
def setup_run_dir(mode, topologies):
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_mode = mode.replace("/", "-").replace(" ", "")
    run_dir = os.path.join("results", f"daily_{stamp}_{safe_mode}_"
                           f"{'-'.join(topologies)}")
    os.makedirs(run_dir, exist_ok=True)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    fileh = logging.FileHandler(os.path.join(run_dir, "run.log"))
    fileh.setFormatter(fmt)
    # Replace any existing handlers so we don't double-log on re-entry.
    root.handlers = [console, fileh]
    return run_dir


# ==========================================================================
# Main
# ==========================================================================
def main():
    topologies = os.getenv("TOPOLOGIES", "full,sparse,sector").split(",")
    topologies = [t.strip() for t in topologies if t.strip()]
    n_rounds = int(os.getenv("ROUNDS", config.N_ROUNDS))
    all_tickers = list(config.UNIVERSE.keys())
    tickers = all_tickers[:int(os.getenv("DEMO_N", len(all_tickers)))]
    sectors = config.UNIVERSE

    # We need the mode string before the run dir, but the LLM banner logs go to
    # the handlers set up here; build a provisional mode, finalise after clients.
    backend = os.getenv("LLM_BACKEND", "none")
    prelim_mode = "rule-based" if backend == "none" else f"LLM-{backend}"
    run_dir = setup_run_dir(prelim_mode, topologies)
    log.info("=" * 70)
    log.info("RESULTS DIR: %s", os.path.abspath(run_dir))
    log.info("=" * 70)

    agent_batch, manager_llm, mode = build_llm_clients()
    # BOOK_SIZING="flexible" hands the position count to the manager itself;
    # "fixed" pins it to N_LONG/N_SHORT. See config.py.
    sizing = os.getenv("BOOK_SIZING", config.BOOK_SIZING)
    manager = ManagerAgent(config.N_LONG, config.N_SHORT,
                           llm_client=manager_llm, sizing=sizing,
                           min_positions=config.MIN_POSITIONS,
                           max_positions=config.MAX_POSITIONS,
                           long_threshold=config.LONG_SCORE_THRESHOLD,
                           short_threshold=config.SHORT_SCORE_THRESHOLD,
                           shortlist=config.MANAGER_SHORTLIST)
    recorder = TranscriptRecorder(os.path.join(run_dir, "transcript.jsonl"),
                                 log_each=True)

    log.info("Loading prices for %d stocks (%s -> %s), cache-first ...",
             len(tickers), START, END)
    prices = load_prices(tickers, START, END)
    dates = rebalance_dates(prices, STEP_DAYS, MOM_LOOKBACK, HORIZON)
    dates = dates[:int(os.getenv("MAX_DATES", len(dates)))]
    # Feature set per config.FEATURE_SET ("extended" adds the technical,
    # risk, volume and cross-sectional features to the prompt; the rule score is
    # unaffected either way). Built once — it indexes the OHLCV history.
    from features import make_feature_fn
    feature_fn = make_feature_fn(prices, tickers, START, END, sectors,
                                 MOM_LOOKBACK, REV_LOOKBACK)
    log.info("MODE=%s | %d DAILY dates (%s -> %s) | topologies=%s | rounds=%d | "
             "manager=%s | features=%s", mode, len(dates), dates[0], dates[-1],
             topologies, n_rounds, "LLM" if manager_llm else "rule",
             config.FEATURE_SET)
    if sizing == "flexible":
        log.info("book sizing=flexible: manager picks %d-%d names per side "
                 "(rule gate: long score>=%.0f, short score<=%.0f)",
                 config.MIN_POSITIONS, config.MAX_POSITIONS,
                 config.LONG_SCORE_THRESHOLD, config.SHORT_SCORE_THRESHOLD)
    else:
        log.info("book sizing=fixed: %d longs / %d shorts every rebalance",
                 config.N_LONG, config.N_SHORT)

    # Per-topology accumulators.
    series = {topo: {"gross": [], "net": [], "ic0": [], "icF": []}
              for topo in topologies}
    prev_w = {topo: None for topo in topologies}
    daily_rows, fb_rows = [], []
    n_prompts_total, n_fallbacks_total = 0, 0

    for i, asof in enumerate(dates, 1):
        log.info("\n" + "#" * 70)
        log.info("# [%d/%d] REBALANCE %s", i, len(dates), asof)
        log.info("#" * 70)
        feats = feature_fn(asof)
        fwd = forward_returns(prices, asof, HORIZON)          # next-day returns

        # ---- Round 0: shared independent assessment (the no-comm baseline) ----
        base_agents = {t: StockAgent(t, sectors[t]) for t in tickers}
        round0, fb0 = run_round0(base_agents, tickers, feats, agent_batch,
                                 recorder, asof, "round0-shared")
        n_prompts_total += len(tickers)
        n_fallbacks_total += len(fb0)
        fb_rows.extend(fb0)
        log.info("Round 0: %d/%d agents answered by LLM, %d fell back to rule",
                 len(tickers) - len(fb0), len(tickers), len(fb0))
        log_ranking("ROUND 0 (independent)", round0, fwd)
        log_memory("ROUND 0", base_agents, tickers)

        # ---- Each topology: revise from the SAME round 0, then manage ----
        for topo in topologies:
            log.info("\n----- topology=%s -----", topo)
            agents = {t: StockAgent(t, sectors[t]) for t in tickers}
            for t in tickers:                       # seed memory with round 0
                agents[t].memory.remember(asof, round0[t].score,
                                          round0[t].direction.value)
            bus = MessageBus(build_topology(topo, tickers, sectors,
                                            degree=config.SPARSE_DEGREE,
                                            seed=config.SEED))
            for msg in round0.values():
                bus.post(msg)

            final = round0
            for r in range(1, n_rounds + 1):
                final, fbr = run_revision(agents, tickers, bus, r, agent_batch,
                                          recorder, asof, topo)
                n_prompts_total += len(fbr) + sum(
                    1 for t in tickers if bus.inbox_for(t))  # rough, informational
                n_fallbacks_total += len(fbr)
                fb_rows.extend(fbr)
                log.info("  Round %d [%s]: %d fell back to rule", r, topo,
                         len(fbr))
                log_ranking(f"ROUND {r} [{topo}]", final, fwd)
                log_memory(f"ROUND {r} [{topo}]", agents, tickers)

            # ---- Manager builds the book from the final opinions ----
            book = manager.build(final, sectors)
            if manager_llm is not None and book.source == "rule":
                rec = {"date": asof, "topology": topo, "round": -1,
                       "ticker": "MANAGER", "error": "unusable manager JSON",
                       "raw": ""}
                fb_rows.append(rec)
                n_fallbacks_total += 1
                log.warning("  ! FALLBACK  MANAGER [%s] -> used RULE book", topo)
            log_manager(book, final, fwd)

            # ---- Grade the book on next-day returns ----
            ic0 = ev.rank_ic({t: m.score for t, m in round0.items()}, fwd)
            icF = ev.rank_ic({t: m.score for t, m in final.items()}, fwd)
            g = ev.portfolio_return(book.weights, fwd)
            turn = ev.weight_turnover(prev_w[topo], book.weights)
            n = ev.apply_cost(g, turn, ROUNDTRIP_BPS)
            prev_w[topo] = book.weights
            series[topo]["gross"].append(g)
            series[topo]["net"].append(n)
            series[topo]["ic0"].append(ic0)
            series[topo]["icF"].append(icF)
            log.info("  GRADE [%s]: gross=%+.3f%% net=%+.3f%% turnover=%.2f "
                     "IC(r0->final)=%+.3f->%+.3f", topo, g * 100, n * 100, turn,
                     ic0, icF)
            daily_rows.append({
                "date": asof, "topology": topo, "gross_ret": g, "net_ret": n,
                "turnover": turn, "rank_ic_round0": ic0, "rank_ic_final": icF,
                "manager_source": book.source, "longs": "|".join(book.longs),
                "shorts": "|".join(book.shorts), "rationale": book.rationale})

    write_outputs(run_dir, mode, topologies, dates, series, daily_rows, fb_rows,
                  recorder, n_prompts_total, n_fallbacks_total)


# ==========================================================================
# Outputs: metrics.json / metrics.csv / daily_returns.csv / fallbacks.csv
# ==========================================================================
def _cum(xs):
    c = 1.0
    for r in xs:
        c *= (1.0 + r)
    return c - 1.0


def write_outputs(run_dir, mode, topologies, dates, series, daily_rows, fb_rows,
                  recorder, n_prompts, n_fallbacks):
    metrics = {}
    for topo in topologies:
        s = series[topo]
        gs = ev.summarize(s["gross"], periods_per_year=PERIODS_PER_YEAR)
        ns = ev.summarize(s["net"], periods_per_year=PERIODS_PER_YEAR)
        hit = (sum(1 for r in s["net"] if r > 0) / len(s["net"])
               if s["net"] else 0.0)
        metrics[topo] = {
            "n_days": len(s["gross"]),
            "gross": {"mean": gs["mean"], "vol": gs["std"],
                      "cum": _cum(s["gross"]), "sharpe": gs["sharpe"]},
            "net": {"mean": ns["mean"], "vol": ns["std"], "cum": _cum(s["net"]),
                    "sharpe": ns["sharpe"], "hit_rate": hit},
            "mean_rank_ic_round0": (sum(s["ic0"]) / len(s["ic0"])
                                    if s["ic0"] else 0.0),
            "mean_rank_ic_final": (sum(s["icF"]) / len(s["icF"])
                                   if s["icF"] else 0.0)}

    fb_rate = (n_fallbacks / n_prompts) if n_prompts else 0.0
    run_meta = {"mode": mode, "topologies": topologies, "n_days": len(dates),
                "start": dates[0], "end": dates[-1],
                "roundtrip_bps": ROUNDTRIP_BPS,
                "llm_prompts_approx": n_prompts, "fallbacks": n_fallbacks,
                "fallback_rate_approx": fb_rate, "per_topology": metrics}

    p_json = os.path.join(run_dir, "metrics.json")
    with open(p_json, "w") as f:
        json.dump(run_meta, f, indent=2)

    p_csv = os.path.join(run_dir, "metrics.csv")
    with open(p_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["topology", "n_days", "gross_sharpe", "gross_cum",
                    "net_sharpe", "net_cum", "hit_rate", "mean_ic_round0",
                    "mean_ic_final"])
        for topo in topologies:
            m = metrics[topo]
            w.writerow([topo, m["n_days"], round(m["gross"]["sharpe"], 3),
                        round(m["gross"]["cum"], 5), round(m["net"]["sharpe"], 3),
                        round(m["net"]["cum"], 5), round(m["net"]["hit_rate"], 3),
                        round(m["mean_rank_ic_round0"], 4),
                        round(m["mean_rank_ic_final"], 4)])

    p_daily = os.path.join(run_dir, "daily_returns.csv")
    with open(p_daily, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(daily_rows[0].keys()))
        w.writeheader()
        w.writerows(daily_rows)

    p_fb = os.path.join(run_dir, "fallbacks.csv")
    with open(p_fb, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "topology", "round", "ticker",
                                          "error", "raw"])
        w.writeheader()
        w.writerows(fb_rows)

    # ---- Console/log summary table ----
    log.info("\n" + "=" * 74)
    log.info("SUMMARY  mode=%s  %d days  %s -> %s", mode, len(dates), dates[0],
             dates[-1])
    log.info("LLM prompts ~%d, fallbacks %d (%.1f%% fell back to rule)",
             n_prompts, n_fallbacks, fb_rate * 100)
    log.info("-" * 74)
    log.info("%-10s %8s %9s %8s %9s %7s %8s %8s", "topology", "grSharpe",
             "grCum%", "ntSharpe", "ntCum%", "hit%", "IC_r0", "IC_fin")
    for topo in topologies:
        m = metrics[topo]
        log.info("%-10s %8.2f %9.2f %8.2f %9.2f %7.0f %8.3f %8.3f", topo,
                 m["gross"]["sharpe"], m["gross"]["cum"] * 100,
                 m["net"]["sharpe"], m["net"]["cum"] * 100,
                 m["net"]["hit_rate"] * 100, m["mean_rank_ic_round0"],
                 m["mean_rank_ic_final"])
    log.info("=" * 74)
    log.info("Sharpe annualised sqrt(%d); dollar-neutral book; %.0fbps round-trip "
             "on turnover.", PERIODS_PER_YEAR, ROUNDTRIP_BPS)
    log.info("Transcript: %d agent messages.", recorder.count())
    log.info("\nRESULTS DIR: %s", os.path.abspath(run_dir))
    log.info("  run.log            full narrative (this log)")
    log.info("  metrics.json/.csv  per-topology Sharpe / cum / IC / hit")
    log.info("  daily_returns.csv  one row per (date, topology)")
    log.info("  fallbacks.csv      every fallback event (%d rows)", len(fb_rows))
    log.info("  transcript.jsonl   every agent message, every round")


if __name__ == "__main__":
    main()
