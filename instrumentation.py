"""
instrumentation.py — token and wall-clock accounting, for GPU budgeting.

A 498-firm walk-forward makes `#dates x (1 + n_rounds) x #stocks` LLM calls, so
"can I afford another topology in this job?" is the question that decides every
experiment grid. That question needs three numbers per round: how many calls, how
many tokens, and how long it took. This module collects them and projects the
total, so a run can be sized before the GPU time is spent rather than after.

Two accuracy tiers, always labelled so an estimate is never mistaken for a
measurement:

  EXACT      vLLM's OpenAI-compatible response carries a `usage` block
             (prompt_tokens / completion_tokens). `HttpBatchClient` records it,
             so an actual GPU run reports real token counts.
  ESTIMATED  no usage available (FAgent's in-process LLMClient returns only
             text, and a rule-based run makes no call at all). Then tokens are
             counted with the model's own HuggingFace tokenizer when one is
             reachable, else approximated at ~4 chars/token.

The estimate path is deliberately useful on its own: a rule-based run can build
the prompts it *would* have sent and report the token budget a matching LLM run
needs — a full-size forecast for zero GPU time.
"""

import csv
import logging
import os
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

log = logging.getLogger("usage")


# ==========================================================================
# Token counting
# ==========================================================================
class TokenCounter:
    """
    Counts tokens in a string, preferring the real tokenizer.

    Tries to load the HuggingFace tokenizer for `model` (free: the tokenizer is
    tiny and Qwen2.5 is already in HF_HOME on this cluster). Falls back to
    chars/4, which is close enough for budgeting English + numbers but is
    reported as ESTIMATED so nobody quotes it as measured.
    """

    # Measured against Qwen2.5's tokenizer on this project's actual prompts:
    # they are number-dense (percentiles, scores, tickers), so tokens average
    # ~3.2 chars, not the usual ~4. Using 4.0 underestimated the budget by 21%,
    # and underestimating is the dangerous direction for a job time limit.
    CHARS_PER_TOKEN = 3.2

    # The project's model, so a rule-based FORECAST still counts exactly (the
    # tokenizer is cached in HF_HOME and costs nothing to load).
    DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

    def __init__(self, model: Optional[str] = None, allow_download: bool = False):
        self.model = model or os.getenv("LLM_MODEL") or self.DEFAULT_MODEL
        self._tok = None
        self.method = "chars/4"
        if self.model:
            try:                            # never fatal — this is telemetry
                from transformers import AutoTokenizer
                kwargs = {} if allow_download else {"local_files_only": True}
                self._tok = AutoTokenizer.from_pretrained(self.model, **kwargs)
                self.method = f"tokenizer:{self.model}"
            except Exception as e:
                log.debug("no tokenizer for %s (%s); using chars/4", self.model, e)

    @property
    def exact(self) -> bool:
        return self._tok is not None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._tok is not None:
            return len(self._tok.encode(text))
        return int(round(len(text) / self.CHARS_PER_TOKEN))


# ==========================================================================
# Per-round record
# ==========================================================================
@dataclass
class RoundUsage:
    """One row of the usage log: a single round of a single topology on a date."""
    date: str
    topology: str
    stage: str                       # "round0", "round1", ..., "manager"
    n_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_s: float = 0.0
    fallbacks: int = 0
    source: str = "estimated"        # "exact" (server usage) or "estimated"
    latencies: List[float] = field(default_factory=list)
    # False when no model was actually called (a rule-based forecast row). Its
    # wall_s is prompt-building time, not inference time, so throughput derived
    # from it would be meaningless — and is therefore suppressed everywhere.
    llm_called: bool = True

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_row(self) -> dict:
        lat = sorted(self.latencies)

        def pct(p):
            if not lat:
                return ""
            return f"{lat[min(len(lat) - 1, int(p * len(lat)))]:.3f}"

        return {
            "date": self.date,
            "topology": self.topology,
            "stage": self.stage,
            "n_calls": self.n_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "tokens_per_call": (round(self.total_tokens / self.n_calls, 1)
                                if self.n_calls else ""),
            "wall_s": round(self.wall_s, 3),
            "calls_per_s": (round(self.n_calls / self.wall_s, 1)
                            if self.llm_called and self.wall_s > 0 else ""),
            "tokens_per_s": (round(self.total_tokens / self.wall_s, 0)
                             if self.llm_called and self.wall_s > 0 else ""),
            "call_latency_p50": pct(0.50),
            "call_latency_p95": pct(0.95),
            "fallbacks": self.fallbacks,
            "token_source": self.source,
        }


# ==========================================================================
# The log
# ==========================================================================
class UsageLog:
    """
    Accumulates RoundUsage rows and reports totals, throughput and projections.

    Thread-safe: `HttpBatchClient` fires a round's prompts from a thread pool and
    each thread records its own call.
    """

    def __init__(self, counter: Optional[TokenCounter] = None,
                 log_each_round: bool = True):
        self.counter = counter or TokenCounter()
        self.rows: List[RoundUsage] = []
        self.log_each_round = log_each_round
        self._lock = threading.Lock()
        self.t_start = time.time()

    # ---- recording ------------------------------------------------------
    def record(self, usage: RoundUsage) -> None:
        with self._lock:
            self.rows.append(usage)
        if self.log_each_round and usage.n_calls:
            r = usage.as_row()
            if usage.llm_called:
                rate = (f", {usage.wall_s:.1f}s, {r['calls_per_s']} calls/s, "
                        f"{int(r['tokens_per_s'] or 0):,} tok/s")
            else:
                rate = "  [FORECAST: no model called]"
            log.info("    usage [%s|%s] %d calls, %s tok "
                     "(%s prompt + %s completion)%s%s",
                     usage.topology, usage.stage, usage.n_calls,
                     f"{usage.total_tokens:,}", f"{usage.prompt_tokens:,}",
                     f"{usage.completion_tokens:,}", rate,
                     "" if (usage.source == "exact" or not usage.llm_called)
                     else "  [tokens ESTIMATED]")

    def estimate_calls(self, prompts, stage: str, date: str, topology: str,
                       wall_s: float = 0.0, completion_tokens_each: int = 0,
                       fallbacks: int = 0) -> RoundUsage:
        """
        Record a round from the prompt TEXTS, without a server's usage block.

        Used by the rule-based path (to forecast what an LLM run would cost) and
        by the in-process client (which reports no usage). `completion_tokens_each`
        lets a forecast include the expected output — pass the client's
        max_tokens for a worst case.
        """
        pt = sum(self.counter.count(p) for p in prompts)
        u = RoundUsage(date=date, topology=topology, stage=stage,
                       n_calls=len(prompts), prompt_tokens=pt,
                       completion_tokens=completion_tokens_each * len(prompts),
                       wall_s=wall_s, fallbacks=fallbacks, source="estimated",
                       llm_called=False)
        self.record(u)
        return u

    # ---- reporting ------------------------------------------------------
    def totals(self) -> dict:
        calls = sum(r.n_calls for r in self.rows)
        pt = sum(r.prompt_tokens for r in self.rows)
        ct = sum(r.completion_tokens for r in self.rows)
        llm_rows = [r for r in self.rows if r.llm_called]
        llm_s = sum(r.wall_s for r in llm_rows)
        llm_calls = sum(r.n_calls for r in llm_rows)
        llm_toks = sum(r.total_tokens for r in llm_rows)
        exact = all(r.source == "exact" for r in self.rows if r.n_calls)
        dates = {r.date for r in self.rows}
        return {
            "n_calls": calls,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
            "llm_wall_s": llm_s,
            "run_wall_s": time.time() - self.t_start,
            "n_dates": len(dates),
            "calls_per_date": round(calls / len(dates), 1) if dates else 0,
            "tokens_per_date": round((pt + ct) / len(dates)) if dates else 0,
            "calls_per_s": round(llm_calls / llm_s, 1) if llm_s > 0 else None,
            "tokens_per_s": round(llm_toks / llm_s) if llm_s > 0 else None,
            "forecast_only": not llm_rows,
            "token_source": "exact" if exact else "estimated",
            "token_method": self.counter.method,
        }

    def project(self, total_dates: int, n_topologies: Optional[int] = None) -> dict:
        """
        Scale this run's measured cost to a full run — the number that decides
        whether another topology fits in a job's time limit.
        """
        t = self.totals()
        if not t["n_dates"]:
            return {}
        scale = total_dates / t["n_dates"]
        out = {
            "for_dates": total_dates,
            "projected_calls": int(t["n_calls"] * scale),
            "projected_tokens": int(t["total_tokens"] * scale),
            "projected_llm_hours": round(t["llm_wall_s"] * scale / 3600.0, 2),
            "projected_wall_hours": round(t["run_wall_s"] * scale / 3600.0, 2),
        }
        if n_topologies:
            # Round 0 is shared across topologies, so a topology costs the
            # revision rounds only — this is the marginal, not the average.
            rev = sum(r.wall_s for r in self.rows if r.stage.startswith("round")
                      and r.stage != "round0")
            per_topo = rev / n_topologies if n_topologies else 0.0
            out["marginal_hours_per_topology"] = round(
                per_topo * scale / 3600.0, 2)
        return out

    def summary_lines(self, total_dates: Optional[int] = None,
                      n_topologies: Optional[int] = None) -> List[str]:
        t = self.totals()
        if not t["n_calls"]:
            return ["usage: nothing recorded"]
        note = ("" if t["token_source"] == "exact"
                else f"  [tokens ESTIMATED via {t['token_method']}]")
        if t["forecast_only"]:
            # Rule-based run: no model ran, so there is no throughput or
            # inference time to report — only what an LLM run WOULD send.
            lines = [
                f"TOKEN BUDGET FORECAST — no model was called{note}",
                f"  would-be calls   {t['n_calls']:,}  "
                f"({t['calls_per_date']:,} per date over {t['n_dates']} dates)",
                f"  would-be tokens  {t['total_tokens']:,}  "
                f"({t['prompt_tokens']:,} prompt + {t['completion_tokens']:,} "
                f"completion at max_tokens, i.e. an upper bound)",
                f"  tokens per date  {t['tokens_per_date']:,}",
                f"  pipeline time    {t['run_wall_s']:.1f}s (rule-based; an LLM "
                f"run's time depends on GPU throughput)",
            ]
        else:
            lines = [
                f"TOKEN / TIME BUDGET{note}",
                f"  calls            {t['n_calls']:,}  "
                f"({t['calls_per_date']:,} per date over {t['n_dates']} dates)",
                f"  tokens           {t['total_tokens']:,}  "
                f"({t['prompt_tokens']:,} prompt + {t['completion_tokens']:,} "
                f"completion)",
                f"  tokens per date  {t['tokens_per_date']:,}",
                f"  LLM wall time    {t['llm_wall_s']:.1f}s of "
                f"{t['run_wall_s']:.1f}s total "
                f"({t['llm_wall_s'] / t['run_wall_s']:.0%} in the model)",
            ]
            if t["calls_per_s"]:
                lines.append(f"  throughput       {t['calls_per_s']:,} calls/s, "
                             f"{t['tokens_per_s']:,} tokens/s")
        by_stage: Dict[str, List[RoundUsage]] = {}
        for r in self.rows:
            by_stage.setdefault(r.stage, []).append(r)
        lines.append("  by stage:")
        for stage, rs in sorted(by_stage.items()):
            calls = sum(r.n_calls for r in rs)
            toks = sum(r.total_tokens for r in rs)
            wall = sum(r.wall_s for r in rs)
            per = f"{toks / calls:.0f}" if calls else "-"
            lines.append(f"    {stage:<10} {calls:>8,} calls  {toks:>12,} tok  "
                         f"{per:>5} tok/call  {wall:>8.1f}s")
        if total_dates and total_dates != t["n_dates"]:
            p = self.project(total_dates, n_topologies)
            lines += [
                f"  PROJECTED to {total_dates} dates:",
                f"    {p['projected_calls']:,} calls, "
                f"{p['projected_tokens']:,} tokens",
            ]
            if t["forecast_only"]:
                lines.append(f"    (no time projection — measure throughput with "
                             f"a short LLM run first)")
            else:
                lines.append(f"    ~{p['projected_wall_hours']}h wall "
                             f"({p['projected_llm_hours']}h in the model)")
                if "marginal_hours_per_topology" in p:
                    lines.append(f"    ~{p['marginal_hours_per_topology']}h per "
                                 f"additional topology (revision rounds only)")
        return lines

    def report(self, total_dates: Optional[int] = None,
               n_topologies: Optional[int] = None) -> None:
        for line in self.summary_lines(total_dates, n_topologies):
            log.info(line)

    def write_csv(self, path: str) -> None:
        if not self.rows:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        rows = [r.as_row() for r in self.rows]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        log.info("    per-round usage -> %s", path)


# ==========================================================================
# Wrapper for single-call clients (the manager)
# ==========================================================================
class CountingLLM:
    """
    Drop-in proxy around an LLMClient that records tokens and latency.

    The manager makes one `generate_json` call per rebalance, through FAgent's
    LLMClient, which returns only parsed JSON — no usage block. So prompt tokens
    are counted locally and completion tokens are unknown (recorded as 0 rather
    than guessed, which would inflate the budget with fiction). Latency is real.
    """

    def __init__(self, client, usage_log: UsageLog, stage: str = "manager"):
        self.client = client
        self.usage = usage_log
        self.stage = stage
        self.date = "?"
        self.topology = "-"

    def __getattr__(self, name):          # anything else passes through
        return getattr(self.client, name)

    def generate_json(self, prompt, schema, system_message=None, **kw):
        t0 = time.time()
        try:
            return self.client.generate_json(prompt, schema,
                                             system_message=system_message, **kw)
        finally:
            dt = time.time() - t0
            pt = self.usage.counter.count(prompt)
            if system_message:
                pt += self.usage.counter.count(system_message)
            self.usage.record(RoundUsage(
                date=self.date, topology=self.topology, stage=self.stage,
                n_calls=1, prompt_tokens=pt, completion_tokens=0, wall_s=dt,
                source="estimated", latencies=[dt]))
