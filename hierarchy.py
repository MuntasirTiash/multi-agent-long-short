"""
hierarchy.py — the three-tier industry-leader setting (T6 in TOPOLOGY_PLAN.md).

Instead of every stock agent shouting into a flat graph, information is routed
through an organisational structure, mirroring how a real fund is arranged:

    tier 1   498 StockAgents      independent opinion on own ticker (unchanged)
                |  star: every member of an industry reports to its leader
    tier 2   11 IndustryLeaders   each reads its whole industry's opinions and
                |                 emits an IndustryReport: ranked longs/shorts
                |  leader council  with per-name scores, plus an industry outlook
                |                 -> then leaders TALK TO EACH OTHER, so
                |                 cross-industry comparison happens here
    tier 3   IndustryManagerAgent reads the 11 reports -> dollar-neutral book

Three things this structure buys that the flat setting cannot:

  * CONTEXT THAT FITS. A flat manager reading 498 analyst rows needs ~12k tokens
    against vLLM's 4096 window. Here each leader reads only its own industry and
    the manager reads 11 summaries, so nothing is truncated and nothing has to be
    withheld.
  * A PLACE FOR RELATIVE JUDGEMENT. A stock agent cannot know whether its 80 is
    better than another sector's 80. The leader council is where that gets
    resolved, before any position is sized.
  * INDUSTRY NEUTRALITY becomes expressible (see IndustryManagerAgent) — the
    flat book has no notion of how much risk sits in one industry.

Both new agents keep the project's dual-brain contract: a rule-based twin is
computed first and used verbatim if the model returns anything unusable, so a run
never crashes on a bad generation. `source` records which brain produced each
result, and a "LLM" run that quietly fell back is visible rather than hidden.

Leader selection is POINT-IN-TIME (see LeaderSelector): the leader is whichever
member is largest *as of that date*, so leadership changes as the cross-section
does, and no future information picks it.
"""

import logging
import math
import random
from typing import Dict, List, Optional, Tuple

from manager_agent import ManagedPortfolio
from messages import (Direction, IndustryPick, IndustryReport, StockMessage,
                      trim_thesis)

log = logging.getLogger("hierarchy")


def _days_between(a: str, b: str) -> int:
    """Calendar days from ISO date `a` to ISO date `b`."""
    import datetime as _dt
    fmt = "%Y-%m-%d"
    return (_dt.datetime.strptime(b, fmt) - _dt.datetime.strptime(a, fmt)).days


# ==========================================================================
# Who leads each industry
# ==========================================================================
class LeaderSelector:
    """
    Picks each industry's leader as its LARGEST member, point-in-time.

    Three metrics, all selectable so the choice can be tested rather than
    assumed:

      "market_cap"     shares outstanding (from the newest SEC filing *filed* on
                       or before `asof`) x that date's close. The real definition;
                       needs data/shares_outstanding/ populated by
                       fetch_market_cap.py.
      "dollar_volume"  trailing mean of close x volume. Needs no extra data, is
                       point-in-time by construction, and correlates strongly
                       with size — the interim metric that unblocks the build.
      "random"         a fixed random member per industry. This is the CONTROL
                       that matters: it separates "the largest firm's view
                       carries information" from "any designated aggregator
                       helps". Stable across dates, like a real leader.
    """

    def __init__(self, ohlcv: Dict[str, List[dict]], industries: Dict[str, str],
                 metric: str = "dollar_volume", window: int = 20,
                 seed: int = 42, shares: Optional[Dict[str, List[Tuple]]] = None,
                 stale_days: int = 400):
        self.ohlcv = ohlcv
        self.industries = industries
        self.metric = metric
        self.window = window
        self.seed = seed
        self.shares = shares or {}     # {ticker: [(filed_date, shares), ...]}
        # A share count is only trusted if it was filed within `stale_days` of
        # the rebalance date. Multi-class filers (Berkshire is the clear case)
        # can return a stale or single-class figure from XBRL — BRK-B came back
        # as 1M shares filed in 2011 — which would compute a market cap orders of
        # magnitude too small and silently hand leadership to the wrong firm. A
        # stale figure is treated as missing rather than believed.
        self.stale_days = stale_days
        self._stale_warned = set()
        self.by_industry: Dict[str, List[str]] = {}
        for ticker, industry in industries.items():
            if ticker in ohlcv:
                self.by_industry.setdefault(industry, []).append(ticker)
        for members in self.by_industry.values():
            members.sort()             # deterministic order
        self._dates = {t: [r["date"] for r in rows]
                       for t, rows in ohlcv.items()}
        self._cache: Dict[str, Dict[str, str]] = {}

    # ------------------------------------------------------------------
    def _rows_upto(self, ticker: str, asof: str) -> List[dict]:
        import bisect
        dates = self._dates.get(ticker) or []
        end = bisect.bisect_right(dates, asof)
        return self.ohlcv[ticker][:end]

    def _shares_asof(self, ticker: str, asof: str) -> Optional[float]:
        """
        Newest share count FILED on or before `asof` — never a later one, and
        None if the newest one is too old to trust (see `stale_days`).
        """
        best = None
        for filed, count in self.shares.get(ticker, []):
            if filed <= asof and (best is None or filed > best[0]):
                best = (filed, count)
        if best is None:
            return None
        if _days_between(best[0], asof) > self.stale_days:
            if ticker not in self._stale_warned:
                self._stale_warned.add(ticker)
                log.warning("share count for %s is stale (newest filing %s, "
                            "%d days before %s) — excluded from leader "
                            "selection", ticker, best[0],
                            _days_between(best[0], asof), asof)
            return None
        return best[1]

    def size(self, ticker: str, asof: str) -> Optional[float]:
        """The size metric for one ticker at one date; None if unavailable."""
        rows = self._rows_upto(ticker, asof)
        if not rows:
            return None
        if self.metric == "dollar_volume":
            window = [r for r in rows[-self.window:]
                      if r.get("volume") and r.get("close")]
            if not window:
                return None
            return sum(r["volume"] * r["close"] for r in window) / len(window)
        if self.metric == "market_cap":
            shares = self._shares_asof(ticker, asof)
            close = rows[-1].get("close")
            if shares is None or close is None:
                return None
            return shares * close
        if self.metric == "random":
            return None                # handled in leaders()
        raise ValueError(f"unknown LEADER_METRIC {self.metric!r}")

    def coverage(self, asof: str) -> dict:
        """
        How many tickers have a usable size metric at `asof`.

        Call this before trusting LEADER_METRIC="market_cap": if coverage is
        partial, some firms simply cannot be elected leader, which biases the
        setting toward whoever happens to have clean XBRL data.
        """
        usable = sum(1 for t in self.industries if t in self.ohlcv
                     and self.size(t, asof) is not None)
        total = sum(1 for t in self.industries if t in self.ohlcv)
        return {"metric": self.metric, "usable": usable, "total": total,
                "pct": round(100.0 * usable / total, 1) if total else 0.0}

    def leaders(self, asof: str) -> Dict[str, str]:
        """{industry: leader_ticker} as of `asof`."""
        if asof in self._cache:
            return self._cache[asof]
        out = {}
        for industry, members in self.by_industry.items():
            if not members:
                continue
            if self.metric == "random":
                # Seeded by industry only, so the control leader is stable over
                # time exactly as a market-cap leader would be.
                rng = random.Random(f"{self.seed}|{industry}")
                out[industry] = rng.choice(members)
                continue
            sized = [(t, self.size(t, asof)) for t in members]
            sized = [(t, s) for t, s in sized if s is not None]
            if not sized and self.metric == "market_cap":
                # No usable share counts anywhere in this industry: rank it by
                # dollar volume instead. Falling back for the WHOLE industry
                # keeps the units consistent within the comparison — mixing caps
                # and volumes across candidates would be meaningless.
                if industry not in self._stale_warned:
                    self._stale_warned.add(industry)
                    log.warning("no usable market caps in %s — ranking that "
                                "industry by dollar volume instead", industry)
                saved, self.metric = self.metric, "dollar_volume"
                sized = [(t, self.size(t, asof)) for t in members]
                sized = [(t, s) for t, s in sized if s is not None]
                self.metric = saved
            if not sized:
                out[industry] = members[0]         # degenerate but deterministic
                continue
            out[industry] = max(sized, key=lambda x: (x[1], x[0]))[0]
        self._cache[asof] = out
        if len(self._cache) > 64:
            self._cache.pop(next(iter(self._cache)))
        return out


# ==========================================================================
# Tier 2: the industry leader
# ==========================================================================
class IndustryLeader:
    """
    Aggregates one industry's analyst opinions into an IndustryReport.

    The leader reviews EVERY member of its industry INCLUDING ITSELF — it is a
    member of the industry, not an outside observer, so its own ticker is
    eligible for its own nominations.
    """

    SYSTEM = ("You are the lead analyst for one industry at a long/short equity "
              "fund. Your junior analysts have each rated one stock in your "
              "industry (score 0-100, higher = better long). You decide which "
              "names in YOUR industry are the most attractive longs and the most "
              "attractive shorts, and give a short industry-level view. Answer "
              "only with the requested JSON.")

    SCHEMA = {
        "longs": ('list of {"ticker": str, "score": number 0-100, '
                  '"thesis": str} - best long first'),
        "shorts": ('list of {"ticker": str, "score": number 0-100, '
                   '"thesis": str} - best short first'),
        "outlook": "<= 40 word view on the industry as a whole",
    }

    def __init__(self, ticker: str, industry: str, members: List[str],
                 llm_client=None, picks_per_side: int = 3,
                 table_cap: int = 40):
        self.ticker = ticker
        self.industry = industry
        self.members = list(members)
        self.llm = llm_client
        self.picks_per_side = picks_per_side
        # Cap how many members go in the prompt. The largest sector here has 79
        # firms; at ~35 tokens a row that is ~2.8k tokens before the instructions,
        # which crowds a 4096 window. Over the cap we show the extremes (the
        # candidates that can actually be picked) and say how many were omitted.
        self.table_cap = table_cap

    # ---- public entry point -------------------------------------------
    def build_report(self, messages: Dict[str, StockMessage],
                     round_num: int = 0) -> IndustryReport:
        mine = [messages[t] for t in self.members if t in messages]
        fallback = self.rule_report(mine, round_num)
        if self.llm is None:
            return fallback
        data = self.llm.generate_json(self.report_prompt(mine), self.SCHEMA,
                                      system_message=self.SYSTEM)
        got = self.report_from_data(data, mine, round_num)
        return got if got is not None else fallback

    # ---- rule brain ----------------------------------------------------
    def rule_report(self, mine: List[StockMessage],
                    round_num: int = 0) -> IndustryReport:
        """Rank the industry by analyst score; take the extremes."""
        ranked = sorted(mine, key=lambda m: m.score, reverse=True)
        k = max(1, min(self.picks_per_side, len(ranked) // 2)) if ranked else 0
        longs = ranked[:k]
        shorts = list(reversed(ranked[len(ranked) - k:])) if k else []
        mean = (sum(m.score for m in ranked) / len(ranked)) if ranked else 50.0
        return IndustryReport(
            industry=self.industry, leader=self.ticker,
            longs=[self._pick(m) for m in longs],
            shorts=[self._pick(m) for m in shorts],
            outlook=(f"{self.industry}: {len(ranked)} names, mean score "
                     f"{mean:.0f}; best {longs[0].ticker if longs else '-'}, "
                     f"worst {shorts[0].ticker if shorts else '-'}."),
            round_num=round_num, source="rule", n_members=len(ranked))

    @staticmethod
    def _pick(m: StockMessage) -> IndustryPick:
        return IndustryPick(ticker=m.ticker, score=m.score,
                            direction=m.direction, thesis=m.thesis)

    # ---- LLM brain -----------------------------------------------------
    def _table(self, mine: List[StockMessage]) -> Tuple[str, int]:
        ranked = sorted(mine, key=lambda m: m.score, reverse=True)
        omitted = 0
        if len(ranked) > self.table_cap:
            half = self.table_cap // 2
            omitted = len(ranked) - 2 * half
            ranked = ranked[:half] + ranked[-half:]
        rows = [f"  {m.ticker:<6} score={m.score:5.1f} {m.direction.value:<7} "
                f"conf={m.confidence:.2f}  {trim_thesis(m.thesis, 18)}"
                for m in ranked]
        return "\n".join(rows), omitted

    def report_prompt(self, mine: List[StockMessage]) -> str:
        table, omitted = self._table(mine)
        note = (f"\n  (... {omitted} mid-ranked names omitted as unremarkable)"
                if omitted else "")
        k = max(1, min(self.picks_per_side, max(1, len(mine) // 2)))
        return (
            f"Industry: {self.industry}. You lead coverage of {len(mine)} "
            f"companies, and you are the analyst for {self.ticker} yourself.\n"
            f"Your analysts' opinions, best long first:\n{table}{note}\n\n"
            f"Choose up to {k} names to go LONG and up to {k} to SHORT, from "
            f"this list only. Give each pick your OWN score 0-100 — you may "
            f"disagree with the analyst — and a one-line reason. Then give a "
            f"short outlook on the industry as a whole. Prefer names where a "
            f"strong score is backed by high confidence and a clear thesis.")

    def report_from_data(self, data: Dict, mine: List[StockMessage],
                         round_num: int = 0) -> Optional[IndustryReport]:
        """Parse the model's JSON report; None (-> rule fallback) if unusable."""
        if not isinstance(data, dict):
            return None
        allowed = {m.ticker: m for m in mine}
        longs = self._clean(data.get("longs"), allowed, self.picks_per_side)
        shorts = self._clean(data.get("shorts"), allowed, self.picks_per_side,
                             exclude={p.ticker for p in longs})
        if not longs and not shorts:
            return None
        return IndustryReport(
            industry=self.industry, leader=self.ticker, longs=longs,
            shorts=shorts, outlook=str(data.get("outlook", ""))[:400],
            round_num=round_num, source="llm", n_members=len(mine))

    @staticmethod
    def _clean(raw, allowed: Dict[str, StockMessage], cap: int,
               exclude=frozenset()) -> List[IndustryPick]:
        """Validate one side; drop unknown tickers, duplicates and cross-side
        collisions (a name cannot be both a long and a short)."""
        if not isinstance(raw, list):
            return []
        out, seen = [], set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            t = str(item.get("ticker", "")).strip().upper()
            if t not in allowed or t in seen or t in exclude:
                continue
            try:
                score = float(item.get("score"))
            except (TypeError, ValueError):
                score = allowed[t].score          # keep the analyst's score
            seen.add(t)
            out.append(IndustryPick(
                ticker=t, score=score,
                direction=(Direction.LONG if score >= 50 else Direction.SHORT),
                thesis=str(item.get("thesis", "")) or allowed[t].thesis))
            if len(out) >= cap:
                break
        return out

    # ==================================================================
    # The leader council: leaders revise after hearing other industries
    # ==================================================================
    COUNCIL_SCHEMA = {
        "adjust": "number -20..+20, how much to shift YOUR industry's scores",
        "outlook": "<= 40 word revised view on your industry versus the others",
    }

    def council_revise(self, mine: IndustryReport,
                       peers: List[IndustryReport]) -> IndustryReport:
        """Revise this industry's report after reading other industries'."""
        if not peers:
            return mine
        fallback = self.rule_council_revise(mine, peers)
        if self.llm is None:
            return fallback
        data = self.llm.generate_json(self.council_prompt(mine, peers),
                                      self.COUNCIL_SCHEMA,
                                      system_message=self.SYSTEM)
        try:
            adjust = float(data["adjust"])
        except (KeyError, TypeError, ValueError):
            return fallback
        adjust = max(-20.0, min(20.0, adjust))
        return self._shifted(mine, adjust,
                             outlook=str(data.get("outlook", "")) or mine.outlook,
                             source="llm")

    def rule_council_revise(self, mine: IndustryReport,
                            peers: List[IndustryReport]) -> IndustryReport:
        """
        Recentre this industry toward the cross-industry average.

        Note this deliberately differs from StockAgent.rule_revise, which moves a
        stock's score AWAY from its peers (amplifying divergence). At the council
        the goal is the opposite: make industries COMPARABLE before the manager
        ranks across them, so an industry whose scores sit far from the rest is
        pulled toward the middle. Both are placeholders for the LLM brain; what
        matters is that the shift is small, signed and documented.
        """
        peer_mean = sum(p.mean_score() for p in peers) / len(peers)
        shift = 0.2 * (peer_mean - mine.mean_score())
        return self._shifted(
            mine, shift,
            outlook=(f"{mine.outlook} Recentred {shift:+.1f} vs {len(peers)} "
                     f"peer industries (their mean {peer_mean:.0f})."),
            source="rule")

    @staticmethod
    def _shifted(report: IndustryReport, adjust: float, outlook: str,
                 source: str) -> IndustryReport:
        def shift(picks):
            return [IndustryPick(ticker=p.ticker, score=p.score + adjust,
                                 direction=p.direction, thesis=p.thesis)
                    for p in picks]
        return IndustryReport(
            industry=report.industry, leader=report.leader,
            longs=shift(report.longs), shorts=shift(report.shorts),
            outlook=outlook, round_num=report.round_num + 1, source=source,
            n_members=report.n_members)

    def council_prompt(self, mine: IndustryReport,
                       peers: List[IndustryReport]) -> str:
        lines = [f"  {p.industry:<24} mean score {p.mean_score():5.1f}  "
                 f"best {p.longs[0].ticker if p.longs else '-':<6} "
                 f"worst {p.shorts[0].ticker if p.shorts else '-':<6} "
                 f"{trim_thesis(p.outlook, 20)}" for p in peers]
        return (
            f"You lead {mine.industry} (your mean pick score "
            f"{mine.mean_score():.1f}). The other industry leads report:\n"
            + "\n".join(lines) +
            f"\n\nScores must be comparable ACROSS industries: a 70 in "
            f"{mine.industry} should mean the same as a 70 anywhere else. If "
            f"your industry is stronger than the others, shift your scores up; "
            f"if weaker, shift them down; if fairly rated, shift by 0.")


# ==========================================================================
# Tier 3: the manager that reads industry reports
# ==========================================================================
class IndustryManagerAgent:
    """
    Builds the book from industry reports rather than from 498 analyst rows.

    Deliberately NOT a subclass of ManagerAgent: its input is a different type
    and its industry-neutrality option has no flat-book equivalent. It returns
    the same `ManagedPortfolio` contract, so evaluation.portfolio_return and
    weight_turnover work unchanged and results are directly comparable with the
    flat manager's.
    """

    SYSTEM = ("You are the portfolio manager of a long/short equity fund. Each "
              "industry lead has nominated the best longs and shorts in their "
              "industry and given you their scores. You build ONE market-neutral "
              "book across all industries. Answer only with the requested JSON.")

    SCHEMA = {
        "longs": 'list of {"ticker": str, "weight": positive number}',
        "shorts": 'list of {"ticker": str, "weight": positive number}',
        "rationale": "<= 40 word explanation of the book",
    }

    def __init__(self, n_long: int, n_short: int, llm_client=None,
                 sizing: str = "flexible", min_positions: int = 5,
                 max_positions: int = 50, long_threshold: float = 75.0,
                 short_threshold: float = 25.0, industry_neutral: bool = False):
        self.n_long = n_long
        self.n_short = n_short
        self.llm = llm_client
        self.sizing = sizing
        self.min_positions = max(1, min_positions)
        self.max_positions = max(self.min_positions, max_positions)
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        # Cap each industry's contribution so one industry cannot dominate the
        # book. Only expressible because positions arrive grouped by industry.
        self.industry_neutral = industry_neutral

    # ---- public entry point -------------------------------------------
    def build(self, reports: List[IndustryReport]) -> ManagedPortfolio:
        fallback = self.rule_build(reports)
        if self.llm is None:
            return fallback
        data = self.llm.generate_json(self.build_prompt(reports), self.SCHEMA,
                                      system_message=self.SYSTEM)
        got = self.portfolio_from_data(data, reports)
        return got if got is not None else fallback

    # ---- rule brain ----------------------------------------------------
    def rule_build(self, reports: List[IndustryReport]) -> ManagedPortfolio:
        long_pool = [(p, r.industry) for r in reports for p in r.longs]
        short_pool = [(p, r.industry) for r in reports for p in r.shorts]
        long_pool.sort(key=lambda x: x[0].score, reverse=True)
        short_pool.sort(key=lambda x: x[0].score)         # worst first

        if self.sizing == "flexible":
            longs = [x for x in long_pool if x[0].score >= self.long_threshold]
            shorts = [x for x in short_pool if x[0].score <= self.short_threshold]
            longs = longs[:self.max_positions] or long_pool[:self.min_positions]
            shorts = shorts[:self.max_positions] or short_pool[:self.min_positions]
            why = (f"Pooled {len(long_pool)}+{len(short_pool)} industry "
                   f"nominations; gated at {self.long_threshold:.0f}/"
                   f"{self.short_threshold:.0f}")
        else:
            longs = long_pool[:self.n_long]
            shorts = short_pool[:self.n_short]
            why = (f"Pooled industry nominations; took top {self.n_long} / "
                   f"bottom {self.n_short}")

        # A ticker nominated long by one industry cannot also be short.
        taken = {p.ticker for p, _ in longs}
        shorts = [(p, i) for p, i in shorts if p.ticker not in taken]

        weights = {}
        weights.update(self._weights(longs, +1.0))
        weights.update(self._weights(shorts, -1.0))
        neutral = " industry-neutral" if self.industry_neutral else ""
        return ManagedPortfolio(
            weights=weights, longs=[p.ticker for p, _ in longs],
            shorts=[p.ticker for p, _ in shorts],
            rationale=f"{why}; conviction-weighted{neutral}.",
            source="rule",
            ranking=[])       # tier 3 never sees the full 498-name ranking

    def _weights(self, side: List[Tuple[IndustryPick, str]],
                 sign: float) -> Dict[str, float]:
        """
        Conviction weights summing to +/-1, optionally industry-neutral.

        Conviction is |score - 50|: an IndustryPick carries the leader's score but
        no confidence field, so unlike ManagerAgent there is no confidence factor
        here rather than a faked one.
        """
        if not side:
            return {}
        if not self.industry_neutral:
            conv = [max(0.0, abs(p.score - 50.0)) for p, _ in side]
            total = sum(conv)
            if total <= 0:
                return {p.ticker: sign / len(side) for p, _ in side}
            return {p.ticker: sign * c / total for (p, _), c in zip(side, conv)}

        # Industry-neutral: every industry with a pick on this side gets an equal
        # share, split within the industry by conviction.
        by_industry: Dict[str, List[IndustryPick]] = {}
        for p, industry in side:
            by_industry.setdefault(industry, []).append(p)
        share = 1.0 / len(by_industry)
        out = {}
        for picks in by_industry.values():
            conv = [max(0.0, abs(p.score - 50.0)) for p in picks]
            total = sum(conv)
            for p, c in zip(picks, conv):
                frac = (c / total) if total > 0 else (1.0 / len(picks))
                out[p.ticker] = sign * share * frac
        return out

    # ---- LLM brain -----------------------------------------------------
    def build_prompt(self, reports: List[IndustryReport]) -> str:
        blocks = []
        for r in sorted(reports, key=lambda x: x.mean_score(), reverse=True):
            longs = ", ".join(f"{p.ticker} ({p.score:.0f})" for p in r.longs)
            shorts = ", ".join(f"{p.ticker} ({p.score:.0f})" for p in r.shorts)
            blocks.append(
                f"  {r.industry} [{r.n_members} firms, mean "
                f"{r.mean_score():.0f}]\n"
                f"    LONG : {longs or '-'}\n"
                f"    SHORT: {shorts or '-'}\n"
                f"    view : {trim_thesis(r.outlook, 30)}")
        rng = (f"at least {self.min_positions} and at most {self.max_positions} "
               f"per side, and the two sides need not match"
               if self.sizing == "flexible"
               else f"up to {self.n_long} longs and {self.n_short} shorts")
        neutral = ("\nSpread risk across industries: do not let one industry "
                   "dominate either side." if self.industry_neutral else "")
        return (
            f"Industry leads' nominations, strongest industry first:\n"
            + "\n".join(blocks) +
            f"\n\nBuild a dollar-neutral long/short book from THESE nominated "
            f"tickers only. Choose {rng}. Give each pick a positive conviction "
            f"weight (longs are normalised together, shorts together). Scores "
            f"are comparable across industries.{neutral}")

    def portfolio_from_data(self, data: Dict, reports: List[IndustryReport]
                            ) -> Optional[ManagedPortfolio]:
        if not isinstance(data, dict):
            return None
        long_ok = {p.ticker: r.industry for r in reports for p in r.longs}
        short_ok = {p.ticker: r.industry for r in reports for p in r.shorts}
        longs = self._clean_side(data.get("longs"), long_ok, self.max_positions)
        shorts = self._clean_side(data.get("shorts"), short_ok,
                                  self.max_positions,
                                  exclude={t for t, _ in longs})
        if not longs or not shorts:
            # A one-sided book is a directional bet, not the market-neutral book
            # being graded — fall back rather than change the exposure.
            return None
        weights = {}
        weights.update(self._normalise(longs, +1.0))
        weights.update(self._normalise(shorts, -1.0))
        return ManagedPortfolio(
            weights=weights, longs=[t for t, _ in longs],
            shorts=[t for t, _ in shorts],
            rationale=str(data.get("rationale", ""))[:300], source="llm",
            ranking=[])

    @staticmethod
    def _clean_side(raw, allowed: Dict[str, str], cap: int,
                    exclude=frozenset()):
        if not isinstance(raw, list):
            return []
        out, seen = [], set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            t = str(item.get("ticker", "")).strip().upper()
            if t not in allowed or t in seen or t in exclude:
                continue          # not nominated by any industry, or duplicated
            try:
                w = float(item.get("weight", 1.0))
            except (TypeError, ValueError):
                w = 1.0
            if w <= 0:
                continue
            seen.add(t)
            out.append((t, w))
            if len(out) >= cap:
                break
        return out

    @staticmethod
    def _normalise(side, sign: float) -> Dict[str, float]:
        total = sum(w for _, w in side)
        if total <= 0:
            return {}
        return {t: sign * w / total for t, w in side}


# ==========================================================================
# Driving the whole structure for one rebalance date
# ==========================================================================
def run_hierarchy(messages: Dict[str, StockMessage], industries: Dict[str, str],
                  leaders: Dict[str, str], llm_client=None,
                  picks_per_side: int = 3, council_rounds: int = 1,
                  council_graph: Optional[Dict[str, List[str]]] = None,
                  table_cap: int = 40) -> Tuple[List[IndustryReport], Dict]:
    """
    Tier 1 opinions -> industry reports -> council rounds -> final reports.

    Returns (reports, stats). `council_graph` maps a leader's ticker to the
    leader tickers it may hear from; None means a full council (every leader
    hears every other), which is cheap at 11 agents.

    As in the flat schedulers, a council round reads one SNAPSHOT of all reports
    before any revision is posted, so no leader sees a half-updated world.
    """
    members: Dict[str, List[str]] = {}
    for ticker, industry in industries.items():
        if ticker in messages:
            members.setdefault(industry, []).append(ticker)

    agents = {industry: IndustryLeader(leader, industry,
                                       members.get(industry, []),
                                       llm_client=llm_client,
                                       picks_per_side=picks_per_side,
                                       table_cap=table_cap)
              for industry, leader in leaders.items()
              if members.get(industry)}

    reports = {i: a.build_report(messages) for i, a in agents.items()}
    stats = {"n_industries": len(reports),
             "llm_reports": sum(1 for r in reports.values() if r.source == "llm"),
             "rule_reports": sum(1 for r in reports.values()
                                 if r.source == "rule"),
             "council_llm": 0, "council_rule": 0}

    ticker_to_industry = {a.ticker: i for i, a in agents.items()}
    for _ in range(max(0, council_rounds)):
        snapshot = dict(reports)              # read-before-write, as tier 1 does
        revised = {}
        for industry, agent in agents.items():
            if council_graph is None:
                peers = [r for i, r in snapshot.items() if i != industry]
            else:
                allowed = council_graph.get(agent.ticker, [])
                peer_industries = {ticker_to_industry[t] for t in allowed
                                   if t in ticker_to_industry}
                peers = [r for i, r in snapshot.items()
                         if i in peer_industries and i != industry]
            revised[industry] = agent.council_revise(snapshot[industry], peers)
            if revised[industry].source == "llm":
                stats["council_llm"] += 1
            else:
                stats["council_rule"] += 1
        reports = revised

    return list(reports.values()), stats
