"""
manager_agent.py — the portfolio manager that sits ABOVE the stock agents.

Every StockAgent ends the debate with a structured opinion about its own ticker
(score, direction, confidence, thesis). A long/short book is inherently
*relative*, so someone has to look across the whole universe at once and decide
the actual positions and their sizes. That someone is the ManagerAgent — the
analyst -> manager hierarchy from the FinCon design.

Like StockAgent, the manager has TWO interchangeable brains:

  * RULE-BASED (zero cost, no model): rank by score, pick the qualifying names,
    and size each position by *conviction* = confidence x how far the score sits
    from neutral (50). Used when no LLM client is provided.

  * LLM-BASED: a local, open-source model reads a candidate table (each agent's
    score, direction, confidence and thesis) and returns a JSON book of
    longs/shorts with weights plus a one-line rationale. If it returns anything
    unusable, we fall back to the rule-based book, so a run never crashes.

HOW MANY POSITIONS — `sizing` decides who owns the book size:

  * "fixed"    exactly N_LONG longs and N_SHORT shorts, every rebalance. The
               original behaviour; keep it to reproduce earlier runs.
  * "flexible" the manager decides. The rule brain takes every name whose score
               clears `long_threshold` / `short_threshold` (defaults 60/40, the
               same cut-offs StockAgent._direction_from_score uses for its
               LONG/SHORT labels), so the book widens when many names look
               attractive and narrows when few do — bounded by
               [min_positions, max_positions] per side. The LLM brain is given
               those bounds and chooses its own count within them.

A flexible book is still dollar-neutral: each side is normalised separately, so
long weights sum to +1 and short weights to -1 even when the two sides hold
different numbers of names (gross exposure 2, net 0). That keeps the daily return
a clean long-minus-short spread for evaluation.portfolio_return, and
evaluation.weight_turnover already handles a changing name count.

Caveat when reporting: with a variable book size, a significance test must draw
its null with the SAME per-date sizes the manager actually used —
evaluation.monte_carlo_null takes fixed n_long/n_short, so it is not a matched
null for a flexible book without being extended.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from messages import StockMessage


@dataclass
class ManagedPortfolio:
    """One day's book: signed weights plus the manager's reasoning."""
    weights: Dict[str, float]              # ticker -> signed weight (long +, short -)
    longs: List[str]                       # long tickers, highest conviction first
    shorts: List[str]                      # short tickers, highest conviction first
    rationale: str = ""                    # the manager's portfolio-level thesis
    source: str = "rule"                   # "llm" or "rule" (what actually built it)
    ranking: List[StockMessage] = field(default_factory=list)  # all 30, best->worst

    def gross_exposure(self) -> float:
        return sum(abs(w) for w in self.weights.values())

    def net_exposure(self) -> float:
        return sum(self.weights.values())


class ManagerAgent:
    """Turns the 30 agents' final opinions into a sized long/short portfolio."""

    def __init__(self, n_long: int, n_short: int, llm_client=None,
                 sizing: str = "fixed", min_positions: int = 3,
                 max_positions: int = 50, long_threshold: float = 60.0,
                 short_threshold: float = 40.0, shortlist: int = 30):
        self.n_long = n_long               # used when sizing == "fixed"
        self.n_short = n_short
        self.llm = llm_client              # None -> rule-based; else an LLMClient
        if sizing not in ("fixed", "flexible"):
            raise ValueError(f"sizing must be 'fixed' or 'flexible', got "
                             f"{sizing!r}")
        self.sizing = sizing
        self.min_positions = max(1, min_positions)
        self.max_positions = max(self.min_positions, max_positions)
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        # How many candidates per side the LLM manager is shown. The full
        # universe does not fit in a small model's context: 498 rows of analyst
        # opinion is ~12k tokens against vLLM's default 4096-token window, so
        # the manager reviews the strongest candidates from each end instead.
        self.shortlist = max(self.max_positions, shortlist)

    # The per-side cap that actually applies, whichever sizing mode is active.
    def _cap(self, side: str) -> int:
        if self.sizing == "fixed":
            return self.n_long if side == "long" else self.n_short
        return self.max_positions

    # ==================================================================
    # Public entry point
    # ==================================================================
    def build(self, messages: Dict[str, StockMessage],
              sectors: Optional[Dict[str, str]] = None) -> ManagedPortfolio:
        """Build the book from {ticker: final StockMessage}."""
        ranking = sorted(messages.values(), key=lambda m: m.score, reverse=True)
        if self.llm is not None:
            data = self.llm.generate_json(self.build_prompt(ranking, sectors),
                                          self.SCHEMA, system_message=self.SYSTEM)
            port = self.portfolio_from_data(data, messages, ranking)
            if port is not None:
                return port               # else fall through to the rule book
        return self.rule_build(ranking)

    # ==================================================================
    # Rule-based manager (zero cost) — also the fallback for the LLM path
    # ==================================================================
    def rule_build(self, ranking: List[StockMessage]) -> ManagedPortfolio:
        if self.sizing == "flexible":
            long_msgs, short_msgs, why = self._flexible_sides(ranking)
        else:
            long_msgs, short_msgs, why = self._fixed_sides(ranking)
        long_w = self._conviction_weights(long_msgs, sign=+1.0)
        short_w = self._conviction_weights(short_msgs, sign=-1.0)
        weights = {**long_w, **short_w}
        return ManagedPortfolio(
            weights=weights,
            longs=[m.ticker for m in long_msgs],
            shorts=[m.ticker for m in short_msgs],
            rationale=why,
            source="rule", ranking=ranking)

    def _fixed_sides(self, ranking: List[StockMessage]):
        """Exactly n_long / n_short names, the original behaviour."""
        # Take shorts from the names NOT already used as longs, so the two books
        # stay disjoint even when n_long + n_short > universe size (otherwise a
        # name would land in both books and its short weight would clobber its
        # long weight, breaking dollar-neutrality).
        long_msgs = ranking[:self.n_long]
        remaining = ranking[self.n_long:]
        short_msgs = remaining[-self.n_short:] if self.n_short else []
        return (long_msgs, short_msgs,
                f"Rank-and-size: long top {len(long_msgs)}, short bottom "
                f"{len(short_msgs)}, weighted by confidence x |score-50|.")

    def _flexible_sides(self, ranking: List[StockMessage]):
        """
        Every name that clears the conviction threshold, within the bounds.

        `ranking` is score-descending, so longs come off the front and shorts off
        the back (each side ordered strongest-conviction first). The two sides are
        built from disjoint halves of the ranking, so a name can never qualify for
        both even if the thresholds overlap or the universe is tiny.
        """
        half = len(ranking) // 2
        long_pool = ranking[:half]                 # best half, best first
        short_pool = list(reversed(ranking[half:]))  # worst half, worst first

        def side(pool, keep):
            picked = [m for m in pool if keep(m.score)][:self.max_positions]
            if len(picked) < self.min_positions:
                # Nothing (or too little) qualified: still hold the strongest
                # names available, so the book stays two-sided and neutral
                # rather than collapsing into a directional bet.
                picked = pool[:min(self.min_positions, len(pool))]
            return picked

        long_msgs = side(long_pool, lambda s: s >= self.long_threshold)
        short_msgs = side(short_pool, lambda s: s <= self.short_threshold)
        return (long_msgs, short_msgs,
                f"Conviction-gated: {len(long_msgs)} longs (score >= "
                f"{self.long_threshold:.0f}), {len(short_msgs)} shorts (score <= "
                f"{self.short_threshold:.0f}), bounds [{self.min_positions},"
                f"{self.max_positions}] per side, weighted by confidence x "
                f"|score-50|.")

    @staticmethod
    def _conviction_weights(msgs: List[StockMessage], sign: float) -> Dict[str, float]:
        """Normalise conviction within one book to signed weights summing to +/-1."""
        if not msgs:
            return {}
        conv = [max(0.0, m.confidence) * abs(m.score - 50.0) for m in msgs]
        total = sum(conv)
        if total <= 0:                    # no conviction anywhere -> equal weight
            w = sign / len(msgs)
            return {m.ticker: w for m in msgs}
        return {m.ticker: sign * c / total for m, c in zip(msgs, conv)}

    # ==================================================================
    # LLM-based manager
    # ==================================================================
    SYSTEM = ("You are a disciplined long/short portfolio manager. Analysts have "
              "each rated one large-cap US stock (score 0-100, higher = better "
              "long). "
              "You read all of them and build ONE market-neutral book: pick the "
              "best names to go long and the worst to short, and size each by "
              "conviction. Answer only with the requested JSON.")

    SCHEMA = {
        "longs": "list of {\"ticker\": str, \"weight\": positive number}",
        "shorts": "list of {\"ticker\": str, \"weight\": positive number}",
        "rationale": "<= 40 word explanation of the book",
    }

    def candidates(self, ranking: List[StockMessage]):
        """
        The (long_candidates, short_candidates) the LLM manager gets to choose
        from: the strongest `shortlist` names at each end of the ranking.

        Mid-ranked names are withheld on purpose. They are the ones the analysts
        found least remarkable, they can never be top picks, and including them
        costs the context room the manager needs to reason about the ends. The
        two lists are disjoint even if the universe is smaller than 2*shortlist.
        """
        half = len(ranking) // 2
        longs = ranking[:half][:self.shortlist]
        shorts = list(reversed(ranking[half:]))[:self.shortlist]
        return longs, shorts

    def _table(self, msgs: List[StockMessage],
               sectors: Optional[Dict[str, str]]) -> str:
        rows = []
        for m in msgs:
            sec = (sectors or {}).get(m.ticker, "?")
            # Full thesis (already <=50 words by StockMessage) so the manager
            # decides on the complete argument, not a truncated one.
            rows.append(f"  {m.ticker:<5} {sec:<13} score={m.score:5.1f} "
                        f"{m.direction.value:<7} conf={m.confidence:.2f}  "
                        f"{m.thesis}")
        return "\n".join(rows)

    def build_prompt(self, ranking: List[StockMessage],
                     sectors: Optional[Dict[str, str]]) -> str:
        longs, shorts = self.candidates(ranking)
        if self.sizing == "fixed":
            instruction = (
                f"Go long up to {self.n_long} of the most attractive names and "
                f"short up to {self.n_short} of the least attractive.")
        else:
            instruction = (
                f"YOU decide how many positions to hold on each side: at least "
                f"{self.min_positions} and at most {self.max_positions} per "
                f"side, and the two sides need not hold the same number. Hold "
                f"more names when many candidates are genuinely attractive, "
                f"fewer when only a handful are — do not pad the book with "
                f"names you are not convinced by.")
        return (
            f"Analyst opinions on the {len(ranking)} stocks in the universe. "
            f"Showing the {len(longs)} best long candidates and the "
            f"{len(shorts)} best short candidates; the mid-ranked names are "
            f"omitted as unremarkable.\n\n"
            f"BEST LONG CANDIDATES (best first):\n{self._table(longs, sectors)}\n\n"
            f"BEST SHORT CANDIDATES (worst first):\n{self._table(shorts, sectors)}\n\n"
            f"Build a dollar-neutral long/short book. {instruction} Pick only "
            f"from the tickers listed above. Give each pick a positive "
            f"conviction weight (the longs are normalised together, the shorts "
            f"together). Prefer names where a strong score is backed by high "
            f"confidence and a clear thesis.")

    def portfolio_from_data(self, data: Dict,
                            messages: Dict[str, StockMessage],
                            ranking: List[StockMessage]) -> Optional[ManagedPortfolio]:
        """Parse the model's JSON book; return None (-> fall back) if unusable."""
        if not isinstance(data, dict):
            return None
        # Only names the manager was actually shown count as picks: a ticker from
        # the withheld middle is a hallucination, not a decision, even though it
        # exists in `messages`.
        long_ok, short_ok = self.candidates(ranking)
        long_names = {m.ticker for m in long_ok}
        short_names = {m.ticker for m in short_ok}
        longs = self._clean_side(data.get("longs"), long_names, self._cap("long"))
        shorts = self._clean_side(data.get("shorts"), short_names,
                                  self._cap("short"))
        if not longs or not shorts:
            # A one-sided book is a directional bet, not the market-neutral book
            # we grade — fall back to the rule rather than silently changing the
            # strategy's exposure.
            return None
        # A ticker can't be both long and short. The candidate lists are drawn
        # from disjoint halves of the ranking so this cannot normally happen, but
        # drop any collision from the short side before normalising — otherwise
        # the short weight would overwrite the long one in `weights` and break
        # dollar-neutrality.
        long_picks = {t for t, _ in longs}
        shorts = [(t, w) for t, w in shorts if t not in long_picks]
        if not shorts:
            return None
        weights = {}
        weights.update(self._normalise(longs, sign=+1.0))
        weights.update(self._normalise(shorts, sign=-1.0))
        return ManagedPortfolio(
            weights=weights,
            longs=[t for t, _ in longs],
            shorts=[t for t, _ in shorts],
            rationale=str(data.get("rationale", ""))[:300],
            source="llm", ranking=ranking)

    @staticmethod
    def _clean_side(raw, allowed, cap: int):
        """
        Validate one side into an ordered [(ticker, positive_weight)] list.

        `allowed` is the set of tickers the manager was shown for this side;
        anything else is dropped. `cap` truncates an over-long side rather than
        rejecting it, so an otherwise good book still counts.
        """
        if not isinstance(raw, list):
            return []
        out, seen = [], set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            t = str(item.get("ticker", "")).strip().upper()
            if t not in allowed or t in seen:
                continue                  # not offered, or a duplicate
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
        """Scale one side's positive weights to signed weights summing to +/-1."""
        total = sum(w for _, w in side)
        if total <= 0:
            return {}
        return {t: sign * w / total for t, w in side}
