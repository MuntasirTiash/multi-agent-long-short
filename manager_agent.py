"""
manager_agent.py — the portfolio manager that sits ABOVE the 30 stock agents.

The 30 StockAgents debate and each ends with a structured opinion about its own
ticker (score, direction, confidence, thesis). A long/short book is inherently
*relative*, so someone has to look across all 30 at once and decide the actual
positions and their sizes. That someone is the ManagerAgent — the analyst ->
manager hierarchy from the FinCon design.

Like StockAgent, the manager has TWO interchangeable brains:

  * RULE-BASED (zero cost, no model): rank by score, take the top-N long and
    bottom-N short, and size each position by *conviction* = confidence x how
    far the score sits from neutral (50). Used when no LLM client is provided.

  * LLM-BASED: a local, open-source model reads the full 30-row table (every
    agent's score, direction, confidence and thesis) and returns a JSON book of
    longs/shorts with weights plus a one-line rationale. If it returns anything
    unusable, we fall back to the rule-based book, so a run never crashes.

Either way the output is a dollar-neutral book: long weights sum to +1, short
weights sum to -1 (gross exposure 2, net 0), so the daily return is a clean
long-minus-short spread that evaluation.portfolio_return can grade directly.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from messages import StockMessage, trim_thesis


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

    def __init__(self, n_long: int, n_short: int, llm_client=None):
        self.n_long = n_long
        self.n_short = n_short
        self.llm = llm_client              # None -> rule-based; else an LLMClient

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
        long_msgs = ranking[:self.n_long]
        short_msgs = ranking[-self.n_short:] if self.n_short else []
        long_w = self._conviction_weights(long_msgs, sign=+1.0)
        short_w = self._conviction_weights(short_msgs, sign=-1.0)
        weights = {**long_w, **short_w}
        return ManagedPortfolio(
            weights=weights,
            longs=[m.ticker for m in long_msgs],
            shorts=[m.ticker for m in short_msgs],
            rationale=(f"Rank-and-size: long top {len(long_msgs)}, short bottom "
                       f"{len(short_msgs)}, weighted by confidence x |score-50|."),
            source="rule", ranking=ranking)

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
              "each rated one Dow-30 stock (score 0-100, higher = better long). "
              "You read all of them and build ONE market-neutral book: pick the "
              "best names to go long and the worst to short, and size each by "
              "conviction. Answer only with the requested JSON.")

    SCHEMA = {
        "longs": "list of {\"ticker\": str, \"weight\": positive number}",
        "shorts": "list of {\"ticker\": str, \"weight\": positive number}",
        "rationale": "<= 40 word explanation of the book",
    }

    def build_prompt(self, ranking: List[StockMessage],
                     sectors: Optional[Dict[str, str]]) -> str:
        rows = []
        for m in ranking:
            sec = (sectors or {}).get(m.ticker, "?")
            rows.append(f"  {m.ticker:<5} {sec:<13} score={m.score:5.1f} "
                        f"{m.direction.value:<7} conf={m.confidence:.2f}  "
                        f"{trim_thesis(m.thesis, 25)}")
        table = "\n".join(rows)
        return (
            f"Analyst opinions on all {len(ranking)} stocks (sorted best long "
            f"first):\n{table}\n\n"
            f"Build a dollar-neutral long/short book. Go long up to {self.n_long} "
            f"of the most attractive names and short up to {self.n_short} of the "
            f"least attractive. Give each pick a positive conviction weight (the "
            f"longs are normalised together, the shorts together). Prefer names "
            f"where a high score is backed by high confidence and a clear thesis.")

    def portfolio_from_data(self, data: Dict,
                            messages: Dict[str, StockMessage],
                            ranking: List[StockMessage]) -> Optional[ManagedPortfolio]:
        """Parse the model's JSON book; return None (-> fall back) if unusable."""
        if not isinstance(data, dict):
            return None
        longs = self._clean_side(data.get("longs"), messages, self.n_long)
        shorts = self._clean_side(data.get("shorts"), messages, self.n_short)
        if not longs and not shorts:
            return None                   # nothing usable -> rule fallback
        weights = {}
        weights.update(self._normalise(longs, sign=+1.0))
        weights.update(self._normalise(shorts, sign=-1.0))
        # A ticker can't be both long and short; the long side wins if duplicated.
        for t in list(weights):
            if t in longs and t in shorts and weights[t] < 0:
                del weights[t]
        return ManagedPortfolio(
            weights=weights,
            longs=[t for t, _ in longs],
            shorts=[t for t, _ in shorts],
            rationale=str(data.get("rationale", ""))[:300],
            source="llm", ranking=ranking)

    @staticmethod
    def _clean_side(raw, messages, cap: int):
        """Validate one side into an ordered [(ticker, positive_weight)] list."""
        if not isinstance(raw, list):
            return []
        out, seen = [], set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            t = str(item.get("ticker", "")).strip().upper()
            if t not in messages or t in seen:
                continue                  # unknown or duplicate ticker
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
