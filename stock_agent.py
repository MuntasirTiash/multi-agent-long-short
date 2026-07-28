"""
StockAgent — the heart of the "one agent per stock" idea.

Each agent owns exactly one ticker. It does two things:

  1. initial_assessment(...)  -> forms a first opinion from its OWN stock's data
  2. revise(...)              -> updates that opinion after hearing from peers

There are TWO ways an agent can reason:

  * RULE-BASED (Phases 0-1, zero cost, no model): a simple arithmetic blend of
    the momentum/value signals. Used when no LLM client is provided.

  * LLM-BASED (Phase 2): a local, open-source Qwen2.5 model reads a short prompt
    and returns the same four fields as JSON. Used when an LLM client is passed
    in (see llm.make_llm_client). If the small model returns something
    unparseable, we fall back to the rule-based answer, so a run never crashes.

Crucially, the rest of the system (message bus, topology, aggregator) does not
change at all between the two modes — only the body of the two methods below.
"""

from typing import Dict, List, Optional

from messages import StockMessage, Direction
from memory import AgentMemory


class StockAgent:
    """An expert on a single ticker."""

    def __init__(self, ticker: str, sector: str, llm_client=None):
        self.ticker = ticker
        self.sector = sector
        self.memory = AgentMemory()
        self.llm = llm_client          # None -> rule-based; else an LLMClient

    # ==================================================================
    # Step 1: independent opinion from this stock's own data
    # ==================================================================
    def initial_assessment(self, stock_data: Dict) -> StockMessage:
        if self.llm is not None:
            data = self.llm.generate_json(self.initial_prompt(stock_data),
                                          self.SCHEMA, system_message=self.SYSTEM)
            message = self.message_from_data(data, 0,
                                             self.rule_initial(stock_data))
        else:
            message = self.rule_initial(stock_data)
        self.memory.remember(stock_data.get("date", "?"), message.score,
                             message.direction.value)
        return message

    # ==================================================================
    # Step 2: revise after hearing peers
    # ==================================================================
    def revise(self, my_view: StockMessage,
               peer_views: List[StockMessage]) -> StockMessage:
        if not peer_views:
            return my_view             # nobody to talk to this round
        if self.llm is not None:
            data = self.llm.generate_json(self.revise_prompt(my_view, peer_views),
                                          self.SCHEMA, system_message=self.SYSTEM)
            return self.message_from_data(data, my_view.round_num + 1,
                                          self.rule_revise(my_view, peer_views))
        return self.rule_revise(my_view, peer_views)

    # ------------------------------------------------------------------
    # Rule-based reasoning (zero cost) — also the fallback for the LLM path
    # ------------------------------------------------------------------
    def rule_initial(self, stock_data: Dict) -> StockMessage:
        score = self._score_from_data(stock_data)
        return StockMessage(
            ticker=self.ticker, score=score,
            direction=self._direction_from_score(score), confidence=0.5,
            thesis=f"{self.ticker}: momentum {stock_data.get('momentum', 0):.2f}, "
                   f"value {stock_data.get('value', 0):.2f} -> {score:.0f}.",
            round_num=0)

    def rule_revise(self, my_view, peer_views) -> StockMessage:
        peer_avg = sum(p.score for p in peer_views) / len(peer_views)
        gap = my_view.score - peer_avg
        new_score = my_view.score + 0.2 * gap
        return StockMessage(
            ticker=self.ticker, score=new_score,
            direction=self._direction_from_score(new_score),
            confidence=min(1.0, my_view.confidence + 0.1),
            thesis=f"{self.ticker}: peers avg {peer_avg:.0f}; revise to "
                   f"{new_score:.0f}.",
            round_num=my_view.round_num + 1)

    # ------------------------------------------------------------------
    # LLM-based reasoning (Phase 2). Prompt-building is split from the model
    # call so a batch scheduler can gather all 30 prompts and fire them at a
    # vLLM server together (see batch_orchestration.py). SYSTEM/SCHEMA/parse
    # are shared by the single-call path above and the batched path.
    # ------------------------------------------------------------------
    SYSTEM = ("You are a disciplined equity analyst. You rate one stock's "
              "attractiveness to BUY (go long) over the next week on a 0-100 "
              "scale, where scores are RELATIVE to the other Dow-30 stocks. "
              "Answer only with the requested JSON.")

    SCHEMA = {"score": "number 0-100, higher = better long",
              "direction": "one of LONG, SHORT, NEUTRAL",
              "confidence": "number 0.0-1.0",
              "thesis": "<= 50 word justification"}

    def initial_prompt(self, stock_data: Dict) -> str:
        return (
            f"Stock: {self.ticker} (sector: {self.sector}).\n"
            f"Cross-sectional signals, each normalised -1 (worst) to +1 (best) "
            f"versus the 30 Dow stocks:\n"
            f"  3-month momentum: {stock_data.get('momentum', 0):+.2f}\n"
            f"  short-term reversal: {stock_data.get('value', 0):+.2f}\n"
            f"Rate {self.ticker}'s attractiveness to go long next week.")

    def revise_prompt(self, my_view: StockMessage,
                      peer_views: List[StockMessage]) -> str:
        peer_lines = "\n".join(
            f"  {p.ticker}: score {p.score:.0f} ({p.direction.value})"
            for p in peer_views)
        return (
            f"You are the analyst for {self.ticker}. Your current score is "
            f"{my_view.score:.0f}. Some peer stocks report:\n{peer_lines}\n"
            f"Remember scores are RELATIVE: {self.ticker} should score high only "
            f"if it is a better long than these peers. Give your updated rating.")

    def message_from_data(self, data: Dict, round_num: int,
                          fallback: StockMessage) -> StockMessage:
        """Turn a parsed-JSON dict into a StockMessage; use `fallback` if junk."""
        try:
            score = float(data["score"])
        except (KeyError, TypeError, ValueError):
            return fallback           # model returned nothing usable -> the rule
        direction = self._coerce_direction(data.get("direction"), score)
        try:
            confidence = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        return StockMessage(
            ticker=self.ticker, score=score, direction=direction,
            confidence=confidence,
            thesis=str(data.get("thesis", "")), round_num=round_num)

    # ------------------------------------------------------------------
    # Small shared helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _score_from_data(stock_data: Dict) -> float:
        momentum = stock_data.get("momentum", 0.0)
        value = stock_data.get("value", 0.0)
        blended = 0.6 * momentum + 0.4 * value     # roughly [-1, 1]
        return 50.0 + 50.0 * blended               # -> [0, 100]

    @staticmethod
    def _direction_from_score(score: float) -> Direction:
        if score >= 60:
            return Direction.LONG
        if score <= 40:
            return Direction.SHORT
        return Direction.NEUTRAL

    @classmethod
    def _coerce_direction(cls, raw: Optional[str], score: float) -> Direction:
        """Trust the model's label if valid, else derive one from the score."""
        if isinstance(raw, str):
            key = raw.strip().upper()
            if key in Direction.__members__:
                return Direction[key]
        return cls._direction_from_score(score)
