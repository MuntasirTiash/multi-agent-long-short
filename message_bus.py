"""
The message bus and the round scheduler — the "post office" of the system.

  MessageBus     stores the latest message from every agent and, given a
                 topology, hands each agent exactly the peer messages it is
                 allowed to see.

  RoundScheduler drives the conversation: round 0 (everyone forms an
                 independent view), then N revision rounds (everyone updates
                 after reading their neighbours). It returns the final set of
                 messages, one per stock, ready for ranking.

Both are plain Python objects using only the standard library. Keeping the
scheduler separate from the agents means we can change the *protocol* (rounds,
who-hears-whom) without touching agent logic — the separation the research
plan depends on.
"""

import logging
import time
from typing import Dict, List

from messages import StockMessage
from stock_agent import StockAgent

logger = logging.getLogger("message_bus")


class MessageBus:
    """Holds the current message from each agent and routes peer messages."""

    def __init__(self, topology: Dict[str, List[str]]):
        self.topology = topology
        self.latest: Dict[str, StockMessage] = {}   # ticker -> its last message

    def post(self, message: StockMessage) -> None:
        """An agent publishes (or overwrites) its current opinion."""
        self.latest[message.ticker] = message

    def inbox_for(self, ticker: str) -> List[StockMessage]:
        """The messages `ticker` is allowed to read, per the topology."""
        neighbours = self.topology.get(ticker, [])
        return [self.latest[n] for n in neighbours if n in self.latest]


class RoundScheduler:
    """Runs the multi-round conversation between stock agents."""

    def __init__(self, agents: Dict[str, StockAgent], bus: MessageBus,
                 n_rounds: int, usage=None, date="?", topology="?"):
        self.agents = agents          # ticker -> StockAgent
        self.bus = bus
        self.n_rounds = n_rounds
        # Optional instrumentation.UsageLog. On this (rule-based) path no LLM
        # call happens, so passing one turns the run into a FORECAST: the prompts
        # that an LLM run would have sent are built and counted, giving a
        # full-size token budget for zero GPU time. Rows are marked "estimated".
        self.usage = usage
        self.date = date
        self.topology = topology

    def _forecast(self, prompts, stage: str, wall_s: float) -> None:
        if self.usage is None or not prompts:
            return
        # max_tokens on the real clients is 256; charge that as the worst-case
        # completion so the forecast is an upper bound, not an underestimate.
        self.usage.estimate_calls(prompts, stage=stage, date=self.date,
                                  topology=self.topology, wall_s=wall_s,
                                  completion_tokens_each=256)

    def run(self, market_data: Dict[str, Dict]) -> Dict[str, StockMessage]:
        """
        Execute the whole debate and return the final message per stock.

        market_data maps ticker -> that stock's data dict.
        """
        # Round 0: every agent forms an independent opinion and posts it.
        logger.info("Round 0: independent assessments")
        t0 = time.time()
        prompts = []
        for ticker, agent in self.agents.items():
            if self.usage is not None:
                prompts.append(agent.initial_prompt(market_data[ticker]))
            msg = agent.initial_assessment(market_data[ticker])
            self.bus.post(msg)
        self._forecast(prompts, "round0", time.time() - t0)

        # Rounds 1..N: every agent reads its neighbours, then revises.
        for r in range(1, self.n_rounds + 1):
            logger.info(f"Round {r}: revisions after reading peers")
            t0 = time.time()
            # Read everyone's current view first, THEN post the revisions, so
            # that within a round agents all react to the same snapshot.
            revised: Dict[str, StockMessage] = {}
            prompts = []
            for ticker, agent in self.agents.items():
                my_view = self.bus.latest[ticker]
                peers = self.bus.inbox_for(ticker)
                if self.usage is not None and peers:
                    prompts.append(agent.revise_prompt(my_view, peers))
                revised[ticker] = agent.revise(my_view, peers)
            for msg in revised.values():
                self.bus.post(msg)
            self._forecast(prompts, f"round{r}", time.time() - t0)

        return dict(self.bus.latest)
