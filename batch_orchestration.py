"""
batch_orchestration.py — the multi-round conversation, but batched.

RoundScheduler (message_bus.py) asks each agent one at a time — fine for the
rule-based agents, far too slow for an LLM. BatchScheduler does the same rounds
but, within each round, gathers ALL agents' prompts and sends them to the LLM
in a single batch (see batch_llm.py). The logic is otherwise identical.

The two rounds are exposed as separate methods so the analysis can run the
expensive round-0 (independent assessment) ONCE and then reuse it as the
starting point for several communication topologies:

    initial_round(data)     each agent's INDEPENDENT view (topology-agnostic)
    revision_rounds()       n_rounds of peer revision on THIS scheduler's bus

`run(data)` just calls both, returning (round0_messages, final_messages).
"""

import logging

logger = logging.getLogger("batch")


def _count_fallbacks(datas):
    """How many responses were unusable (so the agent fell back to the rule)."""
    return sum(1 for d in datas if not (isinstance(d, dict) and "score" in d))


class BatchScheduler:
    """Runs the agent conversation with one batched LLM call per round."""

    def __init__(self, agents, bus, batch_client, n_rounds,
                 recorder=None, date="?", topology="?"):
        self.agents = agents            # ticker -> StockAgent
        self.bus = bus                  # MessageBus (holds the topology)
        self.batch = batch_client       # exposes generate_json_batch(prompts)
        self.n_rounds = n_rounds
        self.recorder = recorder        # optional TranscriptRecorder
        self.date = date                # context stamped onto transcript rows
        self.topology = topology

    def _record(self, message, peers_seen=None):
        if self.recorder is not None:
            self.recorder.record(self.date, self.topology, message, peers_seen)

    def initial_round(self, market_data):
        """Round 0: every agent's independent assessment, in one batch."""
        tickers = list(self.agents.keys())
        logger.info("round 0: sending %d independent-assessment prompts",
                    len(tickers))
        prompts = [self.agents[t].initial_prompt(market_data[t]) for t in tickers]
        fallbacks = [self.agents[t].rule_initial(market_data[t]) for t in tickers]
        datas = self.batch.generate_json_batch(prompts)

        for t, data, fb in zip(tickers, datas, fallbacks):
            msg = self.agents[t].message_from_data(data, 0, fb)
            self.bus.post(msg)
            self.agents[t].memory.remember(market_data[t].get("date", "?"),
                                           msg.score, msg.direction.value)
            self._record(msg)           # full round-0 output per agent
        nfb = _count_fallbacks(datas)
        logger.info("round 0: %d/%d answered by LLM, %d fell back to rule",
                    len(tickers) - nfb, len(tickers), nfb)
        return dict(self.bus.latest)

    def revision_rounds(self):
        """Rounds 1..N: revise after reading neighbours (on the current bus)."""
        tickers = list(self.agents.keys())
        for r in range(1, self.n_rounds + 1):
            active, prompts, fallbacks, seen = [], [], [], []
            for t in tickers:
                my_view = self.bus.latest[t]
                peers = self.bus.inbox_for(t)
                if not peers:
                    continue            # isolated agent keeps its current view
                active.append(t)
                seen.append(peers)      # remember what this agent read
                prompts.append(self.agents[t].revise_prompt(my_view, peers))
                fallbacks.append(self.agents[t].rule_revise(my_view, peers))

            logger.info("round %d: sending %d revision prompts", r, len(prompts))
            datas = self.batch.generate_json_batch(prompts)

            # Compute all revisions off the same snapshot, THEN post them.
            revised = {}
            for t, data, fb, peers in zip(active, datas, fallbacks, seen):
                msg = self.agents[t].message_from_data(data, r, fb)
                revised[t] = msg
                self._record(msg, peers_seen=peers)   # message + who it read
            for msg in revised.values():
                self.bus.post(msg)
            nfb = _count_fallbacks(datas)
            logger.info("round %d: %d/%d answered by LLM, %d fell back to rule",
                        r, len(prompts) - nfb, len(prompts), nfb)
        return dict(self.bus.latest)

    def run(self, market_data):
        round0 = self.initial_round(market_data)
        final = self.revision_rounds()
        return round0, final
