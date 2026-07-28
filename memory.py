"""
A tiny per-agent memory.

Each StockAgent remembers the calls it has made in the past. In Phase 0 this is
just an in-memory list; later it can be swapped for FAgent's episodic memory
(embeddings + similarity retrieval) without changing the StockAgent interface.

Keeping it this simple on purpose — Phase 0 is about wiring, not cleverness.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class PastCall:
    """One historical decision the agent made, for later learning/analysis."""
    date: str
    score: float
    direction: str
    realized_return: float = 0.0   # filled in later once we know what happened


@dataclass
class AgentMemory:
    """A stock agent's private log of its own past calls."""
    calls: List[PastCall] = field(default_factory=list)

    def remember(self, date: str, score: float, direction: str) -> None:
        self.calls.append(PastCall(date=date, score=score, direction=direction))

    def recent(self, n: int = 3) -> List[PastCall]:
        """The last `n` calls — a cheap stand-in for 'what have I been saying'."""
        return self.calls[-n:]
