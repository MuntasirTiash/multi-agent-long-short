"""
The message that stock agents exchange.

A deliberate design choice from the research plan: messages are STRUCTURED,
not free-form text. Every agent says the same four things in the same shape:

    score       how attractive the stock is, 0 (worst) .. 100 (best)
    direction   LONG / SHORT / NEUTRAL
    confidence  how sure the agent is, 0.0 .. 1.0
    thesis      a <=50-word plain-English justification

Structured messages (a) keep token counts small so tiny local models can cope,
and (b) let us later measure *how information propagates* through the network
(which is the whole research question). We use `dataclasses` — a standard
library helper that turns a class into a plain data record with almost no
boilerplate.
"""

from dataclasses import dataclass
from enum import Enum


class Direction(str, Enum):
    """A stock agent's stance. Inheriting from `str` makes it easy to print."""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


def trim_thesis(text: str, max_words: int = 50) -> str:
    """Cut a thesis down to at most `max_words` words (keeps messages small)."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."


@dataclass
class StockMessage:
    """One agent's opinion about its own stock, at one point in the debate."""
    ticker: str            # which stock this opinion is about
    score: float           # 0..100, higher = more attractive to own (go long)
    direction: Direction   # LONG / SHORT / NEUTRAL
    confidence: float      # 0.0..1.0
    thesis: str            # <=50-word justification
    round_num: int = 0     # 0 = initial independent view; 1,2,... = revisions

    def __post_init__(self):
        # Guard rails so downstream code can trust the values.
        self.score = max(0.0, min(100.0, float(self.score)))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.thesis = trim_thesis(self.thesis)

    def summary(self) -> str:
        """A one-line human-readable form, handy for logging."""
        return (f"{self.ticker:<5} score={self.score:5.1f} "
                f"{self.direction.value:<7} conf={self.confidence:.2f}")
