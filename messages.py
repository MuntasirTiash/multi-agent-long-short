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

from dataclasses import dataclass, field
from enum import Enum
from typing import List


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


# --------------------------------------------------------------------------
# Tier-2 wire format: what an industry leader sends upward
# --------------------------------------------------------------------------
# StockMessage stays the format stock agents speak. A leader says something
# structurally different — a RANKED SHORTLIST for its whole industry plus an
# industry-level view — so the hierarchy (see hierarchy.py) needs its own record.
# Keeping them separate means tier-1 code is untouched by tier-2 changes.
@dataclass
class IndustryPick:
    """One name an industry leader nominates, with the score it stands behind."""
    ticker: str
    score: float           # 0..100, the leader's own view (may differ from the
                           # analyst's, which is the point of the tier)
    direction: Direction
    thesis: str

    def __post_init__(self):
        self.score = max(0.0, min(100.0, float(self.score)))
        self.thesis = trim_thesis(self.thesis)


@dataclass
class IndustryReport:
    """One industry leader's verdict on its own industry."""
    industry: str
    leader: str                                    # the leader's own ticker
    longs: List["IndustryPick"] = field(default_factory=list)   # best first
    shorts: List["IndustryPick"] = field(default_factory=list)  # worst first
    outlook: str = ""                              # <=50 words on the industry
    round_num: int = 0                             # 0 = first pass, 1+ = council
    source: str = "rule"                           # "llm" or "rule"
    n_members: int = 0                             # how many firms it reviewed

    def __post_init__(self):
        self.outlook = trim_thesis(self.outlook)

    def picks(self) -> List["IndustryPick"]:
        return list(self.longs) + list(self.shorts)

    def mean_score(self) -> float:
        """Average score across this industry's picks — its relative standing."""
        picks = self.picks()
        return sum(p.score for p in picks) / len(picks) if picks else 50.0

    def summary(self) -> str:
        return (f"{self.industry:<24} leader={self.leader:<6} "
                f"L={[p.ticker for p in self.longs]} "
                f"S={[p.ticker for p in self.shorts]}")
