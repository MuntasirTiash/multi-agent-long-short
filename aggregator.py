"""
Aggregator — turn the agents' final scores into a long/short portfolio.

After the conversation ends we have one score per stock. Building the book is
then trivial: sort by score, take the best names long and the worst names
short. Because a long/short portfolio is inherently *relative*, the whole value
of the earlier communication is that these scores are now comparable to each
other rather than 30 independent guesses.
"""

from dataclasses import dataclass
from typing import Dict, List

from messages import StockMessage


@dataclass
class Portfolio:
    """The output of one rebalance: which names to long and which to short."""
    longs: List[str]
    shorts: List[str]
    ranking: List[StockMessage]   # every stock, best-to-worst, for inspection


def rank_and_split(messages: Dict[str, StockMessage],
                   n_long: int, n_short: int) -> Portfolio:
    """Sort stocks by score and slice off the top/bottom into long/short."""
    ranked = sorted(messages.values(), key=lambda m: m.score, reverse=True)
    longs = [m.ticker for m in ranked[:n_long]]
    shorts = [m.ticker for m in ranked[-n_short:]]
    return Portfolio(longs=longs, shorts=shorts, ranking=ranked)
