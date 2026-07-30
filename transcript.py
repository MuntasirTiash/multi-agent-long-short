"""
transcript.py — record the FULL history of what every agent said, every round.

The summary metrics tell you whether communication helped; the transcript tells
you *why* — exactly what each stock agent produced in its independent round-0
assessment, and how its message changed after it read specific peers.

Every agent message, at every round, of every topology and date, becomes one
JSON line in a `.jsonl` file (one JSON object per line — easy to `grep`, or to
load into pandas with `pd.read_json(path, lines=True)`):

    {"date": "2024-12-31", "topology": "round0-shared", "round": 0,
     "ticker": "AAPL", "score": 72.0, "direction": "LONG", "confidence": 0.6,
     "thesis": "...", "peers_seen": []}

For revision rounds, `peers_seen` lists the neighbour messages the agent read
before revising (ticker + score + direction), so the propagation of information
through the topology is fully reconstructable.

The same records are optionally echoed to the run log as readable one-liners
(controlled by `log_each`), so `tail -f` on the SLURM log shows the live
agent-by-agent play-by-play.
"""

import json
import logging
import os

logger = logging.getLogger("transcript")


class TranscriptRecorder:
    """Append-only recorder for per-agent messages across the whole run."""

    def __init__(self, path, log_each=True):
        self.path = path
        self.log_each = log_each
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Truncate any previous transcript so a fresh run starts clean.
        open(self.path, "w").close()
        self._n = 0

    def record(self, date, topology, message, peers_seen=None):
        """
        Write one agent message.

        Args:
            date (str):      the rebalance date.
            topology (str):  which topology this message belongs to
                             ("round0-shared" for the independent round).
            message:         a StockMessage.
            peers_seen:      list of StockMessage the agent read before this
                             message (empty/None for round 0).
        """
        peers = [{"ticker": p.ticker, "score": round(p.score, 1),
                  "direction": p.direction.value} for p in (peers_seen or [])]
        row = {
            "date": date,
            "topology": topology,
            "round": message.round_num,
            "ticker": message.ticker,
            "score": round(message.score, 2),
            "direction": message.direction.value,
            "confidence": round(message.confidence, 2),
            "thesis": message.thesis,
            "peers_seen": peers,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
        self._n += 1

        if self.log_each:
            seen = (" <- saw " + ",".join(f"{p['ticker']}:{p['score']:.0f}"
                                          for p in peers)) if peers else ""
            # Full thesis, never truncated — we don't want blind spots in the log.
            logger.info("    [%s|%s|r%d] %-5s score=%5.1f %-7s conf=%.2f | %s%s",
                        date, topology, message.round_num, message.ticker,
                        message.score, message.direction.value,
                        message.confidence, message.thesis, seen)

    def count(self):
        return self._n
