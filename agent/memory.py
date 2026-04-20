"""
Rank-based episodic memory for the witness agent.
Stores this episode's own responses by recency (rank 0 = most recent).
Does NOT use absolute indices — lesson from Cascade Containment.
"""
from typing import List
from models import Turn, Speaker


class EpisodicMemory:
    def __init__(self):
        self._statements: List[str] = []

    def store(self, turn: Turn) -> None:
        if turn.speaker == Speaker.WITNESS:
            self._statements.append(turn.text)

    def retrieve(self, k: int = 5) -> List[str]:
        """Returns k most recent witness statements, rank 0 = most recent."""
        return list(reversed(self._statements[-k:]))

    def clear(self) -> None:
        self._statements = []

    def __len__(self) -> int:
        return len(self._statements)
