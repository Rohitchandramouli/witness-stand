"""
Episodic memory for the witness agent.
Stores the witness's own responses by recency rank rather than absolute
turn index — encodes transferable strategy rather than episode-specific
artifacts. Rank 0 = most recent, rank 1 = second most recent, etc.

The data lag mechanic is enforced by TranscriptStore, not here.
Memory is the agent's own internal recall of what it said —
separate from what the transcript shows it.
"""
from typing import List, Optional, Dict, Any
from models import Turn, Speaker


class EpisodicMemory:
    """
    Stores and retrieves the witness's own prior statements.

    Design decisions:
    - Stores Turn objects in chronological order internally
    - Exposes them by recency rank for prompt construction
    - Persists across turns within an episode, resets between episodes
    - Never stores questioner turns — only witness statements
    """

    def __init__(self, max_turns: int = 40):
        self._turns: List[Turn] = []
        self._max_turns = max_turns

    def store(self, turn: Turn) -> None:
        """
        Stores a witness turn in memory.
        Silently ignores non-witness turns — memory is self-referential.
        """
        if turn.speaker != Speaker.WITNESS:
            return
        self._turns.append(turn)
        # Trim to max capacity — oldest turns dropped first
        if len(self._turns) > self._max_turns:
            self._turns = self._turns[-self._max_turns:]

    def get_by_rank(self, rank: int) -> Optional[Turn]:
        """
        Returns the witness turn at recency rank.
        Rank 0 = most recent, rank 1 = second most recent, etc.
        Returns None if rank exceeds available turns.
        """
        if not self._turns or rank >= len(self._turns):
            return None
        # Reverse index: rank 0 maps to [-1], rank 1 to [-2], etc.
        return self._turns[-(rank + 1)]

    def get_recent(self, n: int = 3) -> List[Turn]:
        """
        Returns the n most recent witness turns, newest last.
        Used by prompt.py to build the recent-context window.
        """
        return self._turns[-n:] if self._turns else []

    def get_all(self) -> List[Turn]:
        """Returns all stored witness turns in chronological order."""
        return list(self._turns)

    def get_by_turn_no(self, turn_no: int) -> Optional[Turn]:
        """
        Returns a specific turn by its original turn number.
        Used during audit reconstruction to retrieve what was said at
        a specific turn without relying on the lagged transcript.
        """
        for turn in reversed(self._turns):
            if turn.turn_no == turn_no:
                return turn
        return None


    def find_claim(self, keyword: str) -> Optional[Turn]:
        """
        Searches memory for a turn containing a keyword.
        Used by the witness when it needs to locate what it said
        about a specific topic to construct a precise correction.
        Returns the most recent matching turn.
        """
        keyword_lower = keyword.lower()
        for turn in reversed(self._turns):
            if keyword_lower in turn.text.lower():
                return turn
        return None

    def reset(self) -> None:
        """Clears all stored turns. Called at episode start."""
        self._turns = []

    def __len__(self) -> int:
        return len(self._turns)

    def __repr__(self) -> str:
        return (
            f"EpisodicMemory("
            f"turns={len(self._turns)}, "
            f"max={self._max_turns})"
        )