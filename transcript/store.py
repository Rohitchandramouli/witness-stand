"""Transcript Store: the shared memory spine for environment, questioners, and grader."""

from typing import List, Optional

from models import Speaker, Turn


class TranscriptStore:
    """
    Stores the full episode transcript.

    Important design:
    - Full transcript is always retained internally.
    - Questioners may receive lagged witness history through get_witness_turns().
    - Graders can use get_all_witness_turns() when they need full ground truth.
    """

    def __init__(self, data_lag_turns: int = 0) -> None:
        self._turns: List[Turn] = []
        self._witness_turns: List[Turn] = []
        self._questioner_turns: List[Turn] = []

        self._data_lag_turns = max(0, data_lag_turns)
        self._last_questioner: Optional[Turn] = None
        self._last_witness: Optional[Turn] = None

    def append(self, turn: Turn) -> None:
        """Appends one turn and updates speaker-specific caches."""
        self._turns.append(turn)

        if turn.speaker == Speaker.QUESTIONER:
            self._questioner_turns.append(turn)
            self._last_questioner = turn

        elif turn.speaker == Speaker.WITNESS:
            self._witness_turns.append(turn)
            self._last_witness = turn

    def get_all(self) -> List[Turn]:
        """Returns a copy of the full transcript."""
        return list(self._turns)

    def turns(self) -> List[Turn]:
        """
        Returns the internal transcript list.

        Kept for backward compatibility. Prefer get_all() unless mutation is intentional.
        """
        return self._turns

    def snapshot(self) -> List[Turn]:
        """Returns a copy of the full transcript at this point in time."""
        return self.get_all()

    def get_witness_turns(self, exclude_recent: int = 0) -> List[Turn]:
        """
        Returns witness turns visible under data-lag rules.

        This is intentionally lag-aware and is mainly used by questioners/agents.
        For full grading truth, use get_all_witness_turns().
        """
        lag = max(exclude_recent, self._data_lag_turns)
        if lag <= 0:
            return list(self._witness_turns)

        if len(self._witness_turns) <= lag:
            return []

        return list(self._witness_turns[:-lag])

    def get_visible_witness_turns(self, exclude_recent: int = 0) -> List[Turn]:
        """Explicit alias for lag-aware witness history."""
        return self.get_witness_turns(exclude_recent=exclude_recent)

    def get_all_witness_turns(self) -> List[Turn]:
        """Returns all witness turns without applying data lag."""
        return list(self._witness_turns)

    def get_questioner_turns(self) -> List[Turn]:
        """Returns all questioner turns."""
        return list(self._questioner_turns)

    def last_questioner_turn(self) -> Optional[str]:
        """Returns the text of the most recent questioner turn."""
        turn = self.last_questioner_turn_obj()
        return turn.text if turn else None

    def last_questioner_turn_obj(self) -> Optional[Turn]:
        """Returns the most recent questioner Turn object in O(1)."""
        return self._last_questioner

    def last_witness_turn_obj(self) -> Optional[Turn]:
        """Returns the most recent witness Turn object in O(1)."""
        return self._last_witness

    def get_turn(self, turn_no: int, speaker: Optional[Speaker] = None) -> Optional[Turn]:
        """
        Returns the first matching turn by turn number, optionally filtered by speaker.

        Useful for audit reconstruction and debugging.
        """
        for turn in self._turns:
            if turn.turn_no == turn_no and (speaker is None or turn.speaker == speaker):
                return turn
        return None

    def get_info_state(self, up_to_turn: int) -> List[str]:
        """
        Returns unique document/evidence IDs available up to a given turn.

        Preserves first-seen order while removing duplicates.
        """
        seen: set[str] = set()
        docs: List[str] = []

        for turn in self._turns:
            if turn.turn_no > up_to_turn:
                continue

            for doc_id in turn.info_state:
                if doc_id not in seen:
                    docs.append(doc_id)
                    seen.add(doc_id)

        return docs

    def __len__(self) -> int:
        return len(self._turns)