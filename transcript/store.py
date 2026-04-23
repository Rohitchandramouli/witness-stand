from typing import List, Optional
from models import Turn, Speaker


class TranscriptStore:
    def __init__(self, data_lag_turns: int = 0):
        self._turns: List[Turn] = []
        self._data_lag_turns = data_lag_turns
        self._last_questioner: Optional[Turn] = None

    def append(self, turn: Turn) -> None:
        self._turns.append(turn)
        if turn.speaker == Speaker.QUESTIONER:
            self._last_questioner = turn

    def get_all(self) -> List[Turn]:
        return list(self._turns)

    def get_witness_turns(self, exclude_recent: int = 0) -> List[Turn]:
        witness_turns = [t for t in self._turns if t.speaker == Speaker.WITNESS]
        lag = max(exclude_recent, self._data_lag_turns)
        if lag > 0:
            witness_turns = witness_turns[:-lag] if len(witness_turns) > lag else []
        return witness_turns

    def last_questioner_turn(self) -> Optional[str]:
        t = self.last_questioner_turn_obj()
        return t.text if t else None

    def last_questioner_turn_obj(self) -> Optional[Turn]:
        """Returns the most recent questioner Turn object. O(1) via cached pointer."""
        if self._last_questioner is None:
            for turn in reversed(self._turns):
                if turn.speaker == Speaker.QUESTIONER:
                    self._last_questioner = turn
                    break
        return self._last_questioner

    def snapshot(self) -> List[Turn]:
        return list(self._turns)

    def get_info_state(self, up_to_turn: int) -> List[str]:
        docs = []
        for t in self._turns:
            if t.turn_no <= up_to_turn:
                docs.extend(t.info_state)
        return docs

    def __len__(self) -> int:
        return len(self._turns)