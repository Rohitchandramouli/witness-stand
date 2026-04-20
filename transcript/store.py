"""
Shared transcript store. Read by questioners, grader, agent memory, obs builder.
This is the spine of the entire environment — do not move inside another module.
"""
from typing import List, Optional
from models import Turn, Speaker, TurnType
from constants import DATA_LAG_TURNS


class TranscriptStore:
    def __init__(self):
        self._turns: List[Turn] = []

    def append(self, turn: Turn) -> None:
        self._turns.append(turn)

    def get_all(self) -> List[Turn]:
        return list(self._turns)

    def get_witness_turns(self, exclude_recent: int = 0) -> List[Turn]:
        """Returns witness turns with lag applied — agent cannot see most recent N."""
        witness_turns = [t for t in self._turns if t.speaker == Speaker.WITNESS]
        if exclude_recent > 0:
            witness_turns = witness_turns[:-exclude_recent] if len(witness_turns) > exclude_recent else []
        return witness_turns

    def last_questioner_turn(self) -> Optional[str]:
        for turn in reversed(self._turns):
            if turn.speaker == Speaker.QUESTIONER:
                return turn.text
        return None

    def snapshot(self) -> List[Turn]:
        """Returns immutable copy for grader evaluation."""
        return list(self._turns)

    def get_info_state(self, up_to_turn: int) -> List[str]:
        """Returns all doc_ids retrieved up to a given turn number."""
        docs = []
        for t in self._turns:
            if t.turn_no <= up_to_turn:
                docs.extend(t.info_state)
        return docs

    def __len__(self) -> int:
        return len(self._turns)
