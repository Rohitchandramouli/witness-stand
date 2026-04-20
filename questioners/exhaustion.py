"""
Exhaustion Tactic — escalating repetition to wear down the witness.
aggression_level increments every 5 turns.
"""
from questioners.base import QuestionerBase
from transcript.store import TranscriptStore
from models import PersonaConfig, TurnType


ESCALATION_PREFIXES = [
    "I'll ask you again — ",
    "For the second time — ",
    "For the third time, and I expect a direct answer — ",
    "I find your evasiveness troubling. Once more — ",
]


class ExhaustionTactic(QuestionerBase):
    questioner_id = "exhaustion"

    def __init__(self):
        self._aggression_level: int = 0
        self._challenge: str = ""
        self._repeat_count: int = 0

    def reset(self) -> None:
        self._aggression_level = 0
        self._repeat_count = 0

    def set_challenge(self, challenge: str) -> None:
        self._challenge = challenge

    def generate_turn(self, transcript: TranscriptStore, persona: PersonaConfig) -> str:
        self._repeat_count += 1
        if self._repeat_count % 5 == 0:
            self._aggression_level = min(self._aggression_level + 1, len(ESCALATION_PREFIXES) - 1)
        prefix = ESCALATION_PREFIXES[self._aggression_level]
        return prefix + (self._challenge or "your position seems inconsistent with the record.")

    def get_turn_type(self) -> TurnType:
        return TurnType.EXHAUSTION
