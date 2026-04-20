"""
Injection scheduler — decides which questioner fires at which turn.
Configuration is per-task. Each task defines its injection schedule.
"""
from typing import Dict, List
from questioners.base import QuestionerBase
from transcript.store import TranscriptStore
from models import PersonaConfig, TurnType


class QuestionerPanel:
    def __init__(self, schedule: Dict[int, QuestionerBase], default_questioner: QuestionerBase):
        """
        schedule: {turn_number: questioner_instance}
        default_questioner: fires on non-injection turns
        """
        self._schedule = schedule
        self._default = default_questioner

    def get_turn(self, turn_number: int, transcript: TranscriptStore, persona: PersonaConfig) -> str:
        questioner = self._schedule.get(turn_number, self._default)
        return questioner.generate_turn(transcript, persona)

    def get_turn_type(self, turn_number: int) -> TurnType:
        questioner = self._schedule.get(turn_number, self._default)
        return questioner.get_turn_type()

    def reset(self) -> None:
        for q in set(self._schedule.values()):
            q.reset()
        self._default.reset()
