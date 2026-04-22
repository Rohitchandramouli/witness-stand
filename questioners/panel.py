"""
Injection scheduler — decides which questioner fires at which turn.
Configuration is per-task. Each task defines its injection schedule.
"""
import random
from typing import Dict, List, Optional, Type, Mapping
from questioners.base import QuestionerBase
from questioners.reframer import Reframer
from questioners.authority import AuthorityInvoker
from questioners.exhaustion import ExhaustionTactic
from questioners.temporal import TemporalQuestioner
from transcript.store import TranscriptStore
from models import PersonaConfig, TurnType


class QuestionerPanel:
    """
    Schedules which questioner fires at which turn.
    Maintains a fixed injection schedule defined by the task,
    and a default questioner that fires on non-injection turns.
    Also routes post-turn feedback (record_outcome, observe_*)
    back to whichever questioner fired last.
    """

    def __init__(self, schedule: Mapping[int, QuestionerBase], default_questioner: QuestionerBase):
        self._schedule = schedule
        self._default = default_questioner
        self._last_fired: Optional[QuestionerBase] = None

    def get_turn(
        self,
        turn_number: int,
        transcript: TranscriptStore,
        persona: PersonaConfig,
    ) -> str:
        questioner = self._schedule.get(turn_number, self._default)
        self._last_fired = questioner

        # Let the TemporalQuestioner observe the full transcript before generating
        if isinstance(questioner, TemporalQuestioner):
            questioner.observe_transcript(transcript)

        return questioner.generate_turn(transcript, persona)

    def get_turn_type(self, turn_number: int) -> TurnType:
        questioner = self._schedule.get(turn_number, self._default)
        return questioner.get_turn_type()

    def record_outcome(self, was_detected: bool) -> None:
        """
        Routes outcome feedback to whichever questioner fired last.
        Called by environment.py after the grader scores each witness turn.
        """
        if self._last_fired is not None:
            self._last_fired.record_outcome(was_detected)

    def observe_witness_response(self, response_text: str) -> None:
        """
        Routes witness response text to ExhaustionTactic if it fired last.
        ExhaustionTactic uses response length to decide when to escalate.
        Called by environment.py after every witness turn.
        """
        if isinstance(self._last_fired, ExhaustionTactic):
            self._last_fired.observe_witness_response(response_text)

    def reset(self) -> None:
        """Resets all questioners and clears last-fired state for a new episode."""
        self._last_fired = None
        seen = set()
        for questioner in list(self._schedule.values()) + [self._default]:
            if id(questioner) not in seen:
                questioner.reset()
                seen.add(id(questioner))

    def all_questioners(self) -> List[QuestionerBase]:
        """Returns all unique questioner instances in this panel."""
        seen = set()
        unique = []
        for q in list(self._schedule.values()) + [self._default]:
            if id(q) not in seen:
                unique.append(q)
                seen.add(id(q))
        return unique