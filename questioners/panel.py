"""
Injection scheduler — decides which questioner fires at which turn.
Configuration is per-task. Each task defines its injection schedule.
"""
from typing import List, Optional, Mapping
from questioners.base import QuestionerBase
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
        questioner.observe_transcript(transcript)  # no-op for non-temporal questioners
        return questioner.generate_turn(transcript, persona)

    def get_turn_type(self, turn_number: int) -> TurnType:
        questioner = self._schedule.get(turn_number, self._default)
        return questioner.get_turn_type()

    def get_active_questioner(self, turn_number: int) -> QuestionerBase:
        """Returns the questioner instance that fires at the given turn.""";
        return self._schedule.get(turn_number, self._default)

    def record_outcome(self, was_detected: bool) -> None:
        """
        Routes outcome feedback to whichever questioner fired last.
        Called by environment.py after the grader scores each witness turn.
        """
        if self._last_fired is not None:
            self._last_fired.record_outcome(was_detected)

    def observe_witness_response(self, response_text: str) -> None:
        """Forwards witness response to last-fired questioner (all implement the no-op base).""";
        if self._last_fired is not None:
            self._last_fired.observe_witness_response(response_text)

    def reset(self) -> None:
        """Resets all questioners and clears last-fired state for a new episode."""
        self._last_fired = None
        for q in self.all_questioners():
            q.reset()

    def all_questioners(self) -> List[QuestionerBase]:
        """Returns all unique questioner instances in this panel."""
        seen = set()
        unique = []
        for q in list(self._schedule.values()) + [self._default]:
            if id(q) not in seen:
                unique.append(q)
                seen.add(id(q))
        return unique