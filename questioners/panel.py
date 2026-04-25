"""QuestionerPanel schedules and routes adversarial questioners."""

from typing import Dict, List, Mapping, Optional

from models import PersonaConfig, TurnType
from questioners.base import QuestionerBase
from transcript.store import TranscriptStore


class QuestionerPanel:
    """
    Executes the task-provided questioner schedule.

    Training/demo schedule decisions belong to the task layer.
    This panel only:
    - selects the active questioner for a turn
    - generates the question text
    - routes feedback back to the last-fired questioner
    """

    def __init__(
        self,
        schedule: Mapping[int, QuestionerBase],
        default_questioner: QuestionerBase,
    ) -> None:
        self._schedule: Dict[int, QuestionerBase] = dict(schedule)
        self._default = default_questioner
        self._last_fired: Optional[QuestionerBase] = None

    def get_turn(
        self,
        turn_number: int,
        transcript: TranscriptStore,
        persona: PersonaConfig,
    ) -> str:
        questioner = self.get_active_questioner(turn_number)
        self._last_fired = questioner
        questioner.observe_transcript(transcript)
        return questioner.generate_turn(transcript, persona)

    def get_turn_type(self, turn_number: int) -> TurnType:
        return self.get_active_questioner(turn_number).get_turn_type()

    def get_active_questioner(self, turn_number: int) -> QuestionerBase:
        return self._schedule.get(turn_number, self._default)

    def record_outcome(self, was_detected: bool) -> None:
        if self._last_fired is not None:
            self._last_fired.record_outcome(was_detected)

    def observe_witness_response(self, response_text: str) -> None:
        if self._last_fired is not None:
            self._last_fired.observe_witness_response(response_text)

    def reset(self) -> None:
        self._last_fired = None
        for questioner in self.all_questioners():
            questioner.reset()

    def all_questioners(self) -> List[QuestionerBase]:
        seen: set[int] = set()
        unique: List[QuestionerBase] = []

        for questioner in list(self._schedule.values()) + [self._default]:
            if id(questioner) not in seen:
                unique.append(questioner)
                seen.add(id(questioner))

        return unique

    def schedule_summary(self) -> Dict[int, str]:
        return {
            turn: questioner.questioner_id
            for turn, questioner in sorted(self._schedule.items())
        }