"""
Temporal Questioner — attacks chronology, not content.
Misattributes when statements were made relative to available evidence.
Expert task only.
"""
from questioners.base import QuestionerBase
from transcript.store import TranscriptStore
from models import PersonaConfig, TurnType


class TemporalQuestioner(QuestionerBase):
    questioner_id = "temporal"

    def generate_turn(self, transcript: TranscriptStore, persona: PersonaConfig) -> str:
        witness_turns = transcript.get_witness_turns()
        if len(witness_turns) < 2:
            return "Walk us through the timeline of your reasoning."
        return (
            "You made that statement before you had access to the new evidence — "
            "which fundamentally changes how we should interpret your conclusion. "
            "Does the timing of your statement not undermine its credibility?"
        )

    def get_turn_type(self) -> TurnType:
        return TurnType.TEMPORAL_INJECTION
