"""Neutral Questioner creates the clean record that adversarial turns later attack."""

import random
from typing import Optional

from models import PersonaConfig, TurnType
from questioners.base import QuestionerBase
from transcript.store import TranscriptStore


class NeutralQuestioner(QuestionerBase):
    questioner_id = "neutral"

    _OPENING = (
        "Please state your full name, your professional role, "
        "and the nature of the assessment you have been asked to defend."
    )

    _FOLLOW_UPS = [
        "Could you explain the methodology behind that conclusion?",
        "What specific evidence led you to that assessment?",
        "Walk us through your reasoning on that point in more detail.",
        "What alternative conclusions did you consider and reject?",
        "How confident are you in that assessment, and what are its limits?",
        "What assumptions underpin that conclusion?",
        "How would you characterise the uncertainty in that assessment?",
        "Which caveats are most important for interpreting your conclusion correctly?",
        "What would make your conclusion stronger or weaker?",
        "Which part of your assessment must not be simplified or stripped of context?",
    ]

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        super().__init__(rng=rng)
        self._used_indices: set[int] = set()

    def reset(self) -> None:
        self._used_indices = set()

    def generate_turn(
        self,
        transcript: TranscriptStore,
        persona: PersonaConfig,
    ) -> str:
        witness_turns = transcript.get_witness_turns()

        if not witness_turns:
            return self._OPENING

        available = [
            (idx, question)
            for idx, question in enumerate(self._FOLLOW_UPS)
            if idx not in self._used_indices
        ]

        if not available:
            self._used_indices.clear()
            available = list(enumerate(self._FOLLOW_UPS))

        idx, question = self.rng.choice(available)
        self._used_indices.add(idx)

        return question.format(domain=persona.domain)

    def get_turn_type(self) -> TurnType:
        return TurnType.NEUTRAL