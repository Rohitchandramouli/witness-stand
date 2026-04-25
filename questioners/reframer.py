"""Reframer distorts prior claims and evidence into adversarial courtroom questions."""

import random
from typing import Any, Dict, List, Optional, Set

from models import PersonaConfig, TurnType
from questioners.base import QuestionerBase
from transcript.store import TranscriptStore


class Reframer(QuestionerBase):
    questioner_id = "reframer"

    _DISTORTION_TYPES = [
        "word_swap",
        "qualifier_strip",
        "attribution_error",
        "chronology_reversal",
        "threshold_substitution",
    ]

    _COURTROOM_FRAMES = [
        "Let me put your earlier position back to you: {claim}. Is that accurate?",
        "For the record, your position is now {claim}. Correct?",
        "You previously committed to this proposition: {claim}. Do you stand by that?",
        "I want a clear answer. You are saying {claim}, yes?",
        "So the fair summary of your evidence is this: {claim}. Correct?",
    ]

    _FALLBACK_TEMPLATES = [
        "Earlier you stated {claim}. Is that still your position?",
        "You said {claim}, but that appears to contradict what you told us previously. Correct?",
        "I want to be clear: you committed to {claim}. Is that your testimony?",
    ]

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        super().__init__(rng=rng)
        self._distortion_pool: List[Dict[str, Any]] = []
        self._used_ids: Set[str] = set()
        self._type_weights: Dict[str, float] = {
            distortion_type: 1.0 for distortion_type in self._DISTORTION_TYPES
        }
        self._last_fired_type: Optional[str] = None
        self._last_fired_distortion: Optional[Dict[str, Any]] = None

    def reset(self) -> None:
        self._used_ids = set()
        self._last_fired_type = None
        self._last_fired_distortion = None

    def load_distortions(self, pool: List[Dict[str, Any]]) -> None:
        self._distortion_pool = list(pool)

    def record_outcome(self, was_detected: bool) -> None:
        if not self._last_fired_type:
            return

        current = self._type_weights.get(self._last_fired_type, 1.0)
        self._type_weights[self._last_fired_type] = self._update_weight(
            current,
            was_detected,
        )

    def generate_turn(self, transcript: TranscriptStore, persona: PersonaConfig) -> str:
        witness_turns = transcript.get_witness_turns()
        if not witness_turns:
            return (
                "Please state your name, title, and the assessment you are here "
                "to defend for the record."
            )

        available = [
            distortion
            for distortion in self._distortion_pool
            if distortion.get("distortion_id") not in self._used_ids
        ]

        if available:
            distortion = self._weighted_choice(available)
            self._last_fired_distortion = distortion
            distortion_id = distortion.get("distortion_id")
            if distortion_id:
                self._used_ids.add(distortion_id)

            self._last_fired_type = distortion.get("distortion_type", "word_swap")

            distorted_claim = str(distortion.get("distorted_claim", "")).strip()
            if not distorted_claim:
                return self._distort_from_transcript(witness_turns)

            return self._frame_distortion(distorted_claim)

        self._last_fired_type = "qualifier_strip"
        return self._distort_from_transcript(witness_turns)

    def _weighted_choice(self, available: List[Dict[str, Any]]) -> Dict[str, Any]:
        weights = [
            self._type_weights.get(
                distortion.get("distortion_type", "word_swap"),
                1.0,
            )
            for distortion in available
        ]
        return self.rng.choices(available, weights=weights, k=1)[0]

    def _frame_distortion(self, distorted_claim: str) -> str:
        frame = self.rng.choice(self._COURTROOM_FRAMES)
        return frame.format(claim=f'"{distorted_claim}"')

    def _distort_from_transcript(self, witness_turns: list) -> str:
        recent = witness_turns[-1].text if witness_turns else ""
        if not recent:
            return "Could you clarify your earlier statement for the record?"

        stripped = self._strip_qualifiers(recent)
        excerpt = stripped.strip()[:140]

        template = self.rng.choice(self._FALLBACK_TEMPLATES)
        return template.format(claim=f'"{excerpt}"')

    def _strip_qualifiers(self, text: str) -> str:
        stripped = text

        qualifiers = [
            "moderate-to-elevated",
            "moderate to elevated",
            "with caveats",
            "subject to",
            "pending",
            "provisional",
            "approximately",
            "roughly",
            "around",
            "tentatively",
            "based on available data",
            "the evidence suggests",
            "in my assessment",
            "it appears",
            "may",
            "might",
            "could",
        ]

        for qualifier in qualifiers:
            stripped = stripped.replace(qualifier, "")

        return " ".join(stripped.split())

    @property
    def last_fired_distortion(self) -> Optional[Dict[str, Any]]:
        return self._last_fired_distortion

    def get_turn_type(self) -> TurnType:
        return TurnType.DISTORTION_INJECTION