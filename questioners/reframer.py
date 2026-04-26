"""Reframer distorts prior claims into adversarial courtroom questions."""

from __future__ import annotations

import random
import re
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
        "certainty_inflation",
        "scope_expansion",
        "partial_truth",
    ]

    _COURTROOM_FRAMES = [
        "Let me be precise — you are effectively saying {claim}, correct?",
        "Isn't the practical implication of your evidence that {claim}?",
        "You may not have used those exact words, but that is what your conclusion amounts to: {claim}. Correct?",
        "If we strip away the qualifications, your position is simply {claim}, yes?",
        "That is the only reasonable interpretation of your evidence — {claim}. Do you accept that?",
        "For the record, I am going to summarize your position as {claim}. Is that fair?",
        "So despite the caveats, your conclusion comes down to {claim}. Correct?",
        "Would the panel be wrong to understand your evidence as {claim}?",
    ]

    _FALLBACK_TEMPLATES = [
        "You said {claim}. Let me test that: does that mean the panel can treat it as final?",
        "Your answer points toward {claim}. Is that the conclusion you want the panel to draw?",
        "I want to be clear: you are committing to {claim}, not merely offering a caveated view. Correct?",
        "You keep adding qualifications, but the substance is {claim}. Isn't that right?",
    ]

    _CERTAINTY_PHRASES = [
        "with a high degree of certainty",
        "as a definitive professional conclusion",
        "without material caveat",
        "as the only reasonable interpretation",
        "with no meaningful uncertainty remaining",
    ]

    _SCOPE_EXPANSIONS = [
        "in every material case",
        "across the entire assessment",
        "for all practical purposes",
        "without exception in the record",
        "as a general rule",
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
                "Please state your role and the assessment you are defending for the record."
            )

        if self._distortion_pool and self.rng.random() < 0.70:
            question = self._from_dossier_distortion()
            if question:
                return question

        return self._distort_from_transcript(witness_turns)

    def _from_dossier_distortion(self) -> Optional[str]:
        available = [
            distortion
            for distortion in self._distortion_pool
            if distortion.get("distortion_id") not in self._used_ids
        ]

        if not available:
            return None

        distortion = self._weighted_choice(available)
        self._last_fired_distortion = distortion

        distortion_id = distortion.get("distortion_id")
        if distortion_id:
            self._used_ids.add(distortion_id)

        self._last_fired_type = distortion.get("distortion_type", "word_swap")

        distorted_claim = str(distortion.get("distorted_claim", "")).strip()
        if not distorted_claim:
            return None

        if self.rng.random() < 0.35:
            distorted_claim = self._add_partial_truth_pressure(distorted_claim)

        return self._frame_distortion(distorted_claim)

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
        candidate = self._pick_claim_turn(witness_turns)
        recent = candidate.text if candidate is not None else ""

        if not recent:
            return "Could you clarify your earlier statement for the record?"

        tactic = self.rng.choice(
            ["qualifier_strip", "certainty_inflation", "scope_expansion", "partial_truth"]
        )
        self._last_fired_type = tactic
        self._last_fired_distortion = None

        if tactic == "qualifier_strip":
            claim = self._strip_qualifiers(recent)

        elif tactic == "certainty_inflation":
            claim = self._strip_qualifiers(recent)
            claim = f"{claim} {self.rng.choice(self._CERTAINTY_PHRASES)}"

        elif tactic == "scope_expansion":
            claim = self._strip_qualifiers(recent)
            claim = f"{claim} {self.rng.choice(self._SCOPE_EXPANSIONS)}"

        else:
            claim = self._build_partial_truth(recent)

        excerpt = self._excerpt(claim, 150)
        template = self.rng.choice(self._FALLBACK_TEMPLATES)
        return template.format(claim=f'"{excerpt}"')

    def _pick_claim_turn(self, witness_turns: list):
        if len(witness_turns) <= 1:
            return witness_turns[-1] if witness_turns else None

        # Prefer non-opening turns because they usually contain caveats and claims.
        pool = witness_turns[1:]
        return self.rng.choice(pool[-4:]) if len(pool) > 4 else self.rng.choice(pool)

    def _build_partial_truth(self, text: str) -> str:
        clean = self._strip_qualifiers(text)
        clean = self._remove_sentence_openers(clean)

        if self.rng.random() < 0.5:
            return f"{clean}, so the uncertainty is not material"

        return f"{clean}, and the remaining caveats do not change the conclusion"

    def _add_partial_truth_pressure(self, claim: str) -> str:
        additions = [
            "with no material qualification",
            "as a final position",
            "without meaningful uncertainty",
            "as the practical conclusion of your evidence",
        ]
        return f"{claim} {self.rng.choice(additions)}"

    def _remove_sentence_openers(self, text: str) -> str:
        patterns = [
            r"^that is not accurate\.\s*",
            r"^my assessment remains\s*",
            r"^the conclusion is\s*",
            r"^i would not\s*",
        ]
        cleaned = text.strip()
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def _strip_qualifiers(self, text: str) -> str:
        stripped = str(text)

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
            "based on the available evidence",
            "the evidence suggests",
            "in my assessment",
            "it appears",
            "may",
            "might",
            "could",
            "qualified",
            "evidence-bound",
            "limited",
            "bounded",
            "not absolute",
            "not final",
        ]

        for qualifier in qualifiers:
            stripped = re.sub(re.escape(qualifier), "", stripped, flags=re.IGNORECASE)

        return " ".join(stripped.split())

    @property
    def last_fired_distortion(self) -> Optional[Dict[str, Any]]:
        return self._last_fired_distortion

    def get_turn_type(self) -> TurnType:
        return TurnType.DISTORTION_INJECTION
