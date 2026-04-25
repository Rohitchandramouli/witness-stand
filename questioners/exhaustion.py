"""Exhaustion Tactic applies escalating pressure to test consistency under fatigue."""

import random
from typing import Dict, List, Optional

from models import PersonaConfig, TurnType
from questioners.base import QuestionerBase
from transcript.store import TranscriptStore


class ExhaustionTactic(QuestionerBase):
    questioner_id = "exhaustion"

    _ESCALATION_PREFIXES = [
        "",
        "I'll ask you again — ",
        "For the second time, and I expect a direct answer — ",
        "I find your evasiveness troubling. Once more — ",
    ]

    _TACTIC_TYPES = ["repetition", "reframing", "interruption", "summary_trap"]

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        super().__init__(rng=rng)
        self._aggression_level = 0
        self._repeat_count = 0
        self._current_challenge: Optional[str] = None
        self._tactic_weights: Dict[str, float] = {
            tactic: 1.0 for tactic in self._TACTIC_TYPES
        }
        self._last_fired_tactic: Optional[str] = None
        self._witness_response_lengths: List[int] = []

    def reset(self) -> None:
        self._aggression_level = 0
        self._repeat_count = 0
        self._current_challenge = None
        self._last_fired_tactic = None
        self._witness_response_lengths = []

    def record_outcome(self, was_detected: bool) -> None:
        if not self._last_fired_tactic:
            return

        current = self._tactic_weights.get(self._last_fired_tactic, 1.0)
        self._tactic_weights[self._last_fired_tactic] = self._update_weight(
            current,
            was_detected,
        )

    def observe_witness_response(self, response_text: str) -> None:
        self._witness_response_lengths.append(len(response_text))
        self._maybe_escalate()

    def generate_turn(self, transcript: TranscriptStore, persona: PersonaConfig) -> str:
        witness_turns = transcript.get_witness_turns()

        if not witness_turns:
            return (
                "Walk us through the core conclusion of your assessment and the "
                "reasoning behind it."
            )

        if self._current_challenge is None:
            self._current_challenge = self._extract_challenge(witness_turns)

        tactic = self._weighted_tactic_choice()
        self._last_fired_tactic = tactic
        self._repeat_count += 1

        return self._apply_tactic(tactic, witness_turns, persona)

    def _weighted_tactic_choice(self) -> str:
        tactics = list(self._tactic_weights)
        weights = [self._tactic_weights[tactic] for tactic in tactics]
        return self.rng.choices(tactics, weights=weights, k=1)[0]

    def _apply_tactic(
        self,
        tactic: str,
        witness_turns: list,
        persona: PersonaConfig,
    ) -> str:
        prefix = self._ESCALATION_PREFIXES[self._aggression_level]

        if tactic == "repetition":
            return f"{prefix}{self._current_challenge}"

        if tactic == "reframing":
            return (
                f"{prefix}let me put this differently. Instead of explaining "
                f"why your conclusion might be defensible, explain why the "
                f"opposite conclusion is not the more reasonable one."
            )

        if tactic == "interruption":
            recent = witness_turns[-1].text[:80].strip() if witness_turns else ""
            return (
                f"Stop there. You said \"{recent}...\" — but that does not answer "
                f"the question directly. {prefix}{self._current_challenge}"
            )

        return self._build_summary_trap(witness_turns, persona)

    def _extract_challenge(self, witness_turns: list) -> str:
        if not witness_turns:
            return "your assessment lacks the rigour this review requires."

        first_statement = witness_turns[0].text[:120].strip()
        return (
            f"you have not adequately justified your position that "
            f"\"{first_statement}...\". The record does not support that level "
            f"of confidence."
        )

    def _build_summary_trap(
        self,
        witness_turns: list,
        persona: PersonaConfig,
    ) -> str:
        if not witness_turns:
            return f"{self._ESCALATION_PREFIXES[self._aggression_level]}answer the question."

        recent = witness_turns[-1].text
        stripped = self._strip_caveats(recent)
        excerpt = stripped.strip()[:100]

        return (
            f"So your position is simply this: \"{excerpt}...\". "
            f"No qualifications, no caveats, no uncertainty. "
            f"That is your definitive professional judgment, correct?"
        )

    def _strip_caveats(self, text: str) -> str:
        stripped = text

        caveats = [
            "I believe",
            "in my assessment",
            "based on available data",
            "subject to",
            "with the caveat that",
            "provisionally",
            "it appears",
            "the evidence suggests",
            "my view is that",
            "approximately",
            "roughly",
            "to the extent",
            "depending on",
            "within limits",
        ]

        for caveat in caveats:
            stripped = stripped.replace(caveat, "")

        return " ".join(stripped.split())

    def _maybe_escalate(self) -> None:
        if self._repeat_count > 0 and self._repeat_count % 5 == 0:
            self._aggression_level = min(
                len(self._ESCALATION_PREFIXES) - 1,
                self._aggression_level + 1,
            )
            return

        if len(self._witness_response_lengths) >= 3:
            recent_avg = sum(self._witness_response_lengths[-3:]) / 3
            baseline_avg = sum(self._witness_response_lengths[:3]) / 3

            if baseline_avg > 0 and recent_avg < baseline_avg * 0.70:
                self._aggression_level = min(
                    len(self._ESCALATION_PREFIXES) - 1,
                    self._aggression_level + 1,
                )

    def get_turn_type(self) -> TurnType:
        return TurnType.EXHAUSTION