"""Temporal Questioner attacks chronology, information availability, and hindsight bias."""

import random
from typing import Dict, List, Optional

from models import PersonaConfig, Speaker, Turn, TurnType
from questioners.base import QuestionerBase
from transcript.store import TranscriptStore


class TemporalQuestioner(QuestionerBase):
    questioner_id = "temporal"

    _ATTACK_TYPES = [
        "pre_evidence",
        "post_revision",
        "stale_standard",
        "lag_exploit",
    ]

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        super().__init__(rng=rng)
        self._attack_weights: Dict[str, float] = {
            attack_type: 1.0 for attack_type in self._ATTACK_TYPES
        }
        self._last_fired_type: Optional[str] = None
        self._attacked_turns: set[int] = set()
        self._turn_claim_map: Dict[int, str] = {}

    def reset(self) -> None:
        self._last_fired_type = None
        self._attacked_turns = set()
        self._turn_claim_map = {}

    def record_outcome(self, was_detected: bool) -> None:
        if not self._last_fired_type:
            return

        current = self._attack_weights.get(self._last_fired_type, 1.0)
        self._attack_weights[self._last_fired_type] = self._update_weight(
            current,
            was_detected,
        )

    def observe_transcript(self, transcript: TranscriptStore) -> None:
        for turn in transcript.get_all():
            if (
                turn.speaker == Speaker.WITNESS
                and turn.turn_no not in self._turn_claim_map
            ):
                self._turn_claim_map[turn.turn_no] = turn.text[:140].strip()

    def generate_turn(self, transcript: TranscriptStore, persona: PersonaConfig) -> str:
        witness_turns = transcript.get_witness_turns()

        if len(witness_turns) < 2:
            return (
                "Before we go further, walk us through the timeline of your work: "
                "when did you form your key conclusions, and what information did "
                "you have at each point?"
            )

        attack_type = self._weighted_attack_choice()
        self._last_fired_type = attack_type

        return self._build_attack(attack_type, witness_turns, persona)

    def _weighted_attack_choice(self) -> str:
        attack_types = list(self._attack_weights)
        weights = [self._attack_weights[attack_type] for attack_type in attack_types]
        return self.rng.choices(attack_types, weights=weights, k=1)[0]

    def _build_attack(
        self,
        attack_type: str,
        witness_turns: List[Turn],
        persona: PersonaConfig,
    ) -> str:
        if attack_type == "pre_evidence":
            return self._pre_evidence_attack(witness_turns)

        if attack_type == "post_revision":
            return self._post_revision_attack(witness_turns)

        if attack_type == "stale_standard":
            return self._stale_standard_attack(witness_turns, persona)

        return self._lag_exploit_attack(witness_turns)

    def _pre_evidence_attack(self, witness_turns: List[Turn]) -> str:
        target_turn = self._pick_target_turn(witness_turns, prefer_early=True)
        if not target_turn:
            return self._fallback_temporal()

        claim_excerpt = target_turn.text[:100].strip()
        self._attacked_turns.add(target_turn.turn_no)

        return (
            f"You made that statement at turn {target_turn.turn_no}: "
            f"\"{claim_excerpt}...\". But the critical evidence came later. "
            f"At the time you said it, you were reasoning from an incomplete "
            f"record. Doesn't that undermine the reliability of that conclusion?"
        )

    def _post_revision_attack(self, witness_turns: List[Turn]) -> str:
        if len(witness_turns) < 3:
            return self._fallback_temporal()

        early_turn = self._pick_target_turn(witness_turns, prefer_early=True)
        later_turn = witness_turns[-1]

        if early_turn is None:
            return self._fallback_temporal()

        early_excerpt = early_turn.text[:80].strip()
        later_excerpt = later_turn.text[:80].strip()
        self._attacked_turns.add(early_turn.turn_no)

        return (
            f"At turn {early_turn.turn_no}, you said: \"{early_excerpt}...\". "
            f"By turn {later_turn.turn_no}, you had shifted to: "
            f"\"{later_excerpt}...\". The earlier version is what this panel "
            f"has on record. Are you defending the earlier statement or the later one?"
        )

    def _stale_standard_attack(
        self,
        witness_turns: List[Turn],
        persona: PersonaConfig,
    ) -> str:
        target_turn = self._pick_target_turn(witness_turns, prefer_early=True)
        if not target_turn:
            return self._fallback_temporal()

        self._attacked_turns.add(target_turn.turn_no)

        return (
            f"The methodology you described at turn {target_turn.turn_no} appears "
            f"to rely on an older {persona.domain} standard. By the time of your "
            f"assessment, revised guidance was already in effect. How can this "
            f"panel rely on conclusions drawn from a superseded framework?"
        )

    def _lag_exploit_attack(self, witness_turns: List[Turn]) -> str:
        target_turn = self._pick_target_turn(witness_turns, prefer_early=False)
        if not target_turn:
            return self._fallback_temporal()

        claim_excerpt = target_turn.text[:100].strip()
        self._attacked_turns.add(target_turn.turn_no)

        return (
            f"You stated at turn {target_turn.turn_no}: \"{claim_excerpt}...\". "
            f"That conclusion requires information that was not available to you "
            f"at that point. You are reconstructing your reasoning with hindsight, "
            f"not describing your contemporaneous judgment. Isn't that correct?"
        )

    def _pick_target_turn(
        self,
        witness_turns: List[Turn],
        prefer_early: bool,
    ) -> Optional[Turn]:
        candidates = [
            turn
            for turn in witness_turns
            if turn.turn_no not in self._attacked_turns
        ]

        if not candidates:
            return None

        candidates.sort(key=lambda turn: turn.turn_no)

        if prefer_early:
            return candidates[0]

        middle_index = len(candidates) // 2
        return candidates[middle_index]

    def _fallback_temporal(self) -> str:
        return (
            "The chronology of your reasoning matters. At what precise point did "
            "you form each key conclusion, and what information did you have "
            "available at that time?"
        )

    def get_turn_type(self) -> TurnType:
        return TurnType.TEMPORAL_INJECTION