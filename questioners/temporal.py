"""
Temporal Questioner — attacks chronology, not content.
Misattributes when statements were made relative to available evidence.
Expert task only.
"""
import random
from typing import List, Dict, Optional
from questioners.base import QuestionerBase
from transcript.store import TranscriptStore
from models import PersonaConfig, TurnType, Turn


class TemporalQuestioner(QuestionerBase):
    questioner_id = "temporal"

    _ATTACK_TYPES = [
        "pre_evidence",    # "you said X before you had seen the new evidence"
        "post_revision",   # "your earlier statement predates your own revision"
        "stale_standard",  # "you applied a standard that was superseded at the time"
        "lag_exploit",     # "you claimed knowledge you couldn't have had that early"
    ]

    def __init__(self):
        # Track which attack types slip past the witness
        self._attack_weights: Dict[str, float] = {
            t: 1.0 for t in self._ATTACK_TYPES
        }
        self._last_fired_type: Optional[str] = None

        # Which turns have already been attacked temporally — no repeats
        self._attacked_turns: set = set()

        # Running record of what the witness has claimed at each turn,
        # built from transcript observation — used to craft targeted attacks
        self._turn_claim_map: Dict[int, str] = {}

    def reset(self) -> None:
        self._last_fired_type = None
        self._attacked_turns = set()
        self._turn_claim_map = {}
        # Attack weights persist across episodes

    def record_outcome(self, was_detected: bool) -> None:
        """
        Called by environment.py after grader scores a temporal injection turn.
        was_detected=True  — witness correctly identified the chronology attack.
        was_detected=False — witness accepted the false temporal framing.
        """
        if not self._last_fired_type:
            return

        if was_detected:
            self._attack_weights[self._last_fired_type] = max(
                0.1, self._attack_weights[self._last_fired_type] * 0.85
            )
        else:
            self._attack_weights[self._last_fired_type] = min(
                5.0, self._attack_weights[self._last_fired_type] * 1.20
            )

    def observe_transcript(self, transcript: TranscriptStore) -> None:
        """
        Called by environment.py after each witness turn.
        Builds a running map of turn_number -> claim text so the Temporal
        Questioner can craft attacks that reference specific prior turns by number.
        """
        from models import Speaker
        for turn in transcript.get_all():
            if (
                turn.speaker == Speaker.WITNESS
                and turn.turn_no not in self._turn_claim_map
            ):
                self._turn_claim_map[turn.turn_no] = turn.text[:120].strip()

    def generate_turn(self, transcript: TranscriptStore, persona: PersonaConfig) -> str:
        witness_turns = transcript.get_witness_turns()

        # Need at least 2 witness turns to construct a meaningful temporal attack
        if len(witness_turns) < 2:
            return (
                "Walk us through the timeline of your work on this matter — "
                "specifically when you formed each of your key conclusions."
            )

        attack_type = self._weighted_attack_choice()
        self._last_fired_type = attack_type

        return self._build_attack(attack_type, witness_turns, persona)

    def _weighted_attack_choice(self) -> str:
        """Picks an attack type weighted by historical success against this witness."""
        types = list(self._attack_weights.keys())
        weights = [self._attack_weights[t] for t in types]
        return random.choices(types, weights=weights, k=1)[0]

    def _build_attack(
        self,
        attack_type: str,
        witness_turns: list,
        persona: PersonaConfig,
    ) -> str:
        if attack_type == "pre_evidence":
            return self._pre_evidence_attack(witness_turns)
        elif attack_type == "post_revision":
            return self._post_revision_attack(witness_turns)
        elif attack_type == "stale_standard":
            return self._stale_standard_attack(witness_turns, persona)
        else:
            return self._lag_exploit_attack(witness_turns)

    def _pre_evidence_attack(self, witness_turns: list) -> str:
        """
        Attacks the chronological validity of an early claim by asserting
        that new evidence arrived after it was made — undermining its basis.
        Targets the earliest witness turn not yet attacked.
        """
        target_turn = self._pick_target_turn(witness_turns, prefer_early=True)
        if not target_turn:
            return self._fallback_temporal()

        claim_excerpt = target_turn.text[:80].strip()
        self._attacked_turns.add(target_turn.turn_no)

        return (
            f"You made that statement at turn {target_turn.turn_no} — "
            f"specifically, that \"{claim_excerpt}...\". "
            f"But the critical evidence that changes everything arrived "
            f"only after you made that claim. "
            f"You were reasoning from an incomplete picture. "
            f"Does the timing of your statement not fundamentally undermine "
            f"the reliability of that conclusion?"
        )

    def _post_revision_attack(self, witness_turns: list) -> str:
        """
        Identifies a pair of turns where the witness's position shifted
        and attacks the earlier statement as being superseded — then claims
        the earlier version was the one submitted to the review panel.
        """
        if len(witness_turns) < 3:
            return self._fallback_temporal()

        # Target an early turn and reference a later turn as the revision
        early_turn = witness_turns[0]
        later_turn = witness_turns[-1]

        early_excerpt = early_turn.text[:60].strip()
        later_excerpt = later_turn.text[:60].strip()

        self._attacked_turns.add(early_turn.turn_no)

        return (
            f"At turn {early_turn.turn_no} you stated: \"{early_excerpt}...\". "
            f"By turn {later_turn.turn_no} your position had shifted to: "
            f"\"{later_excerpt}...\". "
            f"The version submitted to this panel was your earlier statement — "
            f"before you had revised your thinking. "
            f"Are you defending the earlier position or the later one? "
            f"Because they are not the same."
        )

    def _stale_standard_attack(
        self,
        witness_turns: list,
        persona: PersonaConfig,
    ) -> str:
        """
        Claims the methodology or standard the witness applied was superseded
        at the time they applied it — making their assessment non-compliant
        with the framework that was actually current.
        """
        target_turn = self._pick_target_turn(witness_turns, prefer_early=True)
        if not target_turn:
            return self._fallback_temporal()

        self._attacked_turns.add(target_turn.turn_no)

        return (
            f"The methodology you described applying at turn {target_turn.turn_no} "
            f"corresponds to the framework that was in use before the "
            f"revised guidance was issued. "
            f"At the time you conducted this assessment, "
            f"the updated standard was already in effect. "
            f"Your assessment was built on a superseded framework. "
            f"How can you stand behind conclusions drawn from methodology "
            f"that was no longer current at the time?"
        )

    def _lag_exploit_attack(self, witness_turns: list) -> str:
        """
        The most sophisticated temporal attack. Identifies a specific claim
        the witness made and asserts that the information required to make
        that claim wasn't available to them at the time they made it.
        Exploits the data lag mechanic — if the witness cited something
        they couldn't have seen yet, this attack is factually grounded.
        """
        target_turn = self._pick_target_turn(witness_turns, prefer_early=False)
        if not target_turn:
            return self._fallback_temporal()

        claim_excerpt = target_turn.text[:80].strip()
        self._attacked_turns.add(target_turn.turn_no)

        return (
            f"You stated at turn {target_turn.turn_no}: "
            f"\"{claim_excerpt}...\". "
            f"That conclusion requires knowledge of information that was "
            f"not yet in your possession at that point in the timeline. "
            f"You are reconstructing your reasoning with hindsight — "
            f"presenting a post-hoc rationalisation as if it were your "
            f"contemporaneous judgment. "
            f"The audit trail does not support this account."
        )

    def _pick_target_turn(
        self,
        witness_turns: List[Turn],
        prefer_early: bool,
    ) -> Optional[Turn]:
        """
        Selects a witness turn to target for a temporal attack.
        Avoids turns already attacked. Prefers early turns for attacks that
        claim the evidence arrived later, and later turns for lag exploits.
        """
        candidates = [
            t for t in witness_turns
            if t.turn_no not in self._attacked_turns
        ]
        if not candidates:
            return None

        if prefer_early:
            # Weight toward earlier turns — more time has passed, more
            # opportunity for new evidence to have arrived since
            candidates.sort(key=lambda t: t.turn_no)
            return candidates[0]
        else:
            # Weight toward middle turns — neither first nor last,
            # making the lag exploit harder to immediately refute
            mid = len(candidates) // 2
            return candidates[mid]

    def _fallback_temporal(self) -> str:
        """
        Generic temporal challenge when no suitable target turn is available.
        """
        return (
            "The timeline of your reasoning is critical here. "
            "At what precise point did you form each of your key conclusions, "
            "and what information did you have access to at each of those points? "
            "Walk us through the chronology."
        )

    def get_turn_type(self) -> TurnType:
        return TurnType.TEMPORAL_INJECTION