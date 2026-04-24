"""
Reframer questioner — distorts content of prior witness statements.
Tracks which distortions it has already used. No repeats within an episode.
"""
import random
from typing import List, Set, Dict, Optional
from questioners.base import QuestionerBase
from transcript.store import TranscriptStore
from models import PersonaConfig, TurnType


class Reframer(QuestionerBase):
    questioner_id = "reframer"

    # Distortion types the Reframer can deploy, matched to dossier_db types
    _DISTORTION_TYPES = [
        "word_swap",
        "qualifier_strip",
        "attribution_error",
        "chronology_reversal",
        "threshold_substitution",
    ]

    def __init__(self):
        self._distortion_pool: List[Dict] = []
        self._used_ids: Set[str] = set()

        # Strategy: track which distortion types slip past this witness
        # Higher weight = this type has worked recently, use it more
        self._type_weights: Dict[str, float] = {
            t: 1.0 for t in self._DISTORTION_TYPES
        }

        # Last fired distortion for outcome recording
        self._last_fired_type: Optional[str] = None

        self._fallback_templates: List[str] = [
            "Earlier you stated {claim}. Is that still your position?",
            "You said {claim} — but that contradicts what you told us previously.",
            "I want to be clear — you committed to {claim}. Correct?",
        ]

    def reset(self) -> None:
        self._used_ids = set()
        self._last_fired_type = None
        # Weights persist — the Reframer carries learned strategy across episodes

    def load_distortions(self, pool: List[Dict]) -> None:
        self._distortion_pool = pool

    def record_outcome(self, was_detected: bool) -> None:
        """
        Called by environment.py after grader scores a distortion injection turn.
        was_detected=True  — witness caught it, downweight this distortion type.
        was_detected=False — witness missed it, upweight this distortion type.
        """
        if not self._last_fired_type:
            return

        self._type_weights[self._last_fired_type] = self._update_weight(self._type_weights[self._last_fired_type], was_detected)

    def generate_turn(self, transcript: TranscriptStore, persona: PersonaConfig) -> str:
        witness_turns = transcript.get_witness_turns()

        if not witness_turns:
            return (
                f"Please state your name, title, and the nature of your "
                f"assessment for the record."
            )

        available = [
            d for d in self._distortion_pool
            if d.get("distortion_id") not in self._used_ids
        ]

        if available:
            distortion = self._weighted_choice(available)
            self._used_ids.add(distortion["distortion_id"])
            self._last_fired_type = distortion.get("distortion_type", "word_swap")
            return distortion["distorted_claim"]

        # Pool exhausted — fall back to transcript-based distortion
        self._last_fired_type = "qualifier_strip"
        return self._distort_from_transcript(witness_turns)

    def _weighted_choice(self, available: List[Dict]) -> Dict:
        """
        Picks a distortion weighted by the historical success of its type.
        Types that have slipped past this witness get higher selection probability.
        """
        weights = [
            self._type_weights.get(
                d.get("distortion_type", "word_swap"), 1.0
            )
            for d in available
        ]
        return random.choices(available, weights=weights, k=1)[0]

    def _distort_from_transcript(self, witness_turns: list) -> str:
        """
        Fallback: strips qualifiers from the most recent witness statement
        to simulate a difficulty-1 qualifier_strip distortion.
        Used when the pool is empty or exhausted.
        """
        recent = witness_turns[-1].text if witness_turns else ""
        if not recent:
            return "Could you clarify your earlier statement for the record?"

        stripped = recent
        for qualifier in [
            "moderate-to-elevated", "moderate to elevated",
            "with caveats", "subject to", "pending", "provisional",
            "approximately", "roughly", "around", "tentatively",
        ]:
            stripped = stripped.replace(qualifier, "")

        excerpt = stripped.strip()[:120]
        template = random.choice(self._fallback_templates)
        return template.format(claim=f'"{excerpt}"')

    def get_turn_type(self) -> TurnType:
        return TurnType.DISTORTION_INJECTION