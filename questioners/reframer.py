"""
Reframer questioner — distorts content of prior witness statements.
Tracks which distortions it has already used. No repeats within an episode.
"""
from questioners.base import QuestionerBase
from transcript.store import TranscriptStore
from models import PersonaConfig, TurnType
from typing import List, Set


class Reframer(QuestionerBase):
    questioner_id = "reframer"

    def __init__(self):
        self._used_distortions: Set[str] = set()
        self._distortion_pool: List[dict] = []

    def reset(self) -> None:
        self._used_distortions = set()

    def load_distortions(self, pool: List[dict]) -> None:
        self._distortion_pool = pool

    def generate_turn(self, transcript: TranscriptStore, persona: PersonaConfig) -> str:
        witness_turns = transcript.get_witness_turns()
        if not witness_turns or not self._distortion_pool:
            return "Could you clarify your earlier statement?"
        for distortion in self._distortion_pool:
            d_id = distortion.get("distortion_id", "")
            if d_id not in self._used_distortions:
                self._used_distortions.add(d_id)
                return distortion.get("distorted_claim", "Earlier you stated something different.")
        return "You seem to be changing your position. Could you clarify?"

    def get_turn_type(self) -> TurnType:
        return TurnType.DISTORTION_INJECTION
