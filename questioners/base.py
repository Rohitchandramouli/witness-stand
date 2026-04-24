"""Abstract base for all questioner types."""
from abc import ABC, abstractmethod
from transcript.store import TranscriptStore
from models import PersonaConfig, TurnType


class QuestionerBase(ABC):
    questioner_id: str = ""

    @abstractmethod
    def generate_turn(self, transcript: TranscriptStore, persona: PersonaConfig) -> str: ...

    @abstractmethod
    def get_turn_type(self) -> TurnType: ...

    def record_outcome(self, was_detected: bool) -> None:
        """No-op default. Stateful questioners override to update strategy weights."""
        pass

    def observe_witness_response(self, response_text: str) -> None:
        """No-op default. ExhaustionTactic overrides to track response length trends."""
        pass

    def reset(self) -> None:
        pass

    # ── Shared weight update ──────────────────────────────────────────
    @staticmethod
    def _update_weight(current: float, was_detected: bool) -> float:
        """Multiplicative update shared by all adaptive questioners.
        Detected → downweight 0.85×, floor 0.1.
        Missed   → upweight  1.20×, ceiling 5.0.
        """
        if was_detected:
            return max(0.1, current * 0.85)
        return min(5.0, current * 1.20)
