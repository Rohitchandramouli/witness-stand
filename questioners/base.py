"""Abstract base for all questioner types."""
from abc import ABC, abstractmethod
from transcript.store import TranscriptStore
from models import PersonaConfig, TurnType


class QuestionerBase(ABC):
    questioner_id: str = ""

    @abstractmethod
    def generate_turn(self, transcript: TranscriptStore, persona: PersonaConfig) -> str:
        ...

    @abstractmethod
    def get_turn_type(self) -> TurnType:
        ...

    def record_outcome(self, was_detected: bool) -> None:
        """
        Default no-op. Stateful questioners override this to update
        their strategy weights based on whether the witness caught the attack.
        Called by environment.py after grader scores each injection turn.
        """
        pass

    def reset(self) -> None:
        pass

    