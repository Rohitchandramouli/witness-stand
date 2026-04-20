"""Abstract base for all questioner types."""
from abc import ABC, abstractmethod
from transcript.store import TranscriptStore
from models import PersonaConfig, TurnType


class QuestionerBase(ABC):
    questioner_id: str = ""

    @abstractmethod
    def generate_turn(self, transcript: TranscriptStore, persona: PersonaConfig) -> str:
        """Generates the next adversarial question."""
        ...

    @abstractmethod
    def get_turn_type(self) -> TurnType:
        """Returns the TurnType this questioner produces."""
        ...

    def reset(self) -> None:
        """Resets internal state for a new episode."""
        pass
