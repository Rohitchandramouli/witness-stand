"""Base contract and shared utilities for all questioner types."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Optional

from models import PersonaConfig, TurnType
from transcript.store import TranscriptStore


class QuestionerBase(ABC):
    questioner_id: str = ""

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        self.rng = rng or random.Random(0)

    @abstractmethod
    def generate_turn(self, transcript: TranscriptStore, persona: PersonaConfig) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_turn_type(self) -> TurnType:
        raise NotImplementedError

    def record_outcome(self, was_detected: bool) -> None:
        """Optional feedback hook. Adaptive questioners override this."""
        pass

    def observe_witness_response(self, response_text: str) -> None:
        """Optional response hook. ExhaustionTactic uses this."""
        pass

    def observe_transcript(self, transcript: TranscriptStore) -> None:
        """Optional transcript hook. TemporalQuestioner uses this."""
        pass

    def reset(self) -> None:
        """Reset per-episode state."""
        pass

    @staticmethod
    def _update_weight(current: float, was_detected: bool) -> float:
        """
        Shared multiplicative update for adaptive adversaries.

        Detected attack  -> reduce future use of that tactic.
        Missed attack    -> increase future use of that tactic.
        """
        if was_detected:
            return max(0.1, current * 0.85)
        return min(5.0, current * 1.20)

    @staticmethod
    def _excerpt(text: str, max_chars: int = 120) -> str:
        clean = " ".join(str(text).split())
        if len(clean) <= max_chars:
            return clean
        return clean[: max_chars - 3].rstrip() + "..."
