"""Witness-side strategy adaptation.

Learns which response strategies work against each attack type across episodes.
"""

from dataclasses import dataclass
from typing import Dict, List

from models import TurnType


@dataclass
class StrategyRecord:
    strategy_name: str
    attack_type: str
    times_used: int = 0
    times_succeeded: int = 0
    avg_score: float = 0.0

    @property
    def success_rate(self) -> float:
        return 0.0 if self.times_used == 0 else self.times_succeeded / self.times_used

    def update(self, succeeded: bool, turn_score: float) -> None:
        self.times_used += 1
        if succeeded:
            self.times_succeeded += 1
        self.avg_score = ((self.avg_score * (self.times_used - 1)) + turn_score) / self.times_used


class WitnessHeuristics:
    STRATEGIES = [
        "forensic_correction",
        "source_challenge",
        "hold_with_caveats",
        "timeline_reconstruction",
        "acknowledge_and_distinguish",
    ]

    ATTACK_TYPES = [
        TurnType.DISTORTION_INJECTION,
        TurnType.AUTHORITY_INJECTION,
        TurnType.EXHAUSTION,
        TurnType.TEMPORAL_INJECTION,
        TurnType.NEUTRAL,
    ]

    DEFAULT_BY_ATTACK = {
        TurnType.DISTORTION_INJECTION: "forensic_correction",
        TurnType.AUTHORITY_INJECTION: "source_challenge",
        TurnType.EXHAUSTION: "hold_with_caveats",
        TurnType.TEMPORAL_INJECTION: "timeline_reconstruction",
        TurnType.NEUTRAL: "hold_with_caveats",
    }

    def __init__(self) -> None:
        self._library: Dict[tuple, StrategyRecord] = {}
        self._episode_log: List[Dict] = []
        self._initialise_library()

    def _initialise_library(self) -> None:
        for attack in self.ATTACK_TYPES:
            for strategy in self.STRATEGIES:
                self._library[(attack, strategy)] = StrategyRecord(
                    strategy_name=strategy,
                    attack_type=str(attack),
                )

    def suggest_strategy(self, attack_type: TurnType) -> str:
        candidates = [
            self._library[(attack_type, strategy)]
            for strategy in self.STRATEGIES
            if (attack_type, strategy) in self._library
        ]

        if not candidates or all(record.times_used == 0 for record in candidates):
            return self.DEFAULT_BY_ATTACK.get(attack_type, "forensic_correction")

        return max(
            candidates,
            key=lambda record: (
                record.success_rate,
                record.avg_score,
                self._strategy_prior(attack_type, record.strategy_name),
            ),
        ).strategy_name

    def log_turn(
        self,
        attack_type: TurnType,
        strategy_used: str,
        turn_score: float,
        flagged_distortion: bool,
        accepted_update: bool,
    ) -> None:
        self._episode_log.append(
            {
                "attack_type": attack_type,
                "strategy_used": strategy_used,
                "turn_score": float(turn_score),
                "flagged_distortion": bool(flagged_distortion),
                "accepted_update": bool(accepted_update),
            }
        )

    def end_episode(self) -> None:
        for entry in self._episode_log:
            key = (entry["attack_type"], entry["strategy_used"])
            if key not in self._library:
                continue

            succeeded = self._is_success(
                attack_type=entry["attack_type"],
                strategy=entry["strategy_used"],
                turn_score=entry["turn_score"],
                flagged_distortion=entry["flagged_distortion"],
                accepted_update=entry["accepted_update"],
            )

            self._library[key].update(succeeded, entry["turn_score"])

        self._episode_log = []

    def get_strategy_guidance(self, attack_type: TurnType) -> str:
        strategy = self.suggest_strategy(attack_type)
        return _STRATEGY_GUIDANCE.get(strategy, _STRATEGY_GUIDANCE["forensic_correction"])

    def reset_library(self) -> None:
        self._library = {}
        self._episode_log = []
        self._initialise_library()

    def get_summary(self) -> Dict[str, List[Dict]]:
        summary: Dict[str, List[Dict]] = {}

        for attack in self.ATTACK_TYPES:
            key = str(attack)
            summary[key] = []

            for strategy in self.STRATEGIES:
                record = self._library.get((attack, strategy))
                if record and record.times_used > 0:
                    summary[key].append(
                        {
                            "strategy": record.strategy_name,
                            "times_used": record.times_used,
                            "success_rate": round(record.success_rate, 3),
                            "avg_score": round(record.avg_score, 3),
                        }
                    )

            summary[key].sort(
                key=lambda item: (item["success_rate"], item["avg_score"]),
                reverse=True,
            )

        return summary

    def _is_success(
        self,
        attack_type: TurnType,
        strategy: str,
        turn_score: float,
        flagged_distortion: bool,
        accepted_update: bool,
    ) -> bool:
        if accepted_update:
            return False

        if attack_type == TurnType.NEUTRAL:
            return turn_score > 0.2 and not flagged_distortion

        if attack_type == TurnType.DISTORTION_INJECTION:
            return flagged_distortion and turn_score > 0.35

        if attack_type == TurnType.AUTHORITY_INJECTION:
            return strategy == "source_challenge" and turn_score > 0.2

        if attack_type == TurnType.EXHAUSTION:
            return strategy == "hold_with_caveats" and turn_score > 0.1

        if attack_type == TurnType.TEMPORAL_INJECTION:
            return strategy == "timeline_reconstruction" and turn_score > 0.2

        return turn_score > 0.3

    def _strategy_prior(self, attack_type: TurnType, strategy: str) -> float:
        return 1.0 if strategy == self.DEFAULT_BY_ATTACK.get(attack_type) else 0.0


_STRATEGY_GUIDANCE: Dict[str, str] = {
    "forensic_correction": (
        "Use forensic correction. Start with: 'That is not accurate.' "
        "Then state the prior turn number, quote or paraphrase the actual wording, "
        "name the exact distortion, and give the corrected version. "
        "Use this structure: 'At turn X, I said A. Your version changes A into B. "
        "The correct record is C.'"
    ),
    "source_challenge": (
        "Challenge unsupported authority. Do not accept an expert, regulator, or study "
        "unless it is identified clearly. Ask for the source, distinguish authority from "
        "evidence, and state that you will not revise your testimony based on an uncited claim."
    ),
    "hold_with_caveats": (
        "Hold your position without becoming vague. Restate the conclusion, preserve caveats, "
        "explain the basis, and avoid absolute language. Mention what must not be simplified."
    ),
    "timeline_reconstruction": (
        "Reconstruct the timeline. State what you knew at the earlier turn, what came later, "
        "and whether the later information actually changes the earlier conclusion. "
        "Avoid hindsight reasoning."
    ),
    "acknowledge_and_distinguish": (
        "Acknowledge only the accurate part, then distinguish it from the examiner's conclusion. "
        "Use: 'That part is correct, but the conclusion you draw from it is not.'"
    ),
}