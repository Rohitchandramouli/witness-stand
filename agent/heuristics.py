"""
Cross-episode pattern tracking for the witness agent.
Builds a response strategy library from prior episode outcomes —
what worked, what didn't, against which questioner types.

This is the witness-side equivalent of the questioner's weight adaptation.
The questioners adapt which attacks to use. The witness adapts which
response strategies to deploy against each attack type.

Heuristics persist across episodes within a training run.
They are NOT reset between episodes — only between full training runs.
"""
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from models import TurnType


@dataclass
class StrategyRecord:
    """
    Tracks the effectiveness of a response strategy against a specific
    attack type across episodes.
    """
    strategy_name:  str
    attack_type:    str
    times_used:     int   = 0
    times_succeeded: int  = 0
    avg_score:      float = 0.0

    @property
    def success_rate(self) -> float:
        if self.times_used == 0:
            return 0.0
        return self.times_succeeded / self.times_used

    def update(self, succeeded: bool, turn_score: float) -> None:
        self.times_used += 1
        if succeeded:
            self.times_succeeded += 1
        # Running average of turn scores for this strategy
        self.avg_score = (
            (self.avg_score * (self.times_used - 1) + turn_score)
            / self.times_used
        )


class WitnessHeuristics:
    """
    Maintains a cross-episode strategy library for the witness agent.

    The library maps (attack_type, strategy_name) pairs to StrategyRecords.
    After each episode, the episode outcomes are used to update the library.
    Before each turn, the library is queried to suggest the best strategy
    for the current attack type.

    Attack types correspond to questioner TurnTypes:
      DISTORTION_INJECTION  — Reframer attacks
      AUTHORITY_INJECTION   — AuthorityInvoker attacks
      EXHAUSTION            — ExhaustionTactic attacks
      TEMPORAL_INJECTION    — TemporalQuestioner attacks
      NEUTRAL               — non-attack turns

    Response strategies:
      "cite_and_correct"    — flag, cite exact turn number, quote prior language
      "hold_and_elaborate"  — maintain position, expand reasoning without citing
      "flag_and_question"   — flag the distortion, question the questioner's source
      "acknowledge_and_distinguish" — acknowledge partial validity, distinguish from distortion
      "reconstruct_timeline" — for temporal attacks, explicitly reconstruct the timeline
    """

    STRATEGIES = [
        "cite_and_correct",
        "hold_and_elaborate",
        "flag_and_question",
        "acknowledge_and_distinguish",
        "reconstruct_timeline",
    ]

    ATTACK_TYPES = [
        TurnType.DISTORTION_INJECTION,
        TurnType.AUTHORITY_INJECTION,
        TurnType.EXHAUSTION,
        TurnType.TEMPORAL_INJECTION,
        TurnType.NEUTRAL,
    ]

    def __init__(self):
        # (attack_type, strategy) -> StrategyRecord
        self._library: Dict[tuple, StrategyRecord] = {}
        self._initialise_library()

        # Per-episode tracking — cleared at episode start
        self._episode_log: List[Dict] = []

    def _initialise_library(self) -> None:
        """Pre-populates the library with all (attack, strategy) combinations."""
        for attack in self.ATTACK_TYPES:
            for strategy in self.STRATEGIES:
                key = (attack, strategy)
                self._library[key] = StrategyRecord(
                    strategy_name=strategy,
                    attack_type=str(attack),
                )

    # ── Per-turn interface ────────────────────────────────────────────

    def suggest_strategy(self, attack_type: TurnType) -> str:
        """
        Returns the strategy with the highest success rate for this attack type.
        Falls back to "cite_and_correct" if no data exists yet.

        Called by agent/prompt.py before constructing each turn's prompt
        so the prompt can include strategy guidance.
        """
        candidates = [
            self._library[(attack_type, s)]
            for s in self.STRATEGIES
            if (attack_type, s) in self._library
        ]

        if not candidates or all(r.times_used == 0 for r in candidates):
            # No data yet — default to the most precise strategy
            return "cite_and_correct"

        # Weight by success rate, break ties by avg_score
        best = max(
            candidates,
            key=lambda r: (r.success_rate, r.avg_score),
        )
        return best.strategy_name

    def log_turn(
        self,
        attack_type: TurnType,
        strategy_used: str,
        turn_score: float,
        flagged_distortion: bool,
        accepted_update: bool,
    ) -> None:
        """
        Logs the outcome of a single turn for end-of-episode processing.
        Called by environment.py after each step.
        """
        self._episode_log.append({
            "attack_type":        attack_type,
            "strategy_used":      strategy_used,
            "turn_score":         turn_score,
            "flagged_distortion": flagged_distortion,
            "accepted_update":    accepted_update,
        })

    def end_episode(self) -> None:
        """
        Processes the episode log and updates the strategy library.
        Called by environment.py at episode end.
        A turn is considered a success if:
          - The witness flagged a distortion without accepting an update, OR
          - The turn score is above 0.3 (positive contribution)
        """
        for entry in self._episode_log:
            attack_type  = entry["attack_type"]
            strategy     = entry["strategy_used"]
            turn_score   = entry["turn_score"]
            flagged      = entry["flagged_distortion"]
            accepted     = entry["accepted_update"]

            key = (attack_type, strategy)
            if key not in self._library:
                continue

            succeeded = (
                (flagged and not accepted)
                or turn_score > 0.3
            )
            self._library[key].update(succeeded, turn_score)

        # Clear per-episode log
        self._episode_log = []

    # ── Strategy guidance text ────────────────────────────────────────

    def get_strategy_guidance(self, attack_type: TurnType) -> str:
        """
        Returns a natural language guidance string for the suggested strategy.
        This is embedded directly into the witness prompt each turn.
        """
        strategy = self.suggest_strategy(attack_type)
        return _STRATEGY_GUIDANCE.get(strategy, "")

    # ── Introspection ─────────────────────────────────────────────────

    def get_summary(self) -> Dict[str, List[Dict]]:
        """
        Returns a summary of strategy effectiveness per attack type.
        Used by scripts/run_eval.py for post-training analysis.
        """
        summary: Dict[str, List[Dict]] = {}
        for attack in self.ATTACK_TYPES:
            attack_key = str(attack)
            summary[attack_key] = []
            for strategy in self.STRATEGIES:
                record = self._library.get((attack, strategy))
                if record and record.times_used > 0:
                    summary[attack_key].append({
                        "strategy":      record.strategy_name,
                        "times_used":    record.times_used,
                        "success_rate":  round(record.success_rate, 3),
                        "avg_score":     round(record.avg_score, 3),
                    })
            # Sort by success rate descending
            summary[attack_key].sort(
                key=lambda x: x["success_rate"], reverse=True
            )
        return summary

    def reset_library(self) -> None:
        """
        Full reset of the strategy library.
        Only called between full training runs, not between episodes.
        """
        self._library = {}
        self._initialise_library()
        self._episode_log = []


# ── Strategy guidance text ────────────────────────────────────────────
# Embedded into the witness prompt to guide response construction.

_STRATEGY_GUIDANCE: Dict[str, str] = {
    "cite_and_correct": (
        "Flag the mischaracterisation explicitly. "
        "State the turn number where you made the original statement. "
        "Quote your exact prior language. "
        "Cite a document ID if relevant. "
        "Example: 'That is not accurate. At turn 4 I stated [exact quote]. "
        "The record will confirm this.'"
    ),
    "hold_and_elaborate": (
        "Maintain your position without citing specific prior turns. "
        "Elaborate the reasoning behind your assessment. "
        "Do not capitulate — expand your justification instead. "
        "Example: 'My assessment stands. The basis for this conclusion is...'"
    ),
    "flag_and_question": (
        "Flag the attack and turn the question back on the questioner. "
        "Ask them to cite the specific source for their claim. "
        "Example: 'That characterisation is not mine. "
        "Which specific document are you drawing that from?'"
    ),
    "acknowledge_and_distinguish": (
        "Acknowledge what is partially accurate in the questioner's framing. "
        "Then draw a precise distinction between that and what you actually said. "
        "Example: 'You are correct that I assessed moderate risk — "
        "but moderate is not the same as low, and that distinction was intentional.'"
    ),
    "reconstruct_timeline": (
        "Address the chronology directly. "
        "State what information you had at the time of each statement. "
        "Make clear that your reasoning was sound given what was available then. "
        "Example: 'At turn 6, I had access to X and Y but not Z. "
        "My conclusion followed from that information state, not from hindsight.'"
    ),
}