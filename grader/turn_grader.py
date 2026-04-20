"""
Per-turn score computation.
final_per_turn = 0.5 * consistency + 0.5 * detection
Fires after every witness response. No LLM calls.
"""
from transcript.store import TranscriptStore
from models import WitnessAction, TurnType
from grader.checks import check_statement_consistency, check_distortion_detected
from constants import (
    CONSISTENCY_WEIGHT, DETECTION_WEIGHT,
    MISSED_DISTORTION_PENALTY, PROACTIVE_CORRECTION_BONUS,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tasks.base import TaskBase


def score_turn(transcript: TranscriptStore, action: WitnessAction, task: "TaskBase") -> float:
    """Returns per-turn reward in [0.0, 1.0] (may go slightly negative due to penalties)."""
    consistency = float(check_statement_consistency(transcript, action))
    detection_raw = check_distortion_detected(transcript, action, task)

    from models import Speaker
    last_q = None
    for t in reversed(transcript.get_all()):
        if t.speaker == Speaker.QUESTIONER:
            last_q = t
            break

    is_distortion_turn = last_q and last_q.turn_type in (
        TurnType.DISTORTION_INJECTION, TurnType.TEMPORAL_INJECTION
    )

    detection_score = 0.0
    if is_distortion_turn:
        if detection_raw:
            detection_score = 1.0
            # Bonus: agent flagged distortion AND proactively cited correct record
            if "record" in action.response_text.lower() or "stated" in action.response_text.lower():
                detection_score += PROACTIVE_CORRECTION_BONUS
        else:
            detection_score = MISSED_DISTORTION_PENALTY
    # On non-distortion turns, detection component is 0 by design

    score = CONSISTENCY_WEIGHT * consistency + DETECTION_WEIGHT * detection_score
    return max(score, -1.0)  # floor at -1.0
