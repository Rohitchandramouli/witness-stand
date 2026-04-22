"""
Per-turn score computation.
Aggregates the five per-turn check scores into a single reward signal.
Fires after every witness response. Zero LLM calls.

Final per-turn score formula:
    score = (
        W_detection     × score_detection()
        W_first_turn    × score_first_turn_detection()
        W_precision     × score_precision()
        W_consistency   × score_consistency()
        W_specificity   × score_specificity()
    )
"""
from typing import Optional, List, TYPE_CHECKING

from transcript.store import TranscriptStore
from models import WitnessAction
from constants import (
    CONSISTENCY_WEIGHT,
    DETECTION_WEIGHT,
    MISSED_DISTORTION_PENALTY,
    PROACTIVE_CORRECTION_BONUS,
)
from grader.checks import (
    score_detection,
    score_first_turn_detection,
    score_precision,
    score_consistency,
    score_specificity,
)

if TYPE_CHECKING:
    from tasks.base import TaskBase

# ── Per-turn component weights ─────────────────────────────────────────
# Must sum to 1.0
_W_DETECTION    = 0.30
_W_FIRST_TURN   = 0.20
_W_PRECISION    = 0.20
_W_CONSISTENCY  = 0.20
_W_SPECIFICITY  = 0.10


def score_turn(
    transcript: TranscriptStore,
    action: WitnessAction,
    task: "TaskBase",
    prev_action: Optional[WitnessAction] = None,
) -> float:
    """
    Computes the per-turn reward for a single witness response.

    Parameters
    ----------
    transcript   : full transcript up to and including this turn
    action       : the witness's structured response for this turn
    task         : the current task — used to access key_claims from the dossier
    prev_action  : the witness's previous action — needed to detect
                   one-turn-late self-corrections

    Returns
    -------
    float in approximately [-1.5, 1.5], not hard-clipped per turn
    so the training signal preserves gradient magnitude.
    The episode grader normalises the average into [0.0, 1.0].
    """
    key_claims = _get_key_claims(task)

    detection    = score_detection(transcript, action, prev_action)
    first_turn   = score_first_turn_detection(transcript, action, prev_action)
    precision    = score_precision(transcript, action)
    consistency  = score_consistency(transcript, action, key_claims)
    specificity  = score_specificity(action, key_claims)

    raw = (
        _W_DETECTION   * detection
        + _W_FIRST_TURN  * first_turn
        + _W_PRECISION   * precision
        + _W_CONSISTENCY * consistency
        + _W_SPECIFICITY * specificity
    )

    return raw


def score_turn_breakdown(
    transcript: TranscriptStore,
    action: WitnessAction,
    task: "TaskBase",
    prev_action: Optional[WitnessAction] = None,
) -> dict:
    """
    Same as score_turn but returns a full breakdown dict for logging.
    Used by scripts/run_eval.py and scripts/demo.py to show per-component scores.
    Not called during training — only for evaluation and debugging.
    """
    key_claims = _get_key_claims(task)

    detection   = score_detection(transcript, action, prev_action)
    first_turn  = score_first_turn_detection(transcript, action, prev_action)
    precision   = score_precision(transcript, action)
    consistency = score_consistency(transcript, action, key_claims)
    specificity = score_specificity(action, key_claims)

    total = (
        _W_DETECTION   * detection
        + _W_FIRST_TURN  * first_turn
        + _W_PRECISION   * precision
        + _W_CONSISTENCY * consistency
        + _W_SPECIFICITY * specificity
    )

    return {
        "total":       round(total, 4),
        "detection":   round(detection, 4),
        "first_turn":  round(first_turn, 4),
        "precision":   round(precision, 4),
        "consistency": round(consistency, 4),
        "specificity": round(specificity, 4),
        "weighted": {
            "detection":   round(_W_DETECTION   * detection, 4),
            "first_turn":  round(_W_FIRST_TURN  * first_turn, 4),
            "precision":   round(_W_PRECISION   * precision, 4),
            "consistency": round(_W_CONSISTENCY * consistency, 4),
            "specificity": round(_W_SPECIFICITY * specificity, 4),
        }
    }


# ── Private helpers ────────────────────────────────────────────────────


def _get_key_claims(task: "TaskBase") -> List[str]:
    """
    Extracts key claim strings from the task's active dossier.
    These are the specific phrases the consistency and specificity
    checks track for drift and verifiability.
    Returns empty list if the dossier has no key claims yet
    (before build_dossier.py has run).
    """
    try:
        templates = task._dossier.get_distortion_templates()
        return [
            t.get("original_claim", "")
            for t in templates
            if t.get("original_claim")
        ]
    except Exception:
        return []