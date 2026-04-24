"""
Per-turn score computation. Fires after every witness response. Zero LLM calls.
score = Σ(weight_i × check_i) across 5 independent turn checks.
"""

from typing import Optional, List, TYPE_CHECKING

from transcript.store import TranscriptStore
from models import WitnessAction

from grader.checks import (
    score_detection,
    score_first_turn_detection,
    score_precision,
    score_consistency,
    score_specificity,
)

if TYPE_CHECKING:
    from tasks.base import TaskBase

# ── Per-turn component weights — imported from constants.py (single source of truth)
from constants import (
    TURN_W_DETECTION   as _W_DETECTION,
    TURN_W_FIRST_TURN  as _W_FIRST_TURN,
    TURN_W_PRECISION   as _W_PRECISION,
    TURN_W_CONSISTENCY as _W_CONSISTENCY,
    TURN_W_SPECIFICITY as _W_SPECIFICITY,
)


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
    """Same as score_turn but returns per-component dict. Evaluation/debug only."""
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
    }

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