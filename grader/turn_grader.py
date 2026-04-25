"""
Per-turn reward computation.

Fires after every witness response.
All checks are deterministic and use no LLM calls.
"""

from typing import Any, List, Optional

from models import WitnessAction
from transcript.store import TranscriptStore

from grader.checks import (
    score_consistency,
    score_detection,
    score_first_turn_detection,
    score_precision,
    score_specificity,
)

from constants import (
    TURN_W_CONSISTENCY as _W_CONSISTENCY,
    TURN_W_DETECTION as _W_DETECTION,
    TURN_W_FIRST_TURN as _W_FIRST_TURN,
    TURN_W_PRECISION as _W_PRECISION,
    TURN_W_SPECIFICITY as _W_SPECIFICITY,
)


def score_turn(
    transcript: TranscriptStore,
    action: WitnessAction,
    task: Any,
    prev_action: Optional[WitnessAction] = None,
) -> float:
    """
    Computes one per-turn reward.

    This rewards:
    - detection
    - immediate detection
    - precise correction
    - consistency
    - specificity
    """
    key_claims = _get_key_claims(task)

    detection = score_detection(transcript, action, prev_action)
    first_turn = score_first_turn_detection(transcript, action, prev_action)
    precision = score_precision(transcript, action)
    consistency = score_consistency(transcript, action, key_claims)
    specificity = score_specificity(action, key_claims)

    raw = (
        _W_DETECTION * detection
        + _W_FIRST_TURN * first_turn
        + _W_PRECISION * precision
        + _W_CONSISTENCY * consistency
        + _W_SPECIFICITY * specificity
    )

    return float(raw)


def score_turn_breakdown(
    transcript: TranscriptStore,
    action: WitnessAction,
    task: Any,
    prev_action: Optional[WitnessAction] = None,
) -> dict:
    """Returns component-level scoring for debugging and demo display."""
    key_claims = _get_key_claims(task)

    detection = score_detection(transcript, action, prev_action)
    first_turn = score_first_turn_detection(transcript, action, prev_action)
    precision = score_precision(transcript, action)
    consistency = score_consistency(transcript, action, key_claims)
    specificity = score_specificity(action, key_claims)

    total = (
        _W_DETECTION * detection
        + _W_FIRST_TURN * first_turn
        + _W_PRECISION * precision
        + _W_CONSISTENCY * consistency
        + _W_SPECIFICITY * specificity
    )

    return {
        "total": round(total, 4),
        "detection": round(detection, 4),
        "first_turn": round(first_turn, 4),
        "precision": round(precision, 4),
        "consistency": round(consistency, 4),
        "specificity": round(specificity, 4),
    }


def _get_key_claims(task: Any) -> List[str]:
    """
    Pulls claim anchors from both:
    - distortion templates
    - genuine evidence key_claims

    This makes consistency/specificity scoring stronger than using distortions alone.
    """
    claims: List[str] = []

    try:
        templates = task._dossier.get_distortion_templates()
        for template in templates:
            claim = str(template.get("original_claim", "")).strip()
            if claim:
                claims.append(claim)
    except Exception:
        pass

    try:
        evidence = task._dossier.get_genuine_evidence()
        for doc in evidence:
            for claim in doc.get("key_claims", []) or []:
                claim = str(claim).strip()
                if claim:
                    claims.append(claim)
    except Exception:
        pass

    return _dedupe_preserve_order(claims)


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []

    for item in items:
        key = item.lower()
        if key not in seen:
            result.append(item)
            seen.add(key)

    return result