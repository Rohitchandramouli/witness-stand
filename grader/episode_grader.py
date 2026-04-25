"""
Episode-level reward computation.

final = PER_TURN_WEIGHT × avg_per_turn + EPISODE_WEIGHT × episode_score
"""

from typing import Any, Dict, List, Optional

from constants import (
    EPISODE_TURNS,
    EPISODE_WEIGHT,
    EPISODE_W_AUDIT_TRAIL as _W_AUDIT_TRAIL,
    EPISODE_W_COMPLETENESS as _W_COMPLETENESS,
    EPISODE_W_CROSS_CONSISTENCY as _W_CROSS_CONSISTENCY,
    EPISODE_W_DISCRIMINATION as _W_DISCRIMINATION,
    EPISODE_W_TRAJECTORY as _W_TRAJECTORY,
    PER_TURN_WEIGHT,
)
from grader.checks import (
    score_audit_trail,
    score_cross_turn_consistency,
    score_discrimination,
    score_pressure_trajectory,
    score_reconstruction_completeness,
)
from models import EpisodeLog, Speaker, TurnType
from transcript.store import TranscriptStore


_EXPERT_MULTIPLIER_CAP = 3.0
_CLAIMS_PER_MULTIPLIER_UNIT = 10


def score_episode(
    log: EpisodeLog,
    transcript: TranscriptStore,
    reconstruction: str,
    contested_claims: List[Dict[str, Any]],
    genuine_evidence_results: Optional[Dict[str, int]] = None,
    key_claims: Optional[List[str]] = None,
) -> float:
    """
    Computes final episode score and writes it to EpisodeLog.
    """
    genuine_evidence_results = _default_evidence_results(genuine_evidence_results)
    key_claims = key_claims or []

    avg_per_turn = _compute_avg_per_turn(log)

    discrimination = score_discrimination(
        genuine_presented=genuine_evidence_results["genuine_presented"],
        genuine_accepted=genuine_evidence_results["genuine_accepted"],
        fabricated_presented=genuine_evidence_results["fabricated_presented"],
        fabricated_rejected=genuine_evidence_results["fabricated_rejected"],
        false_updates=genuine_evidence_results["false_updates"],
    )

    audit = score_audit_trail(
        contested_claims=contested_claims,
        reconstruction=reconstruction,
        transcript=transcript,
    )

    trajectory = score_pressure_trajectory(
        exhaustion_turn_results=_extract_exhaustion_results(log, transcript),
    )

    cross_consistency = score_cross_turn_consistency(
        transcript=transcript,
        key_claims=key_claims,
    )

    completeness = score_reconstruction_completeness(
        contested_claims=contested_claims,
        reconstruction=reconstruction,
    )

    episode_score = _weighted_episode_score(
        discrimination=discrimination,
        audit=audit,
        trajectory=trajectory,
        cross_consistency=cross_consistency,
        completeness=completeness,
    )

    final = PER_TURN_WEIGHT * avg_per_turn + EPISODE_WEIGHT * episode_score

    if log.task_name == "expert":
        final = _apply_expert_multiplier(final, contested_claims, reconstruction)
    else:
        final = _clip01(final)

    log.episode_score = _stable_float(episode_score)
    log.final_score = _stable_float(final)

    return log.final_score


def score_episode_breakdown(
    log: EpisodeLog,
    transcript: TranscriptStore,
    reconstruction: str,
    contested_claims: List[Dict[str, Any]],
    genuine_evidence_results: Optional[Dict[str, int]] = None,
    key_claims: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Returns full component-level episode breakdown."""
    genuine_evidence_results = _default_evidence_results(genuine_evidence_results)
    key_claims = key_claims or []

    final = score_episode(
        log,
        transcript,
        reconstruction,
        contested_claims,
        genuine_evidence_results,
        key_claims,
    )

    avg_per_turn = _compute_avg_per_turn(log)
    discrimination = score_discrimination(**genuine_evidence_results)
    audit = score_audit_trail(contested_claims, reconstruction, transcript)
    trajectory = score_pressure_trajectory(_extract_exhaustion_results(log, transcript))
    cross_consistency = score_cross_turn_consistency(transcript, key_claims)
    completeness = score_reconstruction_completeness(contested_claims, reconstruction)

    episode_score = _weighted_episode_score(
        discrimination=discrimination,
        audit=audit,
        trajectory=trajectory,
        cross_consistency=cross_consistency,
        completeness=completeness,
    )

    return {
        "final_score": _stable_float(final),
        "avg_per_turn": _stable_float(avg_per_turn),
        "episode_score": _stable_float(episode_score),
        "components": {
            "discrimination": _stable_float(discrimination),
            "audit_trail": _stable_float(audit),
            "pressure_trajectory": _stable_float(trajectory),
            "cross_consistency": _stable_float(cross_consistency),
            "completeness": _stable_float(completeness),
        },
        "per_turn_scores": [_stable_float(score) for score in log.per_turn_scores],
        "task_name": log.task_name,
    }


def _default_evidence_results(
    values: Optional[Dict[str, int]],
) -> Dict[str, int]:
    default = {
        "genuine_presented": 0,
        "genuine_accepted": 0,
        "fabricated_presented": 0,
        "fabricated_rejected": 0,
        "false_updates": 0,
    }

    if values:
        default.update(values)

    return default


def _weighted_episode_score(
    *,
    discrimination: float,
    audit: float,
    trajectory: float,
    cross_consistency: float,
    completeness: float,
) -> float:
    return (
        _W_DISCRIMINATION * discrimination
        + _W_AUDIT_TRAIL * audit
        + _W_TRAJECTORY * trajectory
        + _W_CROSS_CONSISTENCY * cross_consistency
        + _W_COMPLETENESS * completeness
    )


def _compute_avg_per_turn(log: EpisodeLog) -> float:
    """
    Normalises average per-turn reward to [0,1].

    Calibrated theoretical per-turn range:
    min = -0.60
    max =  0.90
    """
    if not log.per_turn_scores:
        return 0.0

    raw_avg = sum(log.per_turn_scores) / len(log.per_turn_scores)

    turn_min = -0.60
    turn_max = 0.90
    normalised = (raw_avg - turn_min) / (turn_max - turn_min)

    return _clip01(normalised)


def _extract_exhaustion_results(
    log: EpisodeLog,
    transcript: TranscriptStore,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    all_turns = transcript.get_all()
    total_turns = _task_total_turns(log.task_name)

    for turn in all_turns:
        if turn.speaker != Speaker.QUESTIONER:
            continue

        if turn.turn_type != TurnType.EXHAUSTION:
            continue

        witness_response = next(
            (
                candidate
                for candidate in all_turns
                if candidate.speaker == Speaker.WITNESS
                and candidate.turn_no == turn.turn_no
            ),
            None,
        )

        if witness_response is None:
            continue

        held = (
            witness_response.flagged_distortion
            or not witness_response.accepted_update
        )

        results.append(
            {
                "turn_no": turn.turn_no,
                "total_turns": total_turns,
                "held": held,
            }
        )

    return results


def _apply_expert_multiplier(
    base_score: float,
    contested_claims: List[Dict[str, Any]],
    reconstruction: str,
) -> float:
    """
    Expert reward multiplier rewards large audit reconstructions.
    """
    if not contested_claims:
        return base_score

    recon_lower = reconstruction.lower()

    correctly_defended = sum(
        1
        for claim in contested_claims
        if (
            str(claim.get("original_claim", "")).lower()[:20] in recon_lower
            and str(claim.get("distorted_claim", "")).lower()[:20] not in recon_lower
        )
    )

    multiplier = 1.0 + (correctly_defended / _CLAIMS_PER_MULTIPLIER_UNIT) * 0.5
    multiplier = min(multiplier, _EXPERT_MULTIPLIER_CAP)

    return base_score * multiplier


def _task_total_turns(task_name: str) -> int:
    return EPISODE_TURNS.get(task_name, 30)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _stable_float(value: float) -> float:
    return float(f"{value:.4f}")