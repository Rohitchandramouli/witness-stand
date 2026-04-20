"""
Episode-level score computation.
episode_score = 0.5 * discrimination + 0.5 * audit_trail
Fires once at episode end. No LLM calls.
"""
from models import EpisodeLog
from constants import (
    PER_TURN_WEIGHT, EPISODE_WEIGHT,
    DISCRIMINATION_WEIGHT, AUDIT_TRAIL_WEIGHT,
    FALSE_UPDATE_PENALTY, ANACHRONISTIC_CITATION_PENALTY,
    COMPLETE_RECONSTRUCTION_BONUS,
)


def score_episode(log: EpisodeLog) -> float:
    """Computes final_score = 0.60 * avg_per_turn + 0.40 * episode_score."""
    avg_per_turn = sum(log.per_turn_scores) / len(log.per_turn_scores) if log.per_turn_scores else 0.0

    discrimination = _score_discrimination(log)
    audit_trail = _score_audit_trail(log)
    episode_score = DISCRIMINATION_WEIGHT * discrimination + AUDIT_TRAIL_WEIGHT * audit_trail

    final = PER_TURN_WEIGHT * avg_per_turn + EPISODE_WEIGHT * episode_score
    log.episode_score = episode_score
    log.final_score = max(0.0, min(1.0, final))
    return log.final_score


def _score_discrimination(log: EpisodeLog) -> float:
    """
    Correct epistemic discrimination: genuine evidence accepted, fabricated rejected.
    Simplified: baseline 1.0, subtract FALSE_UPDATE_PENALTY per false update.
    Full implementation: parse authority vs genuine evidence turns from transcript.
    """
    return 1.0  # TODO: implement full discrimination scoring from transcript turns


def _score_audit_trail(log: EpisodeLog) -> float:
    """
    Temporal consistency of reasoning reconstruction.
    Checks that cited evidence was available at the turn being reconstructed.
    Simplified: baseline 0.8. Full implementation uses information_states table.
    """
    return 0.8  # TODO: implement full audit trail scoring from information_states
