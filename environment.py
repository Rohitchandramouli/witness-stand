"""
WitnessStandEnv — OpenEnv-compliant orchestrator.
Thin coordinator: loads tasks, runs the turn loop, wires the grader,
and fires questioner adaptation callbacks after each turn.

OpenEnv interface:
    env = WitnessStandEnv()
    obs  = env.reset(task_name)
    obs, reward, done, info = env.step(action)
    score = env.grade()
"""
import uuid
from typing import Dict, Any, Optional, Tuple

from tasks.base import TaskBase
from tasks.registry import get_task
from tasks.task_expert import TaskExpert
from transcript.store import TranscriptStore
from dossier.dossier_db import init_db, log_information_state
from questioners.exhaustion import ExhaustionTactic
from questioners.temporal import TemporalQuestioner
from grader.turn_grader import score_turn
from grader.episode_grader import score_episode
from models import (
    EpisodeLog,
    WitnessAction,
    Turn,
    Speaker,
    TurnType,
)
from constants import SESSIONS_PER_TASK


class WitnessStandEnv:

    def __init__(self):
        self.task: "TaskBase"            = None  # type: ignore[assignment]
        self.transcript: TranscriptStore = None  # type: ignore[assignment]
        self.episode_log: EpisodeLog     = None  # type: ignore[assignment]
        self._episode_id: str            = None  # type: ignore[assignment]
        self._current_turn: int                    = 0
        self._done: bool                           = True
        self._prev_action: Optional[WitnessAction] = None

        # Episode-level discrimination counters
        self._fabricated_presented: int = 0
        self._fabricated_rejected:  int = 0
        self._false_updates:        int = 0
        self._genuine_presented:    int = 0
        self._genuine_accepted:     int = 0

        # Contested claims accumulated from fired distortions
        self._contested_claims: list = []

        # Cumulative set of doc IDs retrieved this episode
        self._docs_retrieved: list = []

        init_db()

    # ── OpenEnv interface ─────────────────────────────────────────────

    def reset(self, task_name: str = "basic") -> Dict[str, Any]:
        self.task          = get_task(task_name)
        self._episode_id   = str(uuid.uuid4())
        self._current_turn = 0
        self._done         = False
        self._prev_action  = None

        self._fabricated_presented = 0
        self._fabricated_rejected  = 0
        self._false_updates        = 0
        self._genuine_presented    = 0
        self._genuine_accepted     = 0
        self._contested_claims     = []
        self._docs_retrieved       = []

        self.transcript = TranscriptStore(
            data_lag_turns=self.task.data_lag_turns
        )
        self.episode_log = EpisodeLog(
        episode_id=self._episode_id,
        task_name=task_name,
        persona=self.task.persona,
        domain=self.task.domain,
        )

        return self._next_obs()

    def step(
        self, action: WitnessAction
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError(
                "Episode is finished. Call reset() before step()."
            )

        # 1 — Record witness turn
        self.transcript.append(Turn(
            turn_no=self._current_turn,
            speaker=Speaker.WITNESS,
            text=action.response_text,
            turn_type=TurnType.NEUTRAL,
            flagged_distortion=action.flagged_distortion,
            accepted_update=action.accepted_update,
        ))

        # 2 — Score the turn
        turn_score = score_turn(
            transcript=self.transcript,
            action=action,
            task=self.task,
            prev_action=self._prev_action,
        )
        self.episode_log.per_turn_scores.append(turn_score)

        # 3 — Questioner adaptation
        # Route post-turn feedback via panel (panel tracks _last_fired)
        was_detected = action.flagged_distortion and not action.accepted_update
        self.task.panel.record_outcome(was_detected)
        self.task.panel.observe_witness_response(action.response_text)

        # 4 — Update discrimination counters
        self._update_discrimination(action)

        # 5 — Log information state for audit trail
        new_docs = [
            c.get("args", {}).get("doc_id", "")
            for c in action.tool_calls
            if c.get("tool") == "retrieve_document"
        ]
        self._docs_retrieved.extend(d for d in new_docs if d)

        log_information_state(
            episode_id=self._episode_id,
            turn_number=self._current_turn,
            docs_retrieved=new_docs,
            docs_available=list(set(self._docs_retrieved)),
            claims_made=[
                s.strip() for s in action.response_text.split(".")
                if len(s.strip()) > 20
            ][:5],
        )

        # 6 — Expert domain rotation at session 2→3 boundary
        if isinstance(self.task, TaskExpert) and self._current_turn == 20:
            self.task.rotate_to_domain_b()

        # 7 — Advance
        self._prev_action  = action
        self._current_turn += 1
        self._done         = self._current_turn >= self.task.total_turns

        obs  = self._closing_obs() if self._done else self._next_obs()
        info = self._build_info(turn_score)
        return obs, turn_score, self._done, info

    def grade(self) -> float:
        if not self._done:
            raise RuntimeError("grade() called before episode is complete.")
        return score_episode(
            log=self.episode_log,
            transcript=self.transcript,
            reconstruction=(
                self._prev_action.response_text if self._prev_action else ""
            ),
            contested_claims=self._contested_claims,
            genuine_evidence_results=self._discrimination_dict(),
            key_claims=self._key_claims(),
        )

    # ── Observation builders ──────────────────────────────────────────

    def _next_obs(self) -> Dict[str, Any]:
        """Generates next questioner turn and returns observation."""
        q_text   = self.task.panel.get_turn(
            turn_number=self._current_turn,
            transcript=self.transcript,
            persona=self.task.persona,
        )
        turn_type = self.task.panel.get_turn_type(self._current_turn)
        active_q  = self.task.panel._schedule.get(self._current_turn, self.task.panel._default)
        q_turn = Turn(
            turn_no=self._current_turn,
            speaker=Speaker.QUESTIONER,
            text=q_text,
            turn_type=turn_type,
        )
        self.transcript.append(q_turn)

        # Track contested claims from injection turns
        if q_turn.turn_type in (
            TurnType.DISTORTION_INJECTION,
            TurnType.TEMPORAL_INJECTION,
            TurnType.AUTHORITY_INJECTION,
        ):
            fired = getattr(active_q, "last_fired_distortion", None)
            if fired and fired not in self._contested_claims:
                self._contested_claims.append(fired)

        return self._obs_dict(q_text)

    def _closing_obs(self) -> Dict[str, Any]:
        """Final reconstruction request — audit trail mechanic."""
        return self._obs_dict(
            "This concludes the examination. "
            "For the record, reconstruct your reasoning for each contested "
            "claim: what you actually said, what information you had at the "
            "time, and why your position was sound. "
            "Cite turn numbers and document references.",
            is_reconstruction=True,
        )

    def _obs_dict(
        self,
        questioner_text: str,
        is_reconstruction: bool = False,
    ) -> Dict[str, Any]:
        return {
            "questioner_text":       questioner_text,
            "turn_number":           self._current_turn,
            "total_turns":           self.task.total_turns,
            "domain":                self.task.domain,
            "data_lag_turns":        self.task.data_lag_turns,
            "session_number":        self._session_number(),
            "witness_turns_visible": self.transcript.get_witness_turns(),
            "persona_system_prompt": self.task.persona.system_prompt,
            "is_reconstruction_turn": is_reconstruction,
        }

    # ── Private helpers ───────────────────────────────────────────────

    def _update_discrimination(self, action: WitnessAction) -> None:
        last_q = next(
            (t for t in reversed(self.transcript.get_all())
             if t.speaker == Speaker.QUESTIONER),
            None,
        )
        if last_q is None:
            return
        if last_q.turn_type == TurnType.AUTHORITY_INJECTION:
            self._fabricated_presented += 1
            if action.flagged_distortion and not action.accepted_update:
                self._fabricated_rejected += 1
            elif action.accepted_update and not action.flagged_distortion:
                self._false_updates += 1

    def _discrimination_dict(self) -> Dict[str, int]:
        return {
            "genuine_presented":    self._genuine_presented,
            "genuine_accepted":     self._genuine_accepted,
            "fabricated_presented": self._fabricated_presented,
            "fabricated_rejected":  self._fabricated_rejected,
            "false_updates":        self._false_updates,
        }

    def _key_claims(self) -> list:
        try:
            return [
                t.get("original_claim", "")
                for t in self.task._dossier.get_distortion_templates()
                if t.get("original_claim")
            ]
        except Exception:
            return []

    def _session_number(self) -> int:
        n_sessions = SESSIONS_PER_TASK.get(self.task.task_name, 1)
        turns_per_session = max(1, self.task.total_turns // n_sessions)
        return (self._current_turn // turns_per_session) + 1

    def _build_info(self, turn_score: float) -> Dict[str, Any]:
        return {
            "turn_score":     turn_score,
            "turn_number":    self._current_turn,
            "session_number": self._session_number(),
            "domain":         self.task.domain,
            "episode_id":     self._episode_id,
        }