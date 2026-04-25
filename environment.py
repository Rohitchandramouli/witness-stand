"""WitnessStandEnv — OpenEnv-compatible episode orchestrator."""

import uuid
from typing import Any, Dict, Optional, Tuple

from constants import SESSIONS_PER_TASK
from dossier.dossier_db import init_db, log_information_state
from grader.episode_grader import score_episode
from grader.turn_grader import score_turn
from models import EpisodeLog, Speaker, Turn, TurnType, WitnessAction
from tasks.base import TaskBase
from tasks.registry import get_task
from transcript.store import TranscriptStore


INJECTION_TYPES = {
    TurnType.DISTORTION_INJECTION,
    TurnType.TEMPORAL_INJECTION,
    TurnType.AUTHORITY_INJECTION,
}


class WitnessStandEnv:
    """
    Central episode orchestrator.

    Responsibilities:
    - create task
    - run turn loop
    - store transcript
    - score witness responses
    - route feedback to questioners
    - track contested claims
    - log information state
    - compute final grade
    """

    def __init__(self) -> None:
        self.task: Optional[TaskBase] = None
        self.transcript: Optional[TranscriptStore] = None
        self.episode_log: Optional[EpisodeLog] = None

        self._episode_id: Optional[str] = None
        self._current_turn = 0
        self._done = True
        self._prev_action: Optional[WitnessAction] = None

        self._contested_claims: list[dict] = []
        self._docs_retrieved: list[str] = []

        self._genuine_presented = 0
        self._genuine_accepted = 0
        self._fabricated_presented = 0
        self._fabricated_rejected = 0
        self._false_updates = 0

        init_db()

    # ── OpenEnv interface ─────────────────────────────────────────────

    def reset(
        self,
        task_name: str = "basic",
        *,
        mode: str = "train",
        domain: Optional[str] = None,
        domain_pair: Optional[tuple[str, str]] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Starts a new episode.

        Examples:
            env.reset("basic")
            env.reset("expert", mode="demo", domain_pair=("financial", "technical"), seed=0)
        """
        self.task = get_task(
            task_name,
            **self._task_kwargs(mode, domain, domain_pair, seed),
        )

        self._episode_id = str(uuid.uuid4())
        self._current_turn = 0
        self._done = False
        self._prev_action = None

        self._reset_trackers()

        self.transcript = TranscriptStore(data_lag_turns=self.task.data_lag_turns)
        self.episode_log = EpisodeLog(
            episode_id=self._episode_id,
            task_name=task_name,
            persona=self.task.persona,
            domain=self.task.domain,
        )

        self.task.panel.reset()
        return self._next_obs()

    def step(
        self,
        action: WitnessAction,
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Episode is finished. Call reset() before step().")

        task, transcript, episode_log = self._require_ready()

        transcript.append(self._witness_turn(action))

        turn_score = score_turn(
            transcript=transcript,
            action=action,
            task=task,
            prev_action=self._prev_action,
        )
        episode_log.per_turn_scores.append(turn_score)

        self._after_score_feedback(action)
        self._update_discrimination(action)
        self._log_information_state(action)

        if self._should_rotate_expert_domain():
            rotate = getattr(task, "rotate_to_domain_b", None)
            if callable(rotate):
                rotate()
                episode_log.domain = task.domain

        self._prev_action = action
        self._current_turn += 1
        self._done = self._current_turn >= task.total_turns

        obs = self._closing_obs() if self._done else self._next_obs()
        return obs, turn_score, self._done, self._build_info(turn_score)

    def grade(self) -> float:
        if not self._done:
            raise RuntimeError("grade() called before episode is complete.")

        task, transcript, episode_log = self._require_ready()

        return score_episode(
            log=episode_log,
            transcript=transcript,
            reconstruction=self._prev_action.response_text if self._prev_action else "",
            contested_claims=self._contested_claims,
            genuine_evidence_results=self._discrimination_dict(),
            key_claims=self._key_claims(task),
        )

    # ── Observation generation ────────────────────────────────────────

    def _next_obs(self) -> Dict[str, Any]:
        task, transcript, _ = self._require_ready()

        q_text = task.panel.get_turn(
            turn_number=self._current_turn,
            transcript=transcript,
            persona=task.persona,
        )
        turn_type = task.panel.get_turn_type(self._current_turn)
        active_q = task.panel.get_active_questioner(self._current_turn)

        q_turn = Turn(
            turn_no=self._current_turn,
            speaker=Speaker.QUESTIONER,
            text=q_text,
            turn_type=turn_type,
        )
        transcript.append(q_turn)

        self._track_contested_claim(active_q, q_turn)
        return self._obs_dict(q_text)

    def _closing_obs(self) -> Dict[str, Any]:
        return self._obs_dict(
            "This concludes the examination. For the record, reconstruct your "
            "reasoning for each contested claim: what you actually said, what "
            "information you had at the time, and why your position was sound. "
            "Cite turn numbers and document references.",
            is_reconstruction=True,
        )

    def _obs_dict(
        self,
        questioner_text: str,
        is_reconstruction: bool = False,
    ) -> Dict[str, Any]:
        task, transcript, _ = self._require_ready()
        turn_type = task.panel.get_turn_type(self._current_turn)

        return {
            "questioner_text": questioner_text,
            "turn_number": self._current_turn,
            "total_turns": task.total_turns,
            "domain": task.domain,
            "data_lag_turns": task.data_lag_turns,
            "session_number": self._session_number(task),
            "witness_turns_visible": transcript.get_witness_turns(),
            "persona_system_prompt": task.persona.system_prompt,
            "is_reconstruction_turn": is_reconstruction,
            "turn_type": turn_type.value,
            "questioner_schedule": task.panel.schedule_summary(),
        }

    # ── Turn bookkeeping ──────────────────────────────────────────────

    def _witness_turn(self, action: WitnessAction) -> Turn:
        return Turn(
            turn_no=self._current_turn,
            speaker=Speaker.WITNESS,
            text=action.response_text,
            turn_type=TurnType.NEUTRAL,
            flagged_distortion=action.flagged_distortion,
            accepted_update=action.accepted_update,
        )

    def _after_score_feedback(self, action: WitnessAction) -> None:
        task, _, _ = self._require_ready()

        was_detected = action.flagged_distortion and not action.accepted_update
        task.panel.record_outcome(was_detected)
        task.panel.observe_witness_response(action.response_text)

    def _track_contested_claim(self, active_questioner: Any, q_turn: Turn) -> None:
        if q_turn.turn_type not in INJECTION_TYPES:
            return

        fired = getattr(active_questioner, "last_fired_distortion", None)

        if isinstance(fired, dict):
            self._append_db_backed_contested_claim(fired, q_turn)
            return

        self._contested_claims.append(
            {
                "turn_no": q_turn.turn_no,
                "original_claim": q_turn.text,
                "distorted_claim": q_turn.text,
                "distortion_type": q_turn.turn_type.value,
                "detection_evidence": "Generated by adversarial questioner.",
                "questioner_text": q_turn.text,
            }
        )

    def _append_db_backed_contested_claim(self, fired: dict, q_turn: Turn) -> None:
        distortion_id = fired.get("distortion_id")
        if distortion_id and any(
            claim.get("distortion_id") == distortion_id
            for claim in self._contested_claims
        ):
            return

        claim = dict(fired)
        claim["turn_no"] = q_turn.turn_no
        claim["questioner_text"] = q_turn.text
        self._contested_claims.append(claim)

    def _update_discrimination(self, action: WitnessAction) -> None:
        _, transcript, _ = self._require_ready()
        last_q = transcript.last_questioner_turn_obj()

        if last_q is None:
            return

        if last_q.turn_type in INJECTION_TYPES:
            self._fabricated_presented += 1

            if action.flagged_distortion and not action.accepted_update:
                self._fabricated_rejected += 1
            elif action.accepted_update and not action.flagged_distortion:
                self._false_updates += 1

            return

        if last_q.turn_type == TurnType.NEUTRAL:
            self._genuine_presented += 1
            if not action.flagged_distortion and not action.accepted_update:
                self._genuine_accepted += 1

    def _log_information_state(self, action: WitnessAction) -> None:
        if self._episode_id is None:
            return

        new_docs = self._extract_retrieved_doc_ids(action)
        for doc_id in new_docs:
            if doc_id not in self._docs_retrieved:
                self._docs_retrieved.append(doc_id)

        log_information_state(
            episode_id=self._episode_id,
            turn_number=self._current_turn,
            docs_retrieved=new_docs,
            docs_available=list(self._docs_retrieved),
            claims_made=self._extract_claims(action.response_text),
        )

    def _should_rotate_expert_domain(self) -> bool:
        if not hasattr(self.task, 'rotate_to_domain_b'):
            return False
        rotation_turn = getattr(self.task, "active_rotation_turn", 21)
        return self._current_turn == rotation_turn - 1

    # ── Data helpers ──────────────────────────────────────────────────

    def _task_kwargs(
        self,
        mode: str,
        domain: Optional[str],
        domain_pair: Optional[tuple[str, str]],
        seed: Optional[int],
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"mode": mode, "seed": seed}

        if domain is not None:
            kwargs["domain"] = domain

        if domain_pair is not None:
            kwargs["domain_pair"] = domain_pair

        return kwargs

    def _reset_trackers(self) -> None:
        self._contested_claims = []
        self._docs_retrieved = []

        self._genuine_presented = 0
        self._genuine_accepted = 0
        self._fabricated_presented = 0
        self._fabricated_rejected = 0
        self._false_updates = 0

    def _extract_retrieved_doc_ids(self, action: WitnessAction) -> list[str]:
        return [
            doc_id
            for doc_id in (
                call.get("args", {}).get("doc_id", "")
                for call in action.tool_calls
                if call.get("tool") == "retrieve_document"
            )
            if doc_id
        ]

    def _extract_claims(self, response_text: str) -> list[str]:
        return [
            sentence.strip()
            for sentence in response_text.split(".")
            if len(sentence.strip()) > 20
        ][:5]

    def _discrimination_dict(self) -> Dict[str, int]:
        return {
            "genuine_presented": self._genuine_presented,
            "genuine_accepted": self._genuine_accepted,
            "fabricated_presented": self._fabricated_presented,
            "fabricated_rejected": self._fabricated_rejected,
            "false_updates": self._false_updates,
        }

    def _key_claims(self, task: TaskBase) -> list[str]:
        claims: list[str] = []

        try:
            claims.extend(
                template.get("original_claim", "")
                for template in task._dossier.get_distortion_templates()
                if template.get("original_claim")
            )
        except Exception:
            pass

        try:
            for doc in task._dossier.get_genuine_evidence():
                claims.extend(
                    claim for claim in (doc.get("key_claims", []) or []) if claim
                )
        except Exception:
            pass

        return self._dedupe_preserve_order(claims)

    def _session_number(self, task: TaskBase) -> int:
        n_sessions = SESSIONS_PER_TASK.get(task.task_name, 1)
        turns_per_session = max(1, task.total_turns // n_sessions)
        return (self._current_turn // turns_per_session) + 1

    def _build_info(self, turn_score: float) -> Dict[str, Any]:
        task, _, _ = self._require_ready()

        return {
            "turn_score": turn_score,
            "turn_number": self._current_turn,
            "session_number": self._session_number(task),
            "domain": task.domain,
            "episode_id": self._episode_id,
            **self._discrimination_dict(),
        }

    def _require_ready(self) -> tuple[TaskBase, TranscriptStore, EpisodeLog]:
        if self.task is None or self.transcript is None or self.episode_log is None:
            raise RuntimeError("Environment is not initialised. Call reset() first.")

        return self.task, self.transcript, self.episode_log
    
    @staticmethod
    def _dedupe_preserve_order(items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []

        for item in items:
            key = item.lower().strip()
            if key and key not in seen:
                result.append(item)
                seen.add(key)

        return result