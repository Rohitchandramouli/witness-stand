"""
environment.py — OpenEnv orchestration loop.
Thin coordinator only. Delegates to: tasks/, questioners/, transcript/, grader/.
If this file grows beyond ~150 lines, logic belongs in a specific module.
"""
from models import WitnessAction, EpisodeLog, Turn, Speaker
from transcript.store import TranscriptStore
from tasks.registry import TASK_REGISTRY
from grader.turn_grader import score_turn
import uuid


class WitnessStandEnv:
    def __init__(self):
        self.transcript: TranscriptStore = None
        self.task = None
        self.episode_log: EpisodeLog = None
        self.turn_number: int = 0
        self.done: bool = False

    def reset(self, task_name: str) -> dict:
        task_class = TASK_REGISTRY[task_name]
        self.task = task_class()
        self.transcript = TranscriptStore()
        self.turn_number = 1
        self.done = False
        self.episode_log = EpisodeLog(
            episode_id=str(uuid.uuid4()),
            task_name=task_name,
            persona=self.task.persona,
        )
        first_question = self.task.panel.get_turn(1, self.transcript, self.task.persona)
        self.transcript.append(Turn(
            turn_no=self.turn_number,
            speaker=Speaker.QUESTIONER,
            text=first_question,
            turn_type=self.task.panel.get_turn_type(1),
        ))
        return self._build_observation()

    def step(self, action: WitnessAction) -> tuple:
        self.transcript.append(Turn(
            turn_no=self.turn_number,
            speaker=Speaker.WITNESS,
            text=action.response_text,
            flagged_distortion=action.flagged_distortion,
            accepted_update=action.accepted_update,
        ))
        turn_score = score_turn(self.transcript, action, self.task)
        self.episode_log.per_turn_scores.append(turn_score)
        self.turn_number += 1
        if self.turn_number > self.task.total_turns:
            self.done = True
        else:
            next_question = self.task.panel.get_turn(self.turn_number, self.transcript, self.task.persona)
            self.transcript.append(Turn(
                turn_no=self.turn_number,
                speaker=Speaker.QUESTIONER,
                text=next_question,
                turn_type=self.task.panel.get_turn_type(self.turn_number),
            ))
        return self._build_observation(), turn_score, self.done, {}

    def _build_observation(self) -> dict:
        lag = self.task.data_lag_turns
        visible_turns = self.transcript.get_witness_turns(exclude_recent=lag)
        return {
            "questioner_text": self.transcript.last_questioner_turn(),
            "prior_statements": visible_turns,
            "turn_number": self.turn_number,
            "persona_system_prompt": self.task.persona.system_prompt,
        }
