"""Basic task: direct distortion detection with Reframer only."""

from questioners.neutral import NeutralQuestioner
from questioners.panel import QuestionerPanel
from questioners.reframer import Reframer
from tasks.base import TRAIN_ATTACK_COUNTS, TaskBase


class TaskBasic(TaskBase):
    task_name = "basic"

    DEMO_SCHEDULE = {
        2: "reframer",
        4: "reframer",
    }

    def _build_panel(self) -> QuestionerPanel:
        reframer = Reframer(rng=self.rng)
        neutral = NeutralQuestioner(rng=self.rng)

        reframer.load_distortions(self._difficulty_pool())

        if self.mode == "demo":
            schedule = {
                turn: reframer
                for turn, role in self.DEMO_SCHEDULE.items()
                if role == "reframer" and turn <= self.total_turns
            }
        else:
            attack_turns = self._sample_attack_turns(
                count=TRAIN_ATTACK_COUNTS[self.task_name],
                earliest_turn=3,
            )
            schedule = {turn: reframer for turn in attack_turns}

        return QuestionerPanel(schedule=schedule, default_questioner=neutral)