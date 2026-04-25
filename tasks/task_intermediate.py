"""Intermediate task: Reframer + false authority pressure."""

from questioners.authority import AuthorityInvoker
from questioners.neutral import NeutralQuestioner
from questioners.panel import QuestionerPanel
from questioners.reframer import Reframer
from tasks.base import TaskBase


class TaskIntermediate(TaskBase):
    task_name = "intermediate"

    DEMO_SCHEDULE = {
        2: "reframer",
        4: "authority",
        6: "reframer",
    }

    def _build_panel(self) -> QuestionerPanel:
        reframer = Reframer(rng=self.rng)
        neutral = NeutralQuestioner(rng=self.rng)

        authority_pool = self._dossier.get_false_authority_pool()
        if not authority_pool:
            raise RuntimeError(f"No false authority pool available for domain '{self.domain}'.")

        authority = AuthorityInvoker(authority_pool=authority_pool, rng=self.rng)

        reframer.load_distortions(self._difficulty_pool())

        if self.mode == "demo":
            schedule = self._build_demo_schedule(reframer, authority)
        else:
            attack_turns = self._sample_attack_turns(count=6, earliest_turn=3)
            schedule = {}

            for idx, turn in enumerate(attack_turns):
                schedule[turn] = reframer if idx % 2 == 0 else authority

        return QuestionerPanel(schedule=schedule, default_questioner=neutral)

    def _build_demo_schedule(self, reframer: Reframer, authority: AuthorityInvoker) -> dict:
        role_map = {
            "reframer": reframer,
            "authority": authority,
        }

        return {
            turn: role_map[role]
            for turn, role in self.DEMO_SCHEDULE.items()
            if turn <= self.total_turns
        }