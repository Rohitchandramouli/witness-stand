"""Advanced task: Reframer + Authority + Exhaustion under data lag."""

from questioners.authority import AuthorityInvoker
from questioners.exhaustion import ExhaustionTactic
from questioners.neutral import NeutralQuestioner
from questioners.panel import QuestionerPanel
from questioners.reframer import Reframer
from tasks.base import TaskBase


class TaskAdvanced(TaskBase):
    task_name = "advanced"

    DEMO_SCHEDULE = {
        2: "reframer",
        4: "authority",
        6: "exhaustion",
        8: "reframer",
    }

    def _build_panel(self) -> QuestionerPanel:
        reframer = Reframer(rng=self.rng)
        neutral = NeutralQuestioner(rng=self.rng)

        authority_pool = self._dossier.get_false_authority_pool()
        if not authority_pool:
            raise RuntimeError(f"No false authority pool available for domain '{self.domain}'.")

        authority = AuthorityInvoker(authority_pool=authority_pool, rng=self.rng)
        exhaustion = ExhaustionTactic(rng=self.rng)

        reframer.load_distortions(self._difficulty_pool())

        if self.mode == "demo":
            schedule = self._build_demo_schedule(reframer, authority, exhaustion)
        else:
            attack_turns = self._sample_attack_turns(count=12, earliest_turn=3)
            attackers = [reframer, authority, exhaustion]
            schedule = {
                turn: attackers[idx % len(attackers)]
                for idx, turn in enumerate(attack_turns)
            }

        return QuestionerPanel(schedule=schedule, default_questioner=neutral)

    def _build_demo_schedule(
        self,
        reframer: Reframer,
        authority: AuthorityInvoker,
        exhaustion: ExhaustionTactic,
    ) -> dict:
        role_map = {
            "reframer": reframer,
            "authority": authority,
            "exhaustion": exhaustion,
        }

        return {
            turn: role_map[role]
            for turn, role in self.DEMO_SCHEDULE.items()
            if turn <= self.total_turns
        }