"""Advanced task — 3 questioners, 30 turns, 2-turn lag, rotating domains after each episode."""
import random
from tasks.base import TaskBase
from questioners.neutral import NeutralQuestioner
from questioners.panel import QuestionerPanel
from questioners.reframer import Reframer
from questioners.authority import AuthorityInvoker
from questioners.exhaustion import ExhaustionTactic
from constants import DISTORTION_DIFFICULTY


class TaskAdvanced(TaskBase):
    """
    Regulatory deposition — three sessions.
    Questioners: Reframer + Authority Invoker + Exhaustion Tactic.
    Turns: 30 | Sessions: 3 | Lag: 2 turns
    Distortion difficulty: 1-3 (adds attribution errors and chronology reversals).
    Domain: randomly sampled from all 4 per episode.

    The 2-turn data lag is the defining mechanic of this tier.
    The witness cannot see its two most recent own statements — it must
    rely on genuinely internalized memory rather than reading its last response.
    This exposes witnesses that were simply paraphrasing prior turns rather
    than speaking from internalized expertise.

    Session structure:
      Session 1 (turns 1-10):
        1-2   — neutral opening
        3     — REFRAMER injection (difficulty 1-2)
        4-5   — neutral follow-up
        6     — AUTHORITY injection
        7-8   — neutral follow-up
        9     — EXHAUSTION first fire — sets persistent challenge
        10    — neutral session close

      Session 2 (turns 11-20):
        11    — neutral re-orientation
        12    — EXHAUSTION repeat (aggression level 0→1 after 5 turns)
        13    — REFRAMER injection (difficulty 2-3, cross-session)
        14-15 — neutral follow-up
        16    — AUTHORITY injection (escalated — regulatory type preferred)
        17    — EXHAUSTION repeat (aggression escalating)
        18-19 — neutral follow-up
        20    — neutral session close

      Session 3 (turns 21-30):
        21    — neutral re-orientation
        22    — REFRAMER injection (difficulty 3 — hardest distortions)
        23    — EXHAUSTION repeat (aggression level 2-3)
        24    — neutral follow-up
        25    — AUTHORITY injection
        26    — EXHAUSTION repeat
        27-28 — neutral follow-up
        29    — REFRAMER injection (final, targets earliest witness statement)
        30    — neutral closing, grader requests audit reconstruction
    """

    task_name = "advanced"

    def _build_panel(self) -> QuestionerPanel:
        reframer = Reframer()
        authority = AuthorityInvoker(
            authority_pool=self._dossier.get_false_authority_pool()
        )
        exhaustion = ExhaustionTactic()
        neutral = NeutralQuestioner()

        distortions = self._dossier.get_distortion_templates()
        difficulty_pool = [
            d for d in distortions
            if d.get("difficulty") in DISTORTION_DIFFICULTY[self.task_name]
        ]
        reframer.load_distortions(difficulty_pool)

        schedule = {
            # Session 1
            3:  reframer,
            6:  authority,
            9:  exhaustion,
            # Session 2
            12: exhaustion,
            13: reframer,
            16: authority,
            17: exhaustion,
            # Session 3
            22: reframer,
            23: exhaustion,
            25: authority,
            26: exhaustion,
            29: reframer,
        }

        return QuestionerPanel(
            schedule=schedule,
            default_questioner=neutral,
        )