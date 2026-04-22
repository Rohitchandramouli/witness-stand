"""
Basic task — Reframer only, 10 turns, no lag, rotating domains after each episode.
"""
import random
from tasks.base import TaskBase
from questioners.panel import QuestionerPanel
from questioners.reframer import Reframer
from questioners.base import QuestionerBase
from transcript.store import TranscriptStore
from models import PersonaConfig, TurnType
from constants import DISTORTION_DIFFICULTY


class NeutralQuestioner(QuestionerBase):
    """
    Fires on non-injection turns. Asks genuine follow-up questions that
    give the witness space to elaborate, establishing the position record
    that subsequent distortion turns will attack.
    """
    questioner_id = "neutral"

    _FOLLOW_UPS = [
        "Could you elaborate on the methodology behind that conclusion?",
        "What specific evidence led you to that assessment?",
        "Walk us through your reasoning on that point in more detail.",
        "What alternative conclusions did you consider and reject?",
        "How confident are you in that assessment, and what are its limits?",
    ]

    def generate_turn(
        self, transcript: TranscriptStore, persona: PersonaConfig
    ) -> str:
        witness_turns = transcript.get_witness_turns()
        if not witness_turns:
            return (
                f"Please state your full name, your professional role, "
                f"and the nature of the assessment you have been asked to defend."
            )
        return random.choice(self._FOLLOW_UPS)

    def get_turn_type(self) -> TurnType:
        return TurnType.NEUTRAL


class TaskBasic(TaskBase):
    """
    Single-session regulatory inquiry.
    Questioner: Reframer only.
    Turns: 10 | Sessions: 1 | Lag: 0
    Distortion difficulty: 1 only (direct word substitutions).
    Domain: randomly sampled from all 4 per episode.

    Turn structure:
      1     — witness opens, states identity and assessment (neutral)
      2     — neutral follow-up, witness elaborates methodology
      3     — DISTORTION INJECTION (turn 3 chosen so witness has made
                2 substantive statements to distort)
      4     — neutral follow-up
      5     — neutral follow-up, witness continues building record
      6     — DISTORTION INJECTION (mid-episode, targets a different
                statement than turn 3)
      7     — neutral follow-up
      8     — neutral follow-up
      9     — DISTORTION INJECTION (late pressure — witness is tired,
                record is long, harder to track all prior statements)
      10    — neutral closing — grader requests audit reconstruction
    """

    task_name = "basic"

    def _build_panel(self) -> QuestionerPanel:
        reframer = Reframer()
        neutral = NeutralQuestioner()

        # Load only difficulty-1 distortions for this tier
        distortions = self._dossier.get_distortion_templates()
        difficulty_pool = [
            d for d in distortions
            if d.get("difficulty") in DISTORTION_DIFFICULTY[self.task_name]
        ]
        reframer.load_distortions(difficulty_pool)

        # Inject at turns 3, 6, 9 — neutral on all others
        schedule = {
            3: reframer,
            6: reframer,
            9: reframer,
        }

        return QuestionerPanel(
            schedule=schedule,
            default_questioner=neutral,
        )