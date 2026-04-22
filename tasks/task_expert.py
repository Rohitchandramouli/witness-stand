"""Expert task — all 4 questioners, 40 turns, 3-turn lag,rotating domains within each episode, uncapped reward."""
import random
from typing import List, Tuple, Type

from tasks.base import TaskBase
from questioners.neutral import NeutralQuestioner
from questioners.panel import QuestionerPanel
from questioners.reframer import Reframer
from questioners.authority import AuthorityInvoker
from questioners.exhaustion import ExhaustionTactic
from questioners.temporal import TemporalQuestioner
from dossier import DOSSIER_REGISTRY
from dossier.base import DossierBase
from dossier.persona_builder import load_persona
from constants import DISTORTION_DIFFICULTY
from models import PersonaConfig


# All valid two-domain pairs for the expert rotating persona
_DOMAIN_PAIRS: List[Tuple[str, str]] = [
    ("financial",  "technical"),
    ("medical",    "safety"),
    ("safety",     "technical"),
    ("financial",  "medical"),
    ("medical",    "technical"),
    ("financial",  "safety"),
]


class TaskExpert(TaskBase):
    """
    Cross-domain adversarial tribunal — four sessions.
    Questioners: Reframer + Authority Invoker + Exhaustion + Temporal Questioner.
    Turns: 40 | Sessions: 4 | Lag: 3 turns
    Distortion difficulty: 1-4 (full range including threshold substitutions).
    Domain: 2 randomly sampled domains, rotating at session 3 boundary.

    The Temporal Questioner is the defining mechanic of this tier.
    It attacks chronology — not content. It requires the witness to track
    not just what it said but what information it had access to when it said it.

    Domain rotation: the witness begins as one domain expert and transitions
    mid-episode to defending decisions made in a second domain. The rotation
    happens at the session 2→3 boundary. The Temporal Questioner exploits
    this by attacking statements made in session 1-2 using evidence that
    only existed in the session 3-4 domain — a deliberate cross-domain
    temporal trap.

    Reward: uncapped — scales with the number of prior statements
    simultaneously tracked and faithfully defended in audit reconstruction.

    Session structure:
      Session 1 (turns 1-10) — domain A:
        1-2   — neutral opening, witness establishes domain A identity
        3     — REFRAMER (difficulty 1-2)
        4     — neutral
        5     — TEMPORAL first fire — establishes chronology baseline
        6     — AUTHORITY injection
        7-8   — neutral
        9     — EXHAUSTION first fire
        10    — neutral session close

      Session 2 (turns 11-20) — domain A continues:
        11    — neutral re-orientation
        12    — REFRAMER (difficulty 2-3, cross-session)
        13    — TEMPORAL attack on session 1 statement
        14    — neutral
        15    — AUTHORITY (escalated type)
        16    — EXHAUSTION repeat
        17    — neutral
        18    — TEMPORAL attack (lag exploit — hardest type)
        19    — neutral
        20    — neutral session close

      Session 3 (turns 21-30) — domain B begins:
        Domain persona rotates here. The witness now defends decisions
        made in domain B, while the Temporal Questioner attacks domain A
        statements using domain B knowledge as false temporal ammunition.
        21    — neutral re-orientation (domain B intro)
        22    — REFRAMER (difficulty 3)
        23    — TEMPORAL cross-domain attack
        24    — EXHAUSTION repeat (aggression level 2)
        25    — AUTHORITY (domain B false authorities)
        26    — neutral
        27    — REFRAMER (difficulty 3-4)
        28    — TEMPORAL (stale standard attack)
        29    — EXHAUSTION (maximum aggression)
        30    — neutral session close

      Session 4 (turns 31-40) — domain B continues:
        31    — neutral re-orientation
        32    — TEMPORAL (post-revision attack)
        33    — AUTHORITY injection
        34    — REFRAMER (difficulty 4 — hardest distortions)
        35    — EXHAUSTION repeat
        36    — neutral
        37    — TEMPORAL (lag exploit — final)
        38    — REFRAMER (final, targets earliest domain A statement)
        39    — EXHAUSTION (maximum aggression, final pressure)
        40    — neutral closing, grader requests full audit reconstruction
    """

    task_name = "expert"

    # Expert samples 2 domains — override base class single-domain sampling
    def __init__(self):
        # Sample two distinct domains before calling super().__init__()
        # because super().__init__() calls _build_panel() which needs both
        self._domain_a, self._domain_b = self._sample_domain_pair()

        self._dossier_a: DossierBase = DOSSIER_REGISTRY[self._domain_a]()
        self._dossier_b: DossierBase = DOSSIER_REGISTRY[self._domain_b]()

        self._persona_data_a: dict = load_persona(self._domain_a)
        self._persona_data_b: dict = load_persona(self._domain_b)

        # Current active domain — starts as A, switches to B at session 3
        self._active_domain: str = self._domain_a

        # Initialise _dossier to domain A so TaskBase.persona works correctly
        self._dossier = self._dossier_a
        self._persona_data = self._persona_data_a

        # Now build the panel — _dossier is set so _build_panel can use it
        self._panel = self._build_panel()

    @staticmethod
    def _sample_domain_pair() -> Tuple[str, str]:
        """Randomly samples one of the predefined two-domain pairs."""
        return random.choice(_DOMAIN_PAIRS)

    def rotate_to_domain_b(self) -> None:
        """
        Called by environment.py at the session 2→3 boundary.
        Switches the active persona from domain A to domain B.
        The witness's system prompt changes — it now speaks as the domain B expert.
        Prior statements from domain A remain in the transcript and become
        targets for the Temporal Questioner's cross-domain attacks.
        """
        self._active_domain = self._domain_b
        self._dossier = self._dossier_b
        self._persona_data = self._persona_data_b

    @property
    def persona(self) -> PersonaConfig:
        """Returns the currently active persona — A or B depending on session."""
        config = self._dossier.get_persona_config()
        config.system_prompt = self._persona_data["persona"].get(
            "system_prompt", ""
        )
        return config

    @property
    def domain(self) -> str:
        return self._active_domain

    @property
    def domain_a(self) -> str:
        return self._domain_a

    @property
    def domain_b(self) -> str:
        return self._domain_b

    def _build_panel(self) -> QuestionerPanel:
        reframer = Reframer()
        authority = AuthorityInvoker(
            authority_pool=(
                self._dossier_a.get_false_authority_pool() +
                self._dossier_b.get_false_authority_pool()
            )
        )
        exhaustion = ExhaustionTactic()
        temporal = TemporalQuestioner()
        neutral = NeutralQuestioner()

        distortions = (
            self._dossier_a.get_distortion_templates() +
            self._dossier_b.get_distortion_templates()
        )
        difficulty_pool = [
            d for d in distortions
            if d.get("difficulty") in DISTORTION_DIFFICULTY[self.task_name]
        ]
        reframer.load_distortions(difficulty_pool)

        schedule = {
            # Session 1 — domain A
            3:  reframer,
            5:  temporal,
            6:  authority,
            9:  exhaustion,
            # Session 2 — domain A continues
            12: reframer,
            13: temporal,
            15: authority,
            16: exhaustion,
            18: temporal,
            # Session 3 — domain B begins (rotation at turn 21)
            22: reframer,
            23: temporal,
            24: exhaustion,
            25: authority,
            27: reframer,
            28: temporal,
            29: exhaustion,
            # Session 4 — domain B continues
            32: temporal,
            33: authority,
            34: reframer,
            35: exhaustion,
            37: temporal,
            38: reframer,
            39: exhaustion,
        }

        return QuestionerPanel(
            schedule=schedule,
            default_questioner=neutral,
        )