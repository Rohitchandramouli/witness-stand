"""Expert task: all adversaries, domain rotation, temporal pressure, and data lag."""

import random
from typing import Optional, Tuple

from constants import DISTORTION_DIFFICULTY
from dossier import DOSSIER_REGISTRY
from dossier.base import DossierBase
from dossier.persona_builder import load_persona
from models import PersonaConfig
from questioners.authority import AuthorityInvoker
from questioners.exhaustion import ExhaustionTactic
from questioners.neutral import NeutralQuestioner
from questioners.panel import QuestionerPanel
from questioners.reframer import Reframer
from questioners.temporal import TemporalQuestioner
from tasks.base import TaskBase


_DOMAIN_PAIRS: list[Tuple[str, str]] = [
    ("financial", "technical"),
    ("medical", "safety"),
    ("technical", "financial"),
    ("safety", "medical"),
]


class TaskExpert(TaskBase):
    task_name = "expert"

    rotation_turn = 21
    demo_rotation_turn = 7

    DEMO_SCHEDULE = {
        2: "reframer",
        4: "authority",
        6: "temporal",
        8: "exhaustion",
        10: "reframer",
        11: "temporal",
    }

    def __init__(
        self,
        domain: Optional[str] = None,
        domain_pair: Optional[Tuple[str, str]] = None,
        mode: str = "train",
        seed: Optional[int] = None,
    ) -> None:
        self.mode = self._validate_mode(mode)
        self.seed = 0 if seed is None else seed
        self.rng = random.Random(self.seed)

        self._domain_a, self._domain_b = self._resolve_domain_pair(domain, domain_pair)

        self._dossier_a: DossierBase = DOSSIER_REGISTRY[self._domain_a]()
        self._dossier_b: DossierBase = DOSSIER_REGISTRY[self._domain_b]()

        self._persona_data_a = self._load_persona_data(self._domain_a)
        self._persona_data_b = self._load_persona_data(self._domain_b)

        self._active_domain = self._domain_a
        self._dossier = self._dossier_a
        self._persona_data = self._persona_data_a

        self._persona_cache = self._build_persona()
        self._panel = self._build_panel()

    def _resolve_domain_pair(
        self,
        domain: Optional[str],
        domain_pair: Optional[Tuple[str, str]],
    ) -> Tuple[str, str]:
        if domain_pair is not None:
            a, b = domain_pair
            self._validate_domain(a)
            self._validate_domain(b)
            if a == b:
                raise ValueError("Expert domain_pair must contain two different domains.")
            return a, b

        if domain is not None:
            self._validate_domain(domain)
            possible = [pair for pair in _DOMAIN_PAIRS if pair[0] == domain]
            if possible:
                return self.rng.choice(possible)

            alternatives = [d for d in DOSSIER_REGISTRY if d != domain]
            return domain, self.rng.choice(alternatives)

        return self.rng.choice(_DOMAIN_PAIRS)

    def _validate_domain(self, domain: str) -> None:
        if domain not in DOSSIER_REGISTRY:
            valid = ", ".join(sorted(DOSSIER_REGISTRY))
            raise ValueError(f"Unknown domain '{domain}'. Valid domains: {valid}")

    def rotate_to_domain_b(self) -> None:
        self._active_domain = self._domain_b
        self._dossier = self._dossier_b
        self._persona_data = self._persona_data_b
        self._persona_cache = self._build_persona()

    @property
    def active_rotation_turn(self) -> int:
        return self.demo_rotation_turn if self.mode == "demo" else self.rotation_turn

    @property
    def persona(self) -> PersonaConfig:
        return self._persona_cache

    @property
    def domain(self) -> str:
        return self._active_domain

    @property
    def domain_a(self) -> str:
        return self._domain_a

    @property
    def domain_b(self) -> str:
        return self._domain_b

    def _build_persona(self) -> PersonaConfig:
        config = self._dossier.get_persona_config()
        config.system_prompt = self._persona_data.get("persona", {}).get("system_prompt", "")
        return config

    def _difficulty_pool_for_both_domains(self) -> list[dict]:
        allowed = set(DISTORTION_DIFFICULTY[self.task_name])

        distortions = (
            self._dossier_a.get_distortion_templates()
            + self._dossier_b.get_distortion_templates()
        )

        pool = [
            d for d in distortions
            if int(d.get("difficulty", 1)) in allowed
        ]

        if not pool:
            raise RuntimeError(
                f"No expert distortions available for domains "
                f"'{self._domain_a}' and '{self._domain_b}'. "
                "Run scripts/build_dossier.py first."
            )

        return pool

    def _build_panel(self) -> QuestionerPanel:
        reframer = Reframer(rng=self.rng)
        neutral = NeutralQuestioner(rng=self.rng)

        authority_pool = (
            self._dossier_a.get_false_authority_pool()
            + self._dossier_b.get_false_authority_pool()
        )
        if not authority_pool:
            raise RuntimeError(
                f"No false authority pool available for expert domains "
                f"'{self._domain_a}' and '{self._domain_b}'."
            )

        authority = AuthorityInvoker(authority_pool=authority_pool, rng=self.rng)
        exhaustion = ExhaustionTactic(rng=self.rng)
        temporal = TemporalQuestioner(rng=self.rng)

        reframer.load_distortions(self._difficulty_pool_for_both_domains())

        if self.mode == "demo":
            schedule = self._build_demo_schedule(reframer, authority, exhaustion, temporal)
        else:
            attack_turns = self._sample_attack_turns(count=16, earliest_turn=3)
            attackers = [reframer, authority, exhaustion, temporal]
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
        temporal: TemporalQuestioner,
    ) -> dict:
        role_map = {
            "reframer": reframer,
            "authority": authority,
            "exhaustion": exhaustion,
            "temporal": temporal,
        }

        return {
            turn: role_map[role]
            for turn, role in self.DEMO_SCHEDULE.items()
            if turn <= self.total_turns
        }