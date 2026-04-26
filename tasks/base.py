"""Base class and shared curriculum utilities for all task difficulty tiers."""

import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from constants import DATA_LAG_TURNS, DISTORTION_DIFFICULTY, EPISODE_TURNS
from dossier import DOSSIER_REGISTRY
from dossier.base import DossierBase
from dossier.persona_builder import load_persona
from models import PersonaConfig
from questioners.panel import QuestionerPanel


TaskMode = str


DEMO_TURNS: Dict[str, int] = {
    "basic": 6,
    "intermediate": 8,
    "advanced": 10,
    "expert": 12,
}


TRAIN_ATTACK_COUNTS: Dict[str, int] = {
    "basic": 3,
    "intermediate": 6,
    "advanced": 12,
    "expert": 16,
}


class TaskBase(ABC):
    """
    Shared task base.

    Training mode:
        - random domain by default
        - randomized but reproducible attack schedule when seed is supplied
        - full episode horizon

    Demo mode:
        - optional fixed domain
        - fixed compact schedule
        - shorter episode horizon for clear presentation
    """

    task_name: str = ""
    dossier_pool: List[str] = list(DOSSIER_REGISTRY.keys())

    def __init__(
        self,
        domain: Optional[str] = None,
        mode: TaskMode = "train",
        seed: Optional[int] = None,
    ) -> None:
        self.mode = self._validate_mode(mode)
        self.seed = 0 if seed is None else seed
        self.rng = random.Random(self.seed)

        self._dossier: DossierBase = self._sample_dossier(domain)
        self._persona_data: dict = self._load_persona_data(self._dossier.domain)
        self._persona_cache: PersonaConfig = self._build_persona()
        self._panel: QuestionerPanel = self._build_panel()

    def _validate_mode(self, mode: TaskMode) -> TaskMode:
        if mode not in {"train", "demo"}:
            raise ValueError("mode must be either 'train' or 'demo'")
        return mode

    def _sample_dossier(self, domain: Optional[str] = None) -> DossierBase:
        if domain is not None:
            if domain not in DOSSIER_REGISTRY:
                valid = ", ".join(sorted(DOSSIER_REGISTRY))
                raise ValueError(f"Unknown domain '{domain}'. Valid domains: {valid}")
            return DOSSIER_REGISTRY[domain]()

        sampled_domain = self.rng.choice(self.dossier_pool)
        return DOSSIER_REGISTRY[sampled_domain]()

    def _load_persona_data(self, domain: str) -> dict:
        try:
            return load_persona(domain)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Persona JSON not found for domain '{domain}'. "
                "Run: python scripts/build_dossier.py"
            ) from exc

    def _build_persona(self) -> PersonaConfig:
        config = self._dossier.get_persona_config()
        config.system_prompt = self._persona_data.get("persona", {}).get("system_prompt", "")
        return config

    def _difficulty_pool(self) -> List[dict]:
        distortions = self._dossier.get_distortion_templates()
        allowed = set(DISTORTION_DIFFICULTY[self.task_name])

        pool = [
            d for d in distortions
            if int(d.get("difficulty", 1)) in allowed
        ]

        if not pool:
            # Difficulty-filtered pool is empty — fall back to all domain distortions.
            # This happens when a domain generates distortions at higher difficulties only.
            # A warning is printed but training continues normally with the full pool.
            if distortions:
                print(
                    f"Warning: no distortions matched task '{self.task_name}' difficulty "
                    f"{sorted(allowed)} for domain '{self.domain}'. "
                    f"Falling back to all domain distortions."
                )
                return distortions
            raise RuntimeError(
                f"No distortions found for domain '{self.domain}'. "
                f"Run scripts/00_build_dossier.py first."
            )

        return pool

    def _sample_attack_turns(
        self,
        count: int,
        *,
        earliest_turn: int = 3,
        latest_turn: Optional[int] = None,
        min_gap: int = 2,
    ) -> List[int]:
        """
        Samples non-repetitive attack turns.

        This prevents the model from learning fixed attack positions like 3, 6, 9.
        If sampling cannot satisfy the gap constraint, it falls back safely.
        """
        latest_turn = latest_turn or self.total_turns

        candidates = list(range(earliest_turn, latest_turn + 1))
        self.rng.shuffle(candidates)

        selected: List[int] = []

        for turn in candidates:
            if all(abs(turn - chosen) >= min_gap for chosen in selected):
                selected.append(turn)

            if len(selected) >= count:
                break

        if len(selected) < count:
            fallback = list(range(earliest_turn, latest_turn + 1))
            for turn in fallback:
                if turn not in selected:
                    selected.append(turn)
                if len(selected) >= count:
                    break

        return sorted(selected[:count])

    @property
    def total_turns(self) -> int:
        if self.mode == "demo":
            return DEMO_TURNS[self.task_name]
        return EPISODE_TURNS[self.task_name]

    @property
    def data_lag_turns(self) -> int:
        return DATA_LAG_TURNS[self.task_name]

    @property
    def persona(self) -> PersonaConfig:
        return self._persona_cache

    @property
    def panel(self) -> QuestionerPanel:
        return self._panel

    @property
    def domain(self) -> str:
        return self._dossier.domain

    @abstractmethod
    def _build_panel(self) -> QuestionerPanel:
        raise NotImplementedError