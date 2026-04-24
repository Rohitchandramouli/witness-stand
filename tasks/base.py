"""Abstract base for all task difficulty tiers."""
import random
from abc import ABC, abstractmethod
from typing import List, Type

from models import PersonaConfig
from questioners.panel import QuestionerPanel
from constants import EPISODE_TURNS, DATA_LAG_TURNS
from dossier import DOSSIER_REGISTRY
from dossier.base import DossierBase
from dossier.persona_builder import load_persona


class TaskBase(ABC):
    task_name: str = ""

    # Subclasses override this with the domains they draw from.
    # basic/intermediate/advanced use all 4.
    # expert uses all 4 but samples 2 per episode.
    dossier_pool: List[str] = list(DOSSIER_REGISTRY.keys())

    def __init__(self):
        self._dossier: DossierBase = self._sample_dossier()
        self._persona_data: dict = load_persona(self._dossier.domain)
        self._panel: QuestionerPanel = self._build_panel()
        self._persona_cache: PersonaConfig = self._build_persona()

    def _build_persona(self) -> PersonaConfig:
        config = self._dossier.get_persona_config()
        config.system_prompt = self._persona_data["persona"].get("system_prompt", "")
        return config

    # ── Episode-level sampling ────────────────────────────────────────

    def _sample_dossier(self) -> DossierBase:
        """
        Samples one domain randomly from dossier_pool and instantiates it.
        Called at __init__ time — each episode gets a fresh random persona.
        """
        domain = random.choice(self.dossier_pool)
        dossier_class: Type[DossierBase] = DOSSIER_REGISTRY[domain]
        return dossier_class()

    # ── OpenEnv properties ────────────────────────────────────────────

    @property
    def total_turns(self) -> int:
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

    # ── Abstract interface ────────────────────────────────────────────

    @abstractmethod
    def _build_panel(self) -> QuestionerPanel:
        """
        Constructs the QuestionerPanel for this task tier.
        Called once at __init__ after the dossier has been sampled,
        so questioners can be loaded with domain-specific distortion pools.
        """
        ...