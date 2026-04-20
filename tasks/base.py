"""Abstract base for all task difficulty tiers."""
from abc import ABC, abstractmethod
from models import PersonaConfig
from questioners.panel import QuestionerPanel
from constants import EPISODE_TURNS, DATA_LAG_TURNS


class TaskBase(ABC):
    task_name: str = ""

    @property
    def total_turns(self) -> int:
        return EPISODE_TURNS[self.task_name]

    @property
    def data_lag_turns(self) -> int:
        return DATA_LAG_TURNS[self.task_name]

    @property
    @abstractmethod
    def persona(self) -> PersonaConfig:
        ...

    @property
    @abstractmethod
    def panel(self) -> QuestionerPanel:
        ...
