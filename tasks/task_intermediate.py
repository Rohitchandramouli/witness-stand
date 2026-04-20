"""Intermediate task — Reframer + Authority Invoker, 20 turns, medical domain."""
from tasks.base import TaskBase
from models import PersonaConfig
from dossier.medical import MedicalDossier
from dossier.persona_builder import load_persona
from questioners.panel import QuestionerPanel
from questioners.reframer import Reframer
from questioners.authority import AuthorityInvoker


class TaskIntermediate(TaskBase):
    task_name = "intermediate"

    def __init__(self):
        self._dossier = MedicalDossier()
        self._persona_data = load_persona("medical")
        self._reframer = Reframer()
        self._authority = AuthorityInvoker(self._dossier.get_false_authority_pool())

    @property
    def persona(self) -> PersonaConfig:
        config = self._dossier.get_persona_config()
        config.system_prompt = self._persona_data["persona"].get("system_prompt", "")
        return config

    @property
    def panel(self) -> QuestionerPanel:
        schedule = {4: self._reframer, 8: self._authority, 12: self._reframer, 16: self._authority}
        return QuestionerPanel(schedule=schedule, default_questioner=self._reframer)
