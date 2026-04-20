"""
Basic task — Reframer only, 10 turns, no lag, financial domain.
Difficulty-1 distortions: direct word substitutions.
"""
from tasks.base import TaskBase
from models import PersonaConfig
from dossier.financial import FinancialDossier
from dossier.persona_builder import load_persona
from questioners.panel import QuestionerPanel
from questioners.reframer import Reframer


class TaskBasic(TaskBase):
    task_name = "basic"

    def __init__(self):
        self._dossier = FinancialDossier()
        self._persona_data = load_persona("financial")
        self._reframer = Reframer()
        self._reframer.load_distortions(self._dossier.get_distortion_templates())

    @property
    def persona(self) -> PersonaConfig:
        config = self._dossier.get_persona_config()
        config.system_prompt = self._persona_data["persona"].get("system_prompt", "")
        return config

    @property
    def panel(self) -> QuestionerPanel:
        # Reframer fires at turns 3, 6, 9
        schedule = {3: self._reframer, 6: self._reframer, 9: self._reframer}
        return QuestionerPanel(schedule=schedule, default_questioner=self._reframer)
