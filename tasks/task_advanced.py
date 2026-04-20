"""Advanced task — 3 questioners, 30 turns, 2-turn lag, safety domain."""
from tasks.base import TaskBase
from models import PersonaConfig
from dossier.safety import SafetyDossier
from dossier.persona_builder import load_persona
from questioners.panel import QuestionerPanel
from questioners.reframer import Reframer
from questioners.authority import AuthorityInvoker
from questioners.exhaustion import ExhaustionTactic


class TaskAdvanced(TaskBase):
    task_name = "advanced"

    def __init__(self):
        self._dossier = SafetyDossier()
        self._persona_data = load_persona("safety")
        self._reframer = Reframer()
        self._authority = AuthorityInvoker(self._dossier.get_false_authority_pool())
        self._exhaustion = ExhaustionTactic()

    @property
    def persona(self) -> PersonaConfig:
        config = self._dossier.get_persona_config()
        config.system_prompt = self._persona_data["persona"].get("system_prompt", "")
        return config

    @property
    def panel(self) -> QuestionerPanel:
        schedule = {
            4: self._reframer, 8: self._authority,
            12: self._exhaustion, 16: self._reframer,
            20: self._authority, 24: self._exhaustion, 28: self._reframer,
        }
        return QuestionerPanel(schedule=schedule, default_questioner=self._reframer)
