"""Expert task — all 4 questioners, 40 turns, 3-turn lag, rotating domains, uncapped reward."""
from tasks.base import TaskBase
from models import PersonaConfig
from dossier.technical import TechnicalDossier
from dossier.financial import FinancialDossier
from dossier.persona_builder import load_persona
from questioners.panel import QuestionerPanel
from questioners.reframer import Reframer
from questioners.authority import AuthorityInvoker
from questioners.exhaustion import ExhaustionTactic
from questioners.temporal import TemporalQuestioner


class TaskExpert(TaskBase):
    task_name = "expert"

    def __init__(self):
        self._dossier = TechnicalDossier()
        self._persona_data = load_persona("technical")
        self._reframer = Reframer()
        self._authority = AuthorityInvoker(self._dossier.get_false_authority_pool())
        self._exhaustion = ExhaustionTactic()
        self._temporal = TemporalQuestioner()

    @property
    def persona(self) -> PersonaConfig:
        config = self._dossier.get_persona_config()
        config.system_prompt = self._persona_data["persona"].get("system_prompt", "")
        return config

    @property
    def panel(self) -> QuestionerPanel:
        schedule = {
            4: self._reframer, 8: self._authority,
            12: self._exhaustion, 16: self._temporal,
            20: self._reframer, 24: self._authority,
            28: self._exhaustion, 32: self._temporal,
            36: self._reframer, 40: self._temporal,
        }
        return QuestionerPanel(schedule=schedule, default_questioner=self._reframer)
