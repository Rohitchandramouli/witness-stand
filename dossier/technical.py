"""
AI systems auditor dossier.
Real sources: HuggingFace model cards, AI audit reports, EU AI Act documents.
"""
from dossier.base import DossierBase
from models import PersonaConfig
from typing import List, Dict


class TechnicalDossier(DossierBase):
    domain = "technical"
    source_urls = [
        "https://huggingface.co/models",
        "https://algorithmwatch.org",
        "https://partnershiponai.org",
    ]

    def get_persona_config(self) -> PersonaConfig:
        return PersonaConfig(
            domain=self.domain,
            name="Sahana Krishnaswamy",
            title="AI Systems Auditor",
            institution="Responsible AI Institute, Bengaluru",
            years_experience=9,
            specialisation=["fairness auditing", "model card analysis", "EU AI Act compliance"],
            professional_philosophy=(
                "A model card is a commitment. If the deployment diverges from the card, "
                "that is an accountability failure, not a technical one."
            ),
        )

    def get_distortion_templates(self) -> List[Dict]:
        return []

    def get_false_authority_pool(self) -> List[str]:
        return [
            "Dr. Pooja Iyer, NASSCOM AI Ethics Board",
            "Prof. Kiran Rao, IISc AI Safety Lab",
            "Dr. Ananya Bose, MeitY Expert Panel",
        ]

    def get_genuine_evidence(self) -> List[Dict]:
        return []
