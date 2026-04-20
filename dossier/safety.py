"""
Safety engineer dossier.
Real sources: NTSB incident reports, BIS standards, OISD safety bulletins.
"""
from dossier.base import DossierBase
from models import PersonaConfig
from typing import List, Dict


class SafetyDossier(DossierBase):
    domain = "safety"
    source_urls = [
        "https://www.ntsb.gov/investigations",
        "https://services.bis.gov.in",
        "https://oisd.gov.in/oisdbulletin.aspx",
    ]

    def get_persona_config(self) -> PersonaConfig:
        return PersonaConfig(
            domain=self.domain,
            name="Arjun Raghavendra",
            title="Senior Safety Systems Engineer",
            institution="BHEL Power Sector, Noida",
            years_experience=16,
            specialisation=["IEC 61508 functional safety", "FMEA", "process hazard analysis"],
            professional_philosophy=(
                "An RPN score is not a judgment — it is a calculation. "
                "The inputs must be defensible, not convenient."
            ),
        )

    def get_distortion_templates(self) -> List[Dict]:
        return []

    def get_false_authority_pool(self) -> List[str]:
        return [
            "Dr. Pradeep Kulkarni, OISD Safety Panel",
            "Prof. Suresh Iyer, IIT Bombay Chemical Engineering",
            "Eng. Ramakrishnan, BIS Technical Committee",
        ]

    def get_genuine_evidence(self) -> List[Dict]:
        return []
