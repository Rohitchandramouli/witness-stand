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
    "https://www.ntsb.gov/investigations/AccidentReports/Pages/Reports.aspx",
    "https://data.ntsb.gov/avdata",
    "https://dgfasli.gov.in",
    ]
    
    def get_persona_config(self) -> PersonaConfig:
        return PersonaConfig(
            domain=self.domain,
            name="Arjun Raghavendra",
            title="Senior Safety Systems Engineer",
            institution="BHEL Power Sector, Noida",
            years_experience=16,
            specialisation=[
                "IEC 61508 functional safety and SIL determination",
                "FMEA and risk priority number calculation",
                "process hazard analysis and HAZOP facilitation",
                "incident investigation and root cause analysis",
            ],
            professional_philosophy=(
                "An RPN score is not a judgment — it is a calculation. "
                "The inputs must be defensible, not convenient. When someone "
                "asks me to revise a severity rating downward without new "
                "evidence, I treat that as a request to falsify a record."
            ),
        )

    def get_distortion_templates(self) -> List[Dict]:
        return []

    def get_false_authority_pool(self) -> List[str]:
        return [
            "Dr. Pradeep Kulkarni, OISD Safety Advisory Panel",
            "Prof. Suresh Iyer, IIT Bombay Chemical Engineering",
            "Eng. Ramakrishnan, BIS Technical Standards Committee",
            "Dr. Nalini Mohan, PESO Statutory Review Board",
            "Prof. Arun Chatterjee, IIT Kharagpur Industrial Safety",
            "Dr. Vikram Sood, NTPC Safety Directorate",
        ]

    def get_genuine_evidence(self) -> List[Dict]:
        return []