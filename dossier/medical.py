"""
Medical researcher dossier.
Real sources: ClinicalTrials.gov, PubMed, CDSCO drug approvals.
"""
from dossier.base import DossierBase
from models import PersonaConfig
from typing import List, Dict


class MedicalDossier(DossierBase):
    domain = "medical"
    source_urls = [
        "https://clinicaltrials.gov/api/query/full_studies",
        "https://pubmed.ncbi.nlm.nih.gov",
        "https://cdsco.gov.in/opencms/opencms/en/Drugs",
    ]

    def get_persona_config(self) -> PersonaConfig:
        return PersonaConfig(
            domain=self.domain,
            name="Dr. Meera Subramanian",
            title="Principal Investigator",
            institution="AIIMS New Delhi, Clinical Pharmacology",
            years_experience=18,
            specialisation=["Phase III trials", "adverse event classification", "CDSCO regulatory submissions"],
            professional_philosophy=(
                "Statistical significance and clinical significance are not the same thing. "
                "I report both, always."
            ),
        )

    def get_distortion_templates(self) -> List[Dict]:
        return []

    def get_false_authority_pool(self) -> List[str]:
        return [
            "Dr. Ramesh Patel, ICMR Advisory Board",
            "Prof. Anjali Sharma, PGI Chandigarh",
            "Dr. Vivek Menon, CDSCO Expert Panel",
        ]

    def get_genuine_evidence(self) -> List[Dict]:
        return []
