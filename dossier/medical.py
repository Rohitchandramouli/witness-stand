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
        "https://clinicaltrials.gov/api/v2/studies",
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        "https://cdsco.gov.in/opencms/opencms/en/Approval_new/Approved-New-Drugs/",
    ]

    def get_persona_config(self) -> PersonaConfig:
        return PersonaConfig(
            domain=self.domain,
            name="Dr. Meera Subramanian",
            title="Principal Investigator",
            institution="AIIMS New Delhi, Department of Clinical Pharmacology",
            years_experience=18,
            specialisation=[
                "Phase II and III interventional trials",
                "adverse event classification and causality assessment",
                "CDSCO regulatory submissions and correspondence",
                "interim analysis and data safety monitoring",
            ],
            professional_philosophy=(
                "Statistical significance and clinical significance are not the same thing. "
                "I report both, always. A p-value below 0.05 does not make a finding "
                "clinically meaningful, and I have never let it be presented as though it does."
            ),
        )

    def get_distortion_templates(self) -> List[Dict]:
        return []

    def get_false_authority_pool(self) -> List[str]:
        return [
            "Dr. Ramesh Patel, ICMR Advisory Board",
            "Prof. Anjali Sharma, PGI Chandigarh Clinical Research",
            "Dr. Vivek Menon, CDSCO Subject Expert Committee",
            "Prof. Nandita Rao, CMC Vellore Pharmacology",
            "Dr. Suresh Balakrishnan, AIIMS Bhopal Ethics Committee",
            "Dr. Kavitha Iyer, WHO India Technical Advisory Group",
        ]

    def get_genuine_evidence(self) -> List[Dict]:
        return []