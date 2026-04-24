"""
Financial analyst dossier.
Real sources: SEBI enforcement orders, RBI monetary policy reports, NSE/BSE disclosures.
"""
from dossier.base import DossierBase
from models import PersonaConfig
from typing import List, Dict


class FinancialDossier(DossierBase):
    domain = "financial"
    source_urls = [
    "https://www.sebi.gov.in/enforcement/orders.html",
    "https://www.rbi.org.in/scripts/AnnualPublications.aspx?head=Monetary+Policy+Report",
    "https://www.rbi.org.in/Scripts/AnnualReportPublications.aspx",
    ]

    def get_persona_config(self) -> PersonaConfig:
        return PersonaConfig(
            domain=self.domain,
            name="Dr. Priya Venkataraman",
            title="Senior Risk Analyst",
            institution="Edelweiss Alternative Asset Managers",
            years_experience=14,
            specialisation=[
                "NBFC sector risk assessment",
                "Basel-III adapted risk taxonomy",
                "SEBI expert opinion submissions",
                "RBI monetary policy impact analysis",
            ],
            professional_philosophy=(
                "Caveats are not qualifications to be stripped for convenience — "
                "they are load-bearing analytical conclusions. A rating without "
                "its caveats is not a simplified rating, it is a different rating."
            ),
        )


    def get_distortion_templates(self) -> List[Dict]:
        """
        Returns distortion templates for this domain.
        Loads from SQLite DB if available (populated by build_dossier.py).
        Falls back to empty list only if DB has no entries — in that case
        the Reframer falls back to transcript-based distortions.
        """
        try:
            # Load all difficulty levels — tasks filter by their own pool
            rows = get_distortions_for_domain(self.domain, [1, 2, 3, 4])
            if rows:
                return rows
        except Exception:
            pass
        return []


    def get_false_authority_pool(self) -> List[str]:
        return [
            "Prof. Anil Mehta, IIM Ahmedabad Finance Faculty",
            "Dr. Riya Shenoy, RBI Advisory Panel",
            "Dr. K. Nair, SEBI Technical Committee",
            "Prof. Sundar Krishnamurthy, IIMB",
            "Dr. Vikram Anand, CRISIL Research Head",
            "Prof. Leela Rao, NUS Business School",
        ]


    def get_genuine_evidence(self) -> List[Dict]:
        """
        Returns genuine evidence records for this domain from the DB.
        Used for the symmetric discrimination mechanic.
        """
        try:
            return search_record("genuine", domain=self.domain)
        except Exception:
            return []
