"""Domain dossier registry for The Witness Stand."""

from dossier.financial import FinancialDossier
from dossier.medical import MedicalDossier
from dossier.safety import SafetyDossier
from dossier.technical import TechnicalDossier

DOSSIER_REGISTRY = {
    "financial": FinancialDossier,
    "medical": MedicalDossier,
    "safety": SafetyDossier,
    "technical": TechnicalDossier,
}

__all__ = [
    "DOSSIER_REGISTRY",
    "FinancialDossier",
    "MedicalDossier",
    "SafetyDossier",
    "TechnicalDossier",
]