"""
build_dossier.py — runs once offline before training.
Fetches real public documents, runs LLM extraction, generates:
  - data/dossier.db (SQLite evidence archive)
  - data/personas/<domain>.json (synthesised persona system prompts)
  - data/distortions/<domain>.json (distortion tables)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dossier.financial import FinancialDossier
from dossier.medical import MedicalDossier
from dossier.safety import SafetyDossier
from dossier.technical import TechnicalDossier
from dossier.persona_builder import build_persona
from dossier.dossier_db import init_db

DOMAINS = [FinancialDossier, MedicalDossier, SafetyDossier, TechnicalDossier]


def main():
    print("Initialising SQLite evidence archive...")
    init_db()
    print("Building personas from real documents...")
    for dossier_class in DOMAINS:
        try:
            build_persona(dossier_class)
        except Exception as e:
            print(f"  Warning: {dossier_class.domain} persona build failed: {e}")
    print("Done. data/ is ready for training.")


if __name__ == "__main__":
    main()
