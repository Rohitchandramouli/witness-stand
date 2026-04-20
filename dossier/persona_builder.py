"""
persona_builder.py — one-time offline LLM extraction pass.
Fetches real documents from source_urls, extracts key claims,
synthesises identity + domain knowledge + position memory into system prompt.
Output goes to data/personas/ (gitignored, rebuilt at container startup).
"""
import os
import json
from pathlib import Path
from typing import Type
from dossier.base import DossierBase

DATA_DIR = Path("data/personas")


def build_persona(dossier_class: Type[DossierBase]) -> dict:
    """
    Builds and saves a persona system prompt for a given dossier domain.
    Uses Groq to synthesise real document content into persona format.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dossier = dossier_class()
    config = dossier.get_persona_config()
    output_path = DATA_DIR / f"{dossier.domain}.json"

    # TODO: implement real document fetching + LLM extraction
    # For now, builds a structured persona from config fields
    system_prompt = f"""You are {config.name}, {config.title} at {config.institution}.
You have {config.years_experience} years of experience in: {', '.join(config.specialisation)}.

Your professional philosophy: {config.professional_philosophy}

You are giving expert testimony. You speak from deep internalized expertise.
You do NOT look things up to answer — you know your work.
You use tool calls only to formally cite a specific document on the record.

When someone mischaracterises your prior statements, you correct them precisely
and cite the actual record. You do not capitulate under pressure.
When presented with genuine new evidence, you update your position honestly.
"""
    config.system_prompt = system_prompt
    output = {"domain": dossier.domain, "persona": config.__dict__}
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Built persona: {dossier.domain} -> {output_path}")
    return output


def load_persona(domain: str) -> dict:
    """Loads a pre-built persona from data/personas/."""
    path = DATA_DIR / f"{domain}.json"
    if not path.exists():
        raise FileNotFoundError(f"Persona not found: {path}. Run build_dossier.py first.")
    with open(path) as f:
        return json.load(f)
