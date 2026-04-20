"""
Assembles the full LLM prompt for the witness agent each turn.
Components: persona system prompt + episode memory + questioner turn.
The persona system prompt is the primary source of the witness's knowledge.
"""
from typing import List
from models import PersonaConfig
from agent.memory import EpisodicMemory
from constants import MEMORY_TOP_K


def build_system_prompt(persona: PersonaConfig) -> str:
    """Returns the full persona system prompt. This IS the witness's identity."""
    return persona.system_prompt


def build_user_prompt(
    questioner_text: str,
    memory: EpisodicMemory,
    turn_number: int,
) -> str:
    """Assembles the current turn context for the witness."""
    prior = memory.retrieve(k=MEMORY_TOP_K)
    prior_block = ""
    if prior:
        prior_block = "Your prior statements (most recent first):\n"
        for i, stmt in enumerate(prior):
            prior_block += f"  [{i}] {stmt}\n"
        prior_block += "\n"

    instruction = (
        "If the following question misrepresents your prior statements, "
        "flag it explicitly before responding. "
        "Speak from your expertise — you know your work.\n\n"
    )

    return f"{prior_block}{instruction}Questioner (turn {turn_number}): {questioner_text}"
