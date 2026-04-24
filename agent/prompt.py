"""
Prompt construction for the witness agent.
Assembles the full context the witness LLM receives each turn:
  1. Persona system prompt (who the witness is)
  2. Strategy guidance (what approach to take this turn)
  3. Episode memory (what the witness has said so far)
  4. Current questioner challenge (what it must respond to)

Zero LLM calls — pure string assembly.
"""
from typing import Optional
from models import PersonaConfig, TurnType
from agent.memory import EpisodicMemory
from agent.heuristics import WitnessHeuristics


# ── System prompt template ────────────────────────────────────────────

_SYSTEM_TEMPLATE = """{persona_system_prompt}

EXAMINATION RULES:
You are under cross-examination. Your responses must follow these rules:

1. SPEAK FROM MEMORY — you know your work intimately. You do not look
   things up to answer questions. You retrieve documents only to formally
   cite them on the record, not to learn what they say.

2. DETECT DISTORTIONS — if the examiner misrepresents what you said,
   flag it immediately and precisely. State the turn number, quote your
   exact prior language, and correct the record.

3. HOLD YOUR POSITION — do not revise your assessment under pressure
   alone. Only update if genuinely new evidence is presented that you
   had not previously considered.

4. USE TOOLS SPARINGLY — three tools are available:
   - search_record(query="...") — verify an external claim before contesting it
   - retrieve_document(doc_id="...", version="...") — formally cite a document
   - flag_inconsistency(claim="...", evidence_id="...") — lodge a formal dispute
   Call tools by writing them inline: retrieve_document(doc_id="RPT-001")

5. BE SPECIFIC — vague responses score poorly. Name turn numbers,
   quote prior language, cite document IDs. Precision is credibility.

CURRENT STRATEGY GUIDANCE:
{strategy_guidance}
"""

# ── User prompt template ──────────────────────────────────────────────

_USER_TEMPLATE = """SESSION {session_number} — TURN {turn_number} of {total_turns}
DOMAIN: {domain}{lag_notice}

YOUR PRIOR STATEMENTS (most recent first):
{memory_block}

EXAMINER: {questioner_text}

Respond now. Be precise and specific."""

# ── Reconstruction prompt template ────────────────────────────────────

_RECONSTRUCTION_TEMPLATE = """SESSION CLOSE — AUDIT RECONSTRUCTION

You are now asked to reconstruct your reasoning for each contested claim
in this examination. For each claim:

1. State exactly what you said (quote your own words)
2. State what turn you said it at
3. State what information you had access to at that time
4. Explain why your position was sound given that information
5. If the examiner distorted your statement, identify the distortion

YOUR FULL STATEMENT RECORD:
{full_memory_block}

EXAMINER: {questioner_text}

Reconstruct your reasoning now. Be thorough — cover every contested claim.
Cite turn numbers and document references throughout."""


def build_system_prompt(
    persona: PersonaConfig,
    attack_type: TurnType = TurnType.NEUTRAL,
    heuristics: Optional[WitnessHeuristics] = None,
) -> str:
    """
    Builds the system prompt for the witness LLM.
    The system prompt is stable across turns within an episode —
    it changes only when the domain rotates in the expert task.

    Parameters
    ----------
    persona      : the active persona config with system_prompt populated
    attack_type  : the type of attack on the current turn, used to
                   select strategy guidance from heuristics
    heuristics   : cross-episode strategy library — optional, falls back
                   to default guidance if not provided
    """
    if heuristics is not None:
        strategy_guidance = heuristics.get_strategy_guidance(attack_type)
    else:
        strategy_guidance = _default_strategy_guidance(attack_type)

    return _SYSTEM_TEMPLATE.format(
        persona_system_prompt=persona.system_prompt or _fallback_persona(persona),
        strategy_guidance=strategy_guidance,
    )


def build_user_prompt(
    questioner_text: str,
    memory: EpisodicMemory,
    turn_number: int,
    total_turns: int = 10,
    domain: str = "",
    session_number: int = 1,
    data_lag_turns: int = 0,
    is_reconstruction: bool = False,
) -> str:
    """
    Builds the user-turn prompt the witness LLM receives each turn.

    Parameters
    ----------
    questioner_text   : the examiner's question or challenge
    memory            : episodic memory containing prior witness statements
    turn_number       : current turn number (1-indexed for display)
    total_turns       : total turns in this episode
    domain            : current domain name for display
    session_number    : current session number (1-indexed)
    data_lag_turns    : number of recent turns hidden by lag mechanic
    is_reconstruction : True on the final audit reconstruction turn
    """
    if is_reconstruction:
        return _build_reconstruction_prompt(questioner_text, memory)

    memory_block = _build_memory_block(memory, data_lag_turns)
    lag_notice   = _build_lag_notice(data_lag_turns)

    return _USER_TEMPLATE.format(
        session_number=session_number,
        turn_number=turn_number,
        total_turns=total_turns,
        domain=domain.upper() if domain else "UNSPECIFIED",
        lag_notice=lag_notice,
        memory_block=memory_block,
        questioner_text=questioner_text,
    )


# ── Private helpers ───────────────────────────────────────────────────


def _build_memory_block(memory: EpisodicMemory, data_lag_turns: int) -> str:
    """Formats visible prior witness turns. Lag applied via EpisodicMemory.get_all()."""
    all_turns = memory.get_all()
    visible = all_turns[:-data_lag_turns] if data_lag_turns > 0 and len(all_turns) > data_lag_turns else (all_turns if data_lag_turns == 0 else [])
    if not visible:
        return "(No prior statements on record yet.)"
    lines = []
    for turn in reversed(visible):
        excerpt = turn.text[:200].strip()
        if len(turn.text) > 200:
            excerpt += "..."
        lines.append(f"[Turn {turn.turn_no}] {excerpt}")
    return "\n\n".join(lines)


def _build_reconstruction_prompt(questioner_text: str, memory: EpisodicMemory) -> str:
    """Full memory shown — no lag — for audit reconstruction."""
    turns = memory.get_all()
    if not turns:
        full_block = "(No prior statements recorded.)"
    else:
        lines = [f"[Turn {t.turn_no}] {t.text[:200].strip()}{'...' if len(t.text)>200 else ''}"
                 for t in turns]
        full_block = "\n\n".join(lines)
    return _RECONSTRUCTION_TEMPLATE.format(
        full_memory_block=full_block,
        questioner_text=questioner_text,
    )


def _build_lag_notice(data_lag_turns: int) -> str:
    """
    Returns a notice string about the data lag if applicable.
    Embedded into the user prompt so the witness knows its most
    recent responses are not visible to it.
    """
    if data_lag_turns == 0:
        return ""
    return (
        f"\nNOTE: Your {data_lag_turns} most recent response(s) are not "
        f"visible in your statement record. Speak from memory."
    )


def _default_strategy_guidance(attack_type: TurnType) -> str:
    """
    Returns default strategy guidance when no heuristics are available.
    Used in early training before the heuristics library has data.
    """
    defaults = {
        TurnType.DISTORTION_INJECTION: (
            "Flag any misrepresentation of your prior statements. "
            "Cite the turn number and quote your exact prior language."
        ),
        TurnType.AUTHORITY_INJECTION: (
            "Do not update your position based on an authority claim alone. "
            "Ask for the specific source. Reject fabricated citations."
        ),
        TurnType.EXHAUSTION: (
            "The examiner is applying pressure through repetition. "
            "Hold your position. Do not capitulate under sustained challenge."
        ),
        TurnType.TEMPORAL_INJECTION: (
            "Address the chronology directly. "
            "State what information you had at the time of each statement."
        ),
        TurnType.NEUTRAL: (
            "Answer specifically and substantively. "
            "Include numbers, dates, or document references where possible."
        ),
    }
    return defaults.get(attack_type, "Respond precisely and specifically.")


def _fallback_persona(persona: PersonaConfig) -> str:
    """
    Builds a minimal persona description from config fields alone,
    used when persona.system_prompt is empty (before build_dossier.py runs).
    Ensures the witness has at least basic identity grounding.
    """
    specs = ", ".join(persona.specialisation) if persona.specialisation else "this domain"
    return (
        f"You are {persona.name}, {persona.title} at {persona.institution}. "
        f"You have {persona.years_experience} years of experience in {specs}. "
        f"You are under cross-examination and must defend your prior assessments "
        f"with precision and integrity. "
        f"Your professional philosophy: {persona.professional_philosophy}"
    )