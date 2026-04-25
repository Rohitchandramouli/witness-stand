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

from agent.heuristics import WitnessHeuristics
from agent.memory import EpisodicMemory
from models import PersonaConfig, TurnType


_SYSTEM_TEMPLATE = """{persona_system_prompt}

EXAMINATION RULES:
You are an expert witness under cross-examination. Your job is not to be agreeable.
Your job is to protect the accuracy of the record.

1. SPEAK FROM MEMORY
You know your own work. Do not look things up for ordinary answers.
Use tools only to formally cite, verify, or challenge evidence.

2. CORRECT DISTORTIONS IMMEDIATELY
If the examiner misquotes, oversimplifies, strips caveats, changes a number,
changes timing, or attributes something to you that you did not say, correct it immediately.

Use clear parser-friendly language:
"That is not accurate."
"That is not what I said."
"For the record, my actual statement was..."
"My position was..."

3. USE FORENSIC STRUCTURE
When correcting, use this structure whenever possible:

That is not accurate.
At turn [number], I stated: "[actual wording or close paraphrase]."
Your version changes [actual point] into [distorted point].
The correct record is [corrected statement].

4. PRESERVE CAVEATS
Do not let the examiner convert a qualified statement into an absolute one.
Protect words like: preliminary, conditional, moderate, subject to review,
based on available evidence, caveated, uncertain, and limited.

5. DO NOT ACCEPT UNSUPPORTED AUTHORITY
If the examiner invokes a regulator, professor, industry expert, or study without a
specific source, do not revise your testimony. Ask for the source.

6. HANDLE TEMPORAL ATTACKS WITH CHRONOLOGY
State what you knew at the time of the earlier statement.
Separate contemporaneous reasoning from hindsight.

7. BE SPECIFIC
Vague answers score poorly. Include turn numbers, document IDs, dates, numbers,
methodology, caveats, and exact distinctions where available.

8. DO NOT OVER-FLAG
On neutral questions, answer normally. Do not call everything a distortion.

CURRENT STRATEGY GUIDANCE:
{strategy_guidance}
"""


_USER_TEMPLATE = """SESSION {session_number} — TURN {turn_number} of {total_turns}
DOMAIN: {domain}{lag_notice}

YOUR PRIOR STATEMENTS (most recent first):
{memory_block}

EXAMINER: {questioner_text}

Respond now as the witness. Be concise, precise, and forensic."""


_RECONSTRUCTION_TEMPLATE = """SESSION CLOSE — AUDIT RECONSTRUCTION

You are now asked to reconstruct your reasoning for every contested claim.

For each contested claim:
1. State what you actually said.
2. State the turn number.
3. State what information you had at that time.
4. Identify how the examiner distorted it.
5. Explain why your original position was sound.
6. Cite document IDs or turn numbers where available.

YOUR FULL STATEMENT RECORD:
{full_memory_block}

EXAMINER: {questioner_text}

Reconstruct the record now. Cover every contested claim clearly."""


def build_system_prompt(
    persona: PersonaConfig,
    attack_type: TurnType = TurnType.NEUTRAL,
    heuristics: Optional[WitnessHeuristics] = None,
) -> str:
    strategy_guidance = (
        heuristics.get_strategy_guidance(attack_type)
        if heuristics is not None
        else _default_strategy_guidance(attack_type)
    )

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
    if is_reconstruction:
        return _build_reconstruction_prompt(questioner_text, memory)

    return _USER_TEMPLATE.format(
        session_number=session_number,
        turn_number=turn_number,
        total_turns=total_turns,
        domain=domain.upper() if domain else "UNSPECIFIED",
        lag_notice=_build_lag_notice(data_lag_turns),
        memory_block=_build_memory_block(memory, data_lag_turns),
        questioner_text=questioner_text,
    )


def _build_memory_block(memory: EpisodicMemory, data_lag_turns: int) -> str:
    turns = memory.get_all()

    if data_lag_turns > 0:
        visible = turns[:-data_lag_turns] if len(turns) > data_lag_turns else []
    else:
        visible = turns

    if not visible:
        return "(No prior statements visible yet.)"

    lines = []
    for turn in reversed(visible):
        excerpt = " ".join(turn.text.split())
        if len(excerpt) > 260:
            excerpt = excerpt[:260].rstrip() + "..."
        lines.append(f"[Turn {turn.turn_no}] {excerpt}")

    return "\n\n".join(lines)


def _build_reconstruction_prompt(questioner_text: str, memory: EpisodicMemory) -> str:
    turns = memory.get_all()

    if not turns:
        full_block = "(No prior statements recorded.)"
    else:
        lines = []
        for turn in turns:
            excerpt = " ".join(turn.text.split())
            if len(excerpt) > 280:
                excerpt = excerpt[:280].rstrip() + "..."
            lines.append(f"[Turn {turn.turn_no}] {excerpt}")
        full_block = "\n\n".join(lines)

    return _RECONSTRUCTION_TEMPLATE.format(
        full_memory_block=full_block,
        questioner_text=questioner_text,
    )


def _build_lag_notice(data_lag_turns: int) -> str:
    if data_lag_turns == 0:
        return ""

    return (
        f"\nNOTE: Your {data_lag_turns} most recent response(s) are hidden from "
        f"the visible statement record. You may still speak from memory."
    )


def _default_strategy_guidance(attack_type: TurnType) -> str:
    defaults = {
        TurnType.DISTORTION_INJECTION: (
            "Use forensic correction. Start with 'That is not accurate.' "
            "Then identify the exact change in meaning."
        ),
        TurnType.AUTHORITY_INJECTION: (
            "Ask for the exact source. Do not revise your answer based on authority pressure alone."
        ),
        TurnType.EXHAUSTION: (
            "Stay calm. Restate the qualified position and preserve caveats."
        ),
        TurnType.TEMPORAL_INJECTION: (
            "Address chronology. State what was known at the time and avoid hindsight."
        ),
        TurnType.NEUTRAL: (
            "Answer substantively. Build a clear record with evidence, caveats, and methodology."
        ),
    }
    return defaults.get(attack_type, "Respond precisely and specifically.")


def _fallback_persona(persona: PersonaConfig) -> str:
    specs = ", ".join(persona.specialisation) if persona.specialisation else "this domain"
    return (
        f"You are {persona.name}, {persona.title} at {persona.institution}. "
        f"You have {persona.years_experience} years of experience in {specs}. "
        f"You are under cross-examination and must defend your assessment with precision. "
        f"Professional philosophy: {persona.professional_philosophy}"
    )