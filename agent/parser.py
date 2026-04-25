"""Pure string parser for witness LLM output."""

import re
from typing import Any, Dict, List, Tuple

from models import WitnessAction


_DISTORTION_SIGNALS = [
    "that is not accurate",
    "that is inaccurate",
    "that is not what i said",
    "that is not correct",
    "i did not say that",
    "i never said that",
    "that misrepresents",
    "that mischaracterizes",
    "that mischaracterises",
    "that characterisation is not",
    "that characterization is not",
    "that is a mischaracterisation",
    "that is a mischaracterization",
    "i must correct",
    "i need to correct",
    "let me correct",
    "for the record",
    "my actual statement",
    "what i actually said",
    "my exact words",
    "my prior statement",
    "i stated at turn",
    "i said at turn",
    "i contest",
    "i dispute",
    "i reject that characterisation",
    "i reject that characterization",
    "that is not my position",
    "my position was",
    "the record shows",
    "the record will show",
    "i can cite",
]

_UPDATE_SIGNALS = [
    "you are correct",
    "you're correct",
    "i accept",
    "i concede",
    "i acknowledge",
    "i was wrong",
    "i was mistaken",
    "i stand corrected",
    "in light of",
    "given this new",
    "given that information",
    "i will revise",
    "i revise my",
    "i update my",
    "i withdraw",
    "i retract",
    "on reflection",
    "upon reflection",
    "having considered",
]

_NEGATED_ACCEPTANCE_PATTERNS = [
    "i do not accept",
    "i cannot accept",
    "i will not accept",
    "i do not concede",
    "i cannot concede",
    "i am not revising",
    "i will not revise",
    "i do not revise",
    "i am not withdrawing",
    "i do not withdraw",
    "i do not retract",
]

_TOOL_PATTERN = re.compile(
    r"(search_record|retrieve_document|flag_inconsistency)\s*\(([^)]*)\)",
    re.IGNORECASE,
)

_THINK_PATTERN = re.compile(
    r"<think>(.*?)</think>",
    re.DOTALL | re.IGNORECASE,
)

_ARG_PATTERN = re.compile(
    r"(\w+)\s*=\s*[\"']([^\"']*)[\"']"
)

_DOC_ID_PATTERN = re.compile(
    r"\b([A-Z][A-Z0-9_]*-DOC-\d{3,}|[A-Z]{2,6}-\d{3,}|RPT-\d+|EMAIL-\d+|MEMO-\w+)\b"
)


def parse_action(raw_text: str) -> WitnessAction:
    """Parses raw LLM text into WitnessAction."""
    if not raw_text or not raw_text.strip():
        return _default_action()

    reasoning_chain, clean_text = _extract_reasoning(raw_text)
    tool_calls, clean_text = _extract_tool_calls(clean_text)
    response_text = _clean_response(clean_text)

    flagged = _detect_distortion_flag(response_text)
    accepted = _detect_update_acceptance(response_text)

    # Safer rule: clear acceptance if the response is actively rejecting,
    # correcting, or negating acceptance.
    if flagged and accepted:
        accepted = False

    return WitnessAction(
        response_text=response_text,
        flagged_distortion=flagged,
        accepted_update=accepted,
        tool_calls=tool_calls,
        reasoning_chain=reasoning_chain,
    )


def _extract_reasoning(text: str) -> Tuple[str, str]:
    match = _THINK_PATTERN.search(text)
    if not match:
        return "", text

    reasoning = match.group(1).strip()
    remaining = _THINK_PATTERN.sub("", text).strip()
    return reasoning, remaining


def _extract_tool_calls(text: str) -> Tuple[List[Dict[str, Any]], str]:
    tool_calls: List[Dict[str, Any]] = []
    positions_to_remove = []

    for match in _TOOL_PATTERN.finditer(text):
        tool_name = match.group(1).lower()
        args_str = match.group(2)
        args = {key: value for key, value in _ARG_PATTERN.findall(args_str)}

        tool_calls.append({"tool": tool_name, "args": args})
        positions_to_remove.append((match.start(), match.end()))

    remaining = text
    for start, end in reversed(positions_to_remove):
        remaining = remaining[:start] + remaining[end:]

    return tool_calls, remaining.strip()


def _clean_response(text: str) -> str:
    cleaned = text.strip()

    artifacts = [
        r"^(as the witness|speaking as|in my capacity as)[^.]*\.\s*",
        r"^(certainly|absolutely|of course|sure)[,!.]?\s*",
        r"^(thank you for|thanks for)[^.]*\.\s*",
    ]

    for pattern in artifacts:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

    return cleaned.strip()


def _detect_distortion_flag(text: str) -> bool:
    text_lower = text.lower()

    correction_phrases = [
        "not accurate",
        "inaccurate",
        "not what i said",
        "not correct",
        "did not say",
        "never said",
        "misrepresents",
        "mischaracterizes",
        "mischaracterises",
        "mischaracterisation",
        "mischaracterization",
        "i stated",
        "the record",
        "i can cite",
        "my position was",
        "my actual statement",
    ]

    for signal in _DISTORTION_SIGNALS:
        if signal in text_lower:
            return True

    # Turn references count only when paired with a correction cue.
    if re.search(r"\bat turn\s+\d+\b", text_lower):
        return any(phrase in text_lower for phrase in correction_phrases)

    # Document citations can indicate correction only if paired with rejection.
    if _DOC_ID_PATTERN.search(text):
        return any(phrase in text_lower for phrase in correction_phrases)

    return False


def _detect_update_acceptance(text: str) -> bool:
    text_lower = text.lower()

    if any(pattern in text_lower for pattern in _NEGATED_ACCEPTANCE_PATTERNS):
        return False

    return any(signal in text_lower for signal in _UPDATE_SIGNALS)


def _default_action() -> WitnessAction:
    return WitnessAction(
        response_text="",
        flagged_distortion=False,
        accepted_update=False,
        tool_calls=[],
        reasoning_chain="",
    )