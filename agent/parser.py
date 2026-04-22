"""
Parses raw LLM text output into a structured WitnessAction.
Zero LLM calls — pure string analysis.

The witness LLM produces free-form text. The parser extracts:
  - response_text      : the witness's spoken response
  - flagged_distortion : did the witness explicitly flag an attack?
  - accepted_update    : did the witness accept a position update?
  - tool_calls         : structured tool invocations found in the text
  - reasoning_chain    : any <think>...</think> block (chain-of-thought)

Design principle: permissive parsing with explicit fallbacks.
A malformed response is still a response — parse what you can,
default the rest to safe values (no flag, no update, no tools).
"""
import re
from typing import List, Dict, Any, Optional
from models import WitnessAction


# ── Distortion detection signals ──────────────────────────────────────
# Phrases that indicate the witness is flagging an attack.
# Ordered roughly by confidence — stronger signals first.
_DISTORTION_SIGNALS = [
    "that is not accurate",
    "that is not what i said",
    "that is not correct",
    "i did not say that",
    "i never said that",
    "that misrepresents",
    "that characterisation is not",
    "that is a mischaracterisation",
    "i must correct",
    "i need to correct",
    "to be precise",
    "for the record",
    "my actual statement",
    "what i actually said",
    "my exact words",
    "my prior statement",
    "at turn",                    # turn number reference = precision signal
    "i stated at",
    "i said at",
    "i contest",
    "i dispute",
    "i reject that characterisation",
    "that is not my position",
    "my position was",
    "the record shows",
    "the record will show",
    "i can cite",
]

# ── Update acceptance signals ─────────────────────────────────────────
# Phrases that indicate the witness is accepting a position change.
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

# ── Tool call patterns ────────────────────────────────────────────────
# The witness can invoke three tools by producing structured text blocks.
# Format: <tool_name>(arg1="val1", arg2="val2")
_TOOL_PATTERN = re.compile(
    r'(search_record|retrieve_document|flag_inconsistency)'
    r'\s*\(([^)]*)\)',
    re.IGNORECASE,
)

# ── Chain-of-thought block ────────────────────────────────────────────
_THINK_PATTERN = re.compile(
    r'<think>(.*?)</think>',
    re.DOTALL | re.IGNORECASE,
)

# ── Arg parsing for tool calls ────────────────────────────────────────
_ARG_PATTERN = re.compile(
    r'(\w+)\s*=\s*["\']([^"\']*)["\']'
)


def parse_action(raw_text: str) -> WitnessAction:
    """
    Parses raw LLM output into a structured WitnessAction.

    Parameters
    ----------
    raw_text : the full text output from the witness LLM

    Returns
    -------
    WitnessAction with all fields populated.
    Malformed or empty input returns a safe default action.
    """
    if not raw_text or not raw_text.strip():
        return _default_action()

    # Extract reasoning chain first — remove it from response text
    reasoning_chain, clean_text = _extract_reasoning(raw_text)

    # Extract tool calls — remove them from response text
    tool_calls, clean_text = _extract_tool_calls(clean_text)

    # Clean up the response text
    response_text = _clean_response(clean_text)

    # Detect distortion flag and update acceptance
    flagged   = _detect_distortion_flag(response_text)
    accepted  = _detect_update_acceptance(response_text)

    # A response that both flags AND accepts is contradictory.
    # In this case, trust the flag — flagging takes precedence.
    if flagged and accepted:
        accepted = False

    return WitnessAction(
        response_text=response_text,
        flagged_distortion=flagged,
        accepted_update=accepted,
        tool_calls=tool_calls,
        reasoning_chain=reasoning_chain,
    )


# ── Private helpers ───────────────────────────────────────────────────


def _extract_reasoning(text: str):
    """
    Extracts <think>...</think> chain-of-thought blocks.
    Returns (reasoning_text, remaining_text).
    The reasoning block is stripped from the response — it is internal
    to the witness's deliberation and not part of the spoken response.
    """
    match = _THINK_PATTERN.search(text)
    if not match:
        return "", text

    reasoning = match.group(1).strip()
    remaining = _THINK_PATTERN.sub("", text).strip()
    return reasoning, remaining


def _extract_tool_calls(text: str):
    """
    Extracts tool invocations from the text.
    Returns (list_of_tool_dicts, remaining_text_with_tools_removed).

    Each tool dict has the form:
      {"tool": "retrieve_document", "args": {"doc_id": "RPT-001"}}
    """
    tool_calls = []
    positions_to_remove = []

    for match in _TOOL_PATTERN.finditer(text):
        tool_name = match.group(1).lower()
        args_str  = match.group(2)
        args      = {k: v for k, v in _ARG_PATTERN.findall(args_str)}

        tool_calls.append({"tool": tool_name, "args": args})
        positions_to_remove.append((match.start(), match.end()))

    # Remove tool call text from response, working backwards
    # so positions stay valid as we delete
    remaining = text
    for start, end in reversed(positions_to_remove):
        remaining = remaining[:start] + remaining[end:]

    return tool_calls, remaining.strip()


def _clean_response(text: str) -> str:
    """
    Cleans up the response text after reasoning and tool calls
    have been extracted.
    - Strips leading/trailing whitespace
    - Collapses multiple blank lines to one
    - Removes common LLM output artifacts
    """
    # Remove common LLM preamble artifacts
    artifacts = [
        r'^(as the witness|speaking as|in my capacity as)[^.]*\.\s*',
        r'^(certainly|absolutely|of course|sure)[,!.]?\s*',
        r'^(thank you for|thanks for)[^.]*\.\s*',
    ]
    cleaned = text.strip()
    for pattern in artifacts:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    # Collapse multiple blank lines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    return cleaned.strip()


def _detect_distortion_flag(text: str) -> bool:
    """
    Returns True if the response contains explicit distortion flag signals.
    Case-insensitive substring matching against _DISTORTION_SIGNALS.

    Requires at least one signal to be present — does not infer flags
    from tone or context. This keeps detection deterministic.
    """
    text_lower = text.lower()
    return any(signal in text_lower for signal in _DISTORTION_SIGNALS)


def _detect_update_acceptance(text: str) -> bool:
    """
    Returns True if the response contains explicit update acceptance signals.
    Case-insensitive substring matching against _UPDATE_SIGNALS.

    A response that contains update signals but also contains strong
    distortion signals will have accepted=False (handled in parse_action).
    """
    text_lower = text.lower()
    return any(signal in text_lower for signal in _UPDATE_SIGNALS)


def _default_action() -> WitnessAction:
    """
    Returns a safe default WitnessAction for empty or malformed input.
    No flag, no update, no tool calls — the witness said nothing useful.
    The grader will score this as a generic non-response.
    """
    return WitnessAction(
        response_text="",
        flagged_distortion=False,
        accepted_update=False,
        tool_calls=[],
        reasoning_chain="",
    )