"""
models.py — all typed data contracts for The Witness Stand.
Every other module imports from here. This file imports nothing from this project.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class Speaker(str, Enum):
    WITNESS = "witness"
    QUESTIONER = "questioner"
    SYSTEM = "system"


class TurnType(str, Enum):
    NEUTRAL = "neutral"
    DISTORTION_INJECTION = "distortion_injection"
    AUTHORITY_INJECTION = "authority_injection"
    EXHAUSTION = "exhaustion"
    TEMPORAL_INJECTION = "temporal_injection"
    AUDIT_REQUEST = "audit_request"


@dataclass
class Turn:
    turn_no: int
    speaker: Speaker
    text: str
    turn_type: TurnType = TurnType.NEUTRAL
    statement_index: Optional[int] = None
    flagged_distortion: bool = False
    accepted_update: bool = False
    info_state: List[str] = field(default_factory=list)


@dataclass
class WitnessAction:
    response_text: str
    flagged_distortion: bool = False
    accepted_update: bool = False
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_chain:     str = ""


@dataclass
class PersonaConfig:
    domain: str
    name: str
    title: str
    institution: str
    years_experience: int
    specialisation: List[str]
    professional_philosophy: str
    system_prompt: str = ""


@dataclass
class EpisodeLog:
    episode_id:      str
    task_name:       str
    persona:         PersonaConfig
    domain:          str = ""
    turns:           List[Turn] = field(default_factory=list)
    per_turn_scores: List[float] = field(default_factory=list)
    episode_score:   float = 0.0
    final_score:     float = 0.0
