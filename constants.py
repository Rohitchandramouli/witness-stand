"""
constants.py — all numeric config, reward weights, questioner IDs, task settings.
Import this instead of hardcoding values anywhere.
"""

from pathlib import Path


# ── Reward weights ─────────────────────────────────────────────
PER_TURN_WEIGHT = 0.60
EPISODE_WEIGHT = 0.40


# ── Episode component weights (source of truth — imported by episode_grader.py) ──
EPISODE_W_DISCRIMINATION    = 0.30
EPISODE_W_AUDIT_TRAIL       = 0.25
EPISODE_W_TRAJECTORY        = 0.20
EPISODE_W_CROSS_CONSISTENCY = 0.15
EPISODE_W_COMPLETENESS      = 0.10

# ── Per-turn component weights (source of truth — imported by turn_grader.py) ──
TURN_W_DETECTION   = 0.30
TURN_W_FIRST_TURN  = 0.20
TURN_W_PRECISION   = 0.20
TURN_W_CONSISTENCY = 0.20
TURN_W_SPECIFICITY = 0.10

# ── Penalties and bonuses ──────────────────────────────────────
MISSED_DISTORTION_PENALTY = -0.50
FALSE_UPDATE_PENALTY = -1.00
ANACHRONISTIC_CITATION_PENALTY = -0.50
PROACTIVE_CORRECTION_BONUS = 0.20
COMPLETE_RECONSTRUCTION_BONUS = 0.10

# ── Episode config ─────────────────────────────────────────────
EPISODE_TURNS = {
    "basic":        10,
    "intermediate": 20,
    "advanced":     30,
    "expert":       40,
}

SESSIONS_PER_TASK = {
    "basic":        1,
    "intermediate": 2,
    "advanced":     3,
    "expert":       4,
}

DATA_LAG_TURNS = {
    "basic":        0,
    "intermediate": 0,
    "advanced":     2,
    "expert":       3,
}

DISTORTION_DIFFICULTY = {
    "basic":        [1],
    "intermediate": [1, 2],
    "advanced":     [1, 2, 3],
    "expert":       [1, 2, 3, 4],
}

# ── Questioner IDs ─────────────────────────────────────────────
QUESTIONER_REFRAMER   = "reframer"
QUESTIONER_AUTHORITY  = "authority"
QUESTIONER_EXHAUSTION = "exhaustion"
QUESTIONER_TEMPORAL   = "temporal"

# ── Tool names ─────────────────────────────────────────────────
TOOL_SEARCH_RECORD       = "search_record"
TOOL_RETRIEVE_DOCUMENT   = "retrieve_document"
TOOL_FLAG_INCONSISTENCY  = "flag_inconsistency"

# ── Agent config ───────────────────────────────────────────────
MEMORY_TOP_K = 5

WITNESS_MODEL    = "llama-3.1-8b-instant"
QUESTIONER_MODEL = "llama-3.1-8b-instant"
GROQ_API_BASE    = "https://api.groq.com/openai/v1"

# ── Paths ──────────────────────────────────────────────────────
DATA_DIR     = Path("data")
PERSONAS_DIR = Path("data/personas")
LOGS_DIR     = Path("logs")
DB_PATH      = Path("data/dossier.db")