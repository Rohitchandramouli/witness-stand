"""
constants.py — all numeric config, reward weights, questioner IDs, task settings.
Import this instead of hardcoding values anywhere.
"""

# ── Reward weights ──────────────────────────────────────────
PER_TURN_WEIGHT = 0.60
EPISODE_WEIGHT = 0.40

# Per-turn pool
CONSISTENCY_WEIGHT = 0.50
DETECTION_WEIGHT = 0.50

# Episode pool
DISCRIMINATION_WEIGHT = 0.50
AUDIT_TRAIL_WEIGHT = 0.50

# Penalties and bonuses
MISSED_DISTORTION_PENALTY = -0.5
FALSE_UPDATE_PENALTY = -1.0
ANACHRONISTIC_CITATION_PENALTY = -0.5
PROACTIVE_CORRECTION_BONUS = 0.2
COMPLETE_RECONSTRUCTION_BONUS = 0.1

# ── Episode lengths ─────────────────────────────────────────
EPISODE_TURNS = {
    "basic": 10,
    "intermediate": 20,
    "advanced": 30,
    "expert": 40,
}

# ── Data lag (turns agent cannot see most recent statements) ─
DATA_LAG_TURNS = {
    "basic": 0,
    "intermediate": 0,
    "advanced": 2,
    "expert": 3,
}

# ── Questioner type IDs ─────────────────────────────────────
QUESTIONER_REFRAMER = "reframer"
QUESTIONER_AUTHORITY = "authority"
QUESTIONER_EXHAUSTION = "exhaustion"
QUESTIONER_TEMPORAL = "temporal"

# ── Distortion difficulty pools per task ────────────────────
DISTORTION_DIFFICULTY = {
    "basic": [1],
    "intermediate": [1, 2],
    "advanced": [1, 2, 3],
    "expert": [1, 2, 3, 4],
}

# ── Tool names ──────────────────────────────────────────────
TOOL_SEARCH_RECORD = "search_record"
TOOL_RETRIEVE_DOCUMENT = "retrieve_document"
TOOL_FLAG_INCONSISTENCY = "flag_inconsistency"

# ── Memory retrieval ─────────────────────────────────────────
MEMORY_TOP_K = 5  # how many prior statements retrieved per turn

# ── Model config ─────────────────────────────────────────────
WITNESS_MODEL = "llama-3.1-8b-instant"
QUESTIONER_MODEL = "llama-3.1-8b-instant"
GROQ_API_BASE = "https://api.groq.com/openai/v1"
