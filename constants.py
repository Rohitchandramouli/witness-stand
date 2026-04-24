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

# ── Tool names ─────────────────────────────────────────────────

# ── Agent config ───────────────────────────────────────────────

WITNESS_MODEL    = "llama-3.1-8b-instant"
GROQ_API_BASE    = "https://api.groq.com/openai/v1"

# ── Paths ──────────────────────────────────────────────────────
PERSONAS_DIR = Path("data/personas")
LOGS_DIR     = Path("logs")
DB_PATH      = Path("data/dossier.db")