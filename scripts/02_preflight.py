"""Preflight check for local hackathon readiness.

Run:
    python scripts/02_preflight.py
"""

from __future__ import annotations

import importlib
import json
import os
import sys
print("Python path:", sys.executable)

from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from constants import DB_PATH, PERSONAS_DIR  # noqa: E402

from dotenv import load_dotenv
load_dotenv()


LOG_DIR = ROOT / "logs" / "health"
OUT_FILE = LOG_DIR / "preflight.json"

REQUIRED_IMPORTS = [
    "environment",
    "tasks.registry",
    "questioners.panel",
    "transcript.store",
    "grader.turn_grader",
    "grader.episode_grader",
    "agent.memory",
    "agent.prompt",
    "agent.parser",
    "agent.heuristics",
]

REQUIRED_PERSONAS = ["financial", "medical", "safety", "technical"]


def check_imports() -> dict[str, bool]:
    result: dict[str, bool] = {}
    for module in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module)
            result[module] = True
        except Exception as exc:
            print(f"[FAIL] import {module}: {exc}")
            result[module] = False
    return result


def check_files() -> dict[str, Any]:
    personas = {
        domain: (PERSONAS_DIR / f"{domain}.json").exists()
        for domain in REQUIRED_PERSONAS
    }

    return {
        "db_exists": DB_PATH.exists(),
        "personas_dir_exists": PERSONAS_DIR.exists(),
        "personas": personas,
        "logs_dir_exists": (ROOT / "logs").exists(),
        "health_dir_exists": LOG_DIR.exists(),
        "groq_key_present": bool(os.getenv("GROQ_API_KEY")),
    }


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "imports": check_imports(),
        "files": check_files(),
    }

    passed = (
        all(report["imports"].values())
        and report["files"]["db_exists"]
        and all(report["files"]["personas"].values())
    )

    report["passed"] = passed
    OUT_FILE.write_text(json.dumps(report, indent=2))

    print("\n=== PREFLIGHT ===")
    print(f"Imports OK       : {all(report['imports'].values())}")
    print(f"DB exists        : {report['files']['db_exists']}")
    print(f"Personas OK      : {all(report['files']['personas'].values())}")
    print(f"GROQ key present : {report['files']['groq_key_present']}")
    print(f"Written          : {OUT_FILE}")

    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
