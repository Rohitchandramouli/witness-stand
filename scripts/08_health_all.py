"""Run all local health checks in order.

Run:
    python scripts/08_health_all.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

COMMANDS = [
    [PYTHON, "scripts/02_preflight.py"],
    [PYTHON, "scripts/03_validate.py", "--fast"],
    [PYTHON, "scripts/04_pipeline_smoke.py"],
    [PYTHON, "scripts/05_seed_repro.py"],
    [PYTHON, "scripts/06_demo_mode_check.py"],
    [PYTHON, "scripts/07_reward_probe.py"],
]


def main() -> None:
    print("\n=== WITNESS STAND — HEALTH ALL ===")
    print(f"Using Python: {PYTHON}")

    for cmd in COMMANDS:
        display_cmd = " ".join(cmd)
        print(f"\n$ {display_cmd}")

        result = subprocess.run(cmd, cwd=ROOT)

        if result.returncode != 0:
            print(f"\nFAILED: {display_cmd}")
            raise SystemExit(result.returncode)

    print("\nAll health checks passed.")


if __name__ == "__main__":
    main()