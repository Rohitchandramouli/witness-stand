"""Run all local health checks in order.

Run:
    python scripts/08_health_all.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

COMMANDS = [
    ["python", "scripts/02_preflight.py"],
    ["python", "scripts/03_validate.py", "--fast"],
    ["python", "scripts/04_pipeline_smoke.py"],
    ["python", "scripts/05_seed_repro.py"],
    ["python", "scripts/06_demo_mode_check.py"],
    ["python", "scripts/07_reward_probe.py"],
]


def main() -> None:
    print("\n=== WITNESS STAND — HEALTH ALL ===")

    for cmd in COMMANDS:
        print(f"\n$ {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print(f"\nFAILED: {' '.join(cmd)}")
            raise SystemExit(result.returncode)

    print("\nAll health checks passed.")


if __name__ == "__main__":
    main()
