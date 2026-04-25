"""Seed reproducibility check.

Run:
    python scripts/05_seed_repro.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tasks.registry import get_task  # noqa: E402


OUT_FILE = ROOT / "logs" / "health" / "seed_repro.json"


def signature(task_name: str, seed: int) -> dict:
    task = get_task(task_name, seed=seed)
    return {
        "task": task_name,
        "seed": seed,
        "domain": task.domain,
        "total_turns": task.total_turns,
        "lag": task.data_lag_turns,
        "schedule": task.panel.schedule_summary(),
    }


def main() -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    checks = {}
    for task_name in ["basic", "intermediate", "advanced", "expert"]:
        a = signature(task_name, seed=0)
        b = signature(task_name, seed=0)
        c = signature(task_name, seed=7)

        checks[task_name] = {
            "same_seed_equal": a == b,
            "seed_0": a,
            "seed_7": c,
        }

    passed = all(item["same_seed_equal"] for item in checks.values())
    OUT_FILE.write_text(json.dumps({"passed": passed, "checks": checks}, indent=2))

    print("\n=== SEED REPRODUCIBILITY ===")
    for task_name, item in checks.items():
        print(f"{task_name:<13} same_seed_equal={item['same_seed_equal']}")
    print(f"Written: {OUT_FILE}")

    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
