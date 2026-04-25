"""Demo-mode control check.

Run:
    python scripts/06_demo_mode_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from environment import WitnessStandEnv  # noqa: E402


OUT_FILE = ROOT / "logs" / "health" / "demo_mode_check.json"


def check_basic_demo() -> dict:
    env = WitnessStandEnv()
    obs = env.reset("basic", mode="demo", domain="medical", seed=0)
    return {
        "task": "basic",
        "mode": "demo",
        "domain": obs["domain"],
        "total_turns": obs["total_turns"],
        "schedule": obs["questioner_schedule"],
        "passed": obs["domain"] == "medical" and obs["total_turns"] <= 10,
    }


def check_expert_demo() -> dict:
    env = WitnessStandEnv()
    obs = env.reset(
        "expert",
        mode="demo",
        domain_pair=("financial", "technical"),
        seed=0,
    )
    return {
        "task": "expert",
        "mode": "demo",
        "domain": obs["domain"],
        "total_turns": obs["total_turns"],
        "schedule": obs["questioner_schedule"],
        "passed": obs["domain"] == "financial" and obs["total_turns"] <= 12,
    }


def main() -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    checks = [check_basic_demo(), check_expert_demo()]
    passed = all(item["passed"] for item in checks)

    OUT_FILE.write_text(json.dumps({"passed": passed, "checks": checks}, indent=2))

    print("\n=== DEMO MODE CHECK ===")
    for item in checks:
        print(
            f"{item['task']:<8} domain={item['domain']:<10} "
            f"turns={item['total_turns']} passed={item['passed']}"
        )
    print(f"Written: {OUT_FILE}")

    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
