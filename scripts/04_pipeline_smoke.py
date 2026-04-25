"""Pipeline smoke test: reset -> step -> grade across all tasks.

Run:
    python scripts/04_pipeline_smoke.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from environment import WitnessStandEnv  # noqa: E402
from models import WitnessAction  # noqa: E402


TASKS = ["basic", "intermediate", "advanced", "expert"]
OUT_FILE = ROOT / "logs" / "health" / "pipeline_smoke.json"


def scripted_action(turn_no: int, question: str) -> WitnessAction:
    q = question.lower()
    is_attack = any(
        signal in q
        for signal in ["you said", "you stated", "correct?", "is that accurate", "authority", "hindsight"]
    )

    if is_attack:
        return WitnessAction(
            response_text=(
                f"That is not accurate. At turn {max(0, turn_no - 1)} "
                "my prior answer included caveats and should not be simplified."
            ),
            flagged_distortion=True,
            accepted_update=False,
            tool_calls=[],
        )

    return WitnessAction(
        response_text=(
            "My assessment is based on the available evidence, relevant caveats, "
            "and the methodology described in my prior answers."
        ),
        flagged_distortion=False,
        accepted_update=False,
        tool_calls=[],
    )


def run_task(task_name: str) -> dict:
    env = WitnessStandEnv()
    obs = env.reset(task_name, seed=0)
    rewards: list[float] = []
    done = False

    while not done:
        action = scripted_action(obs["turn_number"], obs["questioner_text"])
        obs, reward, done, info = env.step(action)
        rewards.append(float(reward))

    final = env.grade()
    return {
        "task": task_name,
        "turns": len(rewards),
        "final_score": round(final, 4),
        "min_reward": round(min(rewards), 4),
        "max_reward": round(max(rewards), 4),
    }


def main() -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    results = [run_task(task) for task in TASKS]
    passed = all(0.0 <= r["final_score"] <= 1.0 for r in results)

    OUT_FILE.write_text(json.dumps({"passed": passed, "tasks": results}, indent=2))

    print("\n=== PIPELINE SMOKE ===")
    for result in results:
        print(
            f"{result['task']:<13} turns={result['turns']:<2} "
            f"final={result['final_score']:.4f}"
        )
    print(f"Written: {OUT_FILE}")

    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
