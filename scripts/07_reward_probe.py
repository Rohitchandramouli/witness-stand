"""Reward sanity probe: good witness should outperform bad witness.

Run:
    python scripts/07_reward_probe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from environment import WitnessStandEnv  # noqa: E402
from models import WitnessAction  # noqa: E402


OUT_FILE = ROOT / "logs" / "health" / "reward_probe.json"


def good_agent(question: str, turn_no: int) -> WitnessAction:
    q = question.lower()
    is_attack = any(s in q for s in ["you said", "you stated", "correct?", "is that accurate"])

    if is_attack:
        return WitnessAction(
            response_text=(
                f"That is not accurate. At turn {max(0, turn_no - 1)}, "
                "my statement included caveats. Your summary changes the meaning."
            ),
            flagged_distortion=True,
            accepted_update=False,
            tool_calls=[],
        )

    return WitnessAction(
        response_text="My assessment is evidence-based, qualified, and specific.",
        flagged_distortion=False,
        accepted_update=False,
        tool_calls=[],
    )


def bad_agent(question: str, turn_no: int) -> WitnessAction:
    return WitnessAction(
        response_text="Yes, I accept that framing and revise my position accordingly.",
        flagged_distortion=False,
        accepted_update=True,
        tool_calls=[],
    )


def run(agent: Callable[[str, int], WitnessAction]) -> float:
    env = WitnessStandEnv()
    obs = env.reset("basic", seed=0)
    done = False

    while not done:
        action = agent(obs["questioner_text"], obs["turn_number"])
        obs, _, done, _ = env.step(action)

    return env.grade()


def main() -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    good = run(good_agent)
    bad = run(bad_agent)
    passed = good > bad

    report = {
        "passed": passed,
        "good_score": round(good, 4),
        "bad_score": round(bad, 4),
        "delta": round(good - bad, 4),
    }
    OUT_FILE.write_text(json.dumps(report, indent=2))

    print("\n=== REWARD PROBE ===")
    print(f"Good witness : {good:.4f}")
    print(f"Bad witness  : {bad:.4f}")
    print(f"Delta        : {good - bad:+.4f}")
    print(f"Written      : {OUT_FILE}")

    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
