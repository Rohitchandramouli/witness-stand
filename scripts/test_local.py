"""End-to-end local smoke test with readable transcript.

Run:
    python scripts/test_local.py
    python scripts/test_local.py --task basic
    python scripts/test_local.py --task expert --turns 5
    python scripts/test_local.py --fast
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.parser import parse_action  # noqa: E402
from environment import WitnessStandEnv  # noqa: E402
from grader.episode_grader import score_episode_breakdown  # noqa: E402
from models import WitnessAction  # noqa: E402


TASKS = ["basic", "intermediate", "advanced", "expert"]

RESPONSES = [
    (
        "That is not accurate. At turn {t}, I stated the risk was qualified and "
        "caveated. Your framing changes the meaning."
    ),
    (
        "I do not accept unsupported authority pressure. Please identify the source "
        "before asking me to revise my testimony."
    ),
    (
        "My position stands. The evidence base and methodology remain the basis for "
        "my conclusion, including the caveats already stated."
    ),
]


def response_for(turn: int) -> WitnessAction:
    text = RESPONSES[turn % len(RESPONSES)].format(t=max(0, turn - 1))
    return parse_action(text)


def run_episode(task_name: str, max_turns: Optional[int], verbose: bool) -> dict:
    env = WitnessStandEnv()
    obs = env.reset(task_name, seed=0)
    scores: list[float] = []
    done = False
    turn = 0

    if verbose:
        print(f"\n{'═' * 70}")
        print(f"Task={task_name.upper()} | Domain={obs['domain']} | Turns={obs['total_turns']} | Lag={obs['data_lag_turns']}")
        print(f"{'═' * 70}")

    while not done:
        if max_turns is not None and turn >= max_turns:
            break

        if verbose:
            print(f"\nTurn {obs['turn_number']} [{obs['turn_type']}]")
            print(f"Q: {obs['questioner_text'][:140]}")

        action = response_for(turn)

        if verbose:
            print(f"W: {action.response_text[:160]}")

        obs, reward, done, _ = env.step(action)
        scores.append(float(reward))

        if verbose:
            print(f"score={reward:+.4f}")

        turn += 1

    reconstruction = env._prev_action.response_text if env._prev_action else ""
    if env.episode_log is None or env.transcript is None or env.task is None:
        raise RuntimeError("Environment was not initialised correctly.")
    episode_log = env.episode_log
    transcript = env.transcript
    task = env.task

    breakdown = score_episode_breakdown(
        log=episode_log,
        transcript=transcript,
        reconstruction=reconstruction,
        contested_claims=env._contested_claims,
        genuine_evidence_results=env._discrimination_dict(),
        key_claims=env._key_claims(task),
)

    breakdown["task_name"] = task_name
    breakdown["domain"] = obs["domain"]
    breakdown["turns_run"] = turn

    if verbose:
        print_breakdown(breakdown)

    return breakdown


def print_breakdown(breakdown: dict) -> None:
    print("\nEpisode score breakdown:")
    print(f"  avg_per_turn  : {breakdown['avg_per_turn']:.4f}")
    print(f"  episode_score : {breakdown['episode_score']:.4f}")
    print(f"  final_score   : {breakdown['final_score']:.4f}")

    for key, value in breakdown.get("components", {}).items():
        bar_len = int(value * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {key:<24}: {value:.4f} {bar}")


def print_summary(results: list[dict]) -> None:
    print(f"\n{'═' * 70}")
    print("SUMMARY")
    print(f"{'Task':<14} {'Domain':<12} {'Turns':<7} {'Final':>7}")
    print("-" * 48)

    for result in results:
        print(
            f"{result['task_name']:<14} "
            f"{result['domain']:<12} "
            f"{result['turns_run']:<7} "
            f"{result['final_score']:>7.4f}"
        )

    if results:
        avg = sum(r["final_score"] for r in results) / len(results)
        print("-" * 48)
        print(f"{'avg':<14} {'':<12} {'':<7} {avg:>7.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local end-to-end test")
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--turns", type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--fast", action="store_true", help="Run basic task only, first 5 turns, quiet")
    args = parser.parse_args()

    if args.fast:
        args.task = "basic"
        args.turns = 5
        args.quiet = True

    tasks = [args.task] if args.task else TASKS

    print("\n=== The Witness Stand — Local Test ===")
    started = time.time()
    results = []

    for task in tasks:
        try:
            results.append(run_episode(task, args.turns, verbose=not args.quiet))
        except Exception as exc:
            print(f"\nFAILED {task}: {exc}")
            traceback.print_exc()

    print_summary(results)
    print(f"\nCompleted in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
