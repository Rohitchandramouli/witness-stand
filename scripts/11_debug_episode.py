"""Interactive debug episode.

Run:
    python scripts/11_debug_episode.py --task basic
    python scripts/11_debug_episode.py --task expert --demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.parser import parse_action  # noqa: E402
from environment import WitnessStandEnv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive Witness Stand episode")
    parser.add_argument("--task", default="basic", choices=["basic", "intermediate", "advanced", "expert"])
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = WitnessStandEnv()
    kwargs = {"mode": "demo" if args.demo else "train", "seed": args.seed}
    obs = env.reset(args.task, **kwargs)

    print("\n=== DEBUG EPISODE ===")
    print(f"Task: {args.task} | Domain: {obs['domain']} | Turns: {obs['total_turns']}")
    print("Type your witness answer. Use Ctrl+C to quit.\n")

    done = False
    while not done:
        print(f"\nTurn {obs['turn_number']}/{obs['total_turns']} [{obs['turn_type']}]")
        print(f"Q: {obs['questioner_text']}")
        raw = input("\nWitness > ")
        action = parse_action(raw)
        obs, reward, done, info = env.step(action)
        print(f"Reward: {reward:+.4f}")

    print(f"\nFinal grade: {env.grade():.4f}")


if __name__ == "__main__":
    main()
