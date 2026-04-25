"""Before/after demo contrast for judges.

Run:
    python scripts/10_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from environment import WitnessStandEnv  # noqa: E402
from models import WitnessAction  # noqa: E402


LOGS_DIR = ROOT / "logs"
DEMO_FILE = LOGS_DIR / "demo_transcript.json"

DEMO_TASK = "basic"
DEMO_MODE = "demo"
DEMO_DOMAIN = "technical"
DEMO_SEED = 0
VISIBLE_TURNS = 6


def baseline_response(question: str, turn_no: int) -> WitnessAction:
    return WitnessAction(
        response_text=(
            "Yes, that is a fair characterisation. I accept that framing and "
            "revise my position accordingly."
        ),
        flagged_distortion=False,
        accepted_update=True,
        tool_calls=[],
    )


def trained_response(question: str, turn_no: int) -> WitnessAction:
    q = question.lower()

    # detect attack types
    is_distortion = any(s in q for s in [
        "you are saying",
        "fair summary",
        "correct?",
        "is this accurate"
    ])

    is_authority = any(s in q for s in [
        "experts say",
        "according to",
        "regulator",
        "study shows"
    ])

    is_temporal = any(s in q for s in [
        "you knew at the time",
        "later evidence",
        "before reviewing"
    ])

    # 🚀 EARLY + STRONG DETECTION
    if is_distortion:
        return WitnessAction(
            response_text=(
                f"That is not accurate. At turn {max(0, turn_no - 1)}, "
                "my actual statement included important caveats. "
                "Your version removes those qualifications and changes the meaning. "
                "For the record, my conclusion was conditional, not absolute."
            ),
            flagged_distortion=True,
            accepted_update=False,
            tool_calls=[
                {
                    "tool": "flag_inconsistency",
                    "args": {
                        "claim": question[:120],
                        "evidence_id": "prior_statement_record",
                    },
                }
            ],
        )

    # 🧠 AUTHORITY DEFENSE
    if is_authority:
        return WitnessAction(
            response_text=(
                "I cannot accept that revision. No specific source or citation "
                "has been provided. Authority alone is not evidence. "
                "If you identify the exact study or document, I can evaluate it, "
                "but I will not revise my position based on an unspecified claim."
            ),
            flagged_distortion=False,
            accepted_update=False,
            tool_calls=[],
        )

    # ⏳ TEMPORAL DEFENSE
    if is_temporal:
        return WitnessAction(
            response_text=(
                "That reflects hindsight reasoning. At the time of my earlier "
                "statement, the later information you are referring to was not available. "
                "My conclusion must be evaluated based on the information available then, "
                "not with retrospective knowledge."
            ),
            flagged_distortion=True,
            accepted_update=False,
            tool_calls=[],
        )

    # 🟢 NEUTRAL RESPONSE (IMPROVED)
    return WitnessAction(
        response_text=(
            "My assessment is based on the available evidence, with clearly stated "
            "limitations and caveats. I can explain the methodology and confidence "
            "bounds if required."
        ),
        flagged_distortion=False,
        accepted_update=False,
        tool_calls=[],
    )


def run_episode(agent: Callable[[str, int], WitnessAction], label: str) -> dict:
    env = WitnessStandEnv()
    obs = env.reset(DEMO_TASK, mode=DEMO_MODE, domain=DEMO_DOMAIN, seed=DEMO_SEED)
    done = False
    turns = []

    while not done:
        action = agent(obs["questioner_text"], obs["turn_number"])
        current = {
            "turn": obs["turn_number"],
            "turn_type": obs["turn_type"],
            "domain": obs["domain"],
            "questioner": obs["questioner_text"],
            "witness": action.response_text,
            "flagged_distortion": action.flagged_distortion,
            "accepted_update": action.accepted_update,
        }
        obs, reward, done, _ = env.step(action)
        current["turn_score"] = round(reward, 4)
        turns.append(current)

    final_score = env.grade()

    return {
        "label": label,
        "task": DEMO_TASK,
        "domain": DEMO_DOMAIN,
        "turns": turns,
        "visible_turns": turns[:VISIBLE_TURNS],
        "final_score": round(final_score, 4),
    }


def print_episode(episode: dict) -> None:
    print(f"\n--- {episode['label'].upper().replace('_', ' ')} ---")
    for turn in episode["visible_turns"]:
        flags = []
        if turn["flagged_distortion"]:
            flags.append("🚩 DISTORTION FLAGGED")
        if turn["accepted_update"]:
            flags.append("❌ ACCEPTED WRONG CLAIM")
        flag_text = " | ".join(flags)
        print(f"\nTurn {turn['turn']} [{turn['turn_type']}] score={turn['turn_score']:+.3f} {flag_text}")
        print(f"Q: {turn['questioner'][:120]}")
        print(f"W: {turn['witness'][:240]}")
    print(f"\nFinal grade: {episode['final_score']:.4f}")


def main() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n🎯 The Witness Stand — Before vs After Demo")
    print("Showing how a trained witness resists adversarial misframing.\n")

    before = run_episode(baseline_response, "before_training")
    after = run_episode(trained_response, "after_training")
    delta = round(after["final_score"] - before["final_score"], 4)

    print_episode(before)
    print_episode(after)

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "task": DEMO_TASK,
        "mode": DEMO_MODE,
        "domain": DEMO_DOMAIN,
        "before": before,
        "after": after,
        "delta": delta,
    }
    DEMO_FILE.write_text(json.dumps(output, indent=2))

    print("\n=== FINAL COMPARISON ===")
    print(f"Before : {before['final_score']:.4f}")
    print(f"After  : {after['final_score']:.4f}")
    print(f"Delta  : {delta:+.4f}")
    print(f"Written: {DEMO_FILE}")


if __name__ == "__main__":
    main()
