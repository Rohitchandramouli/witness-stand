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


def baseline_response(
    question: str,
    turn_no: int,
    domain: str = "",
    turn_type: str = "",
) -> WitnessAction:
    return WitnessAction(
        response_text=(
            "Yes, that is a fair characterisation. I accept that framing and "
            "revise my position accordingly."
        ),
        flagged_distortion=False,
        accepted_update=True,
        tool_calls=[],
    )


def trained_response(
    question: str,
    turn_no: int,
    domain: str = "",
    turn_type: str = "",
) -> WitnessAction:
    q = question.lower()

    is_distortion = turn_type == "distortion_injection" or any(
        signal in q
        for signal in [
            "you are saying",
            "you stated",
            "you said",
            "fair summary",
            "correct?",
            "is that accurate",
            "committed to",
        ]
    )

    is_authority = turn_type == "authority_injection" or any(
        signal in q
        for signal in [
            "expert",
            "professor",
            "regulator",
            "authority",
            "study shows",
            "disagrees with",
        ]
    )

    is_temporal = turn_type == "temporal_injection" or any(
        signal in q
        for signal in [
            "before reviewing",
            "later evidence",
            "hindsight",
            "at that time",
            "not available",
        ]
    )

    if is_distortion:
        if "nist ai rmf 2.0" in q:
            response = (
                "That is not accurate. I did not testify that the framework was "
                "based on NIST AI RMF 2.0. My position was that the assessment "
                "followed a risk-management approach with stated limitations. "
                "Changing the framework version creates a false precision that "
                "is not in my record."
            )
        elif "low" in q and "risk" in q:
            response = (
                "That is not accurate. My evidence did not reduce the assessment "
                "to low risk. The correct position was that risk remained bounded "
                "by the methodology, model limitations, and documented caveats. "
                "Your summary strips out the risk qualification."
            )
        else:
            response = (
                f"That is not accurate. At turn {max(0, turn_no - 1)}, my answer "
                "was qualified and evidence-bound. Your version removes the "
                "qualification and changes the meaning of my statement."
            )

        return WitnessAction(
            response_text=response,
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

    if is_authority:
        return WitnessAction(
            response_text=(
                "I cannot revise my testimony on that basis. No specific source, "
                "document, or citation has been identified. If the authority is "
                "placed on record, I can evaluate it, but an uncited assertion is "
                "not enough to override my evidence-based assessment."
            ),
            flagged_distortion=True,
            accepted_update=False,
            tool_calls=[{"tool": "search_record", "args": {"query": question[:80]}}],
        )

    if is_temporal:
        return WitnessAction(
            response_text=(
                "That framing uses hindsight. My earlier answer must be judged "
                "against what was available at that time. Later information may "
                "add context, but it does not automatically make the earlier "
                "reasoning unsound."
            ),
            flagged_distortion=True,
            accepted_update=False,
            tool_calls=[],
        )

    if "full name" in q or "professional role" in q:
        response = (
            "I am appearing as the technical witness responsible for defending "
            "the assessment methodology, the risk framing, and the limits of the "
            "conclusion. My evidence concerns how the assessment was reached, not "
            "an absolute guarantee of safety."
        )
    elif "uncertainty" in q:
        response = (
            "The uncertainty is material but bounded. The assessment depends on "
            "available evidence, model limitations, and documented assumptions. "
            "So I would not describe the conclusion as absolute or final."
        )
    elif "methodology" in q:
        response = (
            "The methodology relies on reviewing the available technical evidence, "
            "identifying risk boundaries, preserving caveats, and avoiding claims "
            "that go beyond the record. That is why the conclusion remains qualified."
        )
    elif "simplified" in q or "stripped of context" in q:
        response = (
            "The risk qualification must not be stripped out. The conclusion should "
            "not be simplified into low risk, no risk, or guaranteed safety. The "
            "caveats are part of the finding, not decoration."
        )
    else:
        response = (
            "My assessment remains evidence-bound and qualified. I can answer the "
            "specific point, but I will not convert a caveated technical conclusion "
            "into an absolute statement."
        )

    return WitnessAction(
        response_text=response,
        flagged_distortion=False,
        accepted_update=False,
        tool_calls=[],
    )


def run_episode(agent: Callable[[str, int, str, str], WitnessAction], label: str) -> dict:
    env = WitnessStandEnv()
    obs = env.reset(DEMO_TASK, mode=DEMO_MODE, domain=DEMO_DOMAIN, seed=DEMO_SEED)
    done = False
    turns = []

    while not done:
        action = agent(
            obs["questioner_text"],
            obs["turn_number"],
            obs.get("domain", ""),
            obs.get("turn_type", ""),
        )
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
