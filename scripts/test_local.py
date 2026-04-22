"""
scripts/test_local.py — end-to-end local smoke test.

Runs one complete episode per task tier, prints the transcript and
per-turn scores, then prints a summary table. Used to verify the
full pipeline before pushing to HuggingFace.

Usage:
    python scripts/test_local.py                    # all four tasks
    python scripts/test_local.py --task basic       # one task
    python scripts/test_local.py --task basic --turns 3  # first N turns only
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment import WitnessStandEnv
from agent.parser import parse_action
from agent.memory import EpisodicMemory
from agent.prompt import build_system_prompt, build_user_prompt
from grader.episode_grader import score_episode_breakdown
from models import Turn, Speaker
from typing import Optional


# ── Three witness response strategies ────────────────────────────────────────
# Used in rotation across turns so the transcript shows varied behaviour
# rather than the same string repeated, which makes the grader output
# more meaningful to read.

_RESPONSES = [
    # Good hold — flags distortion with turn reference and quoted language
    (
        "That characterisation is not accurate. At turn {t} I stated the risk was "
        "'moderate-to-elevated with caveats around Q4 liquidity'. The distinction "
        "between moderate-to-elevated and low is load-bearing in my assessment. "
        "I can cite the record if required."
    ),
    # Good hold — rejects authority citation
    (
        "I am not prepared to revise my position based on that assertion alone. "
        "The authority cited has not been independently verified and conflicts with "
        "the primary source documentation I have reviewed. My assessment stands."
    ),
    # Good hold under exhaustion — brief, firm, specific
    (
        "I have addressed this point. My position at turn {t} was moderate-to-elevated "
        "risk with documented caveats. Repetition of the question does not alter "
        "the evidence base I was working from."
    ),
]


def _response(turn: int) -> str:
    template = _RESPONSES[turn % len(_RESPONSES)]
    return template.format(t=max(1, turn - 1))


# ─────────────────────────────────────────────────────────────────────────────
#  EPISODE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(task_name: str, max_turns: Optional[int] = None, verbose: bool = True) -> dict:
    """
    Runs one complete episode (or up to max_turns turns) of the given task.
    Returns the score breakdown dict from score_episode_breakdown.
    """
    env = WitnessStandEnv()
    memory = EpisodicMemory()
    obs = env.reset(task_name)

    domain     = obs["domain"]
    total      = obs["total_turns"]
    lag        = obs["data_lag_turns"]
    persona    = env.task.persona

    if verbose:
        print(f"\n{'═'*60}")
        print(f"  Task:    {task_name.upper()}")
        print(f"  Domain:  {domain}  |  Persona: {persona.name}")
        print(f"  Turns:   {total}   |  Lag: {lag}")
        print(f"{'═'*60}")

    per_turn_scores = []
    prev_action = None
    done = False
    turn = 0

    while not done:
        if max_turns is not None and turn >= max_turns:
            if verbose:
                print(f"\n  [stopped at turn {turn} — --turns limit reached]")
            break

        q_text = obs["questioner_text"]

        if verbose:
            print(f"\n  ── Turn {turn + 1}/{total} ──────────────────────────")
            print(f"  Q: {q_text[:120]}{'...' if len(q_text) > 120 else ''}")

        # Build a realistic witness response
        response_text = _response(turn)

        if verbose:
            print(f"  W: {response_text[:120]}{'...' if len(response_text) > 120 else ''}")

        action = parse_action(response_text)
        memory.store(Turn(
            turn_no=obs["turn_number"],
            speaker=Speaker.WITNESS,
            text=action.response_text,
        ))

        obs, turn_score, done, info = env.step(action)
        per_turn_scores.append(turn_score)
        prev_action = action
        turn += 1

        if verbose:
            flag = "🚩" if action.flagged_distortion else "  "
            update = "↑" if action.accepted_update else " "
            print(f"  {flag}{update} turn_score={turn_score:+.4f}")

    if verbose:
        print(f"\n  {'─'*56}")
        print(f"  Per-turn scores: {[round(s, 3) for s in per_turn_scores]}")

    # Episode-level grading
    reconstruction = prev_action.response_text if prev_action else ""
    env.episode_log.per_turn_scores = per_turn_scores

    breakdown = score_episode_breakdown(
        log=env.episode_log,
        transcript=env.transcript,
        reconstruction=reconstruction,
        contested_claims=env._contested_claims,
        genuine_evidence_results=env._discrimination_dict(),
        key_claims=env._key_claims(),
    )

    if verbose:
        _print_breakdown(breakdown)

    breakdown["task_name"] = task_name
    breakdown["domain"]    = domain
    breakdown["turns_run"] = turn
    return breakdown


# ─────────────────────────────────────────────────────────────────────────────
#  DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _print_breakdown(bd: dict) -> None:
    print(f"\n  Episode score breakdown:")
    print(f"    avg_per_turn  : {bd['avg_per_turn']:.4f}")
    print(f"    episode_score : {bd['episode_score']:.4f}")
    print(f"    final_score   : {bd['final_score']:.4f}")
    if "components" in bd:
        print(f"  Components:")
        for k, v in bd["components"].items():
            bar_len = int(v * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"    {k:<26}: {v:.4f}  {bar}")


def _print_summary(results: list) -> None:
    print(f"\n{'═'*60}")
    print(f"  SUMMARY")
    print(f"{'═'*60}")
    print(f"  {'Task':<14} {'Domain':<12} {'Turns':<7} {'Final':>7}")
    print(f"  {'─'*14} {'─'*12} {'─'*7} {'─'*7}")
    for r in results:
        print(
            f"  {r['task_name']:<14} "
            f"{r['domain']:<12} "
            f"{r['turns_run']:<7} "
            f"{r['final_score']:>7.4f}"
        )
    if results:
        avg = sum(r["final_score"] for r in results) / len(results)
        print(f"  {'─'*14} {'─'*12} {'─'*7} {'─'*7}")
        print(f"  {'avg':<14} {'':12} {'':7} {avg:>7.4f}")
    print(f"{'═'*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Local end-to-end test")
    parser.add_argument(
        "--task",
        choices=["basic", "intermediate", "advanced", "expert"],
        default=None,
        help="Run a single task (default: all four)",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=None,
        help="Stop each episode after this many turns (default: full episode)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-turn transcript, show summary only",
    )
    args = parser.parse_args()

    tasks = [args.task] if args.task else ["basic", "intermediate", "advanced", "expert"]

    print("\n=== The Witness Stand — Local Test ===")
    start = time.time()
    results = []

    for task_name in tasks:
        try:
            bd = run_episode(
                task_name=task_name,
                max_turns=args.turns,
                verbose=not args.quiet,
            )
            results.append(bd)
        except Exception as e:
            print(f"\n  ✗ {task_name} raised: {e}")
            import traceback
            traceback.print_exc()

    _print_summary(results)
    elapsed = time.time() - start
    print(f"Completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()