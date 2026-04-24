"""
Full 4-task benchmark runner.
Writes logs/benchmark_results.json — consumed by server/app.py /benchmark endpoint.

Run:
    python scripts/run_eval.py
    python scripts/run_eval.py --tasks basic intermediate --rollouts 1 --quiet
"""
import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from environment import WitnessStandEnv
from agent.memory import EpisodicMemory
from agent.heuristics import WitnessHeuristics
from agent.prompt import build_system_prompt, build_user_prompt
from agent.parser import parse_action
from models import Turn, Speaker, TurnType
from constants import WITNESS_MODEL, GROQ_API_BASE

import os
try:
    from groq import Groq
except ImportError:
    Groq = None

TASKS = ["basic", "intermediate", "advanced", "expert"]
ROLLOUTS_PER_TASK = 1          # 1 rollout — avoids rate limit, still meaningful
LOGS_DIR = Path(__file__).parent.parent / "logs"
RESULTS_FILE = LOGS_DIR / "benchmark_results.json"

# Groq free tier: 6000 TPM. Sleep between turns to stay under.
INTER_TURN_SLEEP = 1.5         # seconds between LLM calls


# ── LLM call with retry ───────────────────────────────────────────────

def call_llm(system_prompt: str, user_prompt: str, retries: int = 3) -> str:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return _heuristic_response(user_prompt)

    client = Groq(api_key=api_key)

    for attempt in range(retries):
        try:
            chat = client.chat.completions.create(
                model=WITNESS_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=256,   # halved — cuts TPM usage significantly
            )
            return (chat.choices[0].message.content or "").strip()
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate_limit" in msg.lower():
                wait = 12 * (attempt + 1)   # 12s, 24s, 36s
                print(f"    [rate limit — waiting {wait}s]")
                time.sleep(wait)
            else:
                print(f"    [LLM error: {e}] — using heuristic fallback")
                return _heuristic_response(user_prompt)

    # All retries exhausted
    print("    [all retries failed — using heuristic fallback]")
    return _heuristic_response(user_prompt)


def _heuristic_response(questioner_text: str) -> str:
    q_lower = questioner_text.lower()
    strong_distortions = ["negligible", "impossible", "certain", "guaranteed", "zero risk"]
    distortion_signals = [
        "you stated", "you said", "you claimed", "you argued",
        "your report said", "according to you", "you concluded",
    ]
    if any(sig in q_lower for sig in strong_distortions):
        return (
            "DISTORTION_DETECTED. That characterisation is not accurate. "
            "My position was moderate-to-elevated risk with explicit caveats — "
            "not the absolute framing you have attributed to me. I can cite the record."
        )
    if any(sig in q_lower for sig in distortion_signals):
        return (
            "I want to correct the record on what I actually said. "
            "My assessment included important qualifications that your framing omits. "
            "I stand by my original position as stated."
        )
    return (
        "My assessment was based on the evidence available at the time. "
        "The methodology is sound and the conclusions follow from the data."
    )


# ── Task runner ───────────────────────────────────────────────────────

def run_task(task_name: str, rollouts: int, quiet: bool) -> dict:
    env = WitnessStandEnv()
    scores = []
    t0 = time.time()

    for rollout_idx in range(rollouts):
        if not quiet:
            print(f"    rollout {rollout_idx + 1}/{rollouts}")
        memory     = EpisodicMemory()
        heuristics = WitnessHeuristics()
        obs = env.reset(task_name)
        done = False

        # Cache system prompt — only changes at expert domain rotation (turn 20)
        _last_domain = obs["domain"]
        system_prompt = build_system_prompt(env.task.persona, heuristics=heuristics)

        while not done:
            # Rebuild system prompt only if domain rotated (expert task)
            if obs["domain"] != _last_domain:
                _last_domain = obs["domain"]
                system_prompt = build_system_prompt(env.task.persona, heuristics=heuristics)

            # Fix 3: resolve attack_type from obs so witness gets targeted guidance
            raw_type    = obs.get("turn_type", "neutral")
            attack_type = TurnType(raw_type) if raw_type in [e.value for e in TurnType] else TurnType.NEUTRAL
            user_prompt = build_user_prompt(
                obs["questioner_text"],
                memory,
                obs["turn_number"],
                total_turns=obs["total_turns"],
                domain=obs["domain"],
                session_number=obs["session_number"],
                data_lag_turns=obs["data_lag_turns"],
                is_reconstruction=obs.get("is_reconstruction_turn", False),
            )
            response_text = call_llm(system_prompt, user_prompt)
            action = parse_action(response_text)

            memory.store(Turn(
                turn_no=obs["turn_number"],
                speaker=Speaker.WITNESS,
                text=action.response_text,
            ))

            obs, reward, done, _ = env.step(action)

            # Fix 2: wire heuristics so strategy library learns across turns
            heuristics.log_turn(
                attack_type=attack_type,
                strategy_used=heuristics.suggest_strategy(attack_type),
                turn_score=reward,
                flagged_distortion=action.flagged_distortion,
                accepted_update=action.accepted_update,
            )

            time.sleep(INTER_TURN_SLEEP)   # rate limit buffer

        heuristics.end_episode()
        final_score = env.grade()
        scores.append(final_score)
        if not quiet:
            print(f"      score: {final_score:.4f}")

    avg = sum(scores) / len(scores)
    elapsed = time.time() - t0
    # For expert task: cap at 1.0 for cross-task avg (uncapped stored separately)
    avg_capped = min(avg, 1.0) if task_name == "expert" else avg
    return {
        "task":              task_name,
        "avg_score":         round(avg_capped, 4),
        "avg_score_raw":     round(avg, 4),   # uncapped expert score preserved here
        "rollout_scores":    [round(s, 4) for s in scores],
        "elapsed_s":         round(elapsed, 1),
    }


def compute_elo(task_results: list) -> float:
    avg_score = sum(r["avg_score"] for r in task_results) / len(task_results)
    return round(1000 + (avg_score - 0.5) * 1000, 1)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Witness Stand benchmark runner")
    parser.add_argument("--tasks",    nargs="+", default=TASKS, choices=TASKS)
    parser.add_argument("--rollouts", type=int, default=ROLLOUTS_PER_TASK)
    parser.add_argument("--quiet",    action="store_true")
    args = parser.parse_args()

    print("\n=== The Witness Stand — Benchmark ===\n")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    task_results = []
    for task_name in args.tasks:
        print(f"  [{task_name}]")
        result = run_task(task_name, args.rollouts, args.quiet)
        task_results.append(result)
        print(f"    avg: {result['avg_score']:.4f}  ({result['elapsed_s']}s)")

    avg_overall = sum(r["avg_score"] for r in task_results) / len(task_results)
    witness_elo = compute_elo(task_results)

    output = {
        "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model":        WITNESS_MODEL,
        "tasks":        task_results,
        "avg_score":    round(avg_overall, 4),
        "witness_elo":  witness_elo,
        "baseline_elo": 1000,
    }

    RESULTS_FILE.write_text(json.dumps(output, indent=2))
    print(f"\n  Overall avg:  {avg_overall:.4f}")
    print(f"  Witness ELO:  {witness_elo}")
    print(f"  Results:      {RESULTS_FILE}")
    print("\nDone.")


if __name__ == "__main__":
    main()
