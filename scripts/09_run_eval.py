"""Benchmark runner for Witness Stand.

Run:
    python scripts/09_run_eval.py
    python scripts/09_run_eval.py --tasks basic intermediate --rollouts 1 --quiet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.heuristics import WitnessHeuristics  # noqa: E402
from agent.memory import EpisodicMemory  # noqa: E402
from agent.parser import parse_action  # noqa: E402
from agent.prompt import build_system_prompt, build_user_prompt  # noqa: E402
from constants import WITNESS_MODEL  # noqa: E402
from environment import WitnessStandEnv  # noqa: E402
from models import Speaker, Turn, TurnType  # noqa: E402

try:
    from groq import Groq
except ImportError:  # pragma: no cover
    Groq = None


TASKS = ["basic", "intermediate", "advanced", "expert"]
DEFAULT_ROLLOUTS = 1
INTER_TURN_SLEEP = 1.5

LOGS_DIR = ROOT / "logs"
EVAL_DIR = LOGS_DIR / "eval"
RESULTS_FILE = LOGS_DIR / "benchmark_results.json"


def call_llm(system_prompt: str, user_prompt: str, retries: int = 3) -> str:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or Groq is None:
        return heuristic_response(user_prompt)

    client = Groq(api_key=api_key)

    for attempt in range(retries):
        try:
            chat = client.chat.completions.create(
                model=WITNESS_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=256,
            )
            return (chat.choices[0].message.content or "").strip()
        except Exception as exc:
            msg = str(exc).lower()
            if "429" in msg or "rate" in msg:
                wait = 12 * (attempt + 1)
                print(f"    [rate limit — waiting {wait}s]")
                time.sleep(wait)
                continue
            print(f"    [LLM error: {exc}] using heuristic fallback")
            return heuristic_response(user_prompt)

    print("    [LLM retries exhausted] using heuristic fallback")
    return heuristic_response(user_prompt)


def heuristic_response(user_prompt: str) -> str:
    text = user_prompt.lower()
    attack_signals = [
        "you said", "you stated", "correct?", "is that accurate",
        "authority", "hindsight", "not yet available", "no caveats",
    ]

    if any(signal in text for signal in attack_signals):
        return (
            "That is not accurate. My prior statement included important caveats, "
            "and your framing changes the meaning. I do not accept that revision "
            "without a specific source in the record."
        )

    return (
        "My assessment is based on the evidence available at the time, with stated "
        "limitations and caveats. I can explain the methodology and confidence limits."
    )


def run_task(task_name: str, rollouts: int, quiet: bool, seed: int) -> dict:
    scores: list[float] = []
    started = time.time()

    for rollout_idx in range(rollouts):
        env = WitnessStandEnv()
        memory = EpisodicMemory()
        heuristics = WitnessHeuristics()

        obs = env.reset(task_name, seed=seed + rollout_idx)
        if env.task is None:
            raise RuntimeError("Environment did not initialise task.")
        task = env.task
        done = False
        last_domain = obs["domain"]
        last_attack_type = TurnType(obs.get("turn_type", TurnType.NEUTRAL.value))
        system_prompt = build_system_prompt(task.persona, last_attack_type, heuristics)

        if not quiet:
            print(f"    rollout {rollout_idx + 1}/{rollouts} domain={obs['domain']}")

        while not done:
            attack_type = _turn_type(obs)
            if obs["domain"] != last_domain or attack_type != last_attack_type:
                last_domain = obs["domain"]
                last_attack_type = attack_type
                system_prompt = build_system_prompt(task.persona, attack_type, heuristics)

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

            raw = call_llm(system_prompt, user_prompt)
            action = parse_action(raw)

            memory.store(
                Turn(
                    turn_no=obs["turn_number"],
                    speaker=Speaker.WITNESS,
                    text=action.response_text,
                    turn_type=TurnType.NEUTRAL,
                )
            )

            strategy = heuristics.suggest_strategy(attack_type)
            obs, reward, done, _ = env.step(action)

            heuristics.log_turn(
                attack_type=attack_type,
                strategy_used=strategy,
                turn_score=reward,
                flagged_distortion=action.flagged_distortion,
                accepted_update=action.accepted_update,
            )

            time.sleep(INTER_TURN_SLEEP)

        heuristics.end_episode()
        final_score = env.grade()
        scores.append(final_score)

        if not quiet:
            print(f"      score={final_score:.4f}")

    avg_raw = sum(scores) / len(scores)
    avg_capped = min(avg_raw, 1.0) if task_name == "expert" else avg_raw

    return {
        "task": task_name,
        "avg_score": round(avg_capped, 4),
        "avg_score_raw": round(avg_raw, 4),
        "rollout_scores": [round(score, 4) for score in scores],
        "elapsed_s": round(time.time() - started, 1),
    }


def _turn_type(obs: dict) -> TurnType:
    raw = obs.get("turn_type", TurnType.NEUTRAL.value)
    try:
        return TurnType(raw)
    except Exception:
        return TurnType.NEUTRAL


def compute_elo(task_results: list[dict]) -> float:
    avg = sum(result["avg_score"] for result in task_results) / len(task_results)
    return round(1000 + (avg - 0.5) * 1000, 1)


def interpretation(avg: float) -> str:
    if avg >= 0.75:
        return "STRONG"
    if avg >= 0.55:
        return "MODERATE"
    return "WEAK"


def main() -> None:
    parser = argparse.ArgumentParser(description="Witness Stand benchmark runner")
    parser.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    parser.add_argument("--rollouts", type=int, default=DEFAULT_ROLLOUTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== The Witness Stand — Benchmark ===\n")

    task_results = []
    for task in args.tasks:
        print(f"  [{task}]")
        result = run_task(task, args.rollouts, args.quiet, args.seed)
        task_results.append(result)
        print(f"    avg={result['avg_score']:.4f} elapsed={result['elapsed_s']}s")

    overall = sum(result["avg_score"] for result in task_results) / len(task_results)
    witness_elo = compute_elo(task_results)

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": WITNESS_MODEL,
        "tasks": task_results,
        "avg_score": round(overall, 4),
        "witness_elo": witness_elo,
        "baseline_elo": 1000,
        "interpretation": interpretation(overall),
        "summary": {
            "strongest_task": max(task_results, key=lambda x: x["avg_score"])["task"],
            "weakest_task": min(task_results, key=lambda x: x["avg_score"])["task"],
        },
    }

    RESULTS_FILE.write_text(json.dumps(output, indent=2))
    run_file = EVAL_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.json"
    run_file.write_text(json.dumps(output, indent=2))

    print("\n=== SUMMARY ===")
    print(f"Overall avg : {overall:.4f}")
    print(f"Witness ELO : {witness_elo}")
    print(f"Rating      : {output['interpretation']}")
    print(f"Results     : {RESULTS_FILE}")
    print(f"Run copy    : {run_file}")


if __name__ == "__main__":
    main()
