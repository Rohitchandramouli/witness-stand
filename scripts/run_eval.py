"""
Full 4-task benchmark with ELO tracking.
Run: python scripts/run_eval.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment import WitnessStandEnv
from agent.memory import EpisodicMemory
from agent.prompt import build_system_prompt, build_user_prompt
from agent.parser import parse_action
from grader.episode_grader import score_episode
from constants import WITNESS_MODEL, GROQ_API_BASE
import os

TASKS = ["basic", "intermediate", "advanced", "expert"]
ROLLOUTS_PER_TASK = 2


def run_task(task_name: str) -> float:
    env = WitnessStandEnv()
    scores = []
    for _ in range(ROLLOUTS_PER_TASK):
        memory = EpisodicMemory()
        obs = env.reset(task_name)
        done = False
        while not done:
            system_prompt = build_system_prompt(env.task.persona)
            user_prompt = build_user_prompt(
                obs["questioner_text"], memory, obs["turn_number"]
            )
            # TODO: wire to Groq API for real LLM call
            response_text = f"[Placeholder response for {task_name} turn {obs['turn_number']}]"
            action = parse_action(response_text)
            from models import Turn, Speaker, TurnType
            memory.store(Turn(
                turn_no=obs["turn_number"],
                speaker=Speaker.WITNESS,
                text=action.response_text,
            ))
            obs, reward, done, _ = env.step(action)
        final_score = score_episode(env.episode_log)
        scores.append(final_score)
    avg = sum(scores) / len(scores)
    return avg


def main():
    print("\n=== The Witness Stand — Benchmark ===\n")
    results = {}
    for task in TASKS:
        print(f"Running {task}...")
        score = run_task(task)
        results[task] = score
        print(f"  {task}: {score:.4f}")
    avg = sum(results.values()) / len(results)
    print(f"\n  Average: {avg:.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
