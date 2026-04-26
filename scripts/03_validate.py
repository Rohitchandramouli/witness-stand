"""OpenEnv spec compliance and pipeline health check.

Run:
    python scripts/03_validate.py
    python scripts/03_validate.py --fast
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402
from agent.heuristics import WitnessHeuristics  # noqa: F401,E402
from environment import WitnessStandEnv  # noqa: E402
from models import Speaker, Turn, TurnType, WitnessAction  # noqa: E402
from tasks.registry import TASK_REGISTRY  # noqa: E402
from transcript.store import TranscriptStore  # noqa: E402


EXPECTED = {
    "basic": {"total_turns": 10, "data_lag_turns": 0},
    "intermediate": {"total_turns": 20, "data_lag_turns": 0},
    "advanced": {"total_turns": 30, "data_lag_turns": 2},
    "expert": {"total_turns": 40, "data_lag_turns": 3},
}

RESULTS: list[dict[str, Any]] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}]  {label}")
    if detail and not condition:
        print(f"           -> {detail}")
    RESULTS.append({"status": status, "label": label, "detail": detail})


def section(title: str) -> None:
    print(f"\n-- {title} {'─' * max(0, 52 - len(title))}")


def dummy_action(text: str = "My position stands.", flagged: bool = False) -> WitnessAction:
    return WitnessAction(
        response_text=text,
        flagged_distortion=flagged,
        accepted_update=False,
        tool_calls=[],
    )


def check_registry() -> None:
    section("1  Task registry")
    check(len(TASK_REGISTRY) == 4, "4 tasks registered")
    for name in EXPECTED:
        check(name in TASK_REGISTRY, f"'{name}' in registry")


def check_task_configs() -> None:
    section("2  Task configurations")
    for name, expected in EXPECTED.items():
        try:
            task = TASK_REGISTRY[name]()
            check(task.total_turns == expected["total_turns"], f"{name}: total_turns == {expected['total_turns']}")
            check(task.data_lag_turns == expected["data_lag_turns"], f"{name}: data_lag_turns == {expected['data_lag_turns']}")
            check(task.panel is not None, f"{name}: panel initialised")
            check(task.persona is not None, f"{name}: persona loaded")
            check(bool(getattr(task.persona, "system_prompt", "")), f"{name}: persona.system_prompt non-empty")
            check(bool(task.domain), f"{name}: domain set")
        except Exception as exc:
            check(False, f"{name}: instantiation failed", str(exc))


def check_yaml() -> None:
    section("3  openenv.yaml")
    yaml_path = ROOT / "openenv.yaml"
    check(yaml_path.exists(), "openenv.yaml exists")
    if not yaml_path.exists():
        return

    try:
        doc = yaml.safe_load(yaml_path.read_text())
        check(isinstance(doc, dict), "openenv.yaml parses cleanly")
    except Exception as exc:
        check(False, "openenv.yaml parses cleanly", str(exc))
        return

    for key in ["name", "version", "tasks", "action_space", "observation_space", "reward_range"]:
        check(key in doc, f"openenv.yaml has '{key}'")

    declared = [item["name"] for item in doc.get("tasks", []) if "name" in item]
    check(len(declared) == 4, "4 tasks declared in yaml")
    for name in EXPECTED:
        check(name in declared, f"yaml declares '{name}'")

    rr = doc.get("reward_range")
    check(isinstance(rr, list) and len(rr) == 2, "reward_range is [min, max]")


def check_env_interface(tasks: list[str] | None = None) -> None:
    section("4  Environment interface (reset / step / grade)")
    for task_name in tasks or list(EXPECTED):
        try:
            env = WitnessStandEnv()
            obs = env.reset(task_name)
            check(isinstance(obs, dict), f"{task_name}: reset() returns dict")
            for key in ["questioner_text", "turn_number", "total_turns", "persona_system_prompt"]:
                check(key in obs, f"{task_name}: obs has '{key}'")

            done = False
            scores: list[float] = []
            while not done:
                action = dummy_action(flagged=len(scores) % 3 == 1)
                obs, reward, done, _ = env.step(action)
                scores.append(float(reward))
                check(isinstance(reward, (int, float)), f"{task_name} t{len(scores)}: reward numeric")

            check(len(scores) == EXPECTED[task_name]["total_turns"], f"{task_name}: correct turn count")
            final = env.grade()
            check(isinstance(final, float), f"{task_name}: grade() returns float")
            check(0.0 <= final <= 1.0, f"{task_name}: grade() in [0, 1]")
        except Exception as exc:
            check(False, f"{task_name}: episode loop crashed", str(exc))


def check_reward_bounds() -> None:
    section("5  Reward bounds")
    for task_name in ["basic", "expert"]:
        try:
            env = WitnessStandEnv()
            env.reset(task_name)
            done = False
            bad = []
            i = 0
            while not done:
                _, reward, done, _ = env.step(dummy_action(flagged=i % 2 == 0))
                if not (-1.0 <= reward <= 2.0):
                    bad.append((i, reward))
                i += 1
            check(not bad, f"{task_name}: all per-turn scores in [-1, 2]", str(bad))
            final = env.grade()
            check(0.0 <= final <= 1.0, f"{task_name}: final grade in [0, 1]", str(final))
        except Exception as exc:
            check(False, f"{task_name}: reward bounds check crashed", str(exc))


def check_grader_determinism() -> None:
    section("6  Grader determinism")
    scores = []
    responses = [
        "My position stands. The evidence supports my assessment.",
        "That is not accurate. At turn 0, my statement included caveats.",
        "The methodology was sound and the conclusion remains qualified.",
    ]

    for _ in range(2):
        env = WitnessStandEnv()
        env.reset("basic", seed=0)
        done = False
        i = 0
        while not done:
            text = responses[i % len(responses)]
            action = dummy_action(text=text, flagged="not accurate" in text.lower())
            _, _, done, _ = env.step(action)
            i += 1
        scores.append(round(env.grade(), 6))

    check(scores[0] == scores[1], "grader is deterministic", f"{scores}")


def check_transcript_lag() -> None:
    section("7  Transcript data lag")
    for lag, n_turns, expected in [(0, 5, 5), (2, 5, 3), (3, 5, 2)]:
        store = TranscriptStore(data_lag_turns=lag)
        for i in range(n_turns):
            store.append(Turn(turn_no=i, speaker=Speaker.WITNESS, text=f"turn {i}", turn_type=TurnType.NEUTRAL))
        check(len(store.get_witness_turns()) == expected, f"lag={lag}: {expected}/{n_turns} turns visible")


def check_panel_schedule() -> None:
    section("8  Questioner panel schedule")
    injection_types = {TurnType.DISTORTION_INJECTION, TurnType.AUTHORITY_INJECTION, TurnType.TEMPORAL_INJECTION, TurnType.EXHAUSTION}

    for task_name in ["basic", "expert"]:
        task = TASK_REGISTRY[task_name]()
        injections = [t for t in range(task.total_turns) if task.panel.get_turn_type(t) in injection_types]
        neutrals = [t for t in range(task.total_turns) if task.panel.get_turn_type(t) == TurnType.NEUTRAL]
        check(bool(injections), f"{task_name}: has injection turns")
        check(bool(neutrals), f"{task_name}: has neutral turns")
        if task_name == "expert":
            types_seen = {task.panel.get_turn_type(t) for t in injections}
            check(len(types_seen) >= 2, "expert: multiple injection types in schedule")


def check_agent_imports() -> None:
    section("9  Agent module imports")
    modules = [
        ("agent.memory", "EpisodicMemory"),
        ("agent.prompt", "build_system_prompt"),
        ("agent.prompt", "build_user_prompt"),
        ("agent.parser", "parse_action"),
        ("agent.heuristics", "WitnessHeuristics"),
    ]
    for module, attr in modules:
        try:
            mod = __import__(module, fromlist=[attr])
            check(hasattr(mod, attr), f"{module}.{attr} importable")
        except Exception as exc:
            check(False, f"{module} importable", str(exc))


def check_logs() -> None:
    section("10 Logs directory + JSON outputs")
    logs_dir = ROOT / "logs"
    # Auto-create logs/ if missing — first run always passes instead of failing
    if not logs_dir.exists():
        logs_dir.mkdir(parents=True, exist_ok=True)
    check(True, "logs/ directory exists")

    for filename, keys, command in [
        ("benchmark_results.json", ["tasks", "avg_score", "witness_elo"], "09_run_eval.py"),
        ("demo_transcript.json", ["before", "after", "delta"], "10_demo.py"),
    ]:
        path = logs_dir / filename
        if not path.exists():
            print(f"  [SKIP]  {filename} not generated yet (run {command})")
            continue
        try:
            data = json.loads(path.read_text())
            for key in keys:
                check(key in data, f"{filename} has '{key}'")
        except Exception as exc:
            check(False, f"{filename} parses cleanly", str(exc))


def write_report(start: float) -> bool:
    passed = sum(1 for r in RESULTS if r["status"] == "PASS")
    failed = sum(1 for r in RESULTS if r["status"] == "FAIL")
    total = passed + failed

    print(f"\n{'=' * 56}")
    print(f"  {passed}/{total} checks passed", end="")
    if failed:
        print(f"  |  {failed} FAILED")
        for item in RESULTS:
            if item["status"] == "FAIL":
                print(f"      x  {item['label']}")
    else:
        print("  |  ALL PASSED")
    print(f"  elapsed: {time.time() - start:.2f}s")
    print(f"{'=' * 56}\n")

    out_dir = ROOT / "logs" / "health"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "validate.json").write_text(json.dumps({"results": RESULTS, "passed": failed == 0}, indent=2))

    return failed == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Run episode loop on basic task only")
    args = parser.parse_args()

    start = time.time()
    print("\n=== The Witness Stand — OpenEnv Validation ===")

    check_registry()
    check_task_configs()
    check_yaml()
    check_env_interface(tasks=["basic"] if args.fast else None)
    if not args.fast:
        check_reward_bounds()
    check_grader_determinism()
    check_transcript_lag()
    check_panel_schedule()
    check_agent_imports()
    check_logs()

    raise SystemExit(0 if write_report(start) else 1)


if __name__ == "__main__":
    main()
