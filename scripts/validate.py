"""
OpenEnv spec compliance + pipeline health check. 199 assertions across 10 checks.
Run: python scripts/validate.py  |  --fast for quick check on basic task only.
"""

import sys
import json
import argparse
try:
    import yaml
except ImportError:
    yaml = None
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PASS_ALL = True
RESULTS  = []


def check(condition: bool, label: str, detail: str = "") -> bool:
    global PASS_ALL
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}]  {label}"
    if detail and not condition:
        msg += f"\n           -> {detail}"
    print(msg)
    RESULTS.append((status, label))
    if not condition:
        PASS_ALL = False
    return condition


def section(title: str) -> None:
    print(f"\n-- {title} {'─' * max(0, 52 - len(title))}")


# ── 1: Registry ───────────────────────────────────────────────────────

def check_registry():
    section("1  Task registry")
    from tasks.registry import TASK_REGISTRY
    REQUIRED = ["basic", "intermediate", "advanced", "expert"]
    check(len(TASK_REGISTRY) == 4, "4 tasks registered",
          f"got {len(TASK_REGISTRY)}: {list(TASK_REGISTRY.keys())}")
    for name in REQUIRED:
        check(name in TASK_REGISTRY, f"'{name}' in registry")


# ── 2: Task config ────────────────────────────────────────────────────

EXPECTED = {
    "basic":        {"total_turns": 10, "data_lag_turns": 0},
    "intermediate": {"total_turns": 20, "data_lag_turns": 0},
    "advanced":     {"total_turns": 30, "data_lag_turns": 2},
    "expert":       {"total_turns": 40, "data_lag_turns": 3},
}

def check_task_configs():
    section("2  Task configurations")
    from tasks.registry import TASK_REGISTRY
    for name, expected in EXPECTED.items():
        if name not in TASK_REGISTRY:
            check(False, f"{name}: skipped (not in registry)")
            continue
        try:
            task = TASK_REGISTRY[name]()
            check(task.total_turns == expected["total_turns"],
                  f"{name}: total_turns == {expected['total_turns']}",
                  f"got {task.total_turns}")
            check(task.data_lag_turns == expected["data_lag_turns"],
                  f"{name}: data_lag_turns == {expected['data_lag_turns']}",
                  f"got {task.data_lag_turns}")
            check(task.panel is not None, f"{name}: panel initialised")
            check(task.persona is not None, f"{name}: persona loaded")
            check(bool(getattr(task.persona, 'system_prompt', '')),
                  f"{name}: persona.system_prompt non-empty")
            check(bool(task.domain), f"{name}: domain set")
        except Exception as e:
            check(False, f"{name}: instantiation failed", str(e))


# ── 3: openenv.yaml ───────────────────────────────────────────────────

def check_yaml():
    section("3  openenv.yaml")
    yaml_path = Path(__file__).parent.parent / "openenv.yaml"
    check(yaml_path.exists(), "openenv.yaml exists")
    if not yaml_path.exists():
        return
    try:
        doc = yaml.safe_load(yaml_path.read_text())
    except Exception as e:
        check(False, "openenv.yaml parses cleanly", str(e))
        return
    for key in ["name", "version", "tasks", "action_space", "observation_space", "reward_range"]:
        check(key in doc, f"openenv.yaml has '{key}'")
    if "tasks" in doc:
        declared = [t["name"] for t in doc["tasks"]]
        check(len(declared) == 4, "4 tasks declared in yaml",
              f"got {len(declared)}: {declared}")
        for name in ["basic", "intermediate", "advanced", "expert"]:
            check(name in declared, f"yaml declares '{name}'")
    if "reward_range" in doc:
        rr = doc["reward_range"]
        check(isinstance(rr, list) and len(rr) == 2,
              "reward_range is [min, max]", f"got {rr}")


# ── 4: Environment interface ──────────────────────────────────────────

def _dummy_action(response_text="My position stands.", flagged=False):
    from models import WitnessAction
    return WitnessAction(
        response_text=response_text,
        flagged_distortion=flagged,
        accepted_update=False,
        tool_calls=[],
    )

def check_env_interface(tasks=None):
    section("4  Environment interface (reset / step / grade)")
    from environment import WitnessStandEnv
    targets = tasks or list(EXPECTED.keys())
    for task_name in targets:
        try:
            env = WitnessStandEnv()
            obs = env.reset(task_name)
            check(isinstance(obs, dict), f"{task_name}: reset() returns dict")
            for key in ["questioner_text", "turn_number", "total_turns", "persona_system_prompt"]:
                check(key in obs, f"{task_name}: obs has '{key}'")

            done = False
            turn_scores = []
            while not done:
                action = _dummy_action(flagged=len(turn_scores) % 3 == 1)
                obs, reward, done, info = env.step(action)
                turn_scores.append(reward)
                check(isinstance(reward, (int, float)),
                      f"{task_name} t{len(turn_scores)}: reward numeric")

            check(len(turn_scores) == EXPECTED[task_name]["total_turns"],
                  f"{task_name}: correct turn count",
                  f"expected {EXPECTED[task_name]['total_turns']}, got {len(turn_scores)}")

            final = env.grade()
            check(isinstance(final, float), f"{task_name}: grade() returns float")
            check(0.0 <= final <= 1.0, f"{task_name}: grade() in [0, 1]",
                  f"got {final:.4f}")
        except Exception as e:
            check(False, f"{task_name}: episode loop crashed", str(e))


# ── 5: Reward bounds ──────────────────────────────────────────────────

def check_reward_bounds():
    section("5  Reward bounds")
    from environment import WitnessStandEnv
    for task_name in ["basic", "expert"]:
        try:
            env = WitnessStandEnv()
            env.reset(task_name)
            done = False
            bad = []
            i = 0
            while not done:
                _, reward, done, _ = env.step(_dummy_action(flagged=i % 2 == 0))
                if not (-1.0 <= reward <= 2.0):
                    bad.append((i, reward))
                i += 1
            check(len(bad) == 0, f"{task_name}: all per-turn scores in [-1, 2]",
                  f"out-of-range: {bad}")
            final = env.grade()
            check(0.0 <= final <= 1.0, f"{task_name}: final grade in [0, 1]",
                  f"got {final:.4f}")
        except Exception as e:
            check(False, f"{task_name}: reward bounds check crashed", str(e))


# ── 6: Grader determinism ─────────────────────────────────────────────

def check_grader_determinism():
    section("6  Grader determinism")
    from environment import WitnessStandEnv
    responses = [
        "My position stands. The evidence supports my original assessment.",
        "DISTORTION_DETECTED. That is not what I said.",
        "I want to be precise about what I actually stated.",
        "The methodology was sound. I stand by the conclusions.",
        "That framing is not accurate. My original statement included caveats.",
    ]
    scores = []
    for _ in range(2):
        env = WitnessStandEnv()
        env.reset("basic")
        done = False
        i = 0
        while not done:
            action = _dummy_action(
                response_text=responses[i % len(responses)],
                flagged=(i % 3 == 1),
            )
            _, _, done, _ = env.step(action)
            i += 1
        scores.append(round(env.grade(), 6))
    check(scores[0] == scores[1], "grader is deterministic",
          f"run1={scores[0]}  run2={scores[1]}")


# ── 7: Transcript lag ─────────────────────────────────────────────────

def check_transcript_lag():
    section("7  Transcript data lag")
    from transcript.store import TranscriptStore
    from models import Turn, Speaker, TurnType

    def make_turn(i):
        return Turn(turn_no=i, speaker=Speaker.WITNESS,
                    text=f"turn {i}", turn_type=TurnType.NEUTRAL)

    for lag, n_turns, expected_visible in [(0, 5, 5), (2, 5, 3), (3, 5, 2)]:
        store = TranscriptStore(data_lag_turns=lag)
        for i in range(n_turns):
            store.append(make_turn(i))
        visible = store.get_witness_turns()
        check(len(visible) == expected_visible,
              f"lag={lag}: {expected_visible}/{n_turns} turns visible",
              f"got {len(visible)}")


# ── 8: Questioner panel ───────────────────────────────────────────────

def check_panel_schedule():
    section("8  Questioner panel schedule")
    from tasks.registry import TASK_REGISTRY
    from models import TurnType

    INJECTION_TYPES = {
        TurnType.DISTORTION_INJECTION,
        TurnType.AUTHORITY_INJECTION,
        TurnType.TEMPORAL_INJECTION,
        TurnType.EXHAUSTION,
    }

    for task_name in ["basic", "expert"]:
        if task_name not in TASK_REGISTRY:
            continue
        task = TASK_REGISTRY[task_name]()
        injections = [t for t in range(task.total_turns)
                      if task.panel.get_turn_type(t) in INJECTION_TYPES]
        neutrals   = [t for t in range(task.total_turns)
                      if task.panel.get_turn_type(t) == TurnType.NEUTRAL]

        check(len(injections) > 0, f"{task_name}: has injection turns",
              f"none found in {task.total_turns} turns")
        check(len(neutrals) > 0,   f"{task_name}: has neutral turns")

        if task_name == "expert":
            types_seen = {task.panel.get_turn_type(t) for t in injections}
            check(len(types_seen) >= 2,
                  "expert: multiple injection types in schedule",
                  f"only {types_seen}")


# ── 9: Agent imports ──────────────────────────────────────────────────

def check_agent_imports():
    section("9  Agent module imports")
    modules = [
        ("agent.memory",    "EpisodicMemory"),
        ("agent.prompt",    "build_system_prompt"),
        ("agent.prompt",    "build_user_prompt"),
        ("agent.parser",    "parse_action"),
        ("agent.heuristics","WitnessHeuristics"),
    ]
    for module, attr in modules:
        try:
            mod = __import__(module, fromlist=[attr])
            check(hasattr(mod, attr), f"{module}.{attr} importable")
        except Exception as e:
            check(False, f"{module} importable", str(e))


# ── 10: Logs + JSON outputs ───────────────────────────────────────────

def check_logs():
    section("10 Logs directory + JSON outputs")
    root     = Path(__file__).parent.parent
    logs_dir = root / "logs"
    check(logs_dir.exists(), "logs/ directory exists")

    benchmark = logs_dir / "benchmark_results.json"
    if benchmark.exists():
        try:
            data = json.loads(benchmark.read_text())
            for key in ["tasks", "avg_score", "witness_elo"]:
                check(key in data, f"benchmark_results.json has '{key}'")
        except Exception as e:
            check(False, "benchmark_results.json parses cleanly", str(e))
    else:
        print("  [SKIP]  benchmark_results.json not generated yet (run run_eval.py)")

    demo = logs_dir / "demo_transcript.json"
    if demo.exists():
        try:
            data = json.loads(demo.read_text())
            for key in ["before", "after", "delta"]:
                check(key in data, f"demo_transcript.json has '{key}'")
        except Exception as e:
            check(False, "demo_transcript.json parses cleanly", str(e))
    else:
        print("  [SKIP]  demo_transcript.json not generated yet (run demo.py)")


# ── Summary ───────────────────────────────────────────────────────────

def print_summary():
    passed = sum(1 for s, _ in RESULTS if s == "PASS")
    failed = sum(1 for s, _ in RESULTS if s == "FAIL")
    total  = passed + failed
    print(f"\n{'=' * 56}")
    print(f"  {passed}/{total} checks passed", end="")
    if failed:
        print(f"  |  {failed} FAILED:")
        for status, label in RESULTS:
            if status == "FAIL":
                print(f"      x  {label}")
    else:
        print("  |  ALL PASSED")
    print(f"{'=' * 56}\n")


# ── Entry point ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Episode loops on basic task only")
    args = parser.parse_args()

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
    print_summary()
    sys.exit(0 if PASS_ALL else 1)


if __name__ == "__main__":
    main()
