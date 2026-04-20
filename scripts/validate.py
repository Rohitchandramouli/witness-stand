"""OpenEnv spec compliance check across all 4 tasks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks.registry import TASK_REGISTRY

REQUIRED_TASKS = ["basic", "intermediate", "advanced", "expert"]
PASS = True


def check(condition: bool, message: str) -> bool:
    global PASS
    if condition:
        print(f"  PASS  {message}")
    else:
        print(f"  FAIL  {message}")
        PASS = False
    return condition


def main():
    print("\n=== OpenEnv Spec Validation ===\n")
    check(len(TASK_REGISTRY) == 4, "4 tasks registered")
    for task_name in REQUIRED_TASKS:
        check(task_name in TASK_REGISTRY, f"Task '{task_name}' in registry")
        if task_name in TASK_REGISTRY:
            try:
                task = TASK_REGISTRY[task_name]()
                check(task.total_turns > 0, f"{task_name}: total_turns > 0")
                check(task.data_lag_turns >= 0, f"{task_name}: data_lag_turns >= 0")
                check(task.panel is not None, f"{task_name}: panel initialised")
            except Exception as e:
                check(False, f"{task_name}: instantiation failed — {e}")
    print(f"\n{'ALL CHECKS PASSED' if PASS else 'SOME CHECKS FAILED'}\n")
    sys.exit(0 if PASS else 1)


if __name__ == "__main__":
    main()
