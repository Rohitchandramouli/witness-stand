"""Task registry — maps task name strings to task classes."""

from typing import Any

from tasks.base import TaskBase
from tasks.task_basic import TaskBasic
from tasks.task_intermediate import TaskIntermediate
from tasks.task_advanced import TaskAdvanced
from tasks.task_expert import TaskExpert


TASK_REGISTRY: dict[str, type[TaskBase]] = {
    "basic": TaskBasic,
    "intermediate": TaskIntermediate,
    "advanced": TaskAdvanced,
    "expert": TaskExpert,
}


def get_task(task_name: str, **kwargs: Any) -> TaskBase:
    """
    Returns a fresh task instance.

    Default usage remains unchanged:
        get_task("basic")

    Demo/training control is also supported:
        get_task("basic", domain="medical", mode="demo")
        get_task("expert", domain_pair=("financial", "technical"), mode="train")
    """
    if task_name not in TASK_REGISTRY:
        valid = ", ".join(sorted(TASK_REGISTRY))
        raise ValueError(f"Unknown task '{task_name}'. Valid tasks: {valid}")

    return TASK_REGISTRY[task_name](**kwargs)