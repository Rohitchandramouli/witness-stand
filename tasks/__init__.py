"""Task package exports for The Witness Stand."""

from tasks.registry import TASK_REGISTRY, get_task
from tasks.task_basic import TaskBasic
from tasks.task_intermediate import TaskIntermediate
from tasks.task_advanced import TaskAdvanced
from tasks.task_expert import TaskExpert

__all__ = [
    "TASK_REGISTRY",
    "get_task",
    "TaskBasic",
    "TaskIntermediate",
    "TaskAdvanced",
    "TaskExpert",
]