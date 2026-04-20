"""Task registry — maps task name strings to task classes."""
from tasks.task_basic import TaskBasic
from tasks.task_intermediate import TaskIntermediate
from tasks.task_advanced import TaskAdvanced
from tasks.task_expert import TaskExpert

TASK_REGISTRY = {
    "basic": TaskBasic,
    "intermediate": TaskIntermediate,
    "advanced": TaskAdvanced,
    "expert": TaskExpert,
}
