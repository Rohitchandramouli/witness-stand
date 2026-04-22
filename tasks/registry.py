"""Task registry — maps task name strings to task classes."""
from tasks.task_basic import TaskBasic
from tasks.task_intermediate import TaskIntermediate
from tasks.task_advanced import TaskAdvanced
from tasks.task_expert import TaskExpert

TASK_REGISTRY = {
    "basic":        TaskBasic,
    "intermediate": TaskIntermediate,
    "advanced":     TaskAdvanced,
    "expert":       TaskExpert,
}


def get_task(task_name: str):
    """
    Instantiates and returns a task by name.
    Raises ValueError with a clear message if the task name is unknown.
    Called by environment.py at every env.reset().
    """
    if task_name not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task '{task_name}'. "
            f"Valid options: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[task_name]()
