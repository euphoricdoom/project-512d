"""Classic benchmark task generators for sequence experiments."""

from experiments.tasks.adding_task import generate_adding_task, make_adding_task
from experiments.tasks.copy_task import generate_copy_task, make_copy_task
from experiments.tasks.parity_task import generate_parity_task, make_parity_task

__all__ = [
    "generate_adding_task",
    "generate_copy_task",
    "generate_parity_task",
    "make_adding_task",
    "make_copy_task",
    "make_parity_task",
]
