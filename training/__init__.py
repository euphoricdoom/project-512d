"""Modular training package for the weightless continual learning system.

Provides clean, composable building blocks that wrap the core API in
train_hf.py with better separation of concerns:

    training.dataset   — dataset loading and preprocessing
    training.trainer   — training loop
    training.evaluation — metrics and evaluation helpers
    training.cli       — command-line interface (re-exported from train_hf)
"""

from training.dataset import DatasetConfig, DatasetLoader
from training.trainer import WeightlessTrainer
from training.evaluation import accuracy, evaluate_model

__all__ = [
    "DatasetConfig",
    "DatasetLoader",
    "WeightlessTrainer",
    "accuracy",
    "evaluate_model",
]
