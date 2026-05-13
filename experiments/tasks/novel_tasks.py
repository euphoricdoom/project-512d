"""Novel sequence task generators for learned-gate generalization tests."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def generate_reverse_task(
    seq_length: int = 10,
    n_samples: int = 500,
    vocab_size: int = 8,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return one-hot sequences and their reverse-order targets."""
    rng = np.random.default_rng(seed)
    width = vocab_size + 1
    inputs = np.zeros((n_samples, seq_length, width), dtype=float)
    targets = np.zeros_like(inputs)
    for i in range(n_samples):
        seq = rng.integers(1, vocab_size + 1, size=seq_length)
        rev = seq[::-1]
        inputs[i, np.arange(seq_length), seq] = 1.0
        targets[i, np.arange(seq_length), rev] = 1.0
    return inputs, targets


def generate_duplicate_task(
    seq_length: int = 10,
    n_samples: int = 500,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return value/marker inputs and fixed-width marked-value targets."""
    rng = np.random.default_rng(seed)
    inputs = np.zeros((n_samples, seq_length, 2), dtype=float)
    targets = np.zeros((n_samples, seq_length, 1), dtype=float)
    for i in range(n_samples):
        values = rng.random(seq_length)
        n_marked = int(rng.integers(2, 4))
        marked = rng.choice(seq_length, size=n_marked, replace=False)
        inputs[i, :, 0] = values
        inputs[i, marked, 1] = 1.0
        targets[i, :n_marked, 0] = values[marked]
    return inputs, targets


def generate_count_task(
    seq_length: int = 15,
    n_samples: int = 500,
    count_value: int = 1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return binary one-hot sequences and counts of ``count_value``."""
    rng = np.random.default_rng(seed)
    seq = rng.integers(0, 2, size=(n_samples, seq_length))
    inputs = np.zeros((n_samples, seq_length, 2), dtype=float)
    rows = np.arange(n_samples)[:, None]
    cols = np.arange(seq_length)[None, :]
    inputs[rows, cols, seq] = 1.0
    targets = (seq == count_value).sum(axis=1, keepdims=True).astype(float)
    return inputs, targets


def generate_median_task(
    seq_length: int = 11,
    n_samples: int = 500,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return scalar sequences and their median value."""
    rng = np.random.default_rng(seed)
    inputs = rng.random((n_samples, seq_length, 1))
    targets = np.median(inputs[:, :, 0], axis=1, keepdims=True)
    return inputs, targets


def _self_test() -> None:
    assert generate_reverse_task(seq_length=5, n_samples=10)[0].shape == (10, 5, 9)
    assert generate_duplicate_task(seq_length=10, n_samples=10)[0].shape == (10, 10, 2)
    assert generate_count_task(seq_length=15, n_samples=10)[1].shape == (10, 1)
    assert generate_median_task(seq_length=11, n_samples=10)[0].shape == (10, 11, 1)
    print("All novel task generators working")


if __name__ == "__main__":
    _self_test()
