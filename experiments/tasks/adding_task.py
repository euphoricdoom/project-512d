"""Classic adding-problem task generator.

The adding task provides a sequence with two channels: random scalar values and
a binary marker channel with exactly two marked positions per sample.  The model
must output the sum of the two marked values.  This implementation returns a
sequence-shaped target with zeros at intermediate steps and the required sum at
the final time step.
"""

from __future__ import annotations

import numpy as np


def generate_adding_task(
    n_samples: int,
    seq_length: int,
    seed: int | None = 0,
    dtype: np.dtype | type = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a deterministic adding-problem dataset.

    Parameters
    ----------
    n_samples:
        Number of independent sequences to generate.
    seq_length:
        Number of time steps in each sequence.  Must be at least ``2`` so each
        sample can contain two distinct marker positions.
    seed:
        Seed passed to ``numpy.random.default_rng``.  Use the same seed to
        reproduce exactly the same dataset.
    dtype:
        Floating dtype for the returned arrays.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(X, Y)`` arrays.  ``X`` has shape ``(n_samples, seq_length, 2)`` with
        values in channel 0 and markers in channel 1.  ``Y`` has shape
        ``(n_samples, seq_length, 1)`` and stores the sum of the marked values
        at the final time step.
    """

    _validate_positive("n_samples", n_samples)
    _validate_minimum("seq_length", seq_length, 2)

    rng = np.random.default_rng(seed)
    values = rng.random((n_samples, seq_length), dtype=dtype)
    markers = np.zeros((n_samples, seq_length), dtype=dtype)
    targets = np.zeros((n_samples, seq_length, 1), dtype=dtype)

    marker_positions = np.empty((n_samples, 2), dtype=np.int64)
    for sample in range(n_samples):
        marker_positions[sample] = rng.choice(seq_length, size=2, replace=False)

    row_idx = np.arange(n_samples)[:, None]
    markers[row_idx, marker_positions] = 1
    sums = values[row_idx, marker_positions].sum(axis=1)
    targets[:, -1, 0] = sums

    X = np.stack((values, markers), axis=-1).astype(dtype, copy=False)
    return X, targets


def make_adding_task(*args, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Alias for :func:`generate_adding_task`."""

    return generate_adding_task(*args, **kwargs)


def _validate_positive(name: str, value: int) -> None:
    if int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_minimum(name: str, value: int, minimum: int) -> None:
    if int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _self_test() -> None:
    X1, Y1 = generate_adding_task(5, 8, seed=123)
    X2, Y2 = generate_adding_task(5, 8, seed=123)

    assert X1.shape == (5, 8, 2)
    assert Y1.shape == (5, 8, 1)
    assert np.array_equal(X1, X2)
    assert np.array_equal(Y1, Y2)
    assert np.all(X1[:, :, 0] >= 0)
    assert np.all(X1[:, :, 0] < 1)
    assert np.all(X1[:, :, 1].sum(axis=1) == 2)
    assert np.all(Y1[:, :-1, 0] == 0)

    expected = (X1[:, :, 0] * X1[:, :, 1]).sum(axis=1)
    assert np.allclose(Y1[:, -1, 0], expected)

    print("adding_task self-test passed")


if __name__ == "__main__":
    _self_test()
