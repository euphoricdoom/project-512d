"""Classic temporal parity task generator.

Each input sequence is binary.  The target is sequence-shaped: at every time
step it contains the running parity, i.e. the cumulative XOR of all bits seen so
far in that sample.  Using a sequence target makes the task compatible with
recurrent or reservoir-style experiments that train/read out at every step.
"""

from __future__ import annotations

import numpy as np


def generate_parity_task(
    n_samples: int,
    seq_length: int,
    seed: int | None = 0,
    dtype: np.dtype | type = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a deterministic binary temporal parity dataset.

    Parameters
    ----------
    n_samples:
        Number of independent binary sequences to generate.
    seq_length:
        Number of time steps in each sequence.
    seed:
        Seed passed to ``numpy.random.default_rng``.  Use the same seed to
        reproduce exactly the same dataset.
    dtype:
        Floating dtype for the returned arrays.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(X, Y)`` arrays with shapes ``(n_samples, seq_length, 1)``.
        ``X`` contains binary inputs and ``Y`` contains running parity targets.
    """

    _validate_positive("n_samples", n_samples)
    _validate_positive("seq_length", seq_length)

    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(n_samples, seq_length), dtype=np.int8)
    parity = np.bitwise_xor.accumulate(bits, axis=1)

    X = bits[..., None].astype(dtype, copy=False)
    Y = parity[..., None].astype(dtype, copy=False)
    return X, Y


def make_parity_task(*args, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Alias for :func:`generate_parity_task`."""

    return generate_parity_task(*args, **kwargs)


def _validate_positive(name: str, value: int) -> None:
    if int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _self_test() -> None:
    X1, Y1 = generate_parity_task(4, 6, seed=123)
    X2, Y2 = generate_parity_task(4, 6, seed=123)

    assert X1.shape == (4, 6, 1)
    assert Y1.shape == (4, 6, 1)
    assert np.array_equal(X1, X2)
    assert np.array_equal(Y1, Y2)
    assert set(np.unique(X1)).issubset({0.0, 1.0})
    assert set(np.unique(Y1)).issubset({0.0, 1.0})
    assert np.array_equal(Y1[..., 0], np.bitwise_xor.accumulate(X1[..., 0].astype(np.int8), axis=1))

    print("parity_task self-test passed")


if __name__ == "__main__":
    _self_test()
