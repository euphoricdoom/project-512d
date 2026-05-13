"""Classic compact copy-memory task generator.

The copy task presents a random sequence of one-hot symbols, emits a trigger
token, and asks the model to replay the original symbols on later time steps.
This module uses the compact benchmark shape requested for this project:

    X, Y: (n_samples, seq_length + delay_length + 1, vocab_size + 2)

The final two channels are reserved for a trigger token and a blank token.  A
full classic copy task would need another ``seq_length`` time steps after the
trigger for the complete replay.  To keep the compact shape above, this
generator places the trigger immediately after the source sequence and writes as
much of the replay as fits in the remaining ``delay_length`` positions.  When
``delay_length < seq_length`` the target replay is truncated to the first
``delay_length`` source symbols.
"""

from __future__ import annotations

import numpy as np


def generate_copy_task(
    n_samples: int,
    seq_length: int,
    delay_length: int,
    vocab_size: int,
    seed: int | None = 0,
    dtype: np.dtype | type = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a deterministic compact copy-memory dataset.

    Parameters
    ----------
    n_samples:
        Number of independent sequences to generate.
    seq_length:
        Number of random source symbols shown at the beginning of each input.
    delay_length:
        Number of time steps available after the trigger for replay targets.
        This compact format can replay at most this many source symbols.
    vocab_size:
        Number of symbol channels.  Channels ``vocab_size`` and
        ``vocab_size + 1`` are the trigger and blank channels.
    seed:
        Seed passed to ``numpy.random.default_rng``.  Use the same seed to
        reproduce exactly the same dataset.
    dtype:
        Floating dtype for the returned arrays.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(X, Y)`` arrays with shape
        ``(n_samples, seq_length + delay_length + 1, vocab_size + 2)``.
        ``X`` contains one-hot source symbols, one trigger token, and blanks.
        ``Y`` contains blanks except for the truncated replay window after the
        trigger.
    """

    _validate_positive("n_samples", n_samples)
    _validate_positive("seq_length", seq_length)
    _validate_nonnegative("delay_length", delay_length)
    _validate_positive("vocab_size", vocab_size)

    rng = np.random.default_rng(seed)
    total_length = seq_length + delay_length + 1
    trigger_channel = vocab_size
    blank_channel = vocab_size + 1

    X = np.zeros((n_samples, total_length, vocab_size + 2), dtype=dtype)
    Y = np.zeros_like(X)
    X[:, :, blank_channel] = 1
    Y[:, :, blank_channel] = 1

    symbols = rng.integers(0, vocab_size, size=(n_samples, seq_length))
    sample_idx = np.arange(n_samples)[:, None]
    time_idx = np.arange(seq_length)[None, :]

    X[sample_idx, time_idx, symbols] = 1
    X[sample_idx, time_idx, blank_channel] = 0

    trigger_t = seq_length
    X[:, trigger_t, blank_channel] = 0
    X[:, trigger_t, trigger_channel] = 1

    replay_length = min(seq_length, delay_length)
    if replay_length:
        replay_times = trigger_t + 1 + np.arange(replay_length)
        replay_symbols = symbols[:, :replay_length]
        Y[:, replay_times, blank_channel] = 0
        Y[np.arange(n_samples)[:, None], replay_times[None, :], replay_symbols] = 1

    return X, Y


def make_copy_task(*args, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Alias for :func:`generate_copy_task`."""

    return generate_copy_task(*args, **kwargs)


def _validate_positive(name: str, value: int) -> None:
    if int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_nonnegative(name: str, value: int) -> None:
    if int(value) != value or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _self_test() -> None:
    X1, Y1 = generate_copy_task(3, 4, 2, 5, seed=123)
    X2, Y2 = generate_copy_task(3, 4, 2, 5, seed=123)

    assert X1.shape == (3, 7, 7)
    assert Y1.shape == (3, 7, 7)
    assert np.array_equal(X1, X2)
    assert np.array_equal(Y1, Y2)
    assert np.allclose(X1.sum(axis=-1), 1)
    assert np.allclose(Y1.sum(axis=-1), 1)
    assert np.all(X1[:, 4, 5] == 1)

    source = np.argmax(X1[:, :4, :5], axis=-1)
    replay = np.argmax(Y1[:, 5:7, :5], axis=-1)
    assert np.array_equal(source[:, :2], replay)

    print("copy_task self-test passed")


if __name__ == "__main__":
    _self_test()
