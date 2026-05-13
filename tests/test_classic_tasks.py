"""Tests for classic sequence task generators."""
import inspect
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.constants import Config512D
from experiments.run_4_kernel_network import KernelNetwork
from experiments.tasks.adding_task import generate_adding_task
from experiments.tasks.copy_task import generate_copy_task
from experiments.tasks.parity_task import generate_parity_task


def _call_generator(fn, **values):
    """Call a generator with only the parameters it declares."""
    aliases = {
        "n_samples": "batch_size",
        "seq_length": "seq_len",
        "vocab_size": "n_symbols",
    }
    signature = inspect.signature(fn)
    kwargs = {}
    for name, parameter in signature.parameters.items():
        if name in values:
            kwargs[name] = values[name]
        elif name in aliases:
            kwargs[name] = values[aliases[name]]
        elif name == "delay_length":
            kwargs[name] = values.get("delay_length", values["seq_len"])
        elif name == "rng":
            kwargs[name] = np.random.default_rng(values["seed"])
        elif parameter.default is inspect.Parameter.empty:
            raise TypeError(f"{fn.__name__} requires unsupported argument {name!r}")
    return fn(**kwargs)


def _as_pair(result):
    assert isinstance(result, tuple)
    assert len(result) >= 2
    return np.asarray(result[0]), np.asarray(result[1])


def _assert_one_hot(a):
    assert np.all((a == 0.0) | (a == 1.0))
    assert np.allclose(a.sum(axis=-1), 1.0)


def _pad_or_trim(vec, size):
    vec = np.asarray(vec, dtype=float).reshape(-1)
    if vec.size >= size:
        return vec[:size]
    return np.pad(vec, (0, size - vec.size))


def test_copy_task_shapes_one_hot_and_reproducible():
    x1, y1 = _as_pair(
        _call_generator(generate_copy_task, batch_size=3, seq_len=5, n_symbols=4, seed=123)
    )
    x2, y2 = _as_pair(
        _call_generator(generate_copy_task, batch_size=3, seq_len=5, n_symbols=4, seed=123)
    )

    assert x1.shape[0] == 3
    assert y1.shape[0] == 3
    assert x1.ndim == 3
    assert y1.ndim == 3
    assert x1.shape[-1] == 6
    assert y1.shape[-1] == 6
    _assert_one_hot(x1)
    _assert_one_hot(y1)
    assert np.array_equal(x1, x2)
    assert np.array_equal(y1, y2)

    symbol_inputs = x1[:, :5, :4]
    replay_targets = y1[:, 6:, :4]
    assert np.array_equal(np.argmax(replay_targets, axis=-1), np.argmax(symbol_inputs, axis=-1))
    assert np.all(y1.sum(axis=(1, 2)) > 0)


def test_parity_task_shapes_correctness_and_reproducible():
    x1, y1 = _as_pair(_call_generator(generate_parity_task, batch_size=6, seq_len=7, seed=222))
    x2, y2 = _as_pair(_call_generator(generate_parity_task, batch_size=6, seq_len=7, seed=222))

    assert x1.shape == (6, 7, 1)
    assert y1.shape == (6, 7, 1)
    assert np.array_equal(x1, x2)
    assert np.array_equal(y1, y2)
    assert np.all((x1 == 0.0) | (x1 == 1.0))

    expected = np.bitwise_xor.accumulate(x1[..., 0].astype(np.int8), axis=1)
    actual = y1[..., 0].astype(np.int8)
    assert np.array_equal(actual, expected)


def test_adding_task_shapes_correctness_and_reproducible():
    x1, y1 = _as_pair(_call_generator(generate_adding_task, batch_size=4, seq_len=8, seed=333))
    x2, y2 = _as_pair(_call_generator(generate_adding_task, batch_size=4, seq_len=8, seed=333))

    assert x1.shape == (4, 8, 2)
    assert y1.shape == (4, 8, 1)
    assert np.array_equal(x1, x2)
    assert np.array_equal(y1, y2)
    assert np.all((x1[..., 1] == 0.0) | (x1[..., 1] == 1.0))
    assert np.allclose(x1[..., 1].sum(axis=1), 2.0)
    assert np.allclose(y1[:, :-1, 0], 0.0)

    expected = (x1[..., 0] * x1[..., 1]).sum(axis=1)
    assert np.allclose(y1[:, -1, 0], expected)


def test_classic_task_kernel_network_smoke():
    cfg = Config512D(seed=44, process_steps=1)
    network = KernelNetwork(1, cfg, use_stable_readout=False)
    x, _ = _as_pair(_call_generator(generate_copy_task, batch_size=1, seq_len=3, n_symbols=3, seed=444))

    pred = network.process(_pad_or_trim(x[0], cfg.input_dim), "copy", steps=1)

    assert pred.shape == (cfg.output_dim,)
    assert np.all(np.isfinite(pred))
