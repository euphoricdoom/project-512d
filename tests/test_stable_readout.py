"""Stability and compatibility tests for StableNetworkReadout."""
import inspect
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from system.multi_readout import MultiTaskReadout
from system.stable_network_readout import StableNetworkReadout


def _make_readout(
    *,
    output_dim: int = 3,
    feature_dim: int = 5,
    num_kernels: int = 1,
    base_lr: float = 0.04,
    seed: int = 123,
):
    """Construct with the repo API while tolerating common stable-readout names."""
    rng = np.random.default_rng(seed)
    signature = inspect.signature(StableNetworkReadout)
    kwargs = {}
    positional = []

    for name in signature.parameters:
        if name == "self":
            continue
        if name == "output_dim":
            kwargs[name] = output_dim
        elif name == "feature_dim":
            kwargs[name] = feature_dim
        elif name == "rng":
            kwargs[name] = rng
        elif name in {"num_kernels", "kernel_count", "n_kernels", "K"}:
            kwargs[name] = num_kernels
        elif name == "base_lr":
            kwargs[name] = base_lr
        elif name == "lr":
            kwargs[name] = base_lr

    if {"output_dim", "feature_dim", "rng"} - set(kwargs):
        positional = [output_dim, feature_dim, rng]

    return StableNetworkReadout(*positional, **kwargs)


def _effective_lr(readout) -> float:
    if hasattr(readout, "effective_lr"):
        value = readout.effective_lr
        return float(value() if callable(value) else value)
    if hasattr(readout, "get_info"):
        info = readout.get_info()
        if "effective_lr" in info:
            return float(info["effective_lr"])
    raise AssertionError("StableNetworkReadout must expose effective_lr or get_info()['effective_lr']")


def _normalize(readout, features: np.ndarray) -> np.ndarray:
    for name in ("layer_norm", "norm", "normalize_features", "_normalize_features", "_layer_norm"):
        if hasattr(readout, name):
            normalizer = getattr(readout, name)
            if hasattr(normalizer, "forward"):
                return np.asarray(normalizer.forward(features))
            return np.asarray(normalizer(features))
    raise AssertionError("StableNetworkReadout must expose a feature layer-normalization method")


def _task_weights(readout, task_id: str) -> np.ndarray:
    if hasattr(readout, "readouts"):
        return np.asarray(readout.readouts[task_id])
    if hasattr(readout, "weights"):
        weights = readout.weights
        if isinstance(weights, dict):
            return np.asarray(weights[task_id])
        return np.asarray(weights)
    raise AssertionError("StableNetworkReadout must expose task-specific weights via readouts or weights")


def _set_task(readout, task_id: str) -> None:
    if hasattr(readout, "set_task"):
        readout.set_task(task_id)
    elif hasattr(readout, "add_task"):
        readout.add_task(task_id)


@pytest.mark.parametrize("num_kernels", [1, 4, 16, 64, 256])
def test_lr_scaling(num_kernels):
    base_lr = 0.08
    readout = _make_readout(
        output_dim=2,
        feature_dim=4,
        num_kernels=num_kernels,
        base_lr=base_lr,
    )

    assert _effective_lr(readout) == pytest.approx(base_lr / np.sqrt(num_kernels))


def test_layer_norm():
    rng = np.random.default_rng(10)
    readout = _make_readout(output_dim=64, feature_dim=92, num_kernels=64)
    features = rng.normal(size=(32, 5888)) * 10.0 + 5.0

    normalized = _normalize(readout, features)

    assert normalized.shape == features.shape
    assert np.allclose(np.mean(normalized, axis=1), 0.0, atol=1e-6)
    assert np.allclose(np.std(normalized, axis=1), 1.0, atol=1e-5)


def test_gradient_magnitude_remains_bounded_across_kernel_counts():
    rng = np.random.default_rng(20)
    base_lr = 0.05
    update_norms = []

    for num_kernels in [1, 4, 16, 64, 256]:
        feature_dim = 92 * num_kernels
        readout = _make_readout(
            output_dim=8,
            feature_dim=92,
            num_kernels=num_kernels,
            base_lr=base_lr,
            seed=21,
        )
        features = rng.normal(size=feature_dim) * 50.0
        error = rng.normal(size=8) * 20.0
        _set_task(readout, "stable")
        before = _task_weights(readout, "stable").copy()

        readout.update(features, error, lr=base_lr, task_id="stable")

        after = _task_weights(readout, "stable")
        delta_norm = np.linalg.norm(after - before)
        assert np.isfinite(delta_norm)
        assert np.all(np.isfinite(after))
        assert np.max(np.abs(after)) <= 3.0
        update_norms.append(delta_norm)

    assert max(update_norms) / max(min(update_norms), 1e-12) < 20.0


def test_stability_at_64_kernels():
    rng = np.random.default_rng(30)
    output_dim = 16
    readout = _make_readout(
        output_dim=output_dim,
        feature_dim=92,
        num_kernels=64,
        base_lr=0.03,
        seed=31,
    )
    feature_dim = readout.total_feature_dim
    losses = []

    for _ in range(100):
        features = rng.normal(size=feature_dim)
        target = np.tanh(rng.normal(size=output_dim))
        pred = readout.predict(features, task_id="long_run")
        error = target - pred
        losses.append(float(np.mean(error**2)))
        readout.update(features, error, lr=0.03, task_id="long_run")
        weights = _task_weights(readout, "long_run")

        assert np.all(np.isfinite(pred))
        assert np.all(np.isfinite(weights))
        assert np.max(np.abs(weights)) <= 3.0

    assert np.isfinite(losses).all()
    assert max(losses) < 1e6
    assert losses[-1] < losses[0] * 100.0 + 1.0


def test_interface_compatibility():
    rng = np.random.default_rng(40)
    stable = _make_readout(output_dim=7, feature_dim=11, num_kernels=1, seed=41)
    multi = MultiTaskReadout(7, 11, rng)
    features = rng.normal(size=11)
    batch_features = rng.normal(size=(4, 11))
    error = rng.normal(size=7)
    batch_errors = rng.normal(size=(4, 7))

    for readout in (stable, multi):
        _set_task(readout, "task")
        assert readout.predict(features, task_id="task").shape == (7,)
        assert readout.predict_batch(batch_features, task_id="task").shape == (4, 7)
        update_result = readout.update(features, error, lr=0.01, task_id="task")
        batch_result = readout.update_batch(batch_features, batch_errors, lr=0.01, task_id="task")
        assert update_result is None or isinstance(update_result, float)
        assert batch_result is None or isinstance(batch_result, float)
        expected_params = 7 * 11
        if isinstance(readout, StableNetworkReadout):
            expected_params += 2 * 11
        assert readout.stored_params() == expected_params


def test_multi_task_isolation():
    rng = np.random.default_rng(50)
    readout = _make_readout(output_dim=4, feature_dim=9, num_kernels=1, seed=51)
    features = rng.normal(size=9)
    error = rng.normal(size=4)
    _set_task(readout, "a")
    _set_task(readout, "b")
    a_before = _task_weights(readout, "a").copy()
    b_before = _task_weights(readout, "b").copy()

    readout.update(features, error, lr=0.02, task_id="b")

    assert np.allclose(_task_weights(readout, "a"), a_before)
    assert not np.allclose(_task_weights(readout, "b"), b_before)


def test_single_vector_predict_stored_params_and_get_info():
    readout = _make_readout(output_dim=5, feature_dim=13, num_kernels=4, base_lr=0.02)
    features = np.linspace(-1.0, 1.0, 52)

    pred = readout.predict(features, task_id="edge")

    assert pred.shape == (5,)
    assert readout.stored_params() == 5 * 52 + 2 * 52
    assert hasattr(readout, "get_info")
    info = readout.get_info()
    assert info["output_dim"] == 5
    assert info["feature_dim"] == 13
    assert info["total_feature_dim"] == 52
    assert info["num_kernels"] == 4
    assert info["effective_lr"] == pytest.approx(0.02 / np.sqrt(4))


def test_stable_base_lr_changes_update_magnitude_even_with_external_lr():
    rng = np.random.default_rng(60)
    features = rng.normal(size=(8, 64 * 92))
    errors = rng.normal(size=(8, 4))

    slow = StableNetworkReadout(
        num_kernels=64,
        feature_dim=92,
        output_dim=4,
        base_lr=0.01,
        rng=np.random.default_rng(61),
        grad_clip=10.0,
    )
    fast = StableNetworkReadout(
        num_kernels=64,
        feature_dim=92,
        output_dim=4,
        base_lr=0.04,
        rng=np.random.default_rng(61),
        grad_clip=10.0,
    )
    slow.set_task("task")
    fast.set_task("task")
    slow_before = slow.weights["task"].copy()
    fast_before = fast.weights["task"].copy()

    slow.update_batch(features, errors, lr=999.0, task_id="task", as_error=True)
    fast.update_batch(features, errors, lr=999.0, task_id="task", as_error=True)

    slow_delta = np.linalg.norm(slow.weights["task"] - slow_before)
    fast_delta = np.linalg.norm(fast.weights["task"] - fast_before)

    assert fast_delta > slow_delta * 3.5
    assert fast_delta < slow_delta * 4.5
