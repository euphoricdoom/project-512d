"""Tests for frozen-feature sleep consolidation."""

from __future__ import annotations

import numpy as np

from core.constants import Config512D
from core.kernel import CoupledModularKernel
from experiments.run_4_kernel_network import AtomicKernel
from system.sleep_consolidation import sleep_consolidation


def test_coupled_kernel_freeze_blocks_inter_update() -> None:
    """Frozen core kernels skip Hebbian inter-factor updates."""
    cfg = Config512D(seed=7)
    rng = np.random.default_rng(7)
    kernel = CoupledModularKernel(cfg, rng)
    x = rng.normal(size=cfg.dim)
    before_u = kernel._U.copy()
    before_v = kernel._V.copy()

    kernel.freeze()
    kernel.update_inter(x, lr=0.1)

    assert kernel.is_frozen()
    assert np.allclose(kernel._U, before_u)
    assert np.allclose(kernel._V, before_v)

    kernel.unfreeze()
    kernel.update_inter(x, lr=0.1)

    assert not kernel.is_frozen()
    assert not np.allclose(kernel._U, before_u)


def test_network_atomic_kernel_freeze_roundtrip() -> None:
    """Network wrapper kernels expose freeze state without disabling dynamics."""
    kernel = AtomicKernel(0, Config512D(seed=3))
    assert not kernel.is_frozen()
    kernel.freeze()
    assert kernel.is_frozen()
    kernel.receive(np.ones(64))
    kernel.step()
    features = kernel.get_features()
    assert features.shape == (kernel.cfg.num_modules + kernel.cfg.dim_c,)
    assert np.all(np.isfinite(features))
    kernel.unfreeze()
    assert not kernel.is_frozen()


def test_sleep_consolidation_uses_cached_features_and_reduces_error() -> None:
    """Sleep should extract frozen features once, then improve the task head."""

    class Kernel:
        def __init__(self) -> None:
            self.frozen = False

        def freeze(self) -> None:
            self.frozen = True

        def unfreeze(self) -> None:
            self.frozen = False

        def is_frozen(self) -> bool:
            return self.frozen

    class Head:
        def __init__(self) -> None:
            self.w = np.zeros((1, 2))

        def predict_batch(self, features, task_id=None):
            return features @ self.w.T

        def update_batch(self, features, targets, task_id=None, lr=0.1):
            pred = self.predict_batch(features, task_id)
            err = targets - pred
            self.w += lr * (err.T @ features) / len(features)
            return float(np.mean(err**2))

    class Net:
        def __init__(self) -> None:
            self.kernels = [Kernel(), Kernel()]
            self.network_readout = Head()

    calls = {"count": 0}

    def extractor(network, inputs, task_id):
        calls["count"] += 1
        assert all(kernel.is_frozen() for kernel in network.kernels)
        return np.zeros((len(inputs), 1)), inputs[:, 0, :]

    inputs = np.array([[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]], [[2.0, 1.0]]])
    targets = np.array([[1.0], [2.0], [3.0], [4.0]])
    net = Net()

    metrics = sleep_consolidation(
        net,
        "linear_task",
        inputs,
        targets,
        num_cycles=80,
        sleep_lr=0.2,
        batch_size=2,
        verbose=False,
        feature_extractor=extractor,
        seed=11,
    )

    assert calls["count"] == 1
    assert metrics["final_error"] < metrics["initial_error"]
    assert metrics["error_reduction"] > 0
    assert all(not kernel.is_frozen() for kernel in net.kernels)
