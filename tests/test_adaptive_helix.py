"""Phase 2 contract tests for the adaptive Helix temporal adapter."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

@pytest.fixture
def adaptive_helix():
    return pytest.importorskip("system.adaptive_helix")


def _make_adapter(adaptive_helix, seed=123):
    return adaptive_helix.AdaptiveHelixTemporalAdapter(
        input_dim=5,
        input_width=3,
        projection_dim=4,
        ac_decay=0.2,
        dc_decay=0.7,
        seed=seed,
        diagnostics=True,
    )


def test_adaptive_helix_keeps_helix_feature_contract(adaptive_helix):
    adapter = _make_adapter(adaptive_helix, seed=3)
    adapter.reset(batch_size=2)
    inputs = [
        np.array([[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]]),
        np.array([[0.5, 1.0, 0.0], [-0.5, 0.0, 1.0]]),
    ]
    reservoir_features = [
        np.arange(10, dtype=float).reshape(2, 5) / 10.0,
        np.flip(np.arange(10, dtype=float).reshape(2, 5), axis=1) / 10.0,
    ]

    for t, (input_t, features_t) in enumerate(zip(inputs, reservoir_features)):
        projected = adapter.step(
            input_t,
            features_t,
            t=t,
            total_steps=len(inputs),
            task_id="copy_task",
        )

    final_features = adapter.final_features()

    assert projected.shape == (2, 4)
    assert final_features.shape == (2, adapter.feature_dim)
    assert adapter.feature_dim == 27
    assert np.all(np.isfinite(final_features))
    assert np.all((0.0 <= adapter.last_write_gate) & (adapter.last_write_gate <= 1.0))


def test_adaptive_helix_seeded_runs_are_reproducible(adaptive_helix):
    first = _make_adapter(adaptive_helix, seed=41)
    second = _make_adapter(adaptive_helix, seed=41)
    input_t = np.array([[0.0, 1.0, 0.5]])
    features_t = np.array([[0.2, -0.1, 0.4, -0.3, 0.6]])

    first.step(input_t, features_t, t=0, total_steps=1, task_id="adding_task")
    second.step(input_t, features_t, t=0, total_steps=1, task_id="adding_task")

    assert np.allclose(first.last_write_gate, second.last_write_gate)
    assert np.allclose(first.final_features(), second.final_features())


def test_adaptive_gate_training_signal_changes_future_gate_response(adaptive_helix):
    adapter = _make_adapter(adaptive_helix, seed=53)
    input_t = np.array([[1.0, 0.0, 1.0]])
    features_t = np.array([[0.5, -0.5, 0.25, -0.25, 0.1]])

    adapter.step(input_t, features_t, t=0, total_steps=1, task_id="parity_task")
    before = adapter.last_write_gate.copy()
    target_gate = np.ones_like(before)
    metrics = adapter.update_gates(input_t, adapter.ac_state, target_gate, task_id="parity_task")

    adapter.reset(batch_size=1)
    adapter.step(input_t, features_t, t=0, total_steps=1, task_id="parity_task")
    after = adapter.last_write_gate

    assert set(metrics) >= {"loss", "gate_mean"}
    assert np.isfinite(metrics["loss"])
    assert after.mean() > before.mean()


def test_adaptive_diagnostics_include_gate_learning_metrics(adaptive_helix):
    adapter = _make_adapter(adaptive_helix, seed=67)
    adapter.step(
        np.array([[0.0, 1.0, 0.0]]),
        np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]),
        t=0,
        total_steps=1,
        task_id="copy_task",
    )

    diagnostics = adapter.get_diagnostics()

    assert diagnostics["adapter_type"] == "adaptive_helix"
    assert diagnostics["learned_gate_enabled"] is True
    assert "gate_loss" in diagnostics
    assert "gate_trajectory" in diagnostics
    assert diagnostics["gate_trajectory"].shape == (1, 1, 4)
