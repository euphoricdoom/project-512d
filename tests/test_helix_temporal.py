"""Tests for the helix temporal feature adapter."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from system.helix_temporal import HelixTemporalAdapter


def _run_sequence(adapter, sequence):
    projected = []
    for features in sequence:
        projected.append(adapter.step(features))
    return np.asarray(projected), adapter.final_features()


def test_feature_shape_and_segments_are_finite():
    adapter = HelixTemporalAdapter(input_dim=3, projection_dim=5, seed=11)
    sequence = np.array(
        [
            [1.0, 0.0, -1.0],
            [0.5, 0.25, 0.0],
            [-0.25, 0.75, 0.5],
        ]
    )

    projected, features = _run_sequence(adapter, sequence)

    assert projected.shape == (3, 5)
    assert adapter.feature_dim == 33
    assert features.shape == (adapter.feature_dim,)
    assert np.all(np.isfinite(features))
    assert np.allclose(features[:5], projected[-1])
    assert np.allclose(features[5:10], projected.mean(axis=0))
    assert np.allclose(features[10:15], projected.max(axis=0))
    assert np.allclose(features[25:30], adapter.ac_state[0] * adapter.dc_state[0])
    assert features[-3:].shape == (3,)


def test_same_seed_produces_deterministic_output():
    sequence = np.array(
        [
            [0.25, -0.5, 0.75, 1.0],
            [-1.0, 0.25, 0.5, -0.25],
            [0.1, 0.2, 0.3, 0.4],
        ]
    )
    first = HelixTemporalAdapter(input_dim=4, projection_dim=6, seed=123)
    second = HelixTemporalAdapter(input_dim=4, projection_dim=6, seed=123)

    first_projected, first_features = _run_sequence(first, sequence)
    second_projected, second_features = _run_sequence(second, sequence)

    assert np.array_equal(first.W_project, second.W_project)
    assert np.allclose(first_projected, second_projected)
    assert np.allclose(first_features, second_features)


def test_ac_trace_responds_and_fades_faster_than_dc_trace():
    adapter = HelixTemporalAdapter(
        input_dim=2,
        projection_dim=4,
        ac_decay=0.2,
        dc_decay=0.8,
        seed=7,
    )

    adapter.step(np.array([1.0, -0.5]))
    initial_ac_norm = np.linalg.norm(adapter.ac_state)
    initial_dc_norm = np.linalg.norm(adapter.dc_state)

    for _ in range(4):
        adapter.step(np.zeros(2))

    faded_ac_norm = np.linalg.norm(adapter.ac_state)
    faded_dc_norm = np.linalg.norm(adapter.dc_state)

    assert initial_ac_norm > initial_dc_norm
    assert faded_ac_norm < initial_ac_norm * 0.1
    assert faded_dc_norm > initial_dc_norm * 0.05


def test_write_gate_changes_dc_updates():
    low_gate = HelixTemporalAdapter(
        input_dim=2,
        projection_dim=3,
        dc_decay=0.5,
        seed=5,
    )
    high_gate = HelixTemporalAdapter(
        input_dim=2,
        projection_dim=3,
        dc_decay=0.5,
        seed=5,
    )
    for adapter, bias in [(low_gate, -10.0), (high_gate, 10.0)]:
        adapter.projection[:] = 0.5
        adapter.gate_bias[:] = bias

    sample = np.array([1.0, 1.0])
    low_gate.step(sample)
    high_gate.step(sample)

    assert high_gate.last_write_gate.mean() > low_gate.last_write_gate.mean()
    assert np.linalg.norm(high_gate.dc_state) > np.linalg.norm(low_gate.dc_state) * 100.0


def test_diagnostics_are_opt_in_and_have_expected_scalar_shape():
    disabled = HelixTemporalAdapter(input_dim=2, projection_dim=4, seed=1)
    enabled = HelixTemporalAdapter(
        input_dim=2,
        projection_dim=4,
        seed=1,
        diagnostics=True,
    )

    disabled_diagnostics = disabled.get_diagnostics()
    assert disabled_diagnostics["diagnostics_enabled"] is False

    enabled.step(np.array([0.5, -0.25]))
    diagnostics = enabled.get_diagnostics()

    expected_keys = {
        "diagnostics_enabled",
        "input_dim",
        "input_width",
        "projection_dim",
        "feature_dim",
        "batch_size",
        "step_count",
        "ac_decay",
        "dc_decay",
        "last_projected_norm",
        "ac_norm",
        "dc_norm",
        "write_gate_mean",
        "write_gate_min",
        "write_gate_max",
        "ac_trajectory",
        "dc_trajectory",
        "gate_trajectory",
    }
    assert set(diagnostics) == expected_keys
    assert diagnostics["input_dim"] == 2
    assert diagnostics["projection_dim"] == 4
    assert diagnostics["feature_dim"] == enabled.feature_dim
    assert diagnostics["step_count"] == 1
    assert diagnostics["diagnostics_enabled"] is True
    for key, value in diagnostics.items():
        if key.endswith("_trajectory"):
            assert value.shape == (1, 1, 4)
        else:
            assert np.isscalar(value)
    assert diagnostics["write_gate_min"] <= diagnostics["write_gate_mean"]
    assert diagnostics["write_gate_mean"] <= diagnostics["write_gate_max"]


def test_classic_benchmark_data_smoke_if_importable():
    pytest.importorskip("experiments.run_classic_benchmarks")
    from core.constants import Config512D
    from experiments.run_classic_benchmarks import CLASSIC_TASKS, generate_classic_task_data

    cfg = Config512D(seed=99, process_steps=1)

    for offset, task_id in enumerate(CLASSIC_TASKS):
        inputs, targets = generate_classic_task_data(task_id, n_samples=2, seed=99 + offset, cfg=cfg)
        assert inputs.shape[0] == 2
        assert targets.shape == (2, cfg.output_dim)
        assert inputs.shape[-1] == cfg.input_dim
        assert np.all(np.isfinite(inputs))
        assert np.all(np.isfinite(targets))
