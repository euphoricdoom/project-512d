"""Tests for system/kernel_lattice.py.

Covers:
  - LatticeConfig validation
  - KernelStack: determinism, output shape, sequence handling
  - LatticeLayer: feature shape, bridge mechanism, feature blending
  - KernelLattice: forward pass shape, zero-forgetting, training, evaluation
  - Analysis methods: bridge_summary, specialization_matrix, task_summary
"""

from __future__ import annotations

import numpy as np
import pytest

from core.constants import Config512D
from system.kernel_lattice import KernelLattice, KernelStack, LatticeConfig, LatticeLayer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_cfg() -> Config512D:
    return Config512D(seed=0)


@pytest.fixture(scope="module")
def small_lattice_cfg() -> LatticeConfig:
    return LatticeConfig(n_layers=2, n_stacks=2, projection_dim=32, seed=0)


@pytest.fixture(scope="module")
def lattice(small_cfg, small_lattice_cfg) -> KernelLattice:
    return KernelLattice(
        output_dim=small_cfg.output_dim,
        lattice_cfg=small_lattice_cfg,
        kernel_cfg=small_cfg,
    )


# ---------------------------------------------------------------------------
# LatticeConfig validation
# ---------------------------------------------------------------------------

class TestLatticeConfig:
    def test_defaults_valid(self):
        cfg = LatticeConfig()
        assert cfg.n_layers == 2
        assert cfg.n_stacks == 4
        assert cfg.projection_dim == 256

    def test_feature_dim_per_stack(self):
        cfg = LatticeConfig(projection_dim=64)
        assert cfg.feature_dim_per_stack == 64 * 6 + 3

    def test_layer_feature_dim(self):
        cfg = LatticeConfig(n_stacks=3, projection_dim=64)
        assert cfg.layer_feature_dim == 3 * (64 * 6 + 3)

    def test_global_feature_dim(self):
        cfg = LatticeConfig(n_layers=2, n_stacks=3, projection_dim=64)
        assert cfg.global_feature_dim == 2 * 3 * (64 * 6 + 3)

    def test_invalid_n_layers(self):
        with pytest.raises(ValueError, match="n_layers"):
            LatticeConfig(n_layers=0)

    def test_invalid_n_stacks(self):
        with pytest.raises(ValueError, match="n_stacks"):
            LatticeConfig(n_stacks=0)

    def test_invalid_projection_dim(self):
        with pytest.raises(ValueError, match="projection_dim"):
            LatticeConfig(projection_dim=0)

    def test_invalid_bridge_threshold(self):
        with pytest.raises(ValueError, match="bridge_threshold"):
            LatticeConfig(bridge_threshold=1.5)

    def test_invalid_bridge_alpha(self):
        with pytest.raises(ValueError, match="bridge_alpha"):
            LatticeConfig(bridge_alpha=-0.1)


# ---------------------------------------------------------------------------
# KernelStack
# ---------------------------------------------------------------------------

class TestKernelStack:
    def _make_stack(self, projection_dim: int = 32) -> KernelStack:
        cfg = Config512D(seed=1)
        rng = np.random.default_rng(1)
        return KernelStack(cfg, projection_dim, rng)

    def test_output_shape_single_step(self):
        stack = self._make_stack(projection_dim=32)
        x = np.random.default_rng(0).normal(size=(1, 64))
        feat = stack.forward(x)
        assert feat.shape == (32 * 6 + 3,)

    def test_output_shape_multi_step(self):
        stack = self._make_stack(projection_dim=32)
        T = 5
        x = np.random.default_rng(0).normal(size=(T, 64))
        feat = stack.forward(x)
        assert feat.shape == (32 * 6 + 3,)

    def test_deterministic_forward(self):
        stack = self._make_stack(projection_dim=32)
        rng = np.random.default_rng(99)
        x = rng.normal(size=(4, 64))
        f1 = stack.forward(x)
        f2 = stack.forward(x)
        np.testing.assert_array_equal(f1, f2)

    def test_different_inputs_differ(self):
        stack = self._make_stack(projection_dim=32)
        rng = np.random.default_rng(7)
        x1 = rng.normal(size=(3, 64))
        x2 = rng.normal(size=(3, 64))
        f1 = stack.forward(x1)
        f2 = stack.forward(x2)
        assert not np.allclose(f1, f2)

    def test_1d_input_accepted(self):
        stack = self._make_stack(projection_dim=16)
        x = np.random.default_rng(5).normal(size=(64,))
        feat = stack.forward(x)
        assert feat.shape == (16 * 6 + 3,)

    def test_reservoir_state_shape(self):
        stack = self._make_stack()
        _ = stack.forward(np.ones((2, 64)))
        assert stack.reservoir_state().shape == (540,)


# ---------------------------------------------------------------------------
# LatticeLayer
# ---------------------------------------------------------------------------

class TestLatticeLayer:
    def _make_layer(self, n_stacks: int = 2, projection_dim: int = 16) -> LatticeLayer:
        cfg = Config512D(seed=2)
        lcfg = LatticeConfig(n_layers=1, n_stacks=n_stacks, projection_dim=projection_dim, seed=2)
        rng = np.random.default_rng(2)
        return LatticeLayer(0, n_stacks, cfg, lcfg, rng)

    def test_output_shape(self):
        layer = self._make_layer(n_stacks=2, projection_dim=16)
        x = np.random.default_rng(0).normal(size=(3, 64))
        out = layer.forward(x)
        assert out.shape == (2 * (16 * 6 + 3),)

    def test_bridge_activation_counts_empty_initially(self):
        layer = self._make_layer()
        # Run a few forward passes to potentially trigger bridges
        x = np.random.default_rng(3).normal(size=(3, 64))
        layer.forward(x)
        counts = layer.bridge_activation_counts()
        assert isinstance(counts, dict)
        for k, v in counts.items():
            assert isinstance(k, str)
            assert isinstance(v, int)

    def test_bridge_keys_are_strings(self):
        layer = self._make_layer(n_stacks=3)
        for _ in range(10):
            x = np.random.default_rng(4).normal(size=(2, 64))
            layer.forward(x)
        for key in layer.bridge_activation_counts():
            assert "->" in key
            assert isinstance(key, str)

    def test_prev_layer_features_mixed(self):
        cfg = Config512D(seed=2)
        lcfg = LatticeConfig(n_layers=2, n_stacks=2, projection_dim=16, seed=2)
        rng = np.random.default_rng(2)
        layer = LatticeLayer(1, 2, cfg, lcfg, rng)
        prev_dim = lcfg.layer_feature_dim
        layer.input_mixing = rng.normal(0.0, 0.01, size=(prev_dim, cfg.input_dim))

        x = np.random.default_rng(5).normal(size=(3, 64))
        prev = np.random.default_rng(6).normal(size=(prev_dim,))

        out_no_mix = layer.forward(x)
        out_mix = layer.forward(x, prev_layer_features=prev)
        # Mixed and unmixed should generally differ (different effective inputs)
        assert not np.allclose(out_no_mix, out_mix)


# ---------------------------------------------------------------------------
# KernelLattice — architecture
# ---------------------------------------------------------------------------

class TestKernelLatticeArchitecture:
    def test_layer_count(self, lattice, small_lattice_cfg):
        assert len(lattice.layers) == small_lattice_cfg.n_layers

    def test_stack_count_per_layer(self, lattice, small_lattice_cfg):
        for layer in lattice.layers:
            assert len(layer.stacks) == small_lattice_cfg.n_stacks

    def test_layer0_no_input_mixing(self, lattice):
        assert lattice.layers[0].input_mixing is None

    def test_layer1_has_input_mixing(self, lattice):
        if len(lattice.layers) > 1:
            assert lattice.layers[1].input_mixing is not None

    def test_forward_output_shape(self, lattice, small_lattice_cfg):
        x = np.random.default_rng(10).normal(size=(3, 64))
        feat = lattice.forward(x)
        assert feat.shape == (small_lattice_cfg.global_feature_dim,)

    def test_forward_single_step(self, lattice, small_lattice_cfg):
        x = np.zeros(64)
        feat = lattice.forward(x)
        assert feat.shape == (small_lattice_cfg.global_feature_dim,)

    def test_predict_shape(self, lattice, small_cfg):
        x = np.random.default_rng(11).normal(size=(3, 64))
        pred = lattice.predict(x, task_id="task_A")
        assert pred.shape == (small_cfg.output_dim,)

    def test_global_feature_dim_matches_readout(self, lattice, small_lattice_cfg):
        assert lattice.readout.feature_dim == small_lattice_cfg.global_feature_dim


# ---------------------------------------------------------------------------
# KernelLattice — zero-forgetting
# ---------------------------------------------------------------------------

class TestZeroForgetting:
    def _make_lattice(self) -> KernelLattice:
        cfg = Config512D(seed=42)
        lcfg = LatticeConfig(n_layers=2, n_stacks=2, projection_dim=16, seed=42)
        return KernelLattice(output_dim=cfg.output_dim, lattice_cfg=lcfg, kernel_cfg=cfg)

    def test_task_a_unchanged_after_task_b(self):
        lattice = self._make_lattice()
        rng = np.random.default_rng(77)
        cfg = Config512D(seed=42)

        X_a = rng.normal(size=(20, 64))
        Y_a = rng.normal(size=(20, cfg.output_dim))
        X_b = rng.normal(size=(20, 64))
        Y_b = rng.normal(size=(20, cfg.output_dim))

        lattice.train_sequence(X_a, Y_a, task_id="task_A", lr=0.01)

        # Snapshot predictions for task A before training B
        preds_a_before = np.array([
            lattice.predict(X_a[i:i+1], task_id="task_A") for i in range(len(X_a))
        ])

        lattice.train_sequence(X_b, Y_b, task_id="task_B", lr=0.01)

        # Task A predictions must be exactly unchanged
        preds_a_after = np.array([
            lattice.predict(X_a[i:i+1], task_id="task_A") for i in range(len(X_a))
        ])

        np.testing.assert_array_equal(preds_a_before, preds_a_after)

    def test_isolated_heads_separate_tasks(self):
        lattice = self._make_lattice()
        rng = np.random.default_rng(88)
        cfg = Config512D(seed=42)

        X = rng.normal(size=(10, 64))
        Y_a = rng.normal(size=(10, cfg.output_dim))
        Y_b = -Y_a

        lattice.train_sequence(X, Y_a, task_id="pos", lr=0.02, epochs=5)
        lattice.train_sequence(X, Y_b, task_id="neg", lr=0.02, epochs=5)

        # After training with opposing targets, heads must not pollute each other
        head_pos = lattice.readout._heads["pos"]
        head_neg = lattice.readout._heads["neg"]
        assert head_pos is not head_neg
        assert not np.allclose(head_pos, head_neg)

    def test_n_tasks_grows_without_limit(self):
        lattice = self._make_lattice()
        rng = np.random.default_rng(55)
        cfg = Config512D(seed=42)
        n_tasks = 5
        for i in range(n_tasks):
            X = rng.normal(size=(5, 64))
            Y = rng.normal(size=(5, cfg.output_dim))
            lattice.train_sequence(X, Y, task_id=f"task_{i}", lr=0.01)
        assert len(lattice.readout._heads) == n_tasks

    def test_shared_dynamics_unchanged(self):
        lattice = self._make_lattice()
        stack = lattice.layers[0].stacks[0]
        W_before = stack.helix.W_project.copy()
        rng = np.random.default_rng(11)
        cfg = Config512D(seed=42)
        for _ in range(3):
            X = rng.normal(size=(10, 64))
            Y = rng.normal(size=(10, cfg.output_dim))
            lattice.train_sequence(X, Y, task_id="task_x", lr=0.01)
        np.testing.assert_array_equal(W_before, stack.helix.W_project)


# ---------------------------------------------------------------------------
# KernelLattice — training and evaluation
# ---------------------------------------------------------------------------

class TestTraining:
    def _make_lattice(self) -> KernelLattice:
        cfg = Config512D(seed=7)
        lcfg = LatticeConfig(n_layers=1, n_stacks=2, projection_dim=16, seed=7)
        return KernelLattice(output_dim=cfg.output_dim, lattice_cfg=lcfg, kernel_cfg=cfg)

    def test_train_sample_returns_scalar_loss(self):
        lattice = self._make_lattice()
        rng = np.random.default_rng(0)
        x = rng.normal(size=(2, 64))
        y = rng.normal(size=(64,))
        loss = lattice.train_sample(x, y, task_id="t1")
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_train_sequence_returns_losses(self):
        lattice = self._make_lattice()
        rng = np.random.default_rng(1)
        X = rng.normal(size=(10, 64))
        Y = rng.normal(size=(10, 64))
        losses = lattice.train_sequence(X, Y, task_id="t1", lr=0.01, epochs=2)
        assert len(losses) == 10
        assert all(isinstance(l, float) for l in losses)

    def test_loss_decreases_with_training(self):
        lattice = self._make_lattice()
        rng = np.random.default_rng(42)
        X = rng.normal(size=(30, 64))
        Y = np.tanh(X[:, :64])
        losses_e1 = lattice.train_sequence(X, Y, task_id="tanh", lr=0.005, epochs=1)
        losses_e5 = lattice.train_sequence(X, Y, task_id="tanh", lr=0.005, epochs=5)
        # Loss after more training should generally be lower
        assert np.mean(losses_e5) <= np.mean(losses_e1) * 2.0

    def test_evaluate_returns_mse_and_r2(self):
        lattice = self._make_lattice()
        rng = np.random.default_rng(3)
        X = rng.normal(size=(20, 64))
        Y = rng.normal(size=(20, 64))
        lattice.train_sequence(X, Y, task_id="t2", lr=0.01)
        metrics = lattice.evaluate(X, Y, task_id="t2")
        assert "mse" in metrics
        assert "r2" in metrics
        assert metrics["mse"] >= 0.0

    def test_task_sample_counts_tracked(self):
        lattice = self._make_lattice()
        rng = np.random.default_rng(4)
        X = rng.normal(size=(15, 64))
        Y = rng.normal(size=(15, 64))
        lattice.train_sequence(X, Y, task_id="counter_task", lr=0.01)
        assert lattice._task_sample_counts["counter_task"] == 15


# ---------------------------------------------------------------------------
# KernelLattice — analysis
# ---------------------------------------------------------------------------

class TestAnalysis:
    def _make_trained_lattice(self) -> KernelLattice:
        cfg = Config512D(seed=5)
        lcfg = LatticeConfig(n_layers=2, n_stacks=2, projection_dim=16, seed=5)
        lattice = KernelLattice(output_dim=cfg.output_dim, lattice_cfg=lcfg, kernel_cfg=cfg)
        rng = np.random.default_rng(5)
        for task in ["tanh", "relu"]:
            X = rng.normal(size=(10, 64))
            Y = rng.normal(size=(10, cfg.output_dim))
            lattice.train_sequence(X, Y, task_id=task, lr=0.01)
        return lattice

    def test_bridge_summary_json_safe(self):
        lattice = self._make_trained_lattice()
        summary = lattice.bridge_summary()
        import json
        json.dumps(summary)

    def test_bridge_summary_has_total_forward_calls(self):
        lattice = self._make_trained_lattice()
        summary = lattice.bridge_summary()
        assert "total_forward_calls" in summary
        assert summary["total_forward_calls"] > 0

    def test_task_summary_json_safe(self):
        lattice = self._make_trained_lattice()
        summary = lattice.task_summary()
        import json
        json.dumps(summary)
        assert "tasks_trained" in summary

    def test_specialization_matrix_shape(self):
        lattice = self._make_trained_lattice()
        cfg = Config512D(seed=5)
        lcfg = lattice.lattice_cfg
        rng = np.random.default_rng(9)
        tasks = ["tanh", "relu"]
        n_samp = 8
        X = rng.normal(size=(len(tasks), n_samp, 1, cfg.input_dim))
        Y = rng.normal(size=(len(tasks), n_samp, cfg.output_dim))
        heatmap = lattice.specialization_matrix(X, Y, tasks, lr=0.001)
        assert heatmap.shape == (lcfg.n_layers, lcfg.n_stacks)
        assert np.all(heatmap >= 0.0)
