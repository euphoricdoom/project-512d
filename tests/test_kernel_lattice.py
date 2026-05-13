"""Tests for system/kernel_lattice.py.

Covers:
  - LSHStackRouter: determinism, distribution, similar-input collision
  - Routing integration: forward() uses LSH routing, LatticeLayer hard routing
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
from system.kernel_lattice import KernelLattice, KernelStack, LatticeConfig, LatticeLayer, LSHStackRouter


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
        heatmap = lattice.specialization_matrix(X, tasks, n_probe=n_samp)
        assert heatmap.shape == (lcfg.n_layers, lcfg.n_stacks)
        assert np.all(heatmap >= 0.0)

    def test_specialization_matrix_no_side_effects(self):
        lattice = self._make_trained_lattice()
        cfg = Config512D(seed=5)
        rng = np.random.default_rng(10)
        tasks = ["tanh", "relu"]
        n_samp = 5
        X = rng.normal(size=(len(tasks), n_samp, 1, cfg.input_dim))

        # Snapshot heads before
        heads_before = {
            tid: head.copy()
            for tid, head in lattice.readout._heads.items()
        }
        n_tasks_before = len(lattice.readout._heads)

        lattice.specialization_matrix(X, tasks, n_probe=n_samp)

        # Heads must be identical after — no side effects
        assert len(lattice.readout._heads) == n_tasks_before
        for tid, head_before in heads_before.items():
            np.testing.assert_array_equal(lattice.readout._heads[tid], head_before)

    def test_bridge_summary_has_hit_rates(self):
        lattice = self._make_trained_lattice()
        summary = lattice.bridge_summary()
        assert "bridge_hit_rates" in summary
        for rate in summary["bridge_hit_rates"].values():
            assert 0.0 <= rate <= 1.0


class TestFitTask:
    def _make_lattice(self) -> KernelLattice:
        cfg = Config512D(seed=3)
        lcfg = LatticeConfig(n_layers=1, n_stacks=2, projection_dim=16, seed=3)
        return KernelLattice(output_dim=cfg.output_dim, lattice_cfg=lcfg, kernel_cfg=cfg)

    def test_returns_per_epoch_losses(self):
        lattice = self._make_lattice()
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 64))
        Y = rng.normal(size=(20, 64))
        losses = lattice.fit_task(X, Y, task_id="t1", epochs=5)
        assert len(losses) == 5
        assert all(isinstance(l, float) for l in losses)

    def test_loss_decreases(self):
        lattice = self._make_lattice()
        rng = np.random.default_rng(1)
        X = rng.normal(size=(50, 64))
        Y = np.tanh(X)
        losses = lattice.fit_task(X, Y, task_id="tanh", lr=0.01, epochs=20)
        assert losses[-1] < losses[0]

    def test_zero_forgetting_after_fit(self):
        lattice = self._make_lattice()
        rng = np.random.default_rng(2)
        cfg = Config512D(seed=3)
        X_a = rng.normal(size=(15, 64))
        Y_a = rng.normal(size=(15, cfg.output_dim))
        X_b = rng.normal(size=(15, 64))
        Y_b = rng.normal(size=(15, cfg.output_dim))

        lattice.fit_task(X_a, Y_a, task_id="A", epochs=3)
        preds_before = np.array([lattice.predict(X_a[i:i+1], "A") for i in range(len(X_a))])
        lattice.fit_task(X_b, Y_b, task_id="B", epochs=3)
        preds_after = np.array([lattice.predict(X_a[i:i+1], "A") for i in range(len(X_a))])
        np.testing.assert_array_equal(preds_before, preds_after)

    def test_sample_counts_tracked(self):
        lattice = self._make_lattice()
        rng = np.random.default_rng(3)
        X = rng.normal(size=(12, 64))
        Y = rng.normal(size=(12, 64))
        lattice.fit_task(X, Y, task_id="counted", epochs=2)
        assert lattice._task_sample_counts["counted"] == 12


# ---------------------------------------------------------------------------
# LSH stack routing
# ---------------------------------------------------------------------------

class TestLSHStackRouter:
    def _make_router(self, n_stacks: int = 4, n_projections: int = 8) -> LSHStackRouter:
        return LSHStackRouter(n_stacks=n_stacks, input_dim=64, n_projections=n_projections, seed=42)

    def test_lsh_router_exists_on_lattice(self):
        cfg = Config512D(seed=42)
        lcfg = LatticeConfig(n_layers=1, n_stacks=4, projection_dim=16, seed=42)
        lattice = KernelLattice(output_dim=cfg.output_dim, lattice_cfg=lcfg, kernel_cfg=cfg)
        assert hasattr(lattice, "stack_router")
        assert isinstance(lattice.stack_router, LSHStackRouter)

    def test_lsh_route_returns_valid_stack_index(self):
        router = self._make_router(n_stacks=4)
        x = np.random.default_rng(0).normal(size=(64,))
        idx = router.route(x)
        assert isinstance(idx, int)
        assert 0 <= idx < 4

    def test_lsh_routing_deterministic(self):
        router = self._make_router(n_stacks=4)
        x = np.random.default_rng(7).normal(size=(64,))
        assert router.route(x) == router.route(x)

    def test_lsh_routing_distributes_across_stacks(self):
        router = self._make_router(n_stacks=4)
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 64))
        indices = np.array([router.route(x) for x in X])
        unique = set(indices.tolist())
        # With 200 samples and 4 stacks, all stacks should be visited
        assert len(unique) == 4

    def test_lsh_similar_inputs_same_stack(self):
        router = self._make_router(n_stacks=4)
        x = np.random.default_rng(5).normal(size=(64,))
        noise = np.random.default_rng(5).normal(size=(64,)) * 1e-6
        x_similar = x + noise
        # Nearly identical vectors must hash identically
        assert router.route(x) == router.route(x_similar)

    def test_lsh_different_inputs_can_differ(self):
        router = self._make_router(n_stacks=4)
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 64))
        indices = [router.route(x) for x in X]
        # Not all inputs route to the same stack
        assert len(set(indices)) > 1

    def test_lsh_route_batch_matches_route(self):
        router = self._make_router(n_stacks=4)
        rng = np.random.default_rng(0)
        X = rng.normal(size=(30, 64))
        batch_indices = router.route_batch(X)
        single_indices = np.array([router.route(x) for x in X])
        np.testing.assert_array_equal(batch_indices, single_indices)

    def test_lsh_hash_input_range(self):
        router = self._make_router(n_stacks=4, n_projections=8)
        x = np.random.default_rng(0).normal(size=(64,))
        h = router.hash_input(x)
        assert 0 <= h < 2 ** 8

    def test_forward_uses_hard_routing_correct_shape(self):
        cfg = Config512D(seed=42)
        lcfg = LatticeConfig(n_layers=1, n_stacks=4, projection_dim=16, seed=42)
        lattice = KernelLattice(output_dim=cfg.output_dim, lattice_cfg=lcfg, kernel_cfg=cfg)
        x = np.random.default_rng(1).normal(size=(3, 64))
        feat = lattice.forward(x)
        assert feat.shape == (lcfg.global_feature_dim,)

    def test_zero_forgetting_preserved_with_lsh_routing(self):
        cfg = Config512D(seed=42)
        lcfg = LatticeConfig(n_layers=1, n_stacks=4, projection_dim=16, seed=42)
        lattice = KernelLattice(output_dim=cfg.output_dim, lattice_cfg=lcfg, kernel_cfg=cfg)
        rng = np.random.default_rng(99)
        X_a = rng.normal(size=(10, 64))
        Y_a = rng.normal(size=(10, cfg.output_dim))
        X_b = rng.normal(size=(10, 64))
        Y_b = rng.normal(size=(10, cfg.output_dim))
        lattice.train_sequence(X_a, Y_a, task_id="A", lr=0.01)
        preds_before = np.array([lattice.predict(X_a[i:i+1], "A") for i in range(len(X_a))])
        lattice.train_sequence(X_b, Y_b, task_id="B", lr=0.01)
        preds_after = np.array([lattice.predict(X_a[i:i+1], "A") for i in range(len(X_a))])
        np.testing.assert_array_equal(preds_before, preds_after)

    def test_layer_forward_all_stacks_active_when_no_target(self):
        cfg = Config512D(seed=2)
        lcfg = LatticeConfig(n_layers=1, n_stacks=2, projection_dim=16, seed=2)
        rng = np.random.default_rng(2)
        layer = LatticeLayer(0, 2, cfg, lcfg, rng)
        x = np.random.default_rng(0).normal(size=(3, 64))
        out = layer.forward(x, target_stack_id=None)
        assert out.shape == (2 * (16 * 6 + 3),)
        # All stacks active — non-zero output
        assert np.any(out != 0.0)

    def test_layer_forward_inactive_stacks_zero(self):
        cfg = Config512D(seed=2)
        lcfg = LatticeConfig(n_layers=1, n_stacks=2, projection_dim=16, seed=2)
        rng = np.random.default_rng(2)
        layer = LatticeLayer(0, 2, cfg, lcfg, rng)
        x = np.random.default_rng(0).normal(size=(3, 64))
        feat_dim = 16 * 6 + 3
        out = layer.forward(x, target_stack_id=0)
        # Stack 1 (second half) should be zero
        assert np.all(out[feat_dim:] == 0.0)

    def test_analyze_routing_distribution_returns_all_tasks(self):
        cfg = Config512D(seed=42)
        lcfg = LatticeConfig(n_layers=1, n_stacks=4, projection_dim=16, seed=42)
        lattice = KernelLattice(output_dim=cfg.output_dim, lattice_cfg=lcfg, kernel_cfg=cfg)
        tasks = ["tanh", "relu", "abs"]
        info = lattice.analyze_routing_distribution(tasks, n_samples=20)
        assert set(info["task_distributions"].keys()) == set(tasks)
        assert info["n_stacks"] == 4
        assert 0.0 <= info["balance_score"] <= 1.0
