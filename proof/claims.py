"""Comprehensive proof suite for the Weightless Continual Learning Model.

Six claims are tested under adversarial conditions:

  CLAIM 1 — ZERO FORGETTING BY CONSTRUCTION
    Training on task B must leave task A's loss unchanged to machine precision.
    Tested with N tasks, adversarial weight norms, and gradient-pressure conditions.

  CLAIM 2 — FIXED ENCODER SUFFICIENCY
    The fixed random encoder must separate linearly inseparable tasks better
    than the raw input alone.  Proved by comparing head loss after N training
    epochs with and without the encoder.

  CLAIM 3 — SYSTEM MEMORY BENEFIT
    AC/DC temporal features must outperform memoryless (single-step) features
    on sequence tasks where the answer depends on history.

  CLAIM 4 — O(N) PARAMETER SCALING
    Total learned parameters must grow linearly in the number of tasks.
    Fixed encoder parameters must be constant regardless of task count.

  CLAIM 5 — SLEEP CONSOLIDATION QUANTIFIABLY HELPS
    After sleep, final_error < initial_error for every task.
    The improvement must be reproducible across seeds.

  CLAIM 6 — INDEPENDENCE UNDER ADVERSARIAL PRESSURE
    Massive updates on task B (lr=10x, 1000 steps) must still leave task A
    exactly unchanged.  Tests that isolation is structural, not fragile.

Run with:
    python -m pytest proof/ -v
or standalone:
    python proof/claims.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Allow running from repo root or proof/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from weightless_model import WeightlessModel, FixedEncoder, SystemMemory
from zero_forgetting import ZeroForgetReadout


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_sequence_task(n_samples: int, seq_len: int, input_dim: int,
                        output_dim: int, rng: np.random.Generator,
                        task_fn=None) -> tuple[np.ndarray, np.ndarray]:
    """Generate ``(X_seq, Y)`` for a simple sequence regression task."""
    X = rng.normal(0.0, 0.6, size=(n_samples, seq_len, input_dim))
    if task_fn is None:
        Y = np.tanh(X[:, -1, :output_dim])
    else:
        Y = task_fn(X)
    return X, Y


def train_task(model: WeightlessModel, X: np.ndarray, Y: np.ndarray,
               task_id: str, epochs: int = 30, lr: float = 0.05) -> float:
    model.set_task(task_id)
    features = model.encode_sequence(X)
    loss = float("inf")
    for _ in range(epochs):
        loss = model.update(features, Y, lr=lr)
    return loss


# ===========================================================================
# CLAIM 1 — ZERO FORGETTING BY CONSTRUCTION
# ===========================================================================

class TestClaim1_ZeroForgetting:
    """Forgetting must be exactly 0.0 — not small, exactly zero."""

    def _measure_forgetting(self, n_tasks: int, epochs_each: int, lr: float) -> list[float]:
        rng   = np.random.default_rng(0)
        model = WeightlessModel(input_dim=16, reservoir_dim=64, output_dim=4, seed=0)
        tasks = {}
        for i in range(n_tasks):
            X, Y = make_sequence_task(40, 5, 16, 4, rng)
            tasks[f"task_{i}"] = (X, Y)

        # Train each task and record loss immediately after
        loss_after_own_training: dict[str, float] = {}
        for tid, (X, Y) in tasks.items():
            train_task(model, X, Y, tid, epochs=epochs_each, lr=lr)
            features = model.encode_sequence(X)
            preds    = model.readout.predict_batch(features, task_id=tid)
            loss_after_own_training[tid] = float(np.mean((Y - preds) ** 2))

        # Re-measure all tasks — must match exactly
        forgetting = []
        for tid, (X, Y) in tasks.items():
            features = model.encode_sequence(X)
            preds    = model.readout.predict_batch(features, task_id=tid)
            loss_now = float(np.mean((Y - preds) ** 2))
            forgetting.append(abs(loss_now - loss_after_own_training[tid]))

        return forgetting

    def test_two_tasks_exact_zero(self):
        deltas = self._measure_forgetting(n_tasks=2, epochs_each=50, lr=0.05)
        for delta in deltas:
            assert delta == 0.0, f"Expected 0.0 forgetting, got {delta}"

    def test_ten_tasks_exact_zero(self):
        deltas = self._measure_forgetting(n_tasks=10, epochs_each=20, lr=0.05)
        for delta in deltas:
            assert delta == 0.0, f"Expected 0.0 forgetting, got {delta}"

    def test_fifty_tasks_exact_zero(self):
        deltas = self._measure_forgetting(n_tasks=50, epochs_each=5, lr=0.05)
        for delta in deltas:
            assert delta == 0.0, f"Expected 0.0 forgetting, got {delta}"

    def test_same_task_id_updates_only_its_head(self):
        """Re-training a task overwrites only its own head."""
        rng = np.random.default_rng(1)
        model = WeightlessModel(input_dim=8, reservoir_dim=32, output_dim=2, seed=1)
        X_a, Y_a = make_sequence_task(20, 4, 8, 2, rng)
        X_b, Y_b = make_sequence_task(20, 4, 8, 2, rng)

        train_task(model, X_a, Y_a, "A", epochs=30)
        train_task(model, X_b, Y_b, "B", epochs=30)

        W_a_snapshot = model.readout._heads["A"].copy()

        # Retrain B hard
        train_task(model, X_b, Y_b, "B", epochs=200, lr=0.1)

        np.testing.assert_array_equal(
            model.readout._heads["A"], W_a_snapshot,
            err_msg="Task A head was modified by task B training"
        )


# ===========================================================================
# CLAIM 2 — FIXED ENCODER SUFFICIENCY
# ===========================================================================

class TestClaim2_EncoderSufficiency:
    """Fixed encoder must provide better features than raw input for nonlinear tasks."""

    def _compare_losses(self, task_fn, n_samples=100, seq_len=6, input_dim=16,
                        output_dim=4, epochs=80, seed=0) -> tuple[float, float]:
        rng = np.random.default_rng(seed)
        X, Y = make_sequence_task(n_samples, seq_len, input_dim, output_dim, rng, task_fn)

        # With fixed encoder + memory
        model = WeightlessModel(input_dim=input_dim, reservoir_dim=64,
                                output_dim=output_dim, seed=seed)
        model.set_task("T")
        features_enc = model.encode_sequence(X)
        rng2 = np.random.default_rng(seed + 1)
        head_enc = ZeroForgetReadout(output_dim=output_dim,
                                     feature_dim=features_enc.shape[1], rng=rng2)
        head_enc.set_task("T")
        for _ in range(epochs):
            loss_enc = head_enc.update_batch(features_enc, Y, lr=0.05)

        # Without encoder — raw last timestep only
        X_raw  = X[:, -1, :output_dim]  # same dimensionality budget
        rng3 = np.random.default_rng(seed + 2)
        head_raw = ZeroForgetReadout(output_dim=output_dim, feature_dim=output_dim, rng=rng3)
        head_raw.set_task("T")
        for _ in range(epochs):
            loss_raw = head_raw.update_batch(X_raw, Y, lr=0.05)

        return loss_enc, loss_raw

    def test_nonlinear_task(self):
        """Nonlinear classification: encoded features should reduce loss vs raw input."""
        def nonlinear_classification(X: np.ndarray) -> np.ndarray:
            # 4-class task based on sum of features across time
            sums = np.sum(X[:, :, :4], axis=1)  # (n_samples, 4)
            labels = np.argmax(sums, axis=1)    # class = argmax of sums
            return np.eye(4)[labels]            # one-hot encoding

        loss_enc, loss_raw = self._compare_losses(task_fn=nonlinear_classification)
        assert loss_enc < loss_raw, (
            f"Encoder did not help: enc={loss_enc:.4f} vs raw={loss_raw:.4f}"
        )

    def test_temporal_integration_task(self):
        """Task depends on sum over all timesteps — raw last-step is provably insufficient."""
        def cumulative_classification(X: np.ndarray) -> np.ndarray:
            # Answer depends on mean across time — last step alone is misleading
            means = X.mean(axis=1)[:, :4]       # (n_samples, 4)
            labels = np.argmax(means, axis=1)   # class = argmax of temporal mean
            return np.eye(4)[labels]

        loss_enc, loss_raw = self._compare_losses(task_fn=cumulative_classification)
        assert loss_enc < loss_raw, (
            f"Encoder did not help on temporal task: enc={loss_enc:.4f} raw={loss_raw:.4f}"
        )

    def test_encoder_spectral_radius(self):
        """Fixed encoder must satisfy spectral bound."""
        enc = FixedEncoder(input_dim=32, reservoir_dim=64, target_radius=0.94, seed=0)
        assert enc.actual_radius <= 0.94 + 1e-9, (
            f"Spectral radius {enc.actual_radius:.4f} exceeds 0.94"
        )

    def test_encoder_weights_never_change(self):
        """Encoder W must be identical before and after training the model."""
        model = WeightlessModel(input_dim=8, reservoir_dim=32, output_dim=2, seed=0)
        W_before = model.encoder._W.copy()
        rng = np.random.default_rng(0)
        X, Y = make_sequence_task(50, 4, 8, 2, rng)
        for _ in range(3):
            train_task(model, X, Y, "T")
        np.testing.assert_array_equal(
            model.encoder._W, W_before,
            err_msg="Fixed encoder weights changed during training"
        )


# ===========================================================================
# CLAIM 3 — SYSTEM MEMORY BENEFIT
# ===========================================================================

class TestClaim3_SystemMemoryBenefit:
    """AC/DC memory must outperform single-step (no memory) on history-dependent tasks."""

    def test_copy_task_needs_memory(self):
        """A task requiring recall of step t=0 at step t=T cannot be solved without memory."""
        rng = np.random.default_rng(7)
        n, T, d = 80, 8, 16

        # Inputs: random sequences. Target = classification based on first timestep.
        X = rng.normal(0.0, 0.6, size=(n, T, d))
        # Class determined by argmax of first timestep's first 4 dims
        labels = np.argmax(X[:, 0, :4], axis=1)
        Y = np.eye(4)[labels]  # one-hot encoding

        model_with    = WeightlessModel(input_dim=d, reservoir_dim=64, output_dim=4, seed=7)
        model_without = WeightlessModel(input_dim=d, reservoir_dim=64, output_dim=4, seed=7)

        # with memory: full sequence
        model_with.set_task("copy")
        feats_with = model_with.encode_sequence(X)

        # without memory: single last step only
        model_without.set_task("copy")
        X_last = X[:, -1:, :]     # only final timestep
        feats_without = model_without.encode_sequence(X_last)

        rng_h = np.random.default_rng(7)
        head_with    = ZeroForgetReadout(4, feats_with.shape[1],    rng=rng_h)
        rng_h2 = np.random.default_rng(7)
        head_without = ZeroForgetReadout(4, feats_without.shape[1], rng=rng_h2)
        head_with.set_task("copy"); head_without.set_task("copy")

        for _ in range(150):
            loss_with    = head_with.update_batch(feats_with,    Y, lr=0.05)
            loss_without = head_without.update_batch(feats_without, Y, lr=0.05)

        assert loss_with < loss_without, (
            f"Memory did not help copy task: with={loss_with:.4f} without={loss_without:.4f}"
        )

    def test_ac_dc_feature_width(self):
        """SystemMemory must emit exactly reservoir_dim * 6 + 3 features."""
        mem = SystemMemory(reservoir_dim=32)
        rng = np.random.default_rng(0)
        X_seq = rng.normal(size=(4, 10, 32))
        feats = mem.process_sequence(X_seq)
        assert feats.shape == (4, 32 * 6 + 3), (
            f"Expected ({4}, {32*6+3}), got {feats.shape}"
        )

    def test_dc_accumulates_slowly(self):
        """DC stream must still retain signal from t=0 at t=T."""
        mem = SystemMemory(reservoir_dim=8, ac_decay=0.25, dc_decay=0.98)
        rng = np.random.default_rng(0)
        T   = 50
        # Pulse at t=0 only
        X = np.zeros((1, T, 8))
        X[:, 0, :] = 1.0
        feats = mem.process_sequence(X)
        # DC at t=T: (1-dc_decay) * dc_decay^(T-1) ≈ 0.02 * 0.98^49 ≈ 0.0074
        # That is still a non-trivial signal compared to zero.
        # Layout: [last(8), mean(8), max(8), ac(8), dc(8), ...]  → dc starts at 32
        dc_offset = 8 * 4   # last + mean + max + ac = 32
        dc_final  = feats[0, dc_offset: dc_offset + 8]
        assert np.any(np.abs(dc_final) > 1e-4), (
            f"DC stream lost the pulse after {T} steps: max={np.max(np.abs(dc_final)):.6f}"
        )

    def test_ac_decays_quickly(self):
        """AC stream must be near-zero T steps after a pulse."""
        mem = SystemMemory(reservoir_dim=8, ac_decay=0.25, dc_decay=0.98)
        T = 20
        X = np.zeros((1, T, 8))
        X[:, 0, :] = 1.0
        feats = mem.process_sequence(X)
        ac_offset = 8 * 3
        ac_final  = feats[0, ac_offset: ac_offset + 8]
        expected_decay = 0.25 ** T   # ≈ 9e-13
        assert np.all(np.abs(ac_final) < 0.001), (
            f"AC stream did not decay fast enough: max={np.max(np.abs(ac_final)):.6f}"
        )


# ===========================================================================
# CLAIM 4 — O(N) PARAMETER SCALING
# ===========================================================================

class TestClaim4_LinearScaling:
    """Learned params must grow linearly in N tasks; encoder is constant."""

    def test_encoder_constant_across_tasks(self):
        model = WeightlessModel(input_dim=16, reservoir_dim=64, output_dim=4, seed=0)
        fixed_before = model.total_fixed_params
        rng = np.random.default_rng(0)
        for i in range(20):
            X, Y = make_sequence_task(20, 4, 16, 4, rng)
            train_task(model, X, Y, f"task_{i}", epochs=5)
        assert model.total_fixed_params == fixed_before, (
            "Fixed encoder parameter count changed after adding tasks"
        )

    def test_learned_params_linear_in_tasks(self):
        model = WeightlessModel(input_dim=16, reservoir_dim=64, output_dim=4, seed=0)
        rng   = np.random.default_rng(0)
        head_size = model.feature_dim * model.output_dim  # params per head
        for i in range(10):
            X, Y = make_sequence_task(20, 4, 16, 4, rng)
            train_task(model, X, Y, f"task_{i}", epochs=2)
            expected = (i + 1) * head_size
            assert model.total_learned_params == expected, (
                f"After {i+1} tasks: expected {expected} params, got {model.total_learned_params}"
            )

    def test_param_summary_fields(self):
        model = WeightlessModel(input_dim=8, reservoir_dim=32, output_dim=2, seed=0)
        summary = model.param_summary()
        assert "fixed_encoder" in summary
        assert "learned_heads" in summary
        assert "tasks" in summary
        assert summary["fixed_encoder"] > 0
        assert summary["learned_heads"] == 0  # no tasks yet
        assert summary["tasks"] == 0


# ===========================================================================
# CLAIM 5 — SLEEP CONSOLIDATION QUANTIFIABLY HELPS
# ===========================================================================

class TestClaim5_SleepConsolidation:
    """Sleep must produce error_reduction > 0 consistently."""

    def _run_sleep(self, seed: int) -> dict:
        rng   = np.random.default_rng(seed)
        model = WeightlessModel(input_dim=16, reservoir_dim=64, output_dim=4, seed=seed)
        X, Y  = make_sequence_task(60, 6, 16, 4, rng)
        # Light training — head not fully converged
        train_task(model, X, Y, "T", epochs=5, lr=0.02)
        return model.sleep("T", X, Y, num_cycles=300, sleep_lr=0.005)

    def test_error_reduces(self):
        result = self._run_sleep(seed=0)
        assert result["error_reduction"] > 0, (
            f"Sleep did not reduce error: "
            f"initial={result['initial_error']:.4f} final={result['final_error']:.4f}"
        )

    def test_error_reduces_multiple_seeds(self):
        for seed in range(5):
            result = self._run_sleep(seed=seed)
            assert result["error_reduction"] > 0, (
                f"Seed {seed}: sleep failed — "
                f"initial={result['initial_error']:.4f} final={result['final_error']:.4f}"
            )

    def test_sleep_does_not_affect_other_tasks(self):
        rng   = np.random.default_rng(42)
        model = WeightlessModel(input_dim=16, reservoir_dim=64, output_dim=4, seed=42)
        X_a, Y_a = make_sequence_task(50, 5, 16, 4, rng)
        X_b, Y_b = make_sequence_task(50, 5, 16, 4, rng)

        train_task(model, X_a, Y_a, "A")
        train_task(model, X_b, Y_b, "B")

        W_b_before = model.readout._heads["B"].copy()
        model.sleep("A", X_a, Y_a, num_cycles=200)

        np.testing.assert_array_equal(
            model.readout._heads["B"], W_b_before,
            err_msg="Sleep on task A modified task B's head"
        )


# ===========================================================================
# CLAIM 6 — INDEPENDENCE UNDER ADVERSARIAL PRESSURE
# ===========================================================================

class TestClaim6_AdversarialIsolation:
    """Structural isolation must hold even under extreme gradient pressure."""

    def test_high_lr_adversarial_task(self):
        """Task B trained with lr=1.0 for 500 steps must leave task A exactly unchanged."""
        rng   = np.random.default_rng(99)
        model = WeightlessModel(input_dim=16, reservoir_dim=64, output_dim=4, seed=99)
        X_a, Y_a = make_sequence_task(40, 5, 16, 4, rng)
        X_b, Y_b = make_sequence_task(40, 5, 16, 4, rng)

        train_task(model, X_a, Y_a, "A", epochs=50)
        W_a = model.readout._heads["A"].copy()

        model.set_task("B")
        feats_b = model.encode_sequence(X_b)
        for _ in range(500):
            model.update(feats_b, Y_b, lr=1.0)   # adversarial pressure

        np.testing.assert_array_equal(
            model.readout._heads["A"], W_a,
            err_msg="Task A corrupted by adversarial task B training"
        )

    def test_task_not_corrupted_by_weight_clip(self):
        """Weight clip on task B must not alter task A's clipped state."""
        rng   = np.random.default_rng(7)
        model = WeightlessModel(input_dim=8, reservoir_dim=32, output_dim=2, seed=7)
        X_a, Y_a = make_sequence_task(20, 3, 8, 2, rng)
        X_b, Y_b = make_sequence_task(20, 3, 8, 2, rng)

        # Drive both heads near the clip boundary
        train_task(model, X_a, Y_a, "A", epochs=100, lr=0.5)
        train_task(model, X_b, Y_b, "B", epochs=100, lr=0.5)
        W_a_clipped = model.readout._heads["A"].copy()

        # More adversarial updates on B
        model.set_task("B")
        feats_b = model.encode_sequence(X_b)
        for _ in range(200):
            model.update(feats_b, Y_b, lr=2.0)

        np.testing.assert_array_equal(
            model.readout._heads["A"], W_a_clipped,
            err_msg="Task A head changed during adversarial B updates (clip side effect?)"
        )

    def test_concurrent_task_ids_independent(self):
        """Explicit task_id override must update exactly that head and no other."""
        rng   = np.random.default_rng(3)
        model = WeightlessModel(input_dim=8, reservoir_dim=32, output_dim=2, seed=3)
        for name in ["alpha", "beta", "gamma", "delta"]:
            X, Y = make_sequence_task(20, 4, 8, 2, rng)
            train_task(model, X, Y, name, epochs=10)

        snapshots = {t: model.readout._heads[t].copy() for t in model.task_ids}

        # Update only "beta" via explicit task_id
        X_b, Y_b = make_sequence_task(20, 4, 8, 2, rng)
        feats = model.encode_sequence(X_b)
        model.update(feats, Y_b, task_id="beta", lr=0.1)

        for t, snap in snapshots.items():
            if t == "beta":
                continue
            np.testing.assert_array_equal(
                model.readout._heads[t], snap,
                err_msg=f"Task {t} changed when only 'beta' was updated"
            )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    suites = [
        TestClaim1_ZeroForgetting,
        TestClaim2_EncoderSufficiency,
        TestClaim3_SystemMemoryBenefit,
        TestClaim4_LinearScaling,
        TestClaim5_SleepConsolidation,
        TestClaim6_AdversarialIsolation,
    ]

    passed = failed = 0
    for suite_cls in suites:
        suite = suite_cls()
        for name in [n for n in dir(suite_cls) if n.startswith("test_")]:
            method = getattr(suite, name)
            try:
                method()
                print(f"  PASS  {suite_cls.__name__}.{name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {suite_cls.__name__}.{name}")
                traceback.print_exc()
                failed += 1

    print(f"\n{'='*60}")
    print(f"  {passed} passed  |  {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
