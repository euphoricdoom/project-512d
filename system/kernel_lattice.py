"""2D Kernel Lattice: vertical temporal hierarchy × horizontal functional specialization.

Architecture overview
---------------------
A KernelLattice is a 2D grid of KernelStacks:

    Layer 1 (abstract)   [Stack 0] --bridge-- [Stack 1] --bridge-- ...
         ↑ inter-layer projection
    Layer 0 (concrete)   [Stack 0] --bridge-- [Stack 1] --bridge-- ...
              ↑ ↑ ↑ ↑
           input sequence (T × 64)

Each KernelStack combines:
  - ModularFieldSystem: 540-dim coupled recurrent kernel (60 modules × 8 dims)
  - HelixTemporalAdapter: projects reservoir states into AC/DC feature trajectories
  - Output: 1539-dim feature vector (projection_dim=256 → 256×6+3=1539)

Horizontal bridges (feature-level, between stacks in the same layer):
  - Cosine resonance: if similarity exceeds threshold, blend feature vectors
  - Bridges operate on final_features() output — NOT on kernel state (safe)

Vertical connections (between layers):
  - Layer l+1 stacks receive a 64-dim projection of layer l's aggregate features
    added to each timestep of the original input sequence

Global readout:
  - ZeroForgetReadout with one isolated head per task
  - Input: concatenation of all layer × stack features (n_layers × n_stacks × 1539)

Forgetting: structurally zero. Shared dynamics (ModularFieldSystem) are frozen after
construction; only isolated per-task readout weights are ever updated.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.constants import Config512D
from system.helix_temporal import HelixTemporalAdapter
from system.modular_system import ModularFieldSystem
from zero_forgetting import ZeroForgetReadout


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LatticeConfig:
    """Hyperparameters for the 2D Kernel Lattice.

    Attributes:
        n_layers:         Number of vertical abstraction levels.
        n_stacks:         Number of horizontal stacks per layer.
        projection_dim:   AC/DC projection width for HelixTemporalAdapter.
        bridge_threshold: Minimum cosine similarity to activate a bridge.
        bridge_alpha:     Blend weight when a bridge fires (0=no blend, 1=full replace).
        seed:             Master RNG seed; each sub-component offsets from this.
    """

    n_layers: int = 2
    n_stacks: int = 4
    projection_dim: int = 256
    bridge_threshold: float = 0.40
    bridge_alpha: float = 0.25
    seed: int = 42

    def __post_init__(self) -> None:
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be ≥ 1, got {self.n_layers}")
        if self.n_stacks < 1:
            raise ValueError(f"n_stacks must be ≥ 1, got {self.n_stacks}")
        if self.projection_dim < 1:
            raise ValueError(f"projection_dim must be ≥ 1, got {self.projection_dim}")
        if not 0.0 <= self.bridge_threshold <= 1.0:
            raise ValueError(f"bridge_threshold must be in [0, 1], got {self.bridge_threshold}")
        if not 0.0 <= self.bridge_alpha <= 1.0:
            raise ValueError(f"bridge_alpha must be in [0, 1], got {self.bridge_alpha}")

    @property
    def feature_dim_per_stack(self) -> int:
        return self.projection_dim * 6 + 3

    @property
    def layer_feature_dim(self) -> int:
        return self.n_stacks * self.feature_dim_per_stack

    @property
    def global_feature_dim(self) -> int:
        return self.n_layers * self.layer_feature_dim


# ---------------------------------------------------------------------------
# KernelStack
# ---------------------------------------------------------------------------

class KernelStack:
    """One (ModularFieldSystem, HelixTemporalAdapter) pair.

    A KernelStack processes an input sequence of shape (T, input_dim=64)
    and returns a 1D feature vector of shape (feature_dim_per_stack,).

    The kernel state is zero-initialized at the start of each forward pass
    (reset_state(scale=0.0)) so the forward pass is fully deterministic.
    The shared kernel dynamics are never updated during task training —
    only the upstream ZeroForgetReadout weights change.
    """

    _RESERVOIR_DIM: int = 540  # cfg.dim

    def __init__(
        self,
        cfg: Config512D,
        projection_dim: int,
        rng: np.random.Generator,
    ) -> None:
        self.cfg = cfg
        self.kernel = ModularFieldSystem(cfg)
        self.helix = HelixTemporalAdapter(
            input_dim=self._RESERVOIR_DIM,
            input_width=cfg.input_dim,
            projection_dim=projection_dim,
            ac_decay=0.25,
            dc_decay=0.98,
            rng=rng,
        )
        self.feature_dim: int = self.helix.feature_dim

    def forward(
        self,
        inputs: np.ndarray,
        task_id: str | None = None,
    ) -> np.ndarray:
        """Process a sequence (T, 64) and return features (feature_dim,).

        Args:
            inputs:  Sequence array of shape (T, input_dim). Each row is one
                     timestep fed into the kernel and helix adapter.
            task_id: Optional task identifier forwarded to HelixTemporalAdapter
                     for task-specific gating channels.

        Returns:
            Feature vector of shape (feature_dim_per_stack,).
        """
        inputs = np.asarray(inputs, dtype=float)
        if inputs.ndim == 1:
            inputs = inputs[None, :]
        T = inputs.shape[0]

        self.kernel.reset_state(scale=0.0)
        self.helix.reset(batch_size=1)

        for t in range(T):
            self.kernel.inject(inputs[t])
            state = self.kernel.step(t=t)
            self.helix.step(
                input_t=inputs[t][None, :],
                reservoir_features_t=state[None, :],
                t=t,
                total_steps=T,
                task_id=task_id,
            )

        out = self.helix.final_features()
        # final_features() returns (B, feature_dim) or (feature_dim,) when B=1
        return out if out.ndim == 1 else out[0]

    def reservoir_state(self) -> np.ndarray:
        """Return current kernel state (540,) — useful for analysis."""
        return self.kernel.state.copy()


# ---------------------------------------------------------------------------
# LatticeLayer
# ---------------------------------------------------------------------------

class LatticeLayer:
    """One horizontal row of KernelStacks with feature-level bridges.

    Bridges fire when cosine similarity between two stacks' feature vectors
    exceeds bridge_threshold, blending them by bridge_alpha. This operates
    entirely on final_features() output — kernel states are never touched.
    """

    def __init__(
        self,
        layer_id: int,
        n_stacks: int,
        cfg: Config512D,
        lattice_cfg: LatticeConfig,
        rng: np.random.Generator,
    ) -> None:
        self.layer_id = layer_id
        self.n_stacks = n_stacks
        self.lattice_cfg = lattice_cfg
        self._bridge_activations: dict[str, int] = {}

        from dataclasses import replace as dc_replace
        self.stacks: list[KernelStack] = [
            KernelStack(
                dc_replace(cfg, seed=cfg.seed + layer_id * 1000 + stack_id),
                lattice_cfg.projection_dim,
                rng,
            )
            for stack_id in range(n_stacks)
        ]
        self.feature_dim_per_stack: int = self.stacks[0].feature_dim

        # Optional inter-layer input projection (set by KernelLattice for layer > 0)
        # Shape: (previous_layer_feature_dim, cfg.input_dim)
        self.input_mixing: np.ndarray | None = None

    def forward(
        self,
        inputs: np.ndarray,
        task_id: str | None = None,
        prev_layer_features: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run all stacks, apply bridges, return layer features.

        Args:
            inputs:              Raw input sequence (T, input_dim).
            task_id:             Forwarded to each stack.
            prev_layer_features: Aggregate features from the previous layer
                                 (layer_feature_dim,). If set and input_mixing
                                 is initialised, mixes into each input timestep.

        Returns:
            Layer feature vector (n_stacks × feature_dim_per_stack,).
        """
        if prev_layer_features is not None and self.input_mixing is not None:
            mixed = prev_layer_features @ self.input_mixing
            mixed_signal = np.tanh(mixed)
            layer_inputs = inputs + mixed_signal[None, :]
        else:
            layer_inputs = inputs

        raw_features: list[np.ndarray] = [
            stack.forward(layer_inputs, task_id=task_id) for stack in self.stacks
        ]

        blended = self._apply_bridges(raw_features)
        return np.concatenate(blended)

    def _apply_bridges(self, features: list[np.ndarray]) -> list[np.ndarray]:
        """Feature-level cosine resonance bridges between adjacent stacks."""
        result = [f.copy() for f in features]
        alpha = self.lattice_cfg.bridge_alpha
        threshold = self.lattice_cfg.bridge_threshold

        for i in range(len(result) - 1):
            j = i + 1
            a, b = result[i], result[j]
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a < 1e-12 or norm_b < 1e-12:
                continue
            sim = float(np.dot(a, b) / (norm_a * norm_b))
            if sim > threshold:
                key = f"{self.layer_id}:{i}->{self.layer_id}:{j}"
                self._bridge_activations[key] = (
                    self._bridge_activations.get(key, 0) + 1
                )
                blend = (a + b) / 2.0
                result[i] = (1.0 - alpha) * result[i] + alpha * blend
                result[j] = (1.0 - alpha) * result[j] + alpha * blend

        return result

    def bridge_activation_counts(self) -> dict[str, int]:
        return dict(self._bridge_activations)


# ---------------------------------------------------------------------------
# KernelLattice
# ---------------------------------------------------------------------------

class KernelLattice:
    """2D lattice of KernelStacks with vertical hierarchy and zero forgetting.

    The lattice layout is (n_layers × n_stacks). Layer 0 is the most concrete
    (processes raw input). Each subsequent layer receives a projection of the
    previous layer's aggregate features mixed into its input timesteps.

    A single ZeroForgetReadout with one isolated head per task reads from the
    concatenation of ALL layer features. Forgetting is structurally impossible:
    shared kernel dynamics are never updated; only the active task's head is
    written.

    Args:
        output_dim:    Number of output units (e.g. 64 for TASKS_8 regression).
        lattice_cfg:   LatticeConfig instance. Defaults to LatticeConfig().
        kernel_cfg:    Config512D instance. Defaults to Config512D().
    """

    def __init__(
        self,
        output_dim: int,
        lattice_cfg: LatticeConfig | None = None,
        kernel_cfg: Config512D | None = None,
    ) -> None:
        self.output_dim = int(output_dim)
        self.lattice_cfg = lattice_cfg or LatticeConfig()
        self.kernel_cfg = kernel_cfg or Config512D()

        rng = np.random.default_rng(self.lattice_cfg.seed)
        readout_rng = np.random.default_rng(self.lattice_cfg.seed + 9999)

        self.layers: list[LatticeLayer] = []
        for layer_id in range(self.lattice_cfg.n_layers):
            layer = LatticeLayer(
                layer_id=layer_id,
                n_stacks=self.lattice_cfg.n_stacks,
                cfg=self.kernel_cfg,
                lattice_cfg=self.lattice_cfg,
                rng=rng,
            )
            if layer_id > 0:
                prev_feature_dim = self.lattice_cfg.layer_feature_dim
                layer.input_mixing = rng.normal(
                    0.0,
                    1.0 / np.sqrt(prev_feature_dim),
                    size=(prev_feature_dim, self.kernel_cfg.input_dim),
                )
            self.layers.append(layer)

        self.readout = ZeroForgetReadout(
            output_dim=self.output_dim,
            feature_dim=self.lattice_cfg.global_feature_dim,
            rng=readout_rng,
        )

        self._task_sample_counts: dict[str, int] = {}
        self._total_forward_calls: int = 0
        self._training_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        inputs: np.ndarray,
        task_id: str | None = None,
    ) -> np.ndarray:
        """Full forward pass through all layers.

        Args:
            inputs:  Input sequence of shape (T, input_dim=64) or (input_dim,)
                     for a single step.
            task_id: Task identifier forwarded through the lattice.

        Returns:
            Global feature vector of shape (global_feature_dim,).
        """
        inputs = np.asarray(inputs, dtype=float)
        if inputs.ndim == 1:
            inputs = inputs[None, :]

        self._total_forward_calls += 1
        prev_features: np.ndarray | None = None
        all_features: list[np.ndarray] = []

        for layer in self.layers:
            layer_features = layer.forward(
                inputs, task_id=task_id, prev_layer_features=prev_features
            )
            all_features.append(layer_features)
            prev_features = layer_features

        return np.concatenate(all_features)

    def predict(
        self,
        inputs: np.ndarray,
        task_id: str,
    ) -> np.ndarray:
        """Predict for a single sample.

        Returns:
            Output vector of shape (output_dim,).
        """
        features = self.forward(inputs, task_id=task_id)
        return self.readout.predict(features, task_id=task_id)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_sample(
        self,
        inputs: np.ndarray,
        target: np.ndarray,
        task_id: str,
        lr: float = 0.01,
    ) -> float:
        """Online update for one (input sequence, target) pair.

        Only the head for task_id is modified. All other task heads and the
        shared kernel dynamics are untouched.

        Args:
            inputs:  Input sequence (T, 64) or (64,).
            target:  Target output (output_dim,).
            task_id: Which task head to update.
            lr:      Learning rate for the delta rule.

        Returns:
            MSE loss before this update.
        """
        target = np.asarray(target, dtype=float).ravel()
        features = self.forward(inputs, task_id=task_id)

        self.readout.set_task(task_id)
        head = self.readout._heads[task_id]
        pred = head @ features
        error = pred - target
        loss = float(np.mean(error ** 2))

        # Delta rule (MSE gradient): weight update for regression
        grad = np.outer(error, features)
        head -= lr * grad
        np.clip(head, -self.readout.weight_clip, self.readout.weight_clip, out=head)

        self._task_sample_counts[task_id] = (
            self._task_sample_counts.get(task_id, 0) + 1
        )
        return loss

    def train_sequence(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        task_id: str,
        lr: float = 0.01,
        epochs: int = 1,
        verbose: bool = False,
    ) -> list[float]:
        """Train on a dataset of (input_sequence, target) pairs.

        Args:
            X:       Array of shape (n_samples, T, input_dim) or
                     (n_samples, input_dim) for single-step samples.
            Y:       Target array of shape (n_samples, output_dim).
            task_id: Task head to update.
            lr:      Learning rate.
            epochs:  Number of passes over the data.
            verbose: If True, prints per-epoch mean loss.

        Returns:
            List of per-sample losses from the final epoch.
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if X.ndim == 2:
            X = X[:, None, :]

        self.readout.set_task(task_id)
        losses: list[float] = []

        for epoch in range(epochs):
            epoch_losses = []
            for i in range(len(X)):
                loss = self.train_sample(X[i], Y[i], task_id=task_id, lr=lr)
                epoch_losses.append(loss)
            losses = epoch_losses
            if verbose:
                print(f"  epoch {epoch + 1}/{epochs}  loss={np.mean(epoch_losses):.4f}")

        return losses

    def fit_task(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        task_id: str,
        lr: float = 0.005,
        epochs: int = 10,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> list[float]:
        """Train via cached features + mini-batch gradient descent.

        Since all stacks have fixed weights and deterministic forward passes,
        the feature vector for any input is the same every epoch. This method
        computes features once and then runs purely linear regression across
        multiple epochs — much faster than ``train_sequence`` for multi-epoch
        training and converges reliably.

        Args:
            X:          Input array (n_samples, T, input_dim) or (n_samples, input_dim).
            Y:          Target array (n_samples, output_dim).
            task_id:    Task head to train.
            lr:         Learning rate for mini-batch gradient descent.
            epochs:     Number of passes over the cached features.
            batch_size: Mini-batch size.
            verbose:    Print per-epoch loss if True.

        Returns:
            Per-epoch mean MSE loss.
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if X.ndim == 2:
            X = X[:, None, :]
        n = len(X)

        # Compute features once — O(n) forward passes instead of O(n * epochs)
        F = np.zeros((n, self.lattice_cfg.global_feature_dim))
        for i in range(n):
            F[i] = self.forward(X[i], task_id=task_id)

        self.readout.set_task(task_id)
        head = self.readout._heads[task_id]
        epoch_losses: list[float] = []
        shuffle_rng = np.random.default_rng(self.lattice_cfg.seed)

        for epoch in range(epochs):
            idx = shuffle_rng.permutation(n)
            for start in range(0, n, batch_size):
                b_idx = idx[start : start + batch_size]
                F_b = F[b_idx]
                Y_b = Y[b_idx]
                preds = F_b @ head.T
                errors = preds - Y_b
                grad = errors.T @ F_b / len(b_idx)
                head -= lr * grad
                np.clip(head, -self.readout.weight_clip, self.readout.weight_clip, out=head)
            # Post-update epoch loss (full pass over cached features)
            mean_loss = float(np.mean((F @ head.T - Y) ** 2))
            epoch_losses.append(mean_loss)
            if verbose:
                print(f"  epoch {epoch + 1}/{epochs}  loss={mean_loss:.4f}")

        self._task_sample_counts[task_id] = (
            self._task_sample_counts.get(task_id, 0) + n
        )
        return epoch_losses

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        task_id: str,
    ) -> dict[str, float]:
        """Evaluate MSE and R² on a dataset.

        Args:
            X:       Input array (n_samples, T, input_dim) or (n_samples, input_dim).
            Y:       Target array (n_samples, output_dim).
            task_id: Which head to use for prediction.

        Returns:
            Dict with 'mse' and 'r2' keys.
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if X.ndim == 2:
            X = X[:, None, :]

        preds = np.array([self.predict(X[i], task_id=task_id) for i in range(len(X))])
        mse = float(np.mean((preds - Y) ** 2))
        ss_res = np.sum((Y - preds) ** 2)
        ss_tot = np.sum((Y - np.mean(Y, axis=0)) ** 2)
        r2 = float(1.0 - ss_res / (ss_tot + 1e-12))
        return {"mse": mse, "r2": r2}

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def bridge_summary(self) -> dict[str, Any]:
        """Return bridge activation counts and hit rates across all layers (JSON-safe keys)."""
        activations: dict[str, int] = {}
        for layer in self.layers:
            for edge_key, count in layer.bridge_activation_counts().items():
                activations[edge_key] = activations.get(edge_key, 0) + count
        total = self._total_forward_calls
        hit_rates = {
            k: round(v / total, 4) if total > 0 else 0.0
            for k, v in activations.items()
        }
        return {
            "bridge_activations": activations,
            "bridge_hit_rates": hit_rates,
            "total_forward_calls": total,
        }

    def specialization_matrix(
        self,
        X: np.ndarray,
        task_ids: list[str],
        n_probe: int = 50,
    ) -> np.ndarray:
        """Measure per-stack feature-norm contribution per task — read-only.

        Probes the stack outputs for each task without touching any readout
        weights. The stacks are deterministic (fixed weights, zero-init kernel),
        so this is a pure measurement with no side effects.

        Args:
            X:        Array (n_tasks, n_samples, T, input_dim) or
                      (n_tasks, n_samples, input_dim).
            task_ids: Task name for each slice along axis 0.
            n_probe:  Max samples to probe per task (default 50).

        Returns:
            Heatmap array (n_layers, n_stacks) of mean feature norms,
            averaged across all tasks and probe samples.
        """
        n_tasks = len(task_ids)
        heatmap = np.zeros((self.lattice_cfg.n_layers, self.lattice_cfg.n_stacks))
        n_samples = X.shape[1]
        n_probe = min(n_probe, n_samples)

        for ti, task_id in enumerate(task_ids):
            x_task = np.asarray(X[ti], dtype=float)
            if x_task.ndim == 2:
                x_task = x_task[:, None, :]

            for i in range(n_probe):
                inputs = x_task[i]
                prev_features: np.ndarray | None = None
                for layer_id, layer in enumerate(self.layers):
                    if prev_features is not None and layer.input_mixing is not None:
                        mixed = np.tanh(prev_features @ layer.input_mixing)
                        layer_inputs = inputs + mixed[None, :]
                    else:
                        layer_inputs = inputs
                    stack_features = [
                        stack.forward(layer_inputs, task_id=task_id)
                        for stack in layer.stacks
                    ]
                    for si, sf in enumerate(stack_features):
                        heatmap[layer_id, si] += float(np.linalg.norm(sf))
                    prev_features = np.concatenate(stack_features)

        return heatmap / (n_tasks * n_probe)

    def task_summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of training state."""
        return {
            "n_layers": self.lattice_cfg.n_layers,
            "n_stacks": self.lattice_cfg.n_stacks,
            "tasks_trained": list(self._task_sample_counts.keys()),
            "samples_per_task": dict(self._task_sample_counts),
            "global_feature_dim": self.lattice_cfg.global_feature_dim,
            "total_forward_calls": self._total_forward_calls,
        }
