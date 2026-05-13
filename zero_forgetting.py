"""Zero-forgetting continual learning via isolated task heads and sleep consolidation.

The two mechanisms that produce near-zero forgetting in this project, extracted
for reuse with any fixed-feature reservoir or encoder:

  1. ``ZeroForgetReadout`` — one linear head per task, never shared.
     Training task B cannot touch task A's weights.

  2. ``sleep_consolidation`` — after learning a new task, replay cached
     features with the kernel/encoder frozen to tighten each head.

Quick start
-----------
    import numpy as np
    from zero_forgetting import ZeroForgetReadout, sleep_consolidation

    rng  = np.random.default_rng(0)
    head = ZeroForgetReadout(output_dim=4, feature_dim=64, rng=rng)

    # --- train task A ---
    head.set_task("task_A")
    for X_batch, Y_batch in task_a_data:
        features = encoder(X_batch)                  # your reservoir / encoder
        head.update_batch(features, Y_batch, lr=0.01)

    # --- train task B (task A is unaffected) ---
    head.set_task("task_B")
    for X_batch, Y_batch in task_b_data:
        features = encoder(X_batch)
        head.update_batch(features, Y_batch, lr=0.01)

    # --- optional sleep pass on task A ---
    sleep_consolidation(
        freeze_fn  = lambda: None,      # freeze your encoder if it has shared weights
        unfreeze_fn= lambda: None,
        extract_fn = lambda X: encoder(X),
        readout    = head,
        task_id    = "task_A",
        inputs     = X_a,
        targets    = Y_a,
    )

    # --- evaluate: task A is intact ---
    preds_a = head.predict_batch(encoder(X_test_a), task_id="task_A")

Dependencies: numpy only.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# ZeroForgetReadout
# ---------------------------------------------------------------------------

class ZeroForgetReadout:
    """One isolated linear head per task.

    Why this works
    ~~~~~~~~~~~~~~
    Catastrophic forgetting happens when gradient updates for a new task modify
    weights that are load-bearing for old tasks.  The fix is not clever
    regularization — it is isolation: each task owns its own weight matrix and
    nothing else can write to it.

    This class maintains a dictionary of ``(output_dim, feature_dim)`` weight
    matrices, one per task ID.  A delta-rule (online gradient descent) update
    on task B is a no-op for all other task matrices.

    Args:
        output_dim:  Number of output units.
        feature_dim: Dimensionality of the feature vector from the reservoir.
        rng:         NumPy random generator used to initialise new heads.
        scale:       Std dev for random weight initialisation.
        weight_clip: Absolute weight clamp applied after every update (prevents
                     divergence in long online runs).
    """

    def __init__(
        self,
        output_dim: int,
        feature_dim: int,
        rng: np.random.Generator,
        scale: float = 0.02,
        weight_clip: float = 3.0,
    ) -> None:
        if output_dim <= 0 or feature_dim <= 0:
            raise ValueError("output_dim and feature_dim must be positive")
        self.output_dim  = int(output_dim)
        self.feature_dim = int(feature_dim)
        self.scale       = float(scale)
        self.weight_clip = float(weight_clip)
        self.rng         = rng
        self._heads: dict[str, np.ndarray] = {}
        self.current_task: str | None = None

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    def set_task(self, task_id: str) -> None:
        """Switch the active task, creating a new head if needed."""
        self.current_task = task_id
        self._ensure(task_id)

    def _ensure(self, task_id: str) -> None:
        if task_id not in self._heads:
            self._heads[task_id] = self.rng.normal(
                0.0, self.scale, size=(self.output_dim, self.feature_dim)
            )

    def _resolve(self, task_id: str | None) -> str:
        tid = task_id if task_id is not None else self.current_task
        if tid is None:
            tid = "default"
        self._ensure(tid)
        return tid

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, features: np.ndarray, task_id: str | None = None) -> np.ndarray:
        """Predict for a single feature vector ``(feature_dim,)``."""
        return self._heads[self._resolve(task_id)] @ features

    def predict_batch(
        self, features: np.ndarray, task_id: str | None = None
    ) -> np.ndarray:
        """Predict for a batch ``(batch, feature_dim)`` → ``(batch, output_dim)``."""
        return features @ self._heads[self._resolve(task_id)].T

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        """Numerically stable row-wise softmax for a 2-D logit matrix."""
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    @staticmethod
    def _ce_loss(probs: np.ndarray, targets: np.ndarray) -> float:
        """Mean cross-entropy loss. targets must be one-hot."""
        return float(-np.mean(np.sum(targets * np.log(probs + 1e-12), axis=1)))

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(
        self,
        features: np.ndarray,
        target: np.ndarray,
        lr: float,
        task_id: str | None = None,
    ) -> float:
        """Softmax + CE update for a single sample. Returns CE loss."""
        tid    = self._resolve(task_id)
        # single sample: make (1, feature_dim) and (1, output_dim)
        F      = features[np.newaxis, :]          # (1, D)
        Y      = target[np.newaxis, :]            # (1, C)
        probs  = self._softmax(F @ self._heads[tid].T)  # (1, C)
        grad   = (probs - Y).T @ F                # (C, D)
        self._heads[tid] -= lr * grad
        np.clip(self._heads[tid], -self.weight_clip, self.weight_clip,
                out=self._heads[tid])
        return self._ce_loss(probs, Y)

    def update_batch(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        lr: float,
        task_id: str | None = None,
    ) -> float:
        """Batch softmax + CE update. Returns mean CE loss."""
        tid        = self._resolve(task_id)
        batch_size = max(1, features.shape[0])
        logits     = features @ self._heads[tid].T          # (B, C)
        probs      = self._softmax(logits)                  # (B, C)
        grad       = (probs - targets).T @ features         # (C, D)
        self._heads[tid] -= lr * grad / batch_size
        np.clip(self._heads[tid], -self.weight_clip, self.weight_clip,
                out=self._heads[tid])
        return self._ce_loss(probs, targets)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def task_ids(self) -> list[str]:
        return list(self._heads.keys())

    def param_count(self) -> int:
        return int(sum(w.size for w in self._heads.values()))


# ---------------------------------------------------------------------------
# Sleep consolidation
# ---------------------------------------------------------------------------

def sleep_consolidation(
    freeze_fn:    Callable[[], None],
    unfreeze_fn:  Callable[[], None],
    extract_fn:   Callable[[np.ndarray], np.ndarray],
    readout:      ZeroForgetReadout,
    task_id:      str,
    inputs:       np.ndarray,
    targets:      np.ndarray,
    num_cycles:   int = 1000,
    sleep_lr:     float = 0.001,
    batch_size:   int = 32,
    seed:         int = 0,
    verbose:      bool = False,
) -> dict[str, Any]:
    """Replay a frozen feature table to tighten one task head.

    The shared encoder / reservoir is frozen before extraction so the feature
    table does not shift between replay cycles (the "moving-target" problem).
    Only the task-owned readout head is updated.

    Args:
        freeze_fn:   Zero-argument callable that freezes shared encoder weights.
        unfreeze_fn: Zero-argument callable that unfreezes shared encoder weights.
        extract_fn:  Maps ``inputs (N, ...)`` → feature matrix ``(N, feature_dim)``.
        readout:     ``ZeroForgetReadout`` whose head for ``task_id`` to update.
        task_id:     Which task head to replay.
        inputs:      Raw inputs to encode once before replay.
        targets:     Targets aligned with inputs.
        num_cycles:  Number of mini-batch replay passes.
        sleep_lr:    Learning rate for head updates during replay.
        batch_size:  Mini-batch size for each replay step.
        seed:        RNG seed for replay sampling.
        verbose:     Print progress every 100 cycles.

    Returns:
        Dict with ``initial_error``, ``final_error``, ``error_reduction``,
        ``final_accuracy``, ``total_time``, and ``history``.
    """
    inputs  = np.asarray(inputs,  dtype=float)
    targets = np.asarray(targets, dtype=float)

    if num_cycles < 0:
        raise ValueError("num_cycles must be non-negative")
    if len(inputs) != len(targets):
        raise ValueError("inputs and targets must have the same sample count")
    if len(inputs) == 0:
        raise ValueError("sleep_consolidation requires at least one sample")

    start = time.time()
    history: list[dict[str, float]] = []
    rng = np.random.default_rng(seed)

    # 1. Freeze shared encoder.
    freeze_fn()

    try:
        # 2. Extract feature table once — this is the key: no moving target.
        features = np.asarray(extract_fn(inputs), dtype=float)

        # 3. Measure initial head quality (CE loss).
        init_probs    = readout._softmax(readout.predict_batch(features, task_id=task_id))
        initial_error = readout._ce_loss(init_probs, targets)
        if verbose:
            print(f"[sleep:{task_id}] initial_ce={initial_error:.4f}")

        # 4. Replay mini-batches — only the head is updated.
        n = len(features)
        for cycle in range(num_cycles):
            idx   = rng.integers(0, n, size=min(batch_size, n))
            loss  = readout.update_batch(
                features[idx], targets[idx], lr=sleep_lr, task_id=task_id
            )
            if verbose and cycle % 100 == 0:
                history.append({"cycle": cycle, "loss": loss})
                print(f"[sleep:{task_id}] cycle={cycle:4d}  ce={loss:.4f}")

        # 5. Measure final quality (CE loss).
        final_probs = readout._softmax(readout.predict_batch(features, task_id=task_id))
        final_error = readout._ce_loss(final_probs, targets)

    finally:
        unfreeze_fn()

    return {
        "task_id":       task_id,
        "num_cycles":    num_cycles,
        "sleep_lr":      sleep_lr,
        "batch_size":    batch_size,
        "initial_error": initial_error,
        "final_error":   final_error,
        "error_reduction": initial_error - final_error,
        "total_time":    float(time.time() - start),
        "history":       history,
    }
