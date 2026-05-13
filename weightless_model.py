"""Weightless Continual Learning Model.

The thesis
----------
A model can achieve zero catastrophic forgetting **by construction** — not by
approximation, regularisation, or replay tricks — if the following invariant
holds:

    SHARED WEIGHTS ARE NEVER UPDATED.

The only weights that change after initialisation are the per-task output
heads.  Everything else — the feature extractor, the temporal memory state
transitions — is fixed at init time.  Tasks cannot interfere because they
share no mutable parameter state.

Three components
----------------
1. ``FixedEncoder``
   A random projection fixed forever at construction. Takes raw input
   ``(batch, input_dim)`` → ``(batch, reservoir_dim)``.  Spectral radius is
   scaled to ``target_radius`` so dynamics are contractive, not explosive.
   Zero weights are ever updated.

2. ``SystemMemory``
   AC/DC temporal state — the "computationally new memory":
     - AC stream (fast, default decay=0.25): tracks rapid transitions
     - DC stream (slow, default decay=0.98): accumulates persistent structure
     - Phase code: encodes where in time the network is
     - Product term (AC × DC): cross-frequency binding
   This is computational state, not stored weights. It resets each sequence.
   Final memory features = concat(last, mean, max, ac, dc, ac×dc, phase)
                         = reservoir_dim * 6 + 3   wide

3. ``ZeroForgetReadout`` (from zero_forgetting.py)
   One linear head per task. Training task B cannot touch task A's head.
   Forgetting is zero by construction.

Usage
-----
    from weightless_model import WeightlessModel
    import numpy as np

    model = WeightlessModel(input_dim=64, reservoir_dim=256, output_dim=4)

    # --- train on task A (sequence of timesteps) ---
    model.set_task("task_A")
    for seq_batch, targets in task_a_sequences:        # seq_batch: (B, T, input_dim)
        features = model.encode_sequence(seq_batch)    # (B, feature_dim)
        model.update(features, targets, lr=0.01)

    # --- train on task B — task A is untouched ---
    model.set_task("task_B")
    ...

    # --- sleep consolidation pass on A ---
    model.sleep("task_A", X_seq_a, Y_a)

    # --- inference ---
    preds = model.predict(seq_batch, task_id="task_A")

Dependencies: numpy only.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from zero_forgetting import ZeroForgetReadout, sleep_consolidation


# ---------------------------------------------------------------------------
# 1. Fixed Encoder — the weightless substrate
# ---------------------------------------------------------------------------

class FixedEncoder:
    """Random linear projection fixed forever at construction.

    The projection matrix W is drawn from a normal distribution and immediately
    scaled so its largest singular value ≤ ``target_radius``.  After that,
    nothing writes to W.

    This is a classical Echo State / Reservoir Computing approach: expressive
    enough to separate inputs, but deliberately not trained.

    Args:
        input_dim:     Width of raw input vectors.
        reservoir_dim: Width of projected feature vectors.
        target_radius: Spectral scale bound (default 0.94, just inside unit circle).
        seed:          RNG seed.
    """

    def __init__(
        self,
        input_dim: int,
        reservoir_dim: int,
        target_radius: float = 0.94,
        seed: int | None = None,
    ) -> None:
        if input_dim <= 0 or reservoir_dim <= 0:
            raise ValueError("input_dim and reservoir_dim must be positive")
        if not 0.0 < target_radius < 1.0:
            raise ValueError("target_radius must be in (0, 1)")
        self.input_dim     = int(input_dim)
        self.reservoir_dim = int(reservoir_dim)
        self.target_radius = float(target_radius)

        rng = np.random.default_rng(seed)
        W   = rng.normal(0.0, 1.0 / np.sqrt(input_dim), size=(reservoir_dim, input_dim))
        # Scale so largest singular value ≤ target_radius
        sv  = np.linalg.svd(W, compute_uv=False)
        if sv[0] > target_radius:
            W *= target_radius / sv[0]
        self._W: np.ndarray = W  # frozen — never written after this line

    @property
    def actual_radius(self) -> float:
        sv = np.linalg.svd(self._W, compute_uv=False)
        return float(sv[0])

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Project input ``(batch, input_dim)`` → ``(batch, reservoir_dim)``."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        out = np.tanh(X @ self._W.T)
        return out


# ---------------------------------------------------------------------------
# 2. System Memory — AC/DC temporal state
# ---------------------------------------------------------------------------

class SystemMemory:
    """AC/DC temporal feature memory.

    This is the "computationally new system memory": state that evolves over a
    sequence via simple leaky-integrator rules, then emits a rich feature
    summary.  No weights are learned — the temporal dynamics come entirely from
    the decay constants and the reservoir features passing through.

    AC stream (fast, decay≈0.25):  s_ac[t] = decay * s_ac[t-1] + (1-decay) * x[t]
    DC stream (slow, decay≈0.98):  s_dc[t] = decay * s_dc[t-1] + (1-decay) * x[t]

    Final feature vector per sample (width = reservoir_dim * 6 + 3):
        [last_x | mean_x | max_x | ac_final | dc_final | ac×dc | phase_sin, phase_cos, pos]

    The product term (AC × DC) encodes cross-frequency binding: a feature is
    "significant" only when it is both recently active (AC) and persistently
    present (DC).  This is the novel memory mechanism.

    Args:
        reservoir_dim: Feature width from the fixed encoder.
        ac_decay:      Fast stream decay rate.
        dc_decay:      Slow stream decay rate.
    """

    def __init__(
        self,
        reservoir_dim: int,
        ac_decay: float = 0.25,
        dc_decay: float = 0.98,
    ) -> None:
        if reservoir_dim <= 0:
            raise ValueError("reservoir_dim must be positive")
        if not 0.0 <= ac_decay < 1.0:
            raise ValueError("ac_decay must be in [0, 1)")
        if not 0.0 <= dc_decay < 1.0:
            raise ValueError("dc_decay must be in [0, 1)")
        self.reservoir_dim = int(reservoir_dim)
        self.ac_decay      = float(ac_decay)
        self.dc_decay      = float(dc_decay)
        # State — reset per-sequence
        self._ac:   np.ndarray | None = None
        self._dc:   np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._max:  np.ndarray | None = None
        self._last: np.ndarray | None = None

    @property
    def feature_dim(self) -> int:
        """Width of features emitted by :meth:`process_sequence`."""
        return self.reservoir_dim * 6 + 3

    def _reset(self, batch_size: int) -> None:
        shape = (batch_size, self.reservoir_dim)
        self._ac   = np.zeros(shape)
        self._dc   = np.zeros(shape)
        self._mean = np.zeros(shape)
        self._max  = np.full(shape, -np.inf)
        self._last = np.zeros(shape)

    def process_sequence(self, X_seq: np.ndarray) -> np.ndarray:
        """Process a full sequence and return memory features.

        Args:
            X_seq: Reservoir features shaped ``(batch, timesteps, reservoir_dim)``.

        Returns:
            Feature matrix shaped ``(batch, feature_dim)`` —
            ``reservoir_dim * 6 + 3`` wide.
        """
        X_seq = np.asarray(X_seq, dtype=float)
        if X_seq.ndim == 2:
            X_seq = X_seq[None, :, :]   # single sample
        batch, T, d = X_seq.shape
        if d != self.reservoir_dim:
            raise ValueError(f"expected reservoir_dim={self.reservoir_dim}, got {d}")

        self._reset(batch)
        ac, dc = self._ac, self._dc
        mean   = np.zeros((batch, d))
        vmax   = np.full((batch, d), -np.inf)

        for t in range(T):
            x_t  = X_seq[:, t, :]
            ac   = self.ac_decay  * ac  + (1.0 - self.ac_decay)  * x_t
            dc   = self.dc_decay  * dc  + (1.0 - self.dc_decay)  * x_t
            mean = mean + x_t
            np.maximum(vmax, x_t, out=vmax)

        last   = X_seq[:, -1, :]
        mean  /= max(1, T)
        phase  = self._phase(T, batch)  # (batch, 3)

        self._ac   = ac
        self._dc   = dc
        self._last = last

        phase = self._phase(T, batch)  # (batch, 3)
        return np.concatenate([last, mean, vmax, ac, dc, ac * dc, phase], axis=1)

    def _phase(self, T: int, batch: int) -> np.ndarray:
        """Sinusoidal position encoding for the end of a sequence of length T.

        Args:
            T:     Sequence length (number of timesteps).
            batch: Batch size — result is tiled to shape ``(batch, 3)``.

        Returns:
            Phase encoding ``(batch, 3)``:
            ``[sin(position_angle), cos(position_angle), normalized_position]``

        The third component is the normalised final position in [0, 1) so the
        readout can distinguish short from long sequences beyond the sin/cos pair.
        For T <= 1 there is no temporal structure, so all components are zeroed
        (except the bias-like third component which stays 1.0 to avoid dead dims).

        Previously this contained a divide-by-(T-1)/(T-1) bug that collapsed
        every length into the constant vector [0, 1, 1]. The corrected formula
        uses angle = 2π * (T-1) / T, giving a distinct phase per sequence length.
        """
        if T <= 1:
            # No temporal structure — use a neutral, non-zero phase
            phase = np.array([0.0, 0.0, 1.0])
        else:
            # Sinusoidal encoding at the final timestep position (T-1) out of T
            angle = 2.0 * np.pi * (T - 1) / T   # range: (0, 2π) as T → ∞
            phase = np.array([
                np.sin(angle),      # oscillates through quadrants with T
                np.cos(angle),      # orthogonal component
                (T - 1) / T,        # normalised position in [0, 1)
            ])
        return np.tile(phase[None, :], (batch, 1))  # (batch, 3)


# ---------------------------------------------------------------------------
# 3. WeightlessModel — the assembly
# ---------------------------------------------------------------------------

class WeightlessModel:
    """Weightless continual learning model.

    Fixed encoder + AC/DC system memory + isolated per-task heads.
    The only mutable parameters are the output heads, one per task.

    Parameters
    ----------
    input_dim:     Raw input width.
    reservoir_dim: Fixed encoder output width.
    output_dim:    Task output width.
    target_radius: Spectral bound for fixed encoder.
    ac_decay:      AC (fast) stream decay.
    dc_decay:      DC (slow) stream decay.
    lr:            Default learning rate for head updates.
    seed:          Reproducibility seed.
    """

    def __init__(
        self,
        input_dim: int,
        reservoir_dim: int,
        output_dim: int,
        target_radius: float = 0.94,
        ac_decay: float = 0.25,
        dc_decay: float = 0.98,
        lr: float = 0.01,
        seed: int | None = None,
    ) -> None:
        self.lr           = float(lr)
        self.input_dim    = int(input_dim)
        self.reservoir_dim = int(reservoir_dim)
        self.output_dim   = int(output_dim)

        rng  = np.random.default_rng(seed)
        self.encoder = FixedEncoder(
            input_dim=input_dim,
            reservoir_dim=reservoir_dim,
            target_radius=target_radius,
            seed=int(rng.integers(0, 2**32)),
        )
        self.memory  = SystemMemory(
            reservoir_dim=reservoir_dim,
            ac_decay=ac_decay,
            dc_decay=dc_decay,
        )
        self.readout = ZeroForgetReadout(
            output_dim=output_dim,
            feature_dim=self.memory.feature_dim,
            rng=rng,
        )

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def encode_sequence(self, X_seq: np.ndarray) -> np.ndarray:
        """Raw sequences → memory features.

        Args:
            X_seq: ``(batch, timesteps, input_dim)`` or ``(timesteps, input_dim)``
                   for a single sequence.

        Returns:
            ``(batch, feature_dim)`` memory features.
        """
        X_seq = np.asarray(X_seq, dtype=float)
        if X_seq.ndim == 2:
            X_seq = X_seq[None, :, :]
        batch, T, _ = X_seq.shape
        # Apply fixed encoder at every timestep
        enc = np.stack(
            [self.encoder.encode(X_seq[:, t, :]) for t in range(T)], axis=1
        )  # (batch, T, reservoir_dim)
        return self.memory.process_sequence(enc)   # (batch, feature_dim)

    def predict(self, X_seq: np.ndarray, task_id: str | None = None) -> np.ndarray:
        """End-to-end prediction for a sequence batch."""
        features = self.encode_sequence(X_seq)
        return self.readout.predict_batch(features, task_id=task_id)

    def update(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        lr: float | None = None,
        task_id: str | None = None,
    ) -> float:
        """Update active task head from pre-computed features. Returns MSE."""
        return self.readout.update_batch(
            features, targets,
            lr=lr if lr is not None else self.lr,
            task_id=task_id,
        )

    def train_sequence(
        self,
        X_seq: np.ndarray,
        targets: np.ndarray,
        lr: float | None = None,
        task_id: str | None = None,
    ) -> float:
        """Encode sequences then update head. Returns MSE."""
        features = self.encode_sequence(X_seq)
        return self.update(features, targets, lr=lr, task_id=task_id)

    def set_task(self, task_id: str) -> None:
        self.readout.set_task(task_id)

    # ------------------------------------------------------------------
    # Sleep consolidation
    # ------------------------------------------------------------------

    def sleep(
        self,
        task_id: str,
        X_seq: np.ndarray,
        targets: np.ndarray,
        num_cycles: int = 500,
        sleep_lr: float = 0.001,
        batch_size: int = 32,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Sleep-consolidate a task: freeze encoder, replay, update head only.

        The encoder has no mutable weights, so freeze/unfreeze are no-ops here.
        The function call is kept symmetric so the same protocol works for
        models where the encoder does have shared weights.
        """
        X_seq = np.asarray(X_seq, dtype=float)
        features = self.encode_sequence(X_seq)  # extracted once — no moving target
        return sleep_consolidation(
            freeze_fn=lambda: None,
            unfreeze_fn=lambda: None,
            extract_fn=lambda _: features,      # already extracted; ignore re-call input
            readout=self.readout,
            task_id=task_id,
            inputs=np.zeros((len(features), 1)),  # dummy (extract_fn ignores it)
            targets=targets,
            num_cycles=num_cycles,
            sleep_lr=sleep_lr,
            batch_size=batch_size,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def feature_dim(self) -> int:
        return self.memory.feature_dim

    @property
    def task_ids(self) -> list[str]:
        return self.readout.task_ids

    @property
    def total_learned_params(self) -> int:
        """Total learned parameters — only in task heads, never in encoder."""
        return self.readout.param_count()

    @property
    def total_fixed_params(self) -> int:
        """Parameters that are fixed at init and never updated."""
        return self.encoder._W.size

    def param_summary(self) -> dict[str, int]:
        return {
            "fixed_encoder": self.total_fixed_params,
            "learned_heads": self.total_learned_params,
            "tasks":         len(self.task_ids),
        }
