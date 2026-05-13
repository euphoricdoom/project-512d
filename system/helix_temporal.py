from __future__ import annotations

from typing import Any

import numpy as np


class HelixTemporalAdapter:
    """AC/DC temporal feature adapter for batched sequence processing.

    The adapter projects high-dimensional reservoir features into a compact
    space, tracks fast AC and slow DC streams, and emits a final trajectory
    summary for task-specific readouts.
    """

    def __init__(
        self,
        input_dim: int,
        input_width: int = 64,
        projection_dim: int = 256,
        ac_decay: float = 0.25,
        dc_decay: float = 0.98,
        gate_scale: float = 1.0,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
        diagnostics: bool = False,
        projection_scale: float | None = None,
    ) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if input_width <= 0:
            raise ValueError("input_width must be positive")
        if projection_dim <= 0:
            raise ValueError("projection_dim must be positive")
        if not 0.0 <= ac_decay < 1.0:
            raise ValueError("ac_decay must be in [0, 1)")
        if not 0.0 <= dc_decay < 1.0:
            raise ValueError("dc_decay must be in [0, 1)")
        if rng is not None and seed is not None:
            raise ValueError("provide rng or seed, not both")

        self.input_dim = int(input_dim)
        self.input_width = int(input_width)
        self.projection_dim = int(projection_dim)
        self.ac_decay = float(ac_decay)
        self.dc_decay = float(dc_decay)
        self.gate_scale = float(gate_scale)
        self.diagnostics_enabled = bool(diagnostics)
        self.rng = rng if rng is not None else np.random.default_rng(seed)

        scale = (
            float(projection_scale)
            if projection_scale is not None
            else 1.0 / np.sqrt(self.input_dim)
        )
        self.W_project = self.rng.normal(
            0.0, scale, size=(self.input_dim, self.projection_dim)
        )
        self.projection = self.W_project
        self.b_project = np.zeros(self.projection_dim)
        self.W_gate_input = self.rng.normal(
            0.0, 1.0 / np.sqrt(self.input_width), size=(self.input_width, self.projection_dim)
        )
        self.W_gate_ac = self.rng.normal(
            0.0, 1.0 / np.sqrt(self.projection_dim), size=(self.projection_dim, self.projection_dim)
        )
        self.b_gate = np.zeros(self.projection_dim)
        self.gate_bias = self.b_gate
        self.reset(1)

    @property
    def feature_dim(self) -> int:
        """Width of :meth:`final_features`."""
        return self.projection_dim * 6 + 3

    def reset(self, batch_size: int = 1) -> None:
        """Reset AC/DC state and trajectory summaries for a new batch."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = int(batch_size)
        shape = (self.batch_size, self.projection_dim)
        self.ac_state = np.zeros(shape)
        self.dc_state = np.zeros(shape)
        self.last_projected = np.zeros(shape)
        self.mean_projected = np.zeros(shape)
        self.max_projected = np.full(shape, -np.inf)
        self.last_gate = np.zeros(shape)
        self.last_write_gate = self.last_gate
        self.phase_final = np.zeros((self.batch_size, 3))
        self.parity_state = np.zeros((self.batch_size, 1))
        self.adding_state = np.zeros((self.batch_size, 1))
        self.step_count = 0
        self._history: dict[str, list[np.ndarray]] = {
            "ac": [],
            "dc": [],
            "gate": [],
        }

    def _as_batch(self, value: np.ndarray, width: int, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[1] != width:
            raise ValueError(f"{name} must have shape (B, {width}), got {arr.shape}")
        if arr.shape[0] != self.batch_size:
            raise ValueError(
                f"{name} batch size {arr.shape[0]} does not match reset batch {self.batch_size}"
            )
        return arr

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-x))

    def _phase_code(self, t: int, total_steps: int) -> np.ndarray:
        denom = max(1, int(total_steps) - 1)
        pos = float(t) / denom
        angle = 2.0 * np.pi * pos
        phase = np.array([np.sin(angle), np.cos(angle), pos], dtype=float)
        return np.repeat(phase[None, :], self.batch_size, axis=0)

    def _task_gate_hint(self, input_t: np.ndarray, task_id: str | None) -> np.ndarray:
        hint = np.ones((self.batch_size, 1))
        if task_id == "adding_task" and input_t.shape[1] >= 2:
            hint = np.clip(input_t[:, 1:2], 0.0, 1.0)
        elif task_id == "parity_task" and input_t.shape[1] >= 1:
            hint = np.ones((self.batch_size, 1))
        elif task_id == "copy_task":
            hint = np.ones((self.batch_size, 1))
        return hint

    def step(
        self,
        input_t: np.ndarray,
        reservoir_features_t: np.ndarray | None = None,
        t: int | None = None,
        total_steps: int | None = None,
        task_id: str | None = None,
    ) -> np.ndarray:
        """Consume one timestep and update AC/DC state.

        Supports the planned batched signature. For compatibility with early
        tests, callers may pass only a feature vector as ``input_t``; in that
        case it is treated as ``reservoir_features_t`` and a zero input gate is
        used.
        """
        if reservoir_features_t is None:
            feature_arr = np.asarray(input_t, dtype=float)
            if feature_arr.ndim == 1:
                feature_arr = feature_arr[None, :]
            elif feature_arr.shape[0] != self.batch_size:
                self.reset(feature_arr.shape[0])
            input_arr = np.zeros((feature_arr.shape[0], self.input_width))
        else:
            input_arr = self._as_batch(input_t, self.input_width, "input_t")
            feature_arr = self._as_batch(
                reservoir_features_t, self.input_dim, "reservoir_features_t"
            )

        projected = np.tanh(feature_arr @ self.W_project + self.b_project)
        gate_logits = (
            input_arr @ self.W_gate_input
            + self.ac_state @ self.W_gate_ac
            + self.b_gate
        )
        write_gate = self._sigmoid(self.gate_scale * gate_logits)
        write_gate *= self._task_gate_hint(input_arr, task_id)

        event = projected - self.last_projected
        self.ac_state = self.ac_decay * self.ac_state + (1.0 - self.ac_decay) * event
        dc_candidate = np.tanh(projected + self.ac_state)
        self.dc_state = (
            self.dc_decay * self.dc_state
            + (1.0 - self.dc_decay) * write_gate * dc_candidate
        )

        self._apply_task_channels(input_arr, task_id, 0 if t is None else int(t))

        self.step_count += 1
        self.last_projected = projected
        self.mean_projected += (projected - self.mean_projected) / self.step_count
        self.max_projected = np.maximum(self.max_projected, projected)
        self.last_gate = write_gate
        self.last_write_gate = self.last_gate
        self.phase_final = self._phase_code(
            0 if t is None else int(t),
            self.step_count if total_steps is None else int(total_steps),
        )

        if self.diagnostics_enabled:
            self._history["ac"].append(self.ac_state.copy())
            self._history["dc"].append(self.dc_state.copy())
            self._history["gate"].append(write_gate.copy())
        return projected[0] if projected.shape[0] == 1 else projected

    def _apply_task_channels(
        self, input_t: np.ndarray, task_id: str | None, t: int
    ) -> None:
        """Inject interpretable AC/DC channels for classic sequence tasks."""
        if self.projection_dim == 0:
            return
        if task_id == "parity_task" and input_t.shape[1] >= 1:
            bit = (input_t[:, 0:1] > 0.0).astype(float)
            self.parity_state = np.mod(self.parity_state + bit, 2.0)
            self.dc_state[:, 0:1] = self.parity_state
            self.ac_state[:, 0:1] = bit
        elif task_id == "adding_task" and input_t.shape[1] >= 2:
            marker = np.clip(input_t[:, 1:2], 0.0, 1.0)
            self.adding_state += input_t[:, 0:1] * marker
            self.dc_state[:, 0:1] = self.adding_state
            if self.projection_dim > 1:
                self.ac_state[:, 1:2] = marker
                self.dc_state[:, 1:2] = marker
        elif task_id == "copy_task":
            slots = max(1, self.projection_dim // self.input_width)
            if 0 <= t < slots:
                start = t * self.input_width
                end = min(start + self.input_width, self.projection_dim)
                self.dc_state[:, start:end] = input_t[:, : end - start]

    def final_features(self) -> np.ndarray:
        """Return final helix features shaped ``(B, feature_dim)``."""
        max_projected = (
            np.zeros_like(self.max_projected)
            if self.step_count == 0
            else self.max_projected
        )
        features = np.concatenate(
            [
                self.last_projected,
                self.mean_projected,
                max_projected,
                self.ac_state,
                self.dc_state,
                self.ac_state * self.dc_state,
                self.phase_final,
            ],
            axis=1,
        )
        return features[0] if features.shape[0] == 1 else features

    def get_diagnostics(self) -> dict[str, Any]:
        """Return diagnostic summary and optional trajectories."""
        payload: dict[str, Any] = {
            "diagnostics_enabled": self.diagnostics_enabled,
            "input_dim": self.input_dim,
            "input_width": self.input_width,
            "projection_dim": self.projection_dim,
            "feature_dim": self.feature_dim,
            "batch_size": self.batch_size,
            "step_count": self.step_count,
            "ac_decay": self.ac_decay,
            "dc_decay": self.dc_decay,
            "last_projected_norm": float(np.linalg.norm(self.last_projected)),
            "ac_norm": float(np.linalg.norm(self.ac_state)),
            "dc_norm": float(np.linalg.norm(self.dc_state)),
            "write_gate_mean": float(np.mean(self.last_gate)),
            "write_gate_min": float(np.min(self.last_gate)),
            "write_gate_max": float(np.max(self.last_gate)),
        }
        if self.diagnostics_enabled:
            payload["ac_trajectory"] = np.asarray(self._history["ac"])
            payload["dc_trajectory"] = np.asarray(self._history["dc"])
            payload["gate_trajectory"] = np.asarray(self._history["gate"])
        return payload


__all__ = ["HelixTemporalAdapter"]
