from __future__ import annotations

from typing import Any

import numpy as np


class TaskSpecificProjection:
    """Task-specific projection and readout over concatenated kernel features.

    Each task owns two trainable matrices:

    - a projection from ``num_kernels * base_feature_dim`` to ``projection_dim``
    - a readout from ``projection_dim`` to ``output_dim``

    Updates perform plain NumPy backpropagation through both matrices for mean
    squared error. Tasks are created lazily on first use.
    """

    def __init__(
        self,
        num_kernels: int,
        base_feature_dim: int = 92,
        projection_dim: int = 256,
        output_dim: int = 64,
        base_lr: float = 0.01,
        rng: np.random.Generator | None = None,
        grad_clip: float | None = 10.0,
        projection_scale: float | None = None,
        readout_scale: float | None = None,
        learning_rate: float | None = None,
        activation: str = "tanh",
    ) -> None:
        """Initialize an empty multi-task projection/readout bank.

        Args:
            num_kernels: Number of kernel feature blocks in the concatenated input.
            base_feature_dim: Feature width produced by each kernel.
            projection_dim: Task-specific bottleneck dimension.
            output_dim: Output width for each task readout. Defaults to 64.
            base_lr: Default SGD step size used by update methods.
            rng: Optional NumPy random generator for reproducible initialization.
            grad_clip: Optional global gradient-norm clip. ``None`` disables it.
            projection_scale: Optional normal init stddev for projection matrices.
            readout_scale: Optional normal init stddev for readout matrices.
            learning_rate: Backward-compatible alias for ``base_lr``.
            activation: Bottleneck activation. Currently ``"tanh"`` or ``"linear"``.
        """
        if num_kernels <= 0:
            raise ValueError("num_kernels must be positive")
        if base_feature_dim <= 0:
            raise ValueError("base_feature_dim must be positive")
        if projection_dim <= 0:
            raise ValueError("projection_dim must be positive")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if learning_rate is not None:
            base_lr = learning_rate
        if base_lr <= 0:
            raise ValueError("base_lr must be positive")
        if grad_clip is not None and grad_clip <= 0:
            raise ValueError("grad_clip must be positive when provided")
        if activation not in {"tanh", "linear"}:
            raise ValueError("activation must be 'tanh' or 'linear'")

        self.num_kernels = int(num_kernels)
        self.base_feature_dim = int(base_feature_dim)
        self.feature_dim = self.base_feature_dim
        self.total_feature_dim = self.num_kernels * self.base_feature_dim
        self.projection_dim = int(projection_dim)
        self.output_dim = int(output_dim)
        self.base_lr = float(base_lr)
        self.learning_rate = self.base_lr
        self.grad_clip = None if grad_clip is None else float(grad_clip)
        self.activation = activation
        self.rng = rng if rng is not None else np.random.default_rng()

        self.projection_scale = (
            float(projection_scale)
            if projection_scale is not None
            else 1.0 / np.sqrt(self.total_feature_dim)
        )
        self.readout_scale = (
            float(readout_scale)
            if readout_scale is not None
            else 1.0 / np.sqrt(self.projection_dim)
        )

        self.projections: dict[str, np.ndarray] = {}
        self.readouts: dict[str, np.ndarray] = {}
        self.projection_biases: dict[str, np.ndarray] = {}
        self.readout_biases: dict[str, np.ndarray] = {}
        self.projection_matrices = self.projections
        self.readout_matrices = self.readouts
        self.weights = self.readouts
        self.current_task: str | None = None

    def _activate(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "linear":
            return z
        return np.tanh(z)

    def _activation_derivative(self, activated: np.ndarray) -> np.ndarray:
        if self.activation == "linear":
            return np.ones_like(activated)
        return 1.0 - activated**2

    def set_task(self, task_id: str) -> None:
        """Set the active task, creating its parameters if needed."""
        self.current_task = task_id
        self.add_task(task_id)

    def add_task(self, task_id: str) -> None:
        """Create task-specific projection and readout matrices if absent."""
        if task_id not in self.projections:
            self.projections[task_id] = self.rng.normal(
                0.0,
                self.projection_scale,
                size=(self.projection_dim, self.total_feature_dim),
            )
        if task_id not in self.readouts:
            self.readouts[task_id] = self.rng.normal(
                0.0,
                self.readout_scale,
                size=(self.output_dim, self.projection_dim),
            )
        if task_id not in self.projection_biases:
            self.projection_biases[task_id] = np.zeros(self.projection_dim, dtype=float)
        if task_id not in self.readout_biases:
            self.readout_biases[task_id] = np.zeros(self.output_dim, dtype=float)

    def _task(self, task_id: str | None = None) -> str:
        resolved = task_id if task_id is not None else self.current_task
        if resolved is None:
            resolved = "default"
        self.add_task(resolved)
        return resolved

    def _features_2d(self, features: np.ndarray) -> tuple[np.ndarray, bool]:
        arr = np.asarray(features, dtype=float)
        if arr.ndim == 1:
            if arr.shape[0] != self.total_feature_dim:
                raise ValueError(
                    f"expected {self.total_feature_dim} features, got {arr.shape[0]}"
                )
            return arr[None, :], True
        if arr.ndim == 2:
            if arr.shape[1] != self.total_feature_dim:
                raise ValueError(
                    f"expected feature shape (B, {self.total_feature_dim}), got {arr.shape}"
                )
            return arr, False
        raise ValueError(f"expected 1D or 2D features, got {arr.ndim}D")

    def _outputs_2d(self, values: np.ndarray, batch_size: int) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 1:
            if batch_size != 1:
                raise ValueError("1D target/error is only valid for single-sample updates")
            if arr.shape[0] != self.output_dim:
                raise ValueError(
                    f"expected output shape ({self.output_dim},), got {arr.shape}"
                )
            return arr[None, :]
        if arr.ndim == 2:
            if arr.shape != (batch_size, self.output_dim):
                raise ValueError(
                    f"expected output shape ({batch_size}, {self.output_dim}), got {arr.shape}"
                )
            return arr
        raise ValueError(f"expected 1D or 2D targets/errors, got {arr.ndim}D")

    def project(self, features: np.ndarray, task_id: str | None = None) -> np.ndarray:
        """Project input features to the task-specific ``projection_dim`` space."""
        feature_batch, was_single = self._features_2d(features)
        resolved = self._task(task_id)
        projected = self._activate(
            feature_batch @ self.projections[resolved].T
            + self.projection_biases[resolved][None, :]
        )
        return projected[0] if was_single else projected

    def get_task_features(
        self, features: np.ndarray, task_id: str | None = None
    ) -> np.ndarray:
        """Compatibility alias for task-specific projected features."""
        return self.project(features, task_id)

    def forward(self, features: np.ndarray, task_id: str | None = None) -> np.ndarray:
        """Return predictions for a single feature vector or a feature batch."""
        feature_batch, was_single = self._features_2d(features)
        resolved = self._task(task_id)
        projected = self._activate(
            feature_batch @ self.projections[resolved].T
            + self.projection_biases[resolved][None, :]
        )
        prediction = projected @ self.readouts[resolved].T + self.readout_biases[resolved][None, :]
        return prediction[0] if was_single else prediction

    def predict(self, features: np.ndarray, task_id: str | None = None) -> np.ndarray:
        """Compatibility alias for :meth:`forward`."""
        return self.forward(features, task_id)

    def predict_batch(
        self, features: np.ndarray, task_id: str | None = None
    ) -> np.ndarray:
        """Return batch predictions for features shaped ``(B, total_feature_dim)``."""
        feature_batch, _ = self._features_2d(features)
        return self.forward(feature_batch, task_id)

    def update(
        self,
        features: np.ndarray,
        targets_or_errors: np.ndarray,
        task_id: str | None = None,
        lr: float | None = None,
        *,
        as_error: bool = False,
    ) -> float:
        """Update projection and readout weights, returning mean squared loss.

        Args:
            features: Input features shaped ``(total_feature_dim,)`` or
                ``(B, total_feature_dim)``.
            targets_or_errors: Targets by default, or prediction errors when
                ``as_error=True``. Shape must match ``output_dim``.
            task_id: Optional task to update. Uses the current task or
                ``"default"`` when omitted.
            lr: Optional learning rate override for this update.
            as_error: Treat ``targets_or_errors`` as ``target - prediction``.
        """
        feature_batch, _ = self._features_2d(features)
        values = self._outputs_2d(targets_or_errors, feature_batch.shape[0])
        return self._update_batch(feature_batch, values, task_id, lr, as_error)

    def update_batch(
        self,
        features: np.ndarray,
        targets_or_errors: np.ndarray,
        task_id: str | None = None,
        lr: float | None = None,
        *,
        as_error: bool = False,
    ) -> float:
        """Batch update alias for :meth:`update`."""
        return self.update(features, targets_or_errors, task_id, lr, as_error=as_error)

    def _update_batch(
        self,
        features: np.ndarray,
        values: np.ndarray,
        task_id: str | None,
        lr: float | None,
        values_are_errors: bool,
    ) -> float:
        resolved = self._task(task_id)
        projection = self.projections[resolved]
        readout = self.readouts[resolved]
        projection_bias = self.projection_biases[resolved]
        readout_bias = self.readout_biases[resolved]
        step_size = self.learning_rate if lr is None else float(lr)
        if step_size <= 0:
            raise ValueError("lr must be positive")

        projected_pre = features @ projection.T + projection_bias[None, :]
        projected = self._activate(projected_pre)
        prediction = projected @ readout.T + readout_bias[None, :]
        errors = values if values_are_errors else values - prediction
        loss = float(np.mean(errors**2))

        batch_size = max(1, int(features.shape[0]))
        grad_readout = errors.T @ projected / batch_size
        grad_readout_bias = np.mean(errors, axis=0)
        grad_projected = (errors @ readout) * self._activation_derivative(projected)
        grad_projection = grad_projected.T @ features / batch_size
        grad_projection_bias = np.mean(grad_projected, axis=0)

        if self.grad_clip is not None:
            grad_norm = float(
                np.sqrt(
                    np.sum(grad_projection**2)
                    + np.sum(grad_readout**2)
                    + np.sum(grad_projection_bias**2)
                    + np.sum(grad_readout_bias**2)
                )
            )
            if grad_norm > self.grad_clip:
                scale = self.grad_clip / (grad_norm + 1e-12)
                grad_projection *= scale
                grad_readout *= scale
                grad_projection_bias *= scale
                grad_readout_bias *= scale

        projection += step_size * grad_projection
        readout += step_size * grad_readout
        projection_bias += step_size * grad_projection_bias
        readout_bias += step_size * grad_readout_bias
        return loss

    def stored_params(self) -> int:
        """Return the number of currently allocated trainable parameters."""
        return int(
            sum(matrix.size for matrix in self.projections.values())
            + sum(matrix.size for matrix in self.readouts.values())
            + sum(vector.size for vector in self.projection_biases.values())
            + sum(vector.size for vector in self.readout_biases.values())
        )

    def get_info(self) -> dict[str, Any]:
        """Return configuration and task allocation details."""
        return {
            "type": self.__class__.__name__,
            "num_kernels": self.num_kernels,
            "base_feature_dim": self.base_feature_dim,
            "feature_dim": self.feature_dim,
            "total_feature_dim": self.total_feature_dim,
            "projection_dim": self.projection_dim,
            "output_dim": self.output_dim,
            "learning_rate": self.learning_rate,
            "base_lr": self.base_lr,
            "grad_clip": self.grad_clip,
            "activation": self.activation,
            "num_tasks": len(self.projections),
            "tasks": sorted(self.projections),
            "stored_params": self.stored_params(),
        }
