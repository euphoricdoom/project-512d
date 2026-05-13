from __future__ import annotations

from typing import Any

import numpy as np


class LayerNorm:
    """Feature-wise layer normalization with learnable affine parameters."""

    def __init__(self, feature_dim: int) -> None:
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self.feature_dim = int(feature_dim)
        self.gamma = np.ones(self.feature_dim, dtype=float)
        self.beta = np.zeros(self.feature_dim, dtype=float)

    def forward(self, x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Normalize over the final feature dimension for 1D or 2D inputs."""
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            if arr.shape[0] != self.feature_dim:
                raise ValueError(
                    f"expected feature shape ({self.feature_dim},), got {arr.shape}"
                )
            mean = np.mean(arr)
            var = np.mean((arr - mean) ** 2)
            normalized = (arr - mean) / np.sqrt(var + eps)
            return normalized * self.gamma + self.beta
        if arr.ndim == 2:
            if arr.shape[1] != self.feature_dim:
                raise ValueError(
                    f"expected feature shape (B, {self.feature_dim}), got {arr.shape}"
                )
            mean = np.mean(arr, axis=1, keepdims=True)
            var = np.mean((arr - mean) ** 2, axis=1, keepdims=True)
            normalized = (arr - mean) / np.sqrt(var + eps)
            return normalized * self.gamma[None, :] + self.beta[None, :]
        raise ValueError(f"expected 1D or 2D input, got {arr.ndim}D")


class StableNetworkReadout:
    """Scale-aware multi-task linear readout over concatenated kernel features."""

    def __init__(
        self,
        num_kernels: int,
        feature_dim: int = 92,
        output_dim: int = 64,
        base_lr: float = 0.01,
        rng: np.random.Generator | None = None,
        grad_clip: float = 10.0,
    ) -> None:
        if num_kernels <= 0:
            raise ValueError("num_kernels must be positive")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if base_lr <= 0:
            raise ValueError("base_lr must be positive")
        if grad_clip <= 0:
            raise ValueError("grad_clip must be positive")

        self.num_kernels = int(num_kernels)
        self.feature_dim = int(feature_dim)
        self.output_dim = int(output_dim)
        self.total_feature_dim = self.num_kernels * self.feature_dim
        self.base_lr = float(base_lr)
        self.effective_lr = self.base_lr / np.sqrt(self.num_kernels)
        self.grad_clip = float(grad_clip)
        self.rng = rng if rng is not None else np.random.default_rng()

        self.norm = LayerNorm(self.total_feature_dim)
        self.layer_norm = self.norm
        self.weights: dict[str, np.ndarray] = {}
        self.readouts = self.weights
        self.current_task: str | None = None
        self._init_scale = 0.02 / np.sqrt(self.num_kernels)
        self._warned_lr_override: set[str] = set()

    def set_task(self, task_id: str) -> None:
        """Set and lazily create the active task head."""
        self.current_task = task_id
        self.add_task(task_id)

    def add_task(self, task_id: str) -> None:
        """Create a task-specific weight matrix if absent."""
        if task_id not in self.weights:
            self.weights[task_id] = self.rng.normal(
                0.0,
                self._init_scale,
                size=(self.output_dim, self.total_feature_dim),
            )

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

    def _targets_2d(self, values: np.ndarray, batch_size: int) -> np.ndarray:
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

    def _norm_cache(
        self, features: np.ndarray, eps: float = 1e-8
    ) -> tuple[np.ndarray, np.ndarray]:
        mean = np.mean(features, axis=1, keepdims=True)
        var = np.mean((features - mean) ** 2, axis=1, keepdims=True)
        x_hat = (features - mean) / np.sqrt(var + eps)
        normalized = x_hat * self.norm.gamma[None, :] + self.norm.beta[None, :]
        return normalized, x_hat

    def predict(self, features: np.ndarray, task_id: str | None = None) -> np.ndarray:
        """Return one prediction for 1D features or batch predictions for 2D features."""
        feature_batch, was_single = self._features_2d(features)
        resolved = self._task(task_id)
        normalized = self.norm.forward(feature_batch)
        prediction = normalized @ self.weights[resolved].T
        return prediction[0] if was_single else prediction

    def predict_batch(
        self, features: np.ndarray, task_id: str | None = None
    ) -> np.ndarray:
        """Return batch predictions for features shaped (B, total_feature_dim)."""
        feature_batch, _ = self._features_2d(features)
        resolved = self._task(task_id)
        return self.norm.forward(feature_batch) @ self.weights[resolved].T

    def update(
        self,
        features: np.ndarray,
        targets_or_errors: np.ndarray,
        task_id: str | float | None = None,
        lr: float | str | None = None,
        *,
        as_error: bool | None = None,
    ) -> float:
        """Update a task head from targets, or from errors when using old call style."""
        task_id, lr, legacy_error_call = self._resolve_update_args(task_id, lr)
        feature_batch, _ = self._features_2d(features)
        values = self._targets_2d(targets_or_errors, feature_batch.shape[0])
        use_errors = legacy_error_call if as_error is None else as_error
        return self._update_batch(feature_batch, values, lr, task_id, use_errors)

    def update_batch(
        self,
        features: np.ndarray,
        errors_or_targets: np.ndarray,
        task_id: str | float | None = None,
        lr: float | str | None = None,
        *,
        as_error: bool | None = None,
    ) -> float:
        """Batch update compatible with old error-style and new target-style calls."""
        task_id, lr, legacy_error_call = self._resolve_update_args(task_id, lr)
        feature_batch, _ = self._features_2d(features)
        values = self._targets_2d(errors_or_targets, feature_batch.shape[0])
        use_errors = legacy_error_call if as_error is None else as_error
        return self._update_batch(feature_batch, values, lr, task_id, use_errors)

    def update_from_targets(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        task_id: str | None = None,
        lr: float | None = None,
    ) -> float:
        """Explicit target-based update path."""
        feature_batch, _ = self._features_2d(features)
        target_batch = self._targets_2d(targets, feature_batch.shape[0])
        return self._update_batch(feature_batch, target_batch, lr, task_id, False)

    def _update_batch(
        self,
        features: np.ndarray,
        values: np.ndarray,
        lr: float | None,
        task_id: str | None,
        values_are_errors: bool,
    ) -> float:
        resolved = self._task(task_id)
        weights = self.weights[resolved]
        step_size = (
            float(lr)
            if lr is not None and getattr(self, "honor_external_lr", False)
            else self.effective_lr
        )

        normalized, x_hat = self._norm_cache(features)
        prediction = normalized @ weights.T
        errors = values if values_are_errors else values - prediction
        loss = float(np.mean(errors**2))

        batch_size = max(1, int(features.shape[0]))
        d_pred = -errors / batch_size
        grad_w = d_pred.T @ normalized
        grad_normalized = d_pred @ weights
        grad_gamma = np.sum(grad_normalized * x_hat, axis=0)
        grad_beta = np.sum(grad_normalized, axis=0)

        grad_norm = float(
            np.sqrt(
                np.sum(grad_w**2)
                + np.sum(grad_gamma**2)
                + np.sum(grad_beta**2)
            )
        )
        if grad_norm > self.grad_clip:
            scale = self.grad_clip / (grad_norm + 1e-12)
            grad_w *= scale
            grad_gamma *= scale
            grad_beta *= scale

        weights -= step_size * grad_w
        self.norm.gamma -= step_size * grad_gamma
        self.norm.beta -= step_size * grad_beta
        np.clip(weights, -3.0, 3.0, out=weights)
        np.clip(self.norm.gamma, -10.0, 10.0, out=self.norm.gamma)
        np.clip(self.norm.beta, -10.0, 10.0, out=self.norm.beta)
        return loss

    def _resolve_update_args(
        self,
        task_id: str | float | None,
        lr: float | str | None,
    ) -> tuple[str | None, float | None, bool]:
        """Resolve new `(task_id, lr)` and legacy `(lr, task_id)` positional calls."""
        if isinstance(task_id, (int, float, np.floating)):
            resolved_lr = float(task_id)
            resolved_task = lr if isinstance(lr, str) else None
            return resolved_task, resolved_lr, True
        if lr is not None and not isinstance(lr, (int, float, np.floating)):
            raise TypeError("lr must be numeric")
        return task_id, None if lr is None else float(lr), False

    def stored_params(self) -> int:
        """Return currently stored trainable parameter count."""
        return int(
            sum(w.size for w in self.weights.values())
            + self.norm.gamma.size
            + self.norm.beta.size
        )

    def get_info(self) -> dict[str, Any]:
        """Return readout configuration and allocation details."""
        return {
            "type": self.__class__.__name__,
            "num_kernels": self.num_kernels,
            "feature_dim": self.feature_dim,
            "total_feature_dim": self.total_feature_dim,
            "output_dim": self.output_dim,
            "base_lr": self.base_lr,
            "effective_lr": float(self.effective_lr),
            "grad_clip": self.grad_clip,
            "num_tasks": len(self.weights),
            "tasks": sorted(self.weights),
            "stored_params": self.stored_params(),
        }
