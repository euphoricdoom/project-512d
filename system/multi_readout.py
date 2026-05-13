from __future__ import annotations

import numpy as np


class MultiTaskReadout:
    """Separate linear readout per task.

    This isolates output-layer learning: training one task updates only that
    task's readout head while the reservoir, router, and modules remain shared.
    """

    def __init__(
        self,
        output_dim: int,
        feature_dim: int,
        rng: np.random.Generator,
        scale: float = 0.02,
    ) -> None:
        self.output_dim = output_dim
        self.feature_dim = feature_dim
        self.scale = scale
        self.rng = rng
        self.readouts: dict[str, np.ndarray] = {}
        self.current_task: str | None = None

    def set_task(self, task_id: str) -> None:
        self.current_task = task_id
        self.add_task(task_id)

    def add_task(self, task_id: str) -> None:
        if task_id not in self.readouts:
            self.readouts[task_id] = self.rng.normal(
                0.0, self.scale, size=(self.output_dim, self.feature_dim)
            )

    def _task(self, task_id: str | None = None) -> str:
        resolved = task_id if task_id is not None else self.current_task
        if resolved is None:
            resolved = "default"
        self.add_task(resolved)
        return resolved

    def predict(self, features: np.ndarray, task_id: str | None = None) -> np.ndarray:
        resolved = self._task(task_id)
        return self.readouts[resolved] @ features

    def predict_batch(
        self, features: np.ndarray, task_id: str | None = None
    ) -> np.ndarray:
        resolved = self._task(task_id)
        return features @ self.readouts[resolved].T

    def update(
        self,
        features: np.ndarray,
        error: np.ndarray,
        lr: float,
        task_id: str | None = None,
    ) -> None:
        resolved = self._task(task_id)
        self.readouts[resolved] += lr * np.outer(error, features)
        np.clip(self.readouts[resolved], -3.0, 3.0, out=self.readouts[resolved])

    def update_batch(
        self,
        features: np.ndarray,
        errors: np.ndarray,
        lr: float,
        task_id: str | None = None,
    ) -> None:
        resolved = self._task(task_id)
        batch_size = max(1, int(features.shape[0]))
        self.readouts[resolved] += lr * (errors.T @ features) / batch_size
        np.clip(self.readouts[resolved], -3.0, 3.0, out=self.readouts[resolved])

    def stored_params(self) -> int:
        return int(sum(w.size for w in self.readouts.values()))
