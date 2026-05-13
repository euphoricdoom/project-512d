"""Distillation trainer with cached, stable feature streams.

V2 exists to test the "moving feature target" diagnosis directly. It uses a
deterministic cache per task/batch shape, but keeps the first feature channels
input-derived so parity/addition targets remain learnable.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from system.distillation_trainer import DistillationTrainer


class DistillationTrainerV2(DistillationTrainer):
    """Distillation trainer using fixed cached features for teacher/student."""

    def __init__(self, *args, random_feature_scale: float = 0.1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_feature_scale = float(random_feature_scale)
        self.feature_cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _stable_seed(key: str) -> int:
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "little") % (2**32)

    def get_fixed_features(
        self,
        task_id: str,
        batch_size: int,
        seq_length: int,
        inputs: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return stable features for a task/batch/sequence shape.

        The random tail is cached. If ``inputs`` is supplied, the leading
        channels are overwritten with current and cumulative inputs so the
        feature stream is both stable and task-informative.
        """
        cache_key = f"{task_id}:{batch_size}:{seq_length}:{self.feature_dim}"
        if cache_key not in self.feature_cache:
            rng = np.random.default_rng(self._stable_seed(cache_key))
            self.feature_cache[cache_key] = (
                rng.normal(0.0, self.random_feature_scale, size=(batch_size, seq_length, self.feature_dim))
                .astype(np.float32)
            )
        features = self.feature_cache[cache_key].copy()
        if inputs is not None:
            inputs = np.asarray(inputs, dtype=np.float32)
            width = min(inputs.shape[2], self.feature_dim)
            features[:, :, :width] = inputs[:, :, :width]
            if self.feature_dim > width:
                cumulative = np.cumsum(inputs[:, :, :width], axis=1)
                end = min(self.feature_dim, width * 2)
                features[:, :, width:end] = cumulative[:, :, : end - width]
        return features

    def make_feature_stream(self, inputs: np.ndarray, task_id: str = "default") -> np.ndarray:
        """Create a fixed, cached feature stream for ``inputs``."""
        inputs = np.asarray(inputs, dtype=np.float32)
        return self.get_fixed_features(
            task_id=task_id,
            batch_size=inputs.shape[0],
            seq_length=inputs.shape[1],
            inputs=inputs,
        )

    def extract_teacher_trajectory(
        self,
        task_id: str,
        inputs: np.ndarray,
        features: np.ndarray | None = None,
    ) -> np.ndarray:
        """Extract teacher trajectory with fixed cached features."""
        if features is None:
            features = self.make_feature_stream(inputs, task_id=task_id)
        return super().extract_teacher_trajectory(task_id, inputs, features)

    def train_epoch(
        self,
        tasks_data: dict[str, tuple[np.ndarray, np.ndarray]],
        batch_size: int = 16,
    ) -> dict[str, object]:
        """Train one epoch using V2 fixed feature streams."""
        self.gate_network.train()
        losses: dict[str, float] = {}
        correlations: dict[str, float] = {}
        for task_id, (inputs, _) in tasks_data.items():
            n = len(inputs)
            idx = self.rng.choice(n, size=min(batch_size, n), replace=False)
            batch_inputs = np.asarray(inputs[idx], dtype=np.float32)
            if batch_inputs.shape[2] != self.input_width:
                padded = np.zeros(
                    (batch_inputs.shape[0], batch_inputs.shape[1], self.input_width),
                    dtype=np.float32,
                )
                m = min(batch_inputs.shape[2], self.input_width)
                padded[:, :, :m] = batch_inputs[:, :, :m]
                batch_inputs = padded
            features = self.make_feature_stream(batch_inputs, task_id=task_id)
            teacher = self.extract_teacher_trajectory(task_id, batch_inputs, features)
            import torch

            support = torch.as_tensor(
                batch_inputs[: min(self.k_support, len(batch_inputs))],
                dtype=torch.float32,
                device=self.device,
            )
            gate_config = self.gate_network.configure_for_task(support)
            student = self.extract_student_trajectory(
                gate_config,
                torch.as_tensor(batch_inputs, dtype=torch.float32, device=self.device),
                torch.as_tensor(features, dtype=torch.float32, device=self.device),
            )
            loss = self.compute_distillation_loss(student, teacher)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.gate_network.parameters(), self.gradient_clip)
            self.optimizer.step()
            losses[task_id] = float(loss.detach().cpu())
            correlations[task_id] = self._correlation(student.detach().cpu().numpy(), teacher)
        return {
            "loss": float(np.mean(list(losses.values()))),
            "task_losses": losses,
            "correlation": float(np.mean(list(correlations.values()))),
            "task_correlations": correlations,
        }


def _self_test() -> None:
    from system.learned_gates import LearnedGateNetwork

    gate = LearnedGateNetwork(input_dim=8, task_embedding_dim=16, num_dc_channels=16, feature_dim=16)
    trainer = DistillationTrainerV2(gate, input_width=8, feature_dim=16, lr=0.002, k_support=4)
    x = np.random.default_rng(4).normal(size=(8, 5, 8)).astype(np.float32)
    f1 = trainer.make_feature_stream(x, "copy_task")
    f2 = trainer.make_feature_stream(x, "copy_task")
    assert np.allclose(f1, f2)
    tasks = {"copy_task": (x, np.zeros((8, 8), dtype=np.float32))}
    metrics = trainer.train_epoch(tasks, batch_size=4)
    assert metrics["loss"] >= 0
    print("DistillationTrainerV2 tests passed")


if __name__ == "__main__":
    _self_test()
