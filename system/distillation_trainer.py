"""Gate distillation trainer.

Trains ``LearnedGateNetwork`` to mimic explicit Helix DC trajectories from the
working Phase 1 teacher adapter.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

if __package__ in {None, ""}:  # pragma: no cover - script execution path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from system.adaptive_helix import AdaptiveHelixTemporalAdapter
from system.helix_temporal import HelixTemporalAdapter
from system.learned_gates import LearnedGateNetwork
from system.pytorch_helix import PyTorchHelix


class DistillationTrainer:
    """Knowledge-distill learned gates from explicit Helix channel behavior."""

    def __init__(
        self,
        gate_network: LearnedGateNetwork,
        input_width: int = 64,
        feature_dim: int = 256,
        lr: float = 0.001,
        device: str = "cpu",
        gradient_clip: float = 1.0,
        k_support: int = 10,
        seed: int = 42,
    ) -> None:
        self.gate_network = gate_network.to(device)
        self.input_width = int(input_width)
        self.feature_dim = int(feature_dim)
        self.device = torch.device(device)
        self.gradient_clip = float(gradient_clip)
        self.k_support = int(k_support)
        self.rng = np.random.default_rng(seed)
        self.teacher = HelixTemporalAdapter(
            input_dim=self.feature_dim,
            input_width=self.input_width,
            projection_dim=self.feature_dim,
            seed=seed + 11,
        )
        self.optimizer = torch.optim.Adam(
            self.gate_network.parameters(), lr=lr, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )
        self.history: dict[str, object] = {
            "epoch": [],
            "total_loss": [],
            "task_losses": {},
            "correlations": {},
        }

    def make_feature_stream(self, inputs: np.ndarray) -> np.ndarray:
        """Create deterministic feature stream from sequence inputs.

        This keeps teacher and student aligned during distillation and exposes
        task inputs to the learned gate generator through projected features.
        """
        inputs = np.asarray(inputs, dtype=np.float32)
        batch, steps, width = inputs.shape
        features = np.zeros((batch, steps, self.feature_dim), dtype=np.float32)
        n = min(width, self.feature_dim)
        features[:, :, :n] = inputs[:, :, :n]
        if self.feature_dim > n:
            cumulative = np.cumsum(inputs[:, :, :n], axis=1)
            end = min(self.feature_dim, n + n)
            features[:, :, n:end] = cumulative[:, :, : end - n]
        return features

    def extract_teacher_trajectory(
        self,
        task_id: str,
        inputs: np.ndarray,
        features: np.ndarray | None = None,
    ) -> np.ndarray:
        """Extract explicit teacher DC trajectory shaped ``(T, B, D)``."""
        inputs = np.asarray(inputs, dtype=np.float32)
        if features is None:
            features = self.make_feature_stream(inputs)
        batch, seq_length, _ = inputs.shape
        self.teacher.reset(batch)
        trajectory = []
        for t in range(seq_length):
            self.teacher.step(
                inputs[:, t, :],
                features[:, t, :],
                t=t,
                total_steps=seq_length,
                task_id=task_id,
            )
            trajectory.append(self.teacher.dc_state.copy())
        return np.stack(trajectory, axis=0).astype(np.float32)

    def extract_student_trajectory(
        self,
        gate_config: dict[str, torch.Tensor],
        inputs: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Extract differentiable student DC trajectory shaped ``(T, B, D)``."""
        batch, seq_length, _ = inputs.shape
        student = PyTorchHelix(
            input_dim=self.feature_dim,
            input_width=self.input_width,
            projection_dim=self.feature_dim,
            device=str(self.device),
            projection_matrix=self.teacher.W_project,
            projection_bias=self.teacher.b_project,
        )
        student.reset(batch)
        for t in range(seq_length):
            student.step(inputs[:, t, :], features[:, t, :], gate_config, t, seq_length)
        return student.get_dc_trajectory()

    def compute_distillation_loss(
        self, student_trajectory: torch.Tensor, teacher_trajectory: np.ndarray
    ) -> torch.Tensor:
        """Compute MSE trajectory loss."""
        teacher = torch.as_tensor(
            teacher_trajectory, dtype=torch.float32, device=self.device
        )
        return F.mse_loss(student_trajectory, teacher)

    @staticmethod
    def _correlation(student: np.ndarray, teacher: np.ndarray) -> float:
        if np.std(student) < 1e-12 or np.std(teacher) < 1e-12:
            return 0.0
        return float(np.corrcoef(student.reshape(-1), teacher.reshape(-1))[0, 1])

    def train_epoch(
        self,
        tasks_data: dict[str, tuple[np.ndarray, np.ndarray]],
        batch_size: int = 16,
    ) -> dict[str, object]:
        """Train one epoch over all tasks."""
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
            features = self.make_feature_stream(batch_inputs)
            teacher = self.extract_teacher_trajectory(task_id, batch_inputs, features)
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
            torch.nn.utils.clip_grad_norm_(
                self.gate_network.parameters(), self.gradient_clip
            )
            self.optimizer.step()
            losses[task_id] = float(loss.detach().cpu())
            correlations[task_id] = self._correlation(
                student.detach().cpu().numpy(), teacher
            )
        return {
            "loss": float(np.mean(list(losses.values()))),
            "task_losses": losses,
            "correlation": float(np.mean(list(correlations.values()))),
            "task_correlations": correlations,
        }

    def train(
        self,
        tasks_data: dict[str, tuple[np.ndarray, np.ndarray]],
        num_epochs: int = 100,
        batch_size: int = 16,
        verbose: bool = True,
        save_path: Optional[str] = None,
        sleep_cycles: int = 0,
        sleep_lr: float | None = None,
    ) -> None:
        """Run full distillation loop, optionally followed by replay consolidation."""
        best_loss = float("inf")
        start = time.time()
        for epoch in range(num_epochs):
            metrics = self.train_epoch(tasks_data, batch_size=batch_size)
            self.scheduler.step(float(metrics["loss"]))
            self.history["epoch"].append(epoch)
            self.history["total_loss"].append(metrics["loss"])
            for task_id, loss in metrics["task_losses"].items():
                self.history["task_losses"].setdefault(task_id, []).append(loss)
            for task_id, corr in metrics["task_correlations"].items():
                self.history["correlations"].setdefault(task_id, []).append(corr)
            if save_path and metrics["loss"] < best_loss:
                best_loss = float(metrics["loss"])
                self.gate_network.save(save_path)
            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(
                    f"Epoch {epoch + 1:3d}/{num_epochs}: "
                    f"loss={metrics['loss']:.6f}, corr={metrics['correlation']:.4f}, "
                    f"elapsed={time.time() - start:.1f}s"
                )
        if sleep_cycles > 0:
            for task_id, (inputs, _) in tasks_data.items():
                metrics = self.sleep_consolidate_gates(
                    task_id,
                    inputs,
                    num_cycles=sleep_cycles,
                    batch_size=batch_size,
                    sleep_lr=sleep_lr,
                    verbose=verbose,
                )
                if save_path and metrics["loss"] < best_loss:
                    best_loss = float(metrics["loss"])
                    self.gate_network.save(save_path)

    def sleep_consolidate_gates(
        self,
        task_id: str,
        inputs: np.ndarray,
        num_cycles: int = 1000,
        batch_size: int = 16,
        sleep_lr: float | None = None,
        verbose: bool = True,
    ) -> dict[str, float]:
        """Consolidate gate parameters through extra frozen-feature replay."""
        if sleep_lr is not None:
            old_lrs = [group["lr"] for group in self.optimizer.param_groups]
            for group in self.optimizer.param_groups:
                group["lr"] = float(sleep_lr)
        else:
            old_lrs = None

        if verbose:
            print(f"Sleep gate consolidation: {task_id}, cycles={num_cycles}")

        last_loss = 0.0
        last_corr = 0.0
        for cycle in range(num_cycles):
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
            try:
                features = self.make_feature_stream(batch_inputs, task_id=task_id)
            except TypeError:
                features = self.make_feature_stream(batch_inputs)
            teacher = self.extract_teacher_trajectory(task_id, batch_inputs, features)
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
            last_loss = float(loss.detach().cpu())
            last_corr = self._correlation(student.detach().cpu().numpy(), teacher)
            if verbose and ((cycle + 1) % max(1, num_cycles // 10) == 0 or cycle == 0):
                print(f"  sleep cycle {cycle + 1:5d}/{num_cycles}: loss={last_loss:.6f}, corr={last_corr:.4f}")

        self.history.setdefault("sleep", {})[task_id] = {
            "cycles": int(num_cycles),
            "loss": last_loss,
            "correlation": last_corr,
        }
        if old_lrs is not None:
            for group, lr in zip(self.optimizer.param_groups, old_lrs):
                group["lr"] = lr
        return {"loss": last_loss, "correlation": last_corr}

    def save_history(self, path: str | Path) -> None:
        """Save history JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")

    def make_numpy_adapter(
        self, gate_network: LearnedGateNetwork | None = None
    ) -> AdaptiveHelixTemporalAdapter:
        """Create a NumPy adaptive adapter aligned with the distillation teacher."""
        adapter = AdaptiveHelixTemporalAdapter(
            input_dim=self.feature_dim,
            input_width=self.input_width,
            projection_dim=self.feature_dim,
            gate_network=gate_network or self.gate_network,
        )
        adapter.W_project = self.teacher.W_project.copy()
        adapter.projection = adapter.W_project
        adapter.b_project = self.teacher.b_project.copy()
        return adapter


def _self_test() -> None:
    gate = LearnedGateNetwork(input_dim=8, task_embedding_dim=16, num_dc_channels=16, feature_dim=16)
    trainer = DistillationTrainer(gate, input_width=8, feature_dim=16, lr=0.002, k_support=4)
    tasks = {
        "copy_task": (
            np.random.default_rng(1).normal(size=(24, 5, 8)).astype(np.float32),
            np.zeros((24, 8), dtype=np.float32),
        )
    }
    before = trainer.train_epoch(tasks, batch_size=8)["loss"]
    for _ in range(3):
        metrics = trainer.train_epoch(tasks, batch_size=8)
    assert metrics["loss"] >= 0
    print(f"DistillationTrainer tests passed; first_loss={before:.6f}, last_loss={metrics['loss']:.6f}")


if __name__ == "__main__":
    _self_test()
