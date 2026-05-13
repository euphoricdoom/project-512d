"""Learned gate interfaces for Adaptive Helix.

This module provides two layers:

* ``LearnedGateNetwork``: the PyTorch task-example encoder and generator from
  the Phase 2 learned-gates design.
* ``LearnedGateController``: a tiny NumPy supervised gate controller used by
  fast unit tests and lightweight diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

try:
    from system.gate_generator import GateGenerator
    from system.task_embedder import TaskEmbedder
except ImportError:  # Allows ``python system/learned_gates.py``.
    from gate_generator import GateGenerator
    from task_embedder import TaskEmbedder


class LearnedGateNetwork(nn.Module):
    """Complete PyTorch gate generator from K task examples."""

    def __init__(
        self,
        input_dim: int = 64,
        task_embedding_dim: int = 64,
        num_dc_channels: int = 32,
        feature_dim: int = 64,
        decay_min: float = 0.9,
        decay_max: float = 0.99,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.task_embedding_dim = int(task_embedding_dim)
        self.num_dc_channels = int(num_dc_channels)
        self.feature_dim = int(feature_dim)
        self.decay_min = float(decay_min)
        self.decay_max = float(decay_max)
        self.task_embedder = TaskEmbedder(
            input_dim=self.input_dim,
            embedding_dim=self.task_embedding_dim,
        )
        self.gate_generator = GateGenerator(
            task_embedding_dim=self.task_embedding_dim,
            num_dc_channels=self.num_dc_channels,
            feature_dim=self.feature_dim,
            decay_min=self.decay_min,
            decay_max=self.decay_max,
        )

    def configure_for_task(self, task_examples: torch.Tensor) -> dict[str, torch.Tensor]:
        """Generate gate config from examples shaped ``(K, T, input_dim)``."""
        embedding = self.task_embedder(task_examples)
        return self.gate_generator(embedding)

    def forward(self, task_examples: torch.Tensor) -> dict[str, torch.Tensor]:
        """Alias for :meth:`configure_for_task`."""
        return self.configure_for_task(task_examples)

    def save(self, path: str | Path) -> None:
        """Save model weights to ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path) -> None:
        """Load model weights from ``path``."""
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # Older PyTorch versions do not expose weights_only.
            state = torch.load(path, map_location="cpu")
        self.load_state_dict(state)


@dataclass
class LearnedGateConfig:
    """Configuration for the lightweight NumPy gate controller."""

    input_width: int
    state_dim: int
    learning_rate: float = 0.05
    seed: int | None = None
    task_embedding_dim: int = 8
    grad_clip: float = 5.0


class LearnedGateController:
    """Small task-conditioned logistic gate controller.

    The controller predicts per-channel gates from ``[input_t, ac_state,
    task_embedding]`` and supports direct supervised updates. It is intentionally
    compact so tests and diagnostics can run without a PyTorch training loop.
    """

    def __init__(self, config: LearnedGateConfig) -> None:
        if config.input_width <= 0 or config.state_dim <= 0:
            raise ValueError("input_width and state_dim must be positive")
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.feature_width = (
            config.input_width + config.state_dim + config.task_embedding_dim + 1
        )
        scale = 1.0 / np.sqrt(self.feature_width)
        self.weights = self.rng.normal(0.0, scale, size=(self.feature_width, config.state_dim))
        self.task_embeddings: dict[str, np.ndarray] = {}
        self.last_metrics: dict[str, float] = {}

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-x))

    def _task_embedding(self, task_id: str | None) -> np.ndarray:
        key = "default" if task_id is None else str(task_id)
        if key not in self.task_embeddings:
            self.task_embeddings[key] = self.rng.normal(
                0.0, 0.1, size=(self.config.task_embedding_dim,)
            )
        return self.task_embeddings[key]

    def _features(self, input_t: np.ndarray, ac_state: np.ndarray, task_id: str | None) -> np.ndarray:
        input_arr = np.asarray(input_t, dtype=float)
        ac_arr = np.asarray(ac_state, dtype=float)
        if input_arr.ndim == 1:
            input_arr = input_arr[None, :]
        if ac_arr.ndim == 1:
            ac_arr = ac_arr[None, :]
        if input_arr.shape[0] != ac_arr.shape[0]:
            raise ValueError("input_t and ac_state batch sizes must match")
        if input_arr.shape[1] != self.config.input_width:
            raise ValueError(f"input_t must have width {self.config.input_width}")
        if ac_arr.shape[1] != self.config.state_dim:
            raise ValueError(f"ac_state must have width {self.config.state_dim}")
        emb = np.repeat(self._task_embedding(task_id)[None, :], input_arr.shape[0], axis=0)
        bias = np.ones((input_arr.shape[0], 1))
        return np.concatenate([input_arr, ac_arr, emb, bias], axis=1)

    def forward(self, input_t: np.ndarray, ac_state: np.ndarray, task_id: str | None = None) -> np.ndarray:
        """Predict gates shaped ``(B, state_dim)`` in ``[0, 1]``."""
        features = self._features(input_t, ac_state, task_id)
        return self._sigmoid(features @ self.weights)

    def loss(
        self,
        input_t: np.ndarray,
        ac_state: np.ndarray,
        target_gate: np.ndarray,
        task_id: str | None = None,
    ) -> float:
        """Return mean squared gate target loss."""
        target = np.asarray(target_gate, dtype=float)
        pred = self.forward(input_t, ac_state, task_id)
        return float(np.mean((pred - target) ** 2))

    def update(
        self,
        input_t: np.ndarray,
        ac_state: np.ndarray,
        target_gate: np.ndarray,
        task_id: str | None = None,
    ) -> dict[str, float]:
        """Perform one supervised gate update and return metrics."""
        features = self._features(input_t, ac_state, task_id)
        target = np.asarray(target_gate, dtype=float)
        pred = self._sigmoid(features @ self.weights)
        if target.shape != pred.shape:
            raise ValueError(f"target_gate must have shape {pred.shape}, got {target.shape}")
        error = pred - target
        grad_logits = (2.0 / pred.size) * error * pred * (1.0 - pred)
        grad = features.T @ grad_logits
        norm = float(np.linalg.norm(grad))
        if norm > self.config.grad_clip:
            grad *= self.config.grad_clip / (norm + 1e-12)
        self.weights -= self.config.learning_rate * grad
        metrics = {"loss": float(np.mean(error**2)), "gate_mean": float(np.mean(pred))}
        self.last_metrics = metrics
        return metrics

    def state_dict(self) -> dict[str, Any]:
        """Return a copy-safe serializable state dictionary."""
        return {
            "config": self.config.__dict__.copy(),
            "weights": self.weights.copy(),
            "task_embeddings": {k: v.copy() for k, v in self.task_embeddings.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load from :meth:`state_dict` without aliasing caller-owned arrays."""
        self.weights = np.asarray(state["weights"], dtype=float).copy()
        self.task_embeddings = {
            str(k): np.asarray(v, dtype=float).copy()
            for k, v in state.get("task_embeddings", {}).items()
        }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def visualize_gate_config(gate_config: dict[str, torch.Tensor], task_name: str = "Unknown") -> None:
    """Print a compact gate configuration summary."""
    decay = gate_config["decay_rates"].detach().cpu()
    write = gate_config["write_gates"].detach().cpu()
    update = gate_config["update_weights"].detach().cpu()
    print(f"\nGate Configuration for: {task_name}")
    print("=" * 60)
    print(f"Decay rates: mean={decay.mean():.3f}, std={decay.std():.3f}")
    print(f"Write gates: mean={write.mean():.3f}, std={write.std():.3f}")
    print(f"Update weights: mean={update.mean():.3f}, std={update.std():.3f}")
    print("=" * 60)


def _self_test() -> None:
    gate_net = LearnedGateNetwork(input_dim=16, task_embedding_dim=32, num_dc_channels=8, feature_dim=16)
    examples = torch.randn(4, 6, 16)
    config = gate_net.configure_for_task(examples)
    assert config["write_gates"].shape == (8, 16)
    assert 10_000 < count_parameters(gate_net) < 250_000

    controller = LearnedGateController(LearnedGateConfig(input_width=3, state_dim=4, seed=1))
    x = np.random.default_rng(1).normal(size=(2, 3))
    ac = np.random.default_rng(2).normal(size=(2, 4))
    y = np.ones((2, 4))
    before = controller.loss(x, ac, y, "demo")
    for _ in range(8):
        controller.update(x, ac, y, "demo")
    after = controller.loss(x, ac, y, "demo")
    assert after < before
    print(f"Total parameters: {count_parameters(gate_net):,}")
    print("LearnedGateNetwork tests passed")


if __name__ == "__main__":
    _self_test()
