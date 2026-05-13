"""Differentiable Helix temporal dynamics for gate distillation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

if __package__ in {None, ""}:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class PyTorchHelix(nn.Module):
    """PyTorch implementation of ``AdaptiveHelixTemporalAdapter`` dynamics.

    The class mirrors the repository's NumPy adapter shape contract:
    ``projection_dim`` is the width of projected, AC, and DC states.
    """

    def __init__(
        self,
        input_dim: int = 256,
        input_width: int = 64,
        projection_dim: int = 256,
        ac_decay: float = 0.25,
        dc_decay: float = 0.98,
        device: str = "cpu",
        projection_matrix: np.ndarray | torch.Tensor | None = None,
        projection_bias: np.ndarray | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or input_width <= 0 or projection_dim <= 0:
            raise ValueError("dimensions must be positive")
        self.input_dim = int(input_dim)
        self.input_width = int(input_width)
        self.projection_dim = int(projection_dim)
        self.ac_dim = self.projection_dim
        self.dc_dim = self.projection_dim
        self.ac_decay = float(ac_decay)
        self.dc_decay = float(dc_decay)
        self.device = torch.device(device)

        if projection_matrix is None:
            matrix = torch.eye(self.input_dim, self.projection_dim, dtype=torch.float32)
            if self.input_dim != self.projection_dim:
                matrix = torch.randn(self.input_dim, self.projection_dim) / np.sqrt(
                    self.input_dim
                )
        else:
            matrix = torch.as_tensor(projection_matrix, dtype=torch.float32)
        if projection_bias is None:
            bias = torch.zeros(self.projection_dim, dtype=torch.float32)
        else:
            bias = torch.as_tensor(projection_bias, dtype=torch.float32)
        if matrix.shape != (self.input_dim, self.projection_dim):
            raise ValueError(
                f"projection_matrix must have shape {(self.input_dim, self.projection_dim)}, got {tuple(matrix.shape)}"
            )
        if bias.shape != (self.projection_dim,):
            raise ValueError(f"projection_bias must have shape ({self.projection_dim},)")

        self.register_buffer("projection_matrix", matrix.to(self.device))
        self.register_buffer("projection_bias", bias.to(self.device))
        self.reset(1)

    @property
    def feature_dim(self) -> int:
        """Width of final Helix features."""
        return self.projection_dim * 6 + 3

    @classmethod
    def from_numpy_adapter(cls, adapter: Any, device: str = "cpu") -> "PyTorchHelix":
        """Create a PyTorchHelix with projection weights copied from a NumPy adapter."""
        return cls(
            input_dim=adapter.input_dim,
            input_width=adapter.input_width,
            projection_dim=adapter.projection_dim,
            ac_decay=adapter.ac_decay,
            dc_decay=adapter.dc_decay,
            device=device,
            projection_matrix=adapter.W_project,
            projection_bias=adapter.b_project,
        )

    def reset(self, batch_size: int) -> None:
        """Reset AC/DC state and trajectory buffers."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = int(batch_size)
        shape = (self.batch_size, self.projection_dim)
        self.ac_state = torch.zeros(shape, device=self.device)
        self.dc_state = torch.zeros(shape, device=self.device)
        self.last_projected = torch.zeros(shape, device=self.device)
        self.mean_projected = torch.zeros(shape, device=self.device)
        self.max_projected = torch.full(shape, -torch.inf, device=self.device)
        self.phase_final = torch.zeros((self.batch_size, 3), device=self.device)
        self.step_count = 0
        self._projected_history: list[torch.Tensor] = []
        self._dc_history: list[torch.Tensor] = []
        self._gate_history: list[torch.Tensor] = []

    def _phase_code(self, t: int, total_steps: int) -> torch.Tensor:
        denom = max(1, int(total_steps) - 1)
        pos = float(t) / denom
        phase = torch.tensor(
            [np.sin(2.0 * np.pi * pos), np.cos(2.0 * np.pi * pos), pos],
            dtype=torch.float32,
            device=self.device,
        )
        return phase.unsqueeze(0).repeat(self.batch_size, 1)

    def project(self, features_t: torch.Tensor) -> torch.Tensor:
        """Project raw reservoir features to compact Helix features."""
        features_t = features_t.to(self.device, dtype=torch.float32)
        return torch.tanh(features_t @ self.projection_matrix + self.projection_bias)

    def step(
        self,
        input_t: torch.Tensor,
        features_t: torch.Tensor,
        gate_config: dict[str, torch.Tensor],
        t: int,
        total_steps: int,
    ) -> torch.Tensor:
        """Run one differentiable learned-gate Helix step."""
        input_t = input_t.to(self.device, dtype=torch.float32)
        del input_t  # Input is kept in the signature for API symmetry.
        projected = self.project(features_t)

        event = projected - self.last_projected
        self.ac_state = self.ac_decay * self.ac_state + (1.0 - self.ac_decay) * event

        write_weights = gate_config["write_gates"].to(self.device, dtype=torch.float32)
        update_weights = gate_config["update_weights"].to(self.device, dtype=torch.float32)
        decay_rates = gate_config["decay_rates"].to(self.device, dtype=torch.float32)
        write_gate = torch.sigmoid(projected @ write_weights.T)
        candidate = torch.tanh(projected @ update_weights.T)
        decay = decay_rates.unsqueeze(0)
        self.dc_state = decay * self.dc_state + (1.0 - decay) * write_gate * candidate

        self.step_count += 1
        self.last_projected = projected
        self.mean_projected = self.mean_projected + (
            projected - self.mean_projected
        ) / self.step_count
        self.max_projected = torch.maximum(self.max_projected, projected)
        self.phase_final = self._phase_code(t, total_steps)
        self._projected_history.append(projected)
        self._dc_history.append(self.dc_state)
        self._gate_history.append(write_gate)
        return self.dc_state

    def get_dc_trajectory(self) -> torch.Tensor:
        """Return DC trajectory shaped ``(T, B, projection_dim)``."""
        if not self._dc_history:
            raise RuntimeError("No trajectory recorded. Call step() first.")
        return torch.stack(self._dc_history, dim=0)

    def get_gate_trajectory(self) -> torch.Tensor:
        """Return write-gate trajectory shaped ``(T, B, projection_dim)``."""
        if not self._gate_history:
            raise RuntimeError("No gate trajectory recorded. Call step() first.")
        return torch.stack(self._gate_history, dim=0)

    def get_final_features(self) -> torch.Tensor:
        """Return final Helix features shaped ``(B, feature_dim)``."""
        max_projected = (
            torch.zeros_like(self.max_projected)
            if self.step_count == 0
            else self.max_projected
        )
        return torch.cat(
            [
                self.last_projected,
                self.mean_projected,
                max_projected,
                self.ac_state,
                self.dc_state,
                self.ac_state * self.dc_state,
                self.phase_final,
            ],
            dim=1,
        )


def _self_test() -> None:
    from system.adaptive_helix import AdaptiveHelixTemporalAdapter
    from system.learned_gates import LearnedGateNetwork

    torch.manual_seed(7)
    gate_net = LearnedGateNetwork(
        input_dim=4, task_embedding_dim=16, num_dc_channels=8, feature_dim=8
    )
    support = torch.randn(5, 3, 4)
    config = gate_net.configure_for_task(support)
    np_adapter = AdaptiveHelixTemporalAdapter(
        input_dim=8, input_width=4, projection_dim=8, seed=11
    )
    np_adapter.current_gate_config = config
    np_adapter.gate_config_np = {
        key: value.detach().numpy().copy() for key, value in config.items()
    }
    pt = PyTorchHelix.from_numpy_adapter(np_adapter)
    np_adapter.reset(2)
    pt.reset(2)
    rng = np.random.default_rng(3)
    max_diff = 0.0
    for t in range(4):
        inp = rng.normal(size=(2, 4)).astype(np.float32)
        feat = rng.normal(size=(2, 8)).astype(np.float32)
        dc_pt = pt.step(torch.from_numpy(inp), torch.from_numpy(feat), config, t, 4)
        np_adapter.step(inp, feat, t=t, total_steps=4)
        max_diff = max(
            max_diff, float(np.max(np.abs(dc_pt.detach().numpy() - np_adapter.dc_state)))
        )
    assert max_diff < 1e-5, max_diff

    write = torch.randn(8, 8, requires_grad=True)
    update = torch.randn(8, 8, requires_grad=True)
    raw_decay = torch.randn(8, requires_grad=True)
    config2 = {
        "write_gates": write,
        "decay_rates": torch.sigmoid(raw_decay) * 0.09 + 0.9,
        "update_weights": update,
    }
    pt.reset(2)
    for t in range(3):
        pt.step(torch.randn(2, 4), torch.randn(2, 8), config2, t, 3)
    loss = pt.get_dc_trajectory().mean()
    loss.backward()
    assert write.grad is not None and update.grad is not None and raw_decay.grad is not None
    print(f"PyTorchHelix tests passed; max NumPy diff={max_diff:.2e}")


if __name__ == "__main__":
    _self_test()
