"""Gate configuration generator for Adaptive Helix."""

from __future__ import annotations

import torch
from torch import nn


class GateGenerator(nn.Module):
    """Generate Helix gate parameters from a task embedding.

    Outputs a dictionary containing:
    - ``write_gates`` shaped ``(num_dc_channels, feature_dim)``
    - ``decay_rates`` shaped ``(num_dc_channels,)`` in the configured range
    - ``update_weights`` shaped ``(num_dc_channels, feature_dim)``
    """

    def __init__(
        self,
        task_embedding_dim: int = 64,
        num_dc_channels: int = 32,
        feature_dim: int = 64,
        hidden_dim: int = 64,
        decay_min: float = 0.9,
        decay_max: float = 0.99,
    ) -> None:
        super().__init__()
        if task_embedding_dim <= 0 or num_dc_channels <= 0 or feature_dim <= 0:
            raise ValueError("dimensions must be positive")

        self.task_embedding_dim = int(task_embedding_dim)
        self.num_dc_channels = int(num_dc_channels)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        if not 0.0 <= decay_min <= decay_max <= 0.999:
            raise ValueError("decay bounds must satisfy 0 <= min <= max <= 0.999")
        self.decay_min = float(decay_min)
        self.decay_max = float(decay_max)

        self.shared_encoder = nn.Sequential(
            nn.Linear(self.task_embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.write_gate_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_dc_channels * self.feature_dim),
        )
        self.decay_head = nn.Sequential(
            nn.Linear(self.hidden_dim, max(1, self.hidden_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(1, self.hidden_dim // 2), self.num_dc_channels),
            nn.Sigmoid(),
        )
        self.update_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_dc_channels * self.feature_dim),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Use small initial weights for stable generated gates."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, task_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        """Generate a gate configuration from ``task_embedding``."""
        if task_embedding.ndim != 1 or task_embedding.shape[0] != self.task_embedding_dim:
            raise ValueError(
                "task_embedding must have shape "
                f"({self.task_embedding_dim},), got {tuple(task_embedding.shape)}"
            )

        hidden = self.shared_encoder(task_embedding)
        write_gates = self.write_gate_head(hidden).reshape(
            self.num_dc_channels, self.feature_dim
        )
        update_weights = self.update_head(hidden).reshape(
            self.num_dc_channels, self.feature_dim
        )
        decay_rates = self.decay_head(hidden) * (self.decay_max - self.decay_min) + self.decay_min
        return {
            "write_gates": write_gates,
            "decay_rates": decay_rates,
            "update_weights": update_weights,
        }


def _self_test() -> None:
    generator = GateGenerator(task_embedding_dim=32, num_dc_channels=8, feature_dim=16, hidden_dim=32)
    config = generator(torch.randn(32))
    assert config["write_gates"].shape == (8, 16)
    assert config["decay_rates"].shape == (8,)
    assert config["update_weights"].shape == (8, 16)
    assert torch.isfinite(config["write_gates"]).all()
    assert torch.all(config["decay_rates"] >= 0.9)
    assert torch.all(config["decay_rates"] <= 0.99)
    print("GateGenerator tests passed")


if __name__ == "__main__":
    _self_test()
