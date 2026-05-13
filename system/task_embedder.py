"""Task embedding network for learned temporal gates.

``TaskEmbedder`` encodes a small support set of K example sequences into one
fixed-width task vector. The learned gate generator consumes that vector to
produce Helix DC write, update, and decay parameters.
"""

from __future__ import annotations

import torch
from torch import nn


class TaskEmbedder(nn.Module):
    """Encode a task from K example sequences.

    Args:
        input_dim: Per-timestep input width.
        hidden_dim: LSTM hidden size.
        embedding_dim: Output task embedding size.
        num_layers: Number of LSTM layers.

    Example:
        >>> embedder = TaskEmbedder(input_dim=64, embedding_dim=128)
        >>> examples = torch.randn(10, 15, 64)
        >>> embedder(examples).shape
        torch.Size([128])
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
        embedding_dim: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.num_layers = int(num_layers)

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.1 if self.num_layers > 1 else 0.0,
        )
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

    def forward(self, examples: torch.Tensor) -> torch.Tensor:
        """Return a task embedding shaped ``(embedding_dim,)``.

        Args:
            examples: Support examples shaped ``(K, seq_length, input_dim)``.
        """
        if examples.ndim != 3:
            raise ValueError(
                f"examples must have shape (K, seq_length, input_dim), got {tuple(examples.shape)}"
            )
        if examples.shape[2] != self.input_dim:
            raise ValueError(
                f"expected input_dim {self.input_dim}, got {examples.shape[2]}"
            )

        _, (hidden, _) = self.lstm(examples)
        sequence_embeddings = hidden[-1]
        scores = self.attention(sequence_embeddings)
        weights = torch.softmax(scores, dim=0)
        pooled = (weights * sequence_embeddings).sum(dim=0)
        return self.projection(pooled)


def _self_test() -> None:
    torch.manual_seed(42)
    embedder = TaskEmbedder(input_dim=16, hidden_dim=16, embedding_dim=32)
    examples = torch.randn(4, 6, 16)
    embedding = embedder(examples)
    assert embedding.shape == (32,)
    assert torch.isfinite(embedding).all()
    print("TaskEmbedder tests passed")


if __name__ == "__main__":
    _self_test()
