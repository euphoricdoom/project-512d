"""Centralized configuration for the weightless learning system.

Type-safe dataclasses for all hyperparameters. Validated at construction.
Import anywhere — no circular dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EncoderConfig:
    """Fixed encoder hyperparameters."""

    input_dim: int
    reservoir_dim: int
    target_radius: float = 0.94
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.reservoir_dim <= 0:
            raise ValueError(
                f"input_dim and reservoir_dim must be positive, "
                f"got input_dim={self.input_dim}, reservoir_dim={self.reservoir_dim}"
            )
        if not 0.0 < self.target_radius < 1.0:
            raise ValueError(
                f"target_radius must be in (0, 1), got {self.target_radius}. "
                f"Values >= 1.0 risk instability; values <= 0.0 lose all dynamics."
            )


@dataclass
class MemoryConfig:
    """AC/DC temporal memory hyperparameters."""

    reservoir_dim: int
    ac_decay: float = 0.25  # Fast stream — ~4-timestep memory window
    dc_decay: float = 0.98  # Slow stream — ~50-timestep memory window

    def __post_init__(self) -> None:
        if self.reservoir_dim <= 0:
            raise ValueError(
                f"reservoir_dim must be positive, got {self.reservoir_dim}"
            )
        if not 0.0 <= self.ac_decay < 1.0:
            raise ValueError(
                f"ac_decay must be in [0, 1), got {self.ac_decay}"
            )
        if not 0.0 <= self.dc_decay < 1.0:
            raise ValueError(
                f"dc_decay must be in [0, 1), got {self.dc_decay}"
            )

    @property
    def feature_dim(self) -> int:
        """Total feature width: reservoir_dim * 6 components + 3 phase dims."""
        return self.reservoir_dim * 6 + 3


@dataclass
class ReadoutConfig:
    """Zero-forget readout head hyperparameters."""

    output_dim: int
    feature_dim: int
    learning_rate: float = 0.05
    weight_init_scale: float = 0.02
    weight_clip: float = 3.0

    def __post_init__(self) -> None:
        if self.output_dim <= 0 or self.feature_dim <= 0:
            raise ValueError(
                f"output_dim and feature_dim must be positive, "
                f"got output_dim={self.output_dim}, feature_dim={self.feature_dim}"
            )
        if self.learning_rate <= 0.0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.weight_clip <= 0.0:
            raise ValueError(
                f"weight_clip must be positive, got {self.weight_clip}"
            )


@dataclass
class TrainingConfig:
    """Training loop hyperparameters."""

    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 0.05
    sleep_cycles: int = 0
    sleep_lr: float = 0.001
    max_train: Optional[int] = None
    max_test: Optional[int] = None

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0.0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.sleep_lr <= 0.0:
            raise ValueError(f"sleep_lr must be positive, got {self.sleep_lr}")
        if self.sleep_cycles < 0:
            raise ValueError(
                f"sleep_cycles must be non-negative, got {self.sleep_cycles}"
            )


@dataclass
class TFIDFConfig:
    """TF-IDF vectorization hyperparameters."""

    tfidf_dim: int = 2000
    max_sentences: int = 6
    sublinear_tf: bool = True
    strip_accents: str = "unicode"

    def __post_init__(self) -> None:
        if self.tfidf_dim <= 0:
            raise ValueError(f"tfidf_dim must be positive, got {self.tfidf_dim}")
        if self.max_sentences <= 0:
            raise ValueError(
                f"max_sentences must be positive, got {self.max_sentences}"
            )


@dataclass
class ModelConfig:
    """Complete weightless model configuration.

    Use the class methods for validated defaults, or construct manually
    and pass components directly.
    """

    encoder: EncoderConfig
    memory: MemoryConfig
    readout: ReadoutConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tfidf: TFIDFConfig = field(default_factory=TFIDFConfig)

    @classmethod
    def default_imdb(cls) -> "ModelConfig":
        """Defaults optimised for IMDB sentiment (78.3% accuracy, 2.6 MB int8).

        Architecture validated in Milestone 0.5:
          - R=1536 gives best accuracy/size tradeoff across tested sizes
          - target_radius=0.94 keeps dynamics contractive
          - ac_decay=0.25, dc_decay=0.98 for AC/DC temporal binding
        """
        reservoir_dim = 1536
        tfidf_dim = 2000
        feature_dim = reservoir_dim * 6 + 3

        return cls(
            encoder=EncoderConfig(
                input_dim=tfidf_dim,
                reservoir_dim=reservoir_dim,
                target_radius=0.94,
            ),
            memory=MemoryConfig(
                reservoir_dim=reservoir_dim,
                ac_decay=0.25,
                dc_decay=0.98,
            ),
            readout=ReadoutConfig(
                output_dim=2,
                feature_dim=feature_dim,
                learning_rate=0.05,
            ),
            training=TrainingConfig(
                epochs=5,
                batch_size=64,
                learning_rate=0.05,
            ),
            tfidf=TFIDFConfig(
                tfidf_dim=tfidf_dim,
                max_sentences=6,
            ),
        )

    @classmethod
    def default_multiclass(cls, num_classes: int) -> "ModelConfig":
        """Defaults for multi-class classification (based on IMDB config)."""
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        config = cls.default_imdb()
        config.readout = ReadoutConfig(
            output_dim=num_classes,
            feature_dim=config.readout.feature_dim,
            learning_rate=config.readout.learning_rate,
            weight_init_scale=config.readout.weight_init_scale,
            weight_clip=config.readout.weight_clip,
        )
        return config
