"""Tests for the centralized configuration system."""

import pytest
import numpy as np

from config.model_config import (
    EncoderConfig,
    MemoryConfig,
    ReadoutConfig,
    TrainingConfig,
    ModelConfig,
)
from config.constants import (
    DEFAULT_RESERVOIR_DIM,
    DEFAULT_TFIDF_DIM,
    DEFAULT_MAX_SENTENCES,
    TARGET_SPECTRAL_RADIUS,
    AC_DECAY_RATE,
    DC_DECAY_RATE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
)


# ---------------------------------------------------------------------------
# EncoderConfig
# ---------------------------------------------------------------------------

class TestEncoderConfig:
    def test_valid_construction(self):
        cfg = EncoderConfig(input_dim=100, reservoir_dim=256)
        assert cfg.input_dim == 100
        assert cfg.reservoir_dim == 256
        assert cfg.target_radius == 0.94  # default

    def test_custom_target_radius(self):
        cfg = EncoderConfig(input_dim=64, reservoir_dim=128, target_radius=0.80)
        assert cfg.target_radius == 0.80

    def test_rejects_non_positive_input_dim(self):
        with pytest.raises(ValueError, match="must be positive"):
            EncoderConfig(input_dim=0, reservoir_dim=256)

    def test_rejects_non_positive_reservoir_dim(self):
        with pytest.raises(ValueError, match="must be positive"):
            EncoderConfig(input_dim=100, reservoir_dim=-1)

    def test_rejects_target_radius_at_one(self):
        with pytest.raises(ValueError, match="target_radius must be in"):
            EncoderConfig(input_dim=100, reservoir_dim=256, target_radius=1.0)

    def test_rejects_target_radius_above_one(self):
        with pytest.raises(ValueError, match="target_radius must be in"):
            EncoderConfig(input_dim=100, reservoir_dim=256, target_radius=1.5)

    def test_rejects_target_radius_at_zero(self):
        with pytest.raises(ValueError, match="target_radius must be in"):
            EncoderConfig(input_dim=100, reservoir_dim=256, target_radius=0.0)


# ---------------------------------------------------------------------------
# MemoryConfig
# ---------------------------------------------------------------------------

class TestMemoryConfig:
    def test_feature_dim_formula(self):
        cfg = MemoryConfig(reservoir_dim=256)
        assert cfg.feature_dim == 256 * 6 + 3

    def test_feature_dim_matches_model(self):
        for r in [64, 256, 1024, 1536]:
            cfg = MemoryConfig(reservoir_dim=r)
            assert cfg.feature_dim == r * 6 + 3

    def test_rejects_non_positive_reservoir_dim(self):
        with pytest.raises(ValueError, match="must be positive"):
            MemoryConfig(reservoir_dim=0)

    def test_rejects_ac_decay_out_of_range(self):
        with pytest.raises(ValueError, match="ac_decay must be in"):
            MemoryConfig(reservoir_dim=64, ac_decay=1.0)

    def test_rejects_dc_decay_out_of_range(self):
        with pytest.raises(ValueError, match="dc_decay must be in"):
            MemoryConfig(reservoir_dim=64, dc_decay=1.1)

    def test_accepts_zero_decay(self):
        cfg = MemoryConfig(reservoir_dim=64, ac_decay=0.0)
        assert cfg.ac_decay == 0.0


# ---------------------------------------------------------------------------
# ReadoutConfig
# ---------------------------------------------------------------------------

class TestReadoutConfig:
    def test_valid_construction(self):
        cfg = ReadoutConfig(output_dim=4, feature_dim=512)
        assert cfg.output_dim == 4
        assert cfg.feature_dim == 512

    def test_rejects_non_positive_output_dim(self):
        with pytest.raises(ValueError, match="must be positive"):
            ReadoutConfig(output_dim=0, feature_dim=512)

    def test_rejects_non_positive_learning_rate(self):
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            ReadoutConfig(output_dim=2, feature_dim=512, learning_rate=0.0)

    def test_rejects_non_positive_weight_clip(self):
        with pytest.raises(ValueError, match="weight_clip must be positive"):
            ReadoutConfig(output_dim=2, feature_dim=512, weight_clip=-1.0)


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------

class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.epochs == 5
        assert cfg.batch_size == 64
        assert cfg.learning_rate == 0.05
        assert cfg.sleep_cycles == 0

    def test_rejects_zero_epochs(self):
        with pytest.raises(ValueError, match="epochs must be positive"):
            TrainingConfig(epochs=0)

    def test_rejects_zero_batch_size(self):
        with pytest.raises(ValueError, match="batch_size must be positive"):
            TrainingConfig(batch_size=0)

    def test_rejects_negative_sleep_cycles(self):
        with pytest.raises(ValueError, match="sleep_cycles must be non-negative"):
            TrainingConfig(sleep_cycles=-1)


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

class TestModelConfigDefaultImdb:
    def setup_method(self):
        self.cfg = ModelConfig.default_imdb()

    def test_encoder_dimensions(self):
        assert self.cfg.encoder.input_dim == DEFAULT_TFIDF_DIM
        assert self.cfg.encoder.reservoir_dim == DEFAULT_RESERVOIR_DIM
        assert self.cfg.encoder.target_radius == TARGET_SPECTRAL_RADIUS

    def test_memory_dimensions(self):
        assert self.cfg.memory.reservoir_dim == DEFAULT_RESERVOIR_DIM
        assert self.cfg.memory.ac_decay == AC_DECAY_RATE
        assert self.cfg.memory.dc_decay == DC_DECAY_RATE

    def test_readout_dimensions(self):
        expected_feature_dim = DEFAULT_RESERVOIR_DIM * 6 + 3
        assert self.cfg.readout.feature_dim == expected_feature_dim
        assert self.cfg.readout.output_dim == 2  # binary sentiment

    def test_tfidf_config(self):
        assert self.cfg.tfidf.tfidf_dim == DEFAULT_TFIDF_DIM
        assert self.cfg.tfidf.max_sentences == DEFAULT_MAX_SENTENCES

    def test_feature_dim_consistency(self):
        assert self.cfg.memory.feature_dim == self.cfg.readout.feature_dim


class TestModelConfigDefaultMulticlass:
    def test_adjusts_output_dim(self):
        cfg = ModelConfig.default_multiclass(num_classes=10)
        assert cfg.readout.output_dim == 10

    def test_inherits_encoder_dims(self):
        cfg = ModelConfig.default_multiclass(num_classes=4)
        assert cfg.encoder.reservoir_dim == DEFAULT_RESERVOIR_DIM

    def test_rejects_non_positive_num_classes(self):
        with pytest.raises(ValueError, match="num_classes must be positive"):
            ModelConfig.default_multiclass(num_classes=0)

    def test_preserves_feature_dim(self):
        cfg = ModelConfig.default_multiclass(num_classes=8)
        assert cfg.readout.feature_dim == DEFAULT_RESERVOIR_DIM * 6 + 3


# ---------------------------------------------------------------------------
# Constants sanity checks
# ---------------------------------------------------------------------------

class TestConstants:
    def test_spectral_radius_is_stable(self):
        assert 0.0 < TARGET_SPECTRAL_RADIUS < 1.0

    def test_default_seed_is_positive(self):
        assert DEFAULT_SEED > 0

    def test_decay_rates_in_range(self):
        assert 0.0 <= AC_DECAY_RATE < 1.0
        assert 0.0 <= DC_DECAY_RATE < 1.0
        assert AC_DECAY_RATE < DC_DECAY_RATE  # AC must be faster than DC

    def test_learning_rate_is_positive(self):
        assert DEFAULT_LEARNING_RATE > 0.0
