"""Stability sweep experiment configuration."""
from __future__ import annotations

from config.default import make_config

# Sweep range
SWEEP_LOW    = 0.001
SWEEP_HIGH   = 0.020
SWEEP_POINTS = 15

# Base config for the sweep
sweep_cfg = make_config(seed=42)
