"""Default configuration entry point.

Usage:
    from config.default import cfg  # singleton default config
    from config.default import make_config
"""
from __future__ import annotations

from core.constants import Config512D

# Default singleton — ready to import anywhere
cfg = Config512D()


def make_config(**overrides) -> Config512D:
    """Construct a Config512D with keyword overrides."""
    import dataclasses
    return dataclasses.replace(cfg, **overrides)
