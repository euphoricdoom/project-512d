"""Test: route_input() returns a valid probability distribution."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from core.constants import Config512D
from system.modular_system import ModularFieldSystem


def test_routing_sums_to_one():
    cfg = Config512D()
    system = ModularFieldSystem(cfg)
    rng = np.random.default_rng(7)
    inp = rng.standard_normal(cfg.input_dim)
    weights = system.route_input(inp)
    assert abs(weights.sum() - 1.0) < 1e-9, (
        f"Routing weights sum to {weights.sum():.10f}, expected 1.0"
    )


def test_routing_non_negative():
    cfg = Config512D()
    system = ModularFieldSystem(cfg)
    rng = np.random.default_rng(13)
    inp = rng.standard_normal(cfg.input_dim)
    weights = system.route_input(inp)
    assert np.all(weights >= 0), "Routing weights contain negative values"


def test_routing_size():
    cfg = Config512D()
    system = ModularFieldSystem(cfg)
    inp = np.zeros(cfg.input_dim)
    weights = system.route_input(inp)
    assert len(weights) == cfg.num_modules, (
        f"Expected {cfg.num_modules} routing weights, got {len(weights)}"
    )
