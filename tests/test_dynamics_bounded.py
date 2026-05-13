"""Test: state norm stays bounded under 200 unconstrained steps."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from core.constants import Config512D
from system.modular_system import ModularFieldSystem


def test_state_bounded_200_steps():
    cfg = Config512D()
    system = ModularFieldSystem(cfg)

    for _ in range(200):
        system.step()

    norm = float(np.linalg.norm(system.state))
    assert norm < 1e4, f"State norm blew up: {norm:.2f}"


def test_state_no_nan():
    cfg = Config512D()
    system = ModularFieldSystem(cfg)

    for _ in range(100):
        system.step()

    assert np.all(np.isfinite(system.state)), "State contains NaN or Inf"


def test_inject_and_step():
    cfg = Config512D()
    system = ModularFieldSystem(cfg)
    rng = np.random.default_rng(999)
    inp = rng.standard_normal(cfg.input_dim) * 0.5
    system.inject(inp)
    system.step()
    assert np.all(np.isfinite(system.state))
