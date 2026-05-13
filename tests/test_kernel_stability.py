"""Test: kernel spectral radius is strictly below target."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from core.constants import Config512D
from core.kernel import CoupledModularKernel, spectral_radius


def test_spectral_radius_below_target():
    cfg = Config512D()
    rng = np.random.default_rng(cfg.seed)
    kernel = CoupledModularKernel(cfg, rng)
    r = spectral_radius(kernel.K)
    assert r < cfg.target_radius, (
        f"Spectral radius {r:.6f} >= target {cfg.target_radius}"
    )


def test_kernel_no_nan_inf():
    cfg = Config512D()
    rng = np.random.default_rng(cfg.seed)
    kernel = CoupledModularKernel(cfg, rng)
    assert np.all(np.isfinite(kernel.K)), "Kernel contains NaN or Inf"


def test_kernel_square():
    cfg = Config512D()
    rng = np.random.default_rng(cfg.seed)
    kernel = CoupledModularKernel(cfg, rng)
    r, c = kernel.K.shape
    assert r == c == cfg.dim, f"Expected ({cfg.dim}, {cfg.dim}), got ({r}, {c})"
