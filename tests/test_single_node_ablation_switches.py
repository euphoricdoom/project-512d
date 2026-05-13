"""Tests for single-node ablation switches."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from core.constants import Config512D
from system.modular_system import ModularFieldSystem
from system.trainer import ModularTrainer


def _inter_factors(system: ModularFieldSystem) -> tuple[np.ndarray, np.ndarray]:
    return system.kernel._U.copy(), system.kernel._V.copy()


def test_sparse_routing_activates_configured_winners():
    cfg = Config512D(k_winners=5)
    system = ModularFieldSystem(cfg)
    rng = np.random.default_rng(5)
    weights = system.route_input(rng.standard_normal(cfg.input_dim))
    assert np.count_nonzero(weights) == cfg.k_winners
    assert abs(weights.sum() - 1.0) < 1e-9


def test_full_routing_escape_hatch_uses_all_modules():
    cfg = Config512D(k_winners=0)
    system = ModularFieldSystem(cfg)
    rng = np.random.default_rng(6)
    weights = system.route_input(rng.standard_normal(cfg.input_dim))
    assert np.count_nonzero(weights) == cfg.num_modules
    assert abs(weights.sum() - 1.0) < 1e-9


def test_fixed_inter_modules_do_not_change_after_training_sample():
    cfg = Config512D(
        seed=11,
        train_epochs=1,
        train_samples=1,
        learnable_inter_modules=False,
    )
    system = ModularFieldSystem(cfg)
    trainer = ModularTrainer(system)
    U_before, V_before = _inter_factors(system)
    rng = np.random.default_rng(12)
    x = rng.normal(0.0, 0.55, size=cfg.input_dim)
    y = np.tanh(2.0 * x[:cfg.output_dim])

    trainer.train_sample(x, y)

    U_after, V_after = _inter_factors(system)
    assert np.allclose(U_before, U_after)
    assert np.allclose(V_before, V_after)


def test_learnable_inter_modules_change_after_training_sample():
    cfg = Config512D(
        seed=11,
        train_epochs=1,
        train_samples=1,
        learnable_inter_modules=True,
    )
    system = ModularFieldSystem(cfg)
    trainer = ModularTrainer(system)
    U_before, V_before = _inter_factors(system)
    rng = np.random.default_rng(12)
    x = rng.normal(0.0, 0.55, size=cfg.input_dim)
    y = np.tanh(2.0 * x[:cfg.output_dim])

    trainer.train_sample(x, y)

    U_after, V_after = _inter_factors(system)
    assert np.linalg.norm(U_after - U_before) > 0.0
    assert np.linalg.norm(V_after - V_before) > 0.0
