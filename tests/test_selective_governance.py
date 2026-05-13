"""Tests for Nintendo-efficient selective governance."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from core.constants import Config512D
from core.governance import BinaryGovernance, CompactGovernance
from system.modular_system import ModularFieldSystem
from system.trainer import ModularTrainer


def test_constraint_space_is_module_scoreboard():
    cfg = Config512D()
    system = ModularFieldSystem(cfg)
    system.step()
    module_norms = system.current_module_norms()

    assert cfg.dim == 540
    assert cfg.dim_c == cfg.num_modules == 60
    assert system.state[cfg.dim_f:].shape == (cfg.num_modules,)
    assert np.allclose(system.state[cfg.dim_f:], module_norms)


def test_compact_governance_damps_overactive_group_more():
    cfg = Config512D()
    governance = CompactGovernance(cfg)
    module_norms = np.zeros(cfg.num_modules)
    module_norms[:20] = 1.0
    field_correction = governance.apply(module_norms)

    assert field_correction.shape == (cfg.dim_f,)
    assert np.linalg.norm(field_correction) > 0
    assert np.mean(field_correction) < 0

    perception = np.mean(np.abs(field_correction[:20 * cfg.module_size]))
    reasoning = np.mean(np.abs(field_correction[20 * cfg.module_size:40 * cfg.module_size]))
    assert perception > reasoning


def test_binary_governance_keeps_top_k_per_group():
    cfg = Config512D()
    governance = BinaryGovernance(cfg, k_winners=2)
    module_norms = np.arange(cfg.num_modules, dtype=float)
    field_correction = governance.apply(module_norms)
    module_damping = -field_correction.reshape(cfg.num_modules, cfg.module_size)[:, 0]

    for group_start, group_end in [(0, 20), (20, 40), (40, 50), (50, 60)]:
        assert np.count_nonzero(module_damping[group_start:group_end] == 0.0) == 2


def test_fixed_governance_does_not_update_self_inhibition():
    cfg = Config512D(seed=21, learnable_constraint_feedback=False)
    system = ModularFieldSystem(cfg)
    trainer = ModularTrainer(system)
    before = system.governance.self_inhibition.copy()
    rng = np.random.default_rng(22)
    x = rng.normal(0.0, 0.55, size=cfg.input_dim)
    y = np.tanh(2.0 * x[:cfg.output_dim])

    trainer.train_sample(x, y)

    assert np.allclose(system.governance.self_inhibition, before)


def test_learnable_governance_updates_self_inhibition():
    cfg = Config512D(seed=21, learnable_constraint_feedback=True)
    system = ModularFieldSystem(cfg)
    trainer = ModularTrainer(system)
    before = system.governance.self_inhibition.copy()
    rng = np.random.default_rng(22)
    x = rng.normal(0.0, 0.55, size=cfg.input_dim)
    y = np.ones(cfg.output_dim) * 5.0

    trainer.train_sample(x, y)

    assert np.linalg.norm(system.governance.self_inhibition - before) > 0.0


def test_projection_can_be_disabled_while_boundary_still_triggers():
    cfg = Config512D(enable_projection=False, boundary_theta=0.01, boundary_delta=0.0)
    system = ModularFieldSystem(cfg)
    system.step()
    summary = system.trace.summary()

    assert summary["triggers"] > 0
    assert summary["projections"] == 0
