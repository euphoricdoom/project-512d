"""Tests for task-specific readout heads."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from core.constants import Config512D
from core.readout import module_features
from system.modular_system import ModularFieldSystem
from system.multi_readout import MultiTaskReadout
from system.trainer import ModularTrainer


def test_multi_readout_updates_only_current_task():
    rng = np.random.default_rng(1)
    readout = MultiTaskReadout(output_dim=3, feature_dim=5, rng=rng)
    features = rng.normal(size=5)
    error = rng.normal(size=3)
    readout.set_task("tanh")
    readout.set_task("smooth")
    smooth_before = readout.readouts["smooth"].copy()
    tanh_before = readout.readouts["tanh"].copy()

    readout.update(features, error, lr=0.1, task_id="tanh")

    assert not np.allclose(readout.readouts["tanh"], tanh_before)
    assert np.allclose(readout.readouts["smooth"], smooth_before)


def test_modular_trainer_sets_task_for_multi_readout():
    cfg = Config512D(seed=5, train_epochs=1, train_samples=1)
    system = ModularFieldSystem(cfg)
    feature_dim = cfg.num_modules + cfg.dim_c
    system.readout = MultiTaskReadout(cfg.output_dim, feature_dim, system.rng)
    trainer = ModularTrainer(system)
    rng = np.random.default_rng(6)
    x = rng.normal(0.0, 0.55, size=cfg.input_dim)
    y = np.tanh(2.0 * x[:cfg.output_dim])

    trainer.set_task("tanh")
    trainer.train_sample(x, y)

    assert system.readout.current_task == "tanh"
    assert set(system.readout.readouts.keys()) == {"tanh"}


def test_system_predict_uses_multi_readout_current_task():
    cfg = Config512D(seed=7)
    system = ModularFieldSystem(cfg)
    feature_dim = cfg.num_modules + cfg.dim_c
    system.readout = MultiTaskReadout(cfg.output_dim, feature_dim, system.rng)
    system.readout.set_task("a")
    pred_a = system.predict()
    system.readout.set_task("b")
    pred_b = system.predict()

    assert pred_a.shape == (cfg.output_dim,)
    assert pred_b.shape == (cfg.output_dim,)
    assert not np.allclose(pred_a, pred_b)
