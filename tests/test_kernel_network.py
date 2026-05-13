"""Tests for the atomic kernel network experiment."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from core.constants import Config512D
from experiments.run_4_kernel_network import AtomicKernel, KernelNetwork, make_extended_dataset


def test_atomic_kernel_interface_shapes():
    cfg = Config512D(process_steps=3)
    kernel = AtomicKernel(0, cfg)
    x = np.ones(cfg.input_dim) * 0.1
    kernel.receive(x)
    kernel.step()

    assert kernel.broadcast().shape == (cfg.output_dim,)
    assert kernel.get_features().shape == (cfg.num_modules + cfg.dim_c,)
    assert kernel.get_spectral_radius() < 1.0


def test_kernel_network_shapes_and_topology():
    cfg = Config512D(process_steps=3)
    network = KernelNetwork(4, cfg)
    x = np.ones(cfg.input_dim) * 0.1
    pred = network.process(x, "tanh", steps=3)

    assert pred.shape == (cfg.output_dim,)
    assert network.get_features().shape == (4 * (cfg.num_modules + cfg.dim_c),)
    assert network.topology.shape == (4, 4)
    assert np.allclose(np.diag(network.topology), 0.0)


def test_kernel_network_train_sample_adds_task_head():
    cfg = Config512D(process_steps=3)
    network = KernelNetwork(4, cfg)
    rng = np.random.default_rng(3)
    X, Y = make_extended_dataset(cfg, "relu", rng, 2)
    loss = network.train_sample(X[0], Y[0], "relu")

    assert loss >= 0.0
    assert "relu" in network.network_readout.readouts
    assert network.resource_usage()["network_readout_params"] == cfg.output_dim * network.feature_dim


def test_batch_evolution_shape():
    cfg = Config512D(process_steps=3)
    network = KernelNetwork(2, cfg)
    rng = np.random.default_rng(14)
    x = rng.normal(0.0, 0.55, size=(5, cfg.input_dim))

    pred, features = network.process_batch(x, "relu", steps=3)

    assert pred.shape == (5, cfg.output_dim)
    assert features.shape == (5, network.feature_dim)


def test_batch_evolution_matches_serial():
    cfg = Config512D(seed=15, process_steps=4)
    rng = np.random.default_rng(16)
    x = rng.normal(0.0, 0.55, size=(3, cfg.input_dim))

    serial = KernelNetwork(2, cfg)
    batch = KernelNetwork(2, cfg)
    serial_preds = []
    for sample in x:
        serial.reset_all()
        serial_preds.append(serial.process(sample, "relu", steps=4))
    serial_preds = np.asarray(serial_preds)

    batch_preds, _ = batch.process_batch(x, "relu", steps=4)

    assert np.allclose(batch_preds, serial_preds, atol=1e-10)


def test_train_batch_uses_batched_dynamics():
    cfg = Config512D(seed=17, process_steps=3)
    network = KernelNetwork(2, cfg)
    rng = np.random.default_rng(18)
    x, y = make_extended_dataset(cfg, "relu", rng, 4)

    loss = network.train_batch(x, y, "relu")

    assert loss >= 0.0
    assert network.timing["batch_process_seconds"] > 0.0
    assert "relu" in network.network_readout.readouts
