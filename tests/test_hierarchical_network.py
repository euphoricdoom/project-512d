"""Tests for piecewise hierarchical kernel networks."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.constants import Config512D
from experiments.hierarchical_kernel_network import HierarchicalKernelNetwork
from experiments.run_4_kernel_network import KernelNetwork, make_extended_dataset


def test_hierarchical_groups_are_four_way():
    cfg = Config512D(process_steps=2)
    network = HierarchicalKernelNetwork(16, cfg)

    assert len(network.groups) == 4
    assert all(len(group) == 4 for group in network.groups)
    assert network.group_assignments()[0] == [0, 1, 2, 3]


def test_hierarchical_topology_is_sparser_than_flat():
    cfg = Config512D(process_steps=2)
    hierarchical = HierarchicalKernelNetwork(16, cfg)
    flat = KernelNetwork(16, cfg)

    assert hierarchical.topology.shape == (16, 16)
    assert hierarchical.communication_edges() < int(np.count_nonzero(flat.topology))
    assert np.allclose(np.diag(hierarchical.topology), 0.0)


def test_hierarchical_route_prefers_task_group():
    cfg = Config512D(process_steps=2)
    network = HierarchicalKernelNetwork(16, cfg)
    x = np.ones(cfg.input_dim) * 0.1

    weights = network.route(x, "smooth")
    group_mass = [
        float(np.sum(weights[group]))
        for group in network.groups
    ]

    assert int(np.argmax(group_mass)) == network.task_group("smooth")
    assert np.isclose(np.sum(weights), 1.0)


def test_hierarchical_batch_training_runs():
    cfg = Config512D(seed=21, process_steps=2)
    network = HierarchicalKernelNetwork(8, cfg)
    rng = np.random.default_rng(22)
    x, y = make_extended_dataset(cfg, "relu", rng, 4)

    loss = network.train_batch(x, y, "relu")
    pred = network.predict(x[0], "relu")

    assert loss >= 0.0
    assert pred.shape == (cfg.output_dim,)
    assert network.timing["batch_process_seconds"] > 0.0
