"""Smoke tests for the kernel scaling-law experiment."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.constants import Config512D
from experiments.run_scaling_law import (
    TASK_SUITE,
    ScalingExperiment,
    generate_task_data,
    generate_task_function,
)


def test_scaling_task_suite_generates_bounded_vectors():
    cfg = Config512D()
    rng = np.random.default_rng(11)

    for task in TASK_SUITE:
        fn = generate_task_function(task)
        x = rng.normal(0.0, 0.55, size=cfg.input_dim)
        y = fn(x)
        assert y.shape == (cfg.output_dim,)
        assert np.all(np.isfinite(y))


def test_generate_task_data_shapes_are_stable():
    cfg = Config512D()
    rng = np.random.default_rng(12)
    x, y = generate_task_data(cfg, "smooth", 4, rng)

    assert x.shape == (4, cfg.input_dim)
    assert y.shape == (4, cfg.output_dim)


def test_scaling_experiment_tiny_run_completes():
    experiment = ScalingExperiment(
        kernel_counts=[1],
        tasks=["relu"],
        epochs=1,
        samples=2,
        eval_samples=1,
        forgetting_threshold=1.0,
        loss_threshold=10.0,
        seed=13,
        steps=2,
        mode="accuracy",
        batch_size=1,
        eval_every=1,
        architecture="flat",
    )

    results = experiment.run()

    assert results[1]["capacity"] == 1
    assert results[1]["retained_tasks"] == 1
    assert "relu" in results[1]["learned_tasks"]
    assert experiment.scaling_formula["formula"].startswith("C(K) =")
