"""Tests for strict mask-aware sequence benchmark metrics."""

from __future__ import annotations

import numpy as np

from experiments.strict_sequence_benchmarks import (
    adding_metrics,
    copy_metrics,
    evaluate_predictions,
    forgetting_drop,
    generate_full_copy_task,
    generate_strict_adding_task,
    generate_strict_parity_task,
    last_input_baseline,
    mean_baseline,
    parity_metrics,
    run_baseline_report,
    zero_baseline,
)


def test_full_copy_zero_predictor_does_not_pass():
    """Regression for old sparse-MSE metric where zero got 100% copy accuracy."""
    task = generate_full_copy_task(
        n_samples=256,
        seq_length=8,
        delay_length=8,
        vocab_size=8,
        seed=123,
    )
    metrics = copy_metrics(zero_baseline(task), task)

    assert task.inputs.shape[1] == 8 + 8 + 1 + 8
    assert task.mask.sum() == 256 * 8
    assert metrics["token_accuracy"] < 0.25
    assert metrics["exact_sequence_accuracy"] == 0.0


def test_full_copy_oracle_scores_perfectly_on_replay_mask():
    task = generate_full_copy_task(n_samples=32, seq_length=5, delay_length=3, vocab_size=5)
    metrics = evaluate_predictions(task.targets.copy(), task)

    assert metrics["token_accuracy"] == 1.0
    assert metrics["exact_sequence_accuracy"] == 1.0
    assert metrics["primary_score"] == 1.0


def test_running_parity_metrics_are_bit_and_sequence_based():
    task = generate_strict_parity_task(n_samples=64, seq_length=32, seed=4)
    oracle = task.targets.copy()
    zero = zero_baseline(task)

    assert parity_metrics(oracle, task)["bit_accuracy"] == 1.0
    zero_metrics = parity_metrics(zero, task)
    assert 0.25 < zero_metrics["bit_accuracy"] < 0.75
    assert zero_metrics["exact_sequence_accuracy"] < 0.25


def test_adding_metrics_score_scalar_regression_not_padded_mse():
    task = generate_strict_adding_task(n_samples=64, seq_length=20, seed=5)
    oracle = task.targets.copy()
    mean_pred = mean_baseline(task)
    zero = zero_baseline(task)

    assert adding_metrics(oracle, task)["tolerance_accuracy"] == 1.0
    assert adding_metrics(oracle, task)["mae"] == 0.0
    assert adding_metrics(mean_pred, task)["mae"] < adding_metrics(zero, task)["mae"]


def test_last_input_baseline_is_not_accidentally_oracle():
    copy_task = generate_full_copy_task(n_samples=128, seq_length=6, delay_length=4, vocab_size=6, seed=7)
    adding_task = generate_strict_adding_task(n_samples=128, seq_length=20, seed=8)

    assert copy_metrics(last_input_baseline(copy_task), copy_task)["token_accuracy"] < 0.5
    assert adding_metrics(last_input_baseline(adding_task), adding_task)["tolerance_accuracy"] < 0.5


def test_forgetting_drop_only_counts_learned_tasks_and_positive_drop():
    assert forgetting_drop(0.97, 0.20, learned_threshold=0.98) == 0.0
    assert forgetting_drop(0.99, 0.995, learned_threshold=0.98) == 0.0
    assert np.isclose(forgetting_drop(0.99, 0.97, learned_threshold=0.98), 0.02)


def test_baseline_report_has_nontrivial_scores_for_all_tasks():
    report = run_baseline_report(samples=64, seed=9)

    assert report["schema_version"] == 1
    assert set(report["tasks"]) == {"copy_task", "parity_task", "adding_task"}
    for task_payload in report["tasks"].values():
        assert set(task_payload["baselines"]) == {"zero", "mean", "last_input"}
        for metrics in task_payload["baselines"].values():
            assert 0.0 <= metrics["primary_score"] <= 1.0
