"""Tests for the strict model benchmark runner."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from experiments.run_strict_sequence_benchmarks import (
    SCHEMA_VERSION,
    StrictTaskHeadBank,
    _flatten_targets,
    _reshape_predictions,
    run,
    validate_fresh_payload,
)
from experiments.strict_sequence_benchmarks import generate_full_copy_task


def _args(tmp_path: Path) -> Namespace:
    return Namespace(
        kernels=1,
        temporal_adapter="none",
        samples=24,
        epochs=1,
        steps=1,
        batch_size=8,
        projection_dim=16,
        stable_base_lr=0.04,
        feature_mode="raw_state",
        seed=123,
        gate_weights=None,
        gate_decay_min=0.0,
        gate_decay_max=0.99,
        learned_threshold=0.98,
        max_forgetting=0.00001,
        baseline_margin=0.05,
        raw_dir=str(tmp_path / "raw"),
        output=str(tmp_path / "strict.json"),
    )


def test_flatten_and_reshape_round_trip():
    task = generate_full_copy_task(n_samples=3, seq_length=4, delay_length=2, vocab_size=4)
    flat, shape = _flatten_targets(task)
    restored = _reshape_predictions(flat, shape)

    assert flat.shape == (3, np.prod(shape))
    assert restored.shape == task.targets.shape
    assert np.array_equal(restored, task.targets)


def test_strict_task_head_bank_isolates_output_widths():
    bank = StrictTaskHeadBank(feature_dim=5, projection_dim=4, base_lr=0.01, seed=1)
    copy_head = bank.head_for("copy_task", output_dim=12)
    parity_head = bank.head_for("parity_task", output_dim=3)

    assert copy_head.output_dim == 12
    assert parity_head.output_dim == 3
    assert copy_head is bank.head_for("copy_task", output_dim=12)
    with pytest.raises(ValueError):
        bank.head_for("copy_task", output_dim=13)


def test_validate_fresh_payload_rejects_stale_settings(tmp_path):
    args = _args(tmp_path)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "settings": {
            "kernels": args.kernels,
            "temporal_adapter": args.temporal_adapter,
            "samples": args.samples,
            "epochs": args.epochs,
            "seed": args.seed,
        },
    }
    validate_fresh_payload(payload, args)
    payload["settings"]["samples"] += 1
    with pytest.raises(ValueError):
        validate_fresh_payload(payload, args)


def test_strict_runner_writes_raw_predictions_and_summary(tmp_path):
    args = _args(tmp_path)
    payload = run(args)

    assert payload["schema_version"] == SCHEMA_VERSION
    assert Path(args.output).exists()
    assert set(payload["tasks"]) == {"copy_task", "parity_task", "adding_task"}
    assert "max_forgetting_drop" in payload["summary"]
    for task in payload["tasks"].values():
        raw_path = Path(task["raw_predictions"])
        assert raw_path.exists()
        raw = np.load(raw_path)
        assert set(raw.files) == {"predictions", "targets", "mask", "inputs"}
        assert raw["predictions"].shape == raw["targets"].shape
        assert "baselines" in task
        assert "best_baseline_score" in task["evaluation"]
