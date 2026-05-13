"""Strict, mask-aware sequence benchmark utilities.

The older classic runner is useful as an architecture smoke test, but its
metrics can be fooled by sparse padded targets. This module defines a stricter
measurement layer for real temporal competence:

- copy is scored only on replay tokens;
- parity is scored as running bit accuracy;
- adding is scored as scalar final-output regression;
- forgetting is measured as positive backward-transfer drop.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass
class StrictTask:
    """Container for one strict benchmark task."""

    task_id: str
    inputs: np.ndarray
    targets: np.ndarray
    mask: np.ndarray
    metadata: dict


def generate_full_copy_task(
    n_samples: int = 512,
    seq_length: int = 8,
    delay_length: int = 8,
    vocab_size: int = 8,
    seed: int = 42,
) -> StrictTask:
    """Generate full copy task with replay mask.

    Shape: ``seq + delay + trigger + seq``. Accuracy is scored only on replay
    positions, so blank/pad positions cannot dominate the metric.
    """
    rng = np.random.default_rng(seed)
    total_steps = seq_length + delay_length + 1 + seq_length
    trigger_channel = vocab_size
    blank_channel = vocab_size + 1
    width = vocab_size + 2
    inputs = np.zeros((n_samples, total_steps, width), dtype=np.float32)
    targets = np.zeros_like(inputs)
    inputs[:, :, blank_channel] = 1.0
    targets[:, :, blank_channel] = 1.0

    symbols = rng.integers(0, vocab_size, size=(n_samples, seq_length))
    rows = np.arange(n_samples)[:, None]
    src_t = np.arange(seq_length)[None, :]
    inputs[rows, src_t, symbols] = 1.0
    inputs[rows, src_t, blank_channel] = 0.0
    trigger_t = seq_length + delay_length
    inputs[:, trigger_t, blank_channel] = 0.0
    inputs[:, trigger_t, trigger_channel] = 1.0

    replay_t = trigger_t + 1 + np.arange(seq_length)
    targets[:, replay_t, blank_channel] = 0.0
    targets[rows, replay_t[None, :], symbols] = 1.0
    mask = np.zeros((n_samples, total_steps), dtype=bool)
    mask[:, replay_t] = True
    return StrictTask(
        "copy_task",
        inputs,
        targets,
        mask,
        {
            "seq_length": seq_length,
            "delay_length": delay_length,
            "vocab_size": vocab_size,
            "target_type": "masked_tokens",
        },
    )


def generate_strict_parity_task(
    n_samples: int = 512,
    seq_length: int = 32,
    seed: int = 43,
) -> StrictTask:
    """Generate running parity task scored at every timestep."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(n_samples, seq_length), dtype=np.int8)
    parity = np.bitwise_xor.accumulate(bits, axis=1).astype(np.float32)
    inputs = bits[..., None].astype(np.float32)
    targets = parity[..., None]
    mask = np.ones((n_samples, seq_length), dtype=bool)
    return StrictTask(
        "parity_task",
        inputs,
        targets,
        mask,
        {"seq_length": seq_length, "target_type": "running_bits"},
    )


def generate_strict_adding_task(
    n_samples: int = 512,
    seq_length: int = 20,
    seed: int = 44,
) -> StrictTask:
    """Generate classic adding task with scalar final target."""
    if seq_length < 2:
        raise ValueError("seq_length must be at least 2")
    rng = np.random.default_rng(seed)
    values = rng.random((n_samples, seq_length)).astype(np.float32)
    markers = np.zeros((n_samples, seq_length), dtype=np.float32)
    positions = np.empty((n_samples, 2), dtype=np.int64)
    for i in range(n_samples):
        positions[i] = rng.choice(seq_length, size=2, replace=False)
    rows = np.arange(n_samples)[:, None]
    markers[rows, positions] = 1.0
    targets = (values * markers).sum(axis=1, keepdims=True).astype(np.float32)
    inputs = np.stack([values, markers], axis=-1)
    mask = np.ones((n_samples, 1), dtype=bool)
    return StrictTask(
        "adding_task",
        inputs,
        targets,
        mask,
        {"seq_length": seq_length, "target_type": "final_scalar"},
    )


def copy_metrics(predictions: np.ndarray, task: StrictTask) -> dict:
    """Compute masked token and exact-sequence accuracy for copy."""
    pred = np.asarray(predictions)
    target = task.targets
    mask = task.mask
    vocab_size = int(task.metadata["vocab_size"])
    if pred.shape != target.shape:
        raise ValueError(f"copy predictions must have shape {target.shape}, got {pred.shape}")
    pred_tokens = np.argmax(pred[:, :, :vocab_size], axis=-1)
    true_tokens = np.argmax(target[:, :, :vocab_size], axis=-1)
    token_correct = pred_tokens[mask] == true_tokens[mask]
    per_sample = np.all((pred_tokens == true_tokens) | ~mask, axis=1)
    return {
        "token_accuracy": float(np.mean(token_correct)),
        "exact_sequence_accuracy": float(np.mean(per_sample)),
        "primary_score": float(np.mean(token_correct)),
    }


def parity_metrics(predictions: np.ndarray, task: StrictTask) -> dict:
    """Compute running parity bit accuracy."""
    pred = np.asarray(predictions)
    if pred.shape != task.targets.shape:
        raise ValueError(f"parity predictions must have shape {task.targets.shape}, got {pred.shape}")
    pred_bits = pred[..., 0] >= 0.5
    true_bits = task.targets[..., 0] >= 0.5
    correct = pred_bits[task.mask] == true_bits[task.mask]
    exact = np.all(pred_bits == true_bits, axis=1)
    return {
        "bit_accuracy": float(np.mean(correct)),
        "exact_sequence_accuracy": float(np.mean(exact)),
        "primary_score": float(np.mean(correct)),
    }


def adding_metrics(predictions: np.ndarray, task: StrictTask, tolerance: float = 0.05) -> dict:
    """Compute scalar adding metrics."""
    pred = np.asarray(predictions)
    if pred.shape != task.targets.shape:
        raise ValueError(f"adding predictions must have shape {task.targets.shape}, got {pred.shape}")
    error = pred[:, 0] - task.targets[:, 0]
    mse = float(np.mean(error**2))
    mae = float(np.mean(np.abs(error)))
    var = float(np.var(task.targets[:, 0]))
    r2 = 1.0 - mse / var if var > 1e-12 else 0.0
    tolerance_accuracy = float(np.mean(np.abs(error) <= tolerance))
    return {
        "mae": mae,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2),
        "tolerance_accuracy": tolerance_accuracy,
        "primary_score": tolerance_accuracy,
    }


METRICS: dict[str, Callable[[np.ndarray, StrictTask], dict]] = {
    "copy_task": copy_metrics,
    "parity_task": parity_metrics,
    "adding_task": adding_metrics,
}


def evaluate_predictions(predictions: np.ndarray, task: StrictTask) -> dict:
    """Evaluate predictions for a strict task."""
    metrics = METRICS[task.task_id](predictions, task)
    metrics["task_id"] = task.task_id
    return metrics


def zero_baseline(task: StrictTask) -> np.ndarray:
    """Return all-zero predictions with the correct task target shape."""
    return np.zeros_like(task.targets, dtype=np.float32)


def mean_baseline(task: StrictTask) -> np.ndarray:
    """Return train-set mean target prediction with correct target shape."""
    mean = np.mean(task.targets, axis=0, keepdims=True)
    return np.repeat(mean, task.targets.shape[0], axis=0).astype(np.float32)


def last_input_baseline(task: StrictTask) -> np.ndarray:
    """Simple last-input baseline for strict tasks."""
    if task.task_id == "adding_task":
        return task.inputs[:, -1, 0:1].astype(np.float32)
    if task.task_id == "parity_task":
        return task.inputs.copy().astype(np.float32)
    if task.task_id == "copy_task":
        pred = np.zeros_like(task.targets)
        blank = int(task.metadata["vocab_size"]) + 1
        pred[:, :, blank] = 1.0
        last_symbol = np.argmax(task.inputs[:, : task.metadata["seq_length"], : task.metadata["vocab_size"]], axis=-1)[:, -1]
        replay_positions = np.where(task.mask[0])[0]
        pred[:, replay_positions, :] = 0.0
        pred[
            np.arange(task.targets.shape[0])[:, None],
            replay_positions[None, :],
            last_symbol[:, None],
        ] = 1.0
        return pred.astype(np.float32)
    raise ValueError(task.task_id)


BASELINES: dict[str, Callable[[StrictTask], np.ndarray]] = {
    "zero": zero_baseline,
    "mean": mean_baseline,
    "last_input": last_input_baseline,
}


def forgetting_drop(best_previous_score: float, current_score: float, learned_threshold: float) -> float:
    """Backward-transfer drop for tasks that were previously learned."""
    if best_previous_score < learned_threshold:
        return 0.0
    return float(max(0.0, best_previous_score - current_score))


def generate_strict_suite(samples: int = 512, seed: int = 42) -> dict[str, StrictTask]:
    """Generate all strict classic tasks."""
    return {
        "copy_task": generate_full_copy_task(n_samples=samples, seed=seed),
        "parity_task": generate_strict_parity_task(n_samples=samples, seed=seed + 1),
        "adding_task": generate_strict_adding_task(n_samples=samples, seed=seed + 2),
    }


def run_baseline_report(samples: int = 512, seed: int = 42) -> dict:
    """Evaluate trivial baselines under strict metrics."""
    tasks = generate_strict_suite(samples=samples, seed=seed)
    results = {
        "experiment": "strict_measurement_baselines",
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "samples": samples,
        "tasks": {},
    }
    for task_id, task in tasks.items():
        task_results = {}
        for name, baseline in BASELINES.items():
            task_results[name] = evaluate_predictions(baseline(task), task)
        results["tasks"][task_id] = {
            "metadata": task.metadata,
            "baselines": task_results,
        }
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict measurement baselines")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/strict_measurement_baselines.json",
    )
    return parser.parse_args()


def main() -> dict:
    """CLI entry point."""
    args = parse_args()
    report = run_baseline_report(samples=args.samples, seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved strict measurement baseline report to {output}")
    for task_id, payload in report["tasks"].items():
        print(task_id)
        for name, metrics in payload["baselines"].items():
            print(f"  {name}: primary={metrics['primary_score']:.3f}")
    return report


if __name__ == "__main__":
    main()
