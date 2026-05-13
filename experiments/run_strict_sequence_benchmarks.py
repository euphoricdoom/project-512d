"""Strict model runner for mask-aware sequence benchmarks.

This runner evaluates actual frozen feature producers plus task-owned heads
against the stricter measurement layer in ``strict_sequence_benchmarks``. It
also saves raw predictions so summaries can be audited later.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.constants import Config512D
from experiments.run_classic_benchmarks import build_network, process_sequence_batch
from experiments.strict_sequence_benchmarks import (
    BASELINES,
    StrictTask,
    evaluate_predictions,
    forgetting_drop,
    generate_strict_suite,
)
from system.task_specific_projection import TaskSpecificProjection
from system.sleep_consolidation import sleep_consolidation


SCHEMA_VERSION = 1


def _fit_sequence_width(inputs: np.ndarray, width: int) -> np.ndarray:
    """Pad/truncate sequence feature width."""
    if inputs.shape[2] == width:
        return inputs.astype(float)
    out = np.zeros((inputs.shape[0], inputs.shape[1], width), dtype=float)
    n = min(width, inputs.shape[2])
    out[:, :, :n] = inputs[:, :, :n]
    return out


def _flatten_targets(task: StrictTask) -> tuple[np.ndarray, tuple[int, ...]]:
    """Flatten task targets to train a final-state task head."""
    shape = task.targets.shape[1:]
    return task.targets.reshape(task.targets.shape[0], -1), shape


def _reshape_predictions(flat: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Restore flat predictions to strict task target shape."""
    return flat.reshape((flat.shape[0],) + shape)


def _split_task(task: StrictTask, train_fraction: float = 0.8) -> tuple[StrictTask, StrictTask]:
    """Split a StrictTask into train/test tasks."""
    split = int(len(task.inputs) * train_fraction)
    train = StrictTask(
        task.task_id,
        task.inputs[:split],
        task.targets[:split],
        task.mask[:split],
        dict(task.metadata),
    )
    test = StrictTask(
        task.task_id,
        task.inputs[split:],
        task.targets[split:],
        task.mask[split:],
        dict(task.metadata),
    )
    return train, test


class StrictTaskHeadBank:
    """Task-owned projection/readout heads with task-specific output widths."""

    def __init__(
        self,
        feature_dim: int,
        projection_dim: int,
        base_lr: float,
        seed: int,
    ) -> None:
        self.feature_dim = int(feature_dim)
        self.projection_dim = int(projection_dim)
        self.base_lr = float(base_lr)
        self.rng = np.random.default_rng(seed)
        self.heads: dict[str, TaskSpecificProjection] = {}

    def head_for(self, task_id: str, output_dim: int) -> TaskSpecificProjection:
        """Return/create isolated head for one task/output width."""
        if task_id not in self.heads:
            self.heads[task_id] = TaskSpecificProjection(
                num_kernels=1,
                base_feature_dim=self.feature_dim,
                projection_dim=self.projection_dim,
                output_dim=output_dim,
                base_lr=self.base_lr,
                rng=self.rng,
            )
        head = self.heads[task_id]
        if head.output_dim != output_dim:
            raise ValueError(f"task {task_id} head output width changed")
        return head


def extract_features(network, inputs: np.ndarray, task_id: str) -> np.ndarray:
    """Extract final frozen sequence features from the benchmark network."""
    _, features = process_sequence_batch(network, inputs, task_id)
    return np.asarray(features, dtype=float)


def train_task_head(
    network,
    heads: StrictTaskHeadBank,
    task: StrictTask,
    epochs: int,
    batch_size: int,
    rng: np.random.Generator,
) -> dict:
    """Train an isolated task head on strict task targets."""
    inputs = _fit_sequence_width(task.inputs, network.cfg.input_dim)
    targets, _ = _flatten_targets(task)
    head = heads.head_for(task.task_id, targets.shape[1])
    losses = []
    start = time.time()
    for _ in range(epochs):
        order = rng.permutation(len(inputs))
        batch_losses = []
        for start_idx in range(0, len(inputs), batch_size):
            idx = order[start_idx : start_idx + batch_size]
            features = extract_features(network, inputs[idx], task.task_id)
            loss = head.update_batch(features, targets[idx], task_id=task.task_id)
            batch_losses.append(loss)
        losses.append(float(np.mean(batch_losses)))
    return {
        "final_loss": losses[-1],
        "epoch_losses": losses,
        "training_time": time.time() - start,
    }


def predict_task(
    network,
    heads: StrictTaskHeadBank,
    task: StrictTask,
) -> np.ndarray:
    """Predict strict task targets with the task-owned head."""
    inputs = _fit_sequence_width(task.inputs, network.cfg.input_dim)
    flat_targets, target_shape = _flatten_targets(task)
    head = heads.head_for(task.task_id, flat_targets.shape[1])
    features = extract_features(network, inputs, task.task_id)
    flat_pred = head.predict_batch(features, task.task_id)
    return _reshape_predictions(flat_pred, target_shape).astype(np.float32)


def baseline_report_for_task(task: StrictTask) -> dict:
    """Evaluate trivial baselines for one task."""
    return {
        name: evaluate_predictions(fn(task), task)
        for name, fn in BASELINES.items()
    }


def validate_fresh_payload(payload: dict, args: argparse.Namespace) -> None:
    """Basic stale/config guard for strict result payloads."""
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("schema_version mismatch")
    settings = payload.get("settings", {})
    required = {
        "kernels": args.kernels,
        "temporal_adapter": args.temporal_adapter,
        "samples": args.samples,
        "epochs": args.epochs,
        "seed": args.seed,
    }
    for key, expected in required.items():
        if settings.get(key) != expected:
            raise ValueError(f"settings mismatch for {key}: {settings.get(key)} != {expected}")


def run(args: argparse.Namespace) -> dict:
    """Run strict sequential benchmark and save raw predictions."""
    cfg = Config512D(
        seed=args.seed,
        train_epochs=args.epochs,
        train_samples=args.samples,
        process_steps=args.steps,
    )
    tasks = generate_strict_suite(samples=args.samples, seed=args.seed)
    split_tasks = {tid: _split_task(task) for tid, task in tasks.items()}
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    network = build_network(
        args.kernels,
        cfg,
        use_projection=True,
        projection_dim=args.projection_dim,
        stable_base_lr=args.stable_base_lr,
        feature_mode=args.feature_mode,
        temporal_adapter=args.temporal_adapter,
        gate_weights=args.gate_weights,
        gate_decay_min=args.gate_decay_min,
        gate_decay_max=args.gate_decay_max,
    )
    heads = StrictTaskHeadBank(
        feature_dim=network.feature_dim,
        projection_dim=args.projection_dim,
        base_lr=args.stable_base_lr,
        seed=args.seed + 9000,
    )
    rng = np.random.default_rng(args.seed + 4100)
    payload = {
        "experiment": "strict_sequence_benchmarks",
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "kernels": args.kernels,
            "temporal_adapter": args.temporal_adapter,
            "samples": args.samples,
            "epochs": args.epochs,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "projection_dim": args.projection_dim,
            "stable_base_lr": args.stable_base_lr,
            "feature_mode": args.feature_mode,
            "seed": args.seed,
            "learned_threshold": args.learned_threshold,
            "max_forgetting": args.max_forgetting,
            "baseline_margin": args.baseline_margin,
            "sleep_cycles": getattr(args, "sleep_cycles", 0),
            "sleep_lr": getattr(args, "sleep_lr", 0.001),
        },
        "tasks": {},
        "task_order": list(split_tasks),
    }

    best_scores: dict[str, float] = {}
    for task_idx, task_id in enumerate(payload["task_order"]):
        print(f"[{task_idx + 1}/{len(payload['task_order'])}] strict train {task_id}")
        train_task, test_task = split_tasks[task_id]
        training = train_task_head(network, heads, train_task, args.epochs, args.batch_size, rng)
        train_inputs = _fit_sequence_width(train_task.inputs, network.cfg.input_dim)
        train_targets, _ = _flatten_targets(train_task)
        head = heads.head_for(task_id, train_targets.shape[1])
        sleep_metrics = None
        sleep_cycles = getattr(args, "sleep_cycles", 0)
        sleep_lr = getattr(args, "sleep_lr", 0.001)
        if sleep_cycles > 0:
            sleep_metrics = sleep_consolidation(
                network,
                task_id,
                train_inputs,
                train_targets,
                num_cycles=sleep_cycles,
                sleep_lr=sleep_lr,
                batch_size=args.batch_size,
                verbose=True,
                readout=head,
                feature_extractor=lambda net, batch, tid: (
                    np.zeros((len(batch), train_targets.shape[1]), dtype=float),
                    extract_features(net, batch, tid),
                ),
                seed=args.seed + 9100 + task_idx,
            )
        predictions = predict_task(network, heads, test_task)
        metrics = evaluate_predictions(predictions, test_task)
        baselines = baseline_report_for_task(test_task)
        best_baseline = max(v["primary_score"] for v in baselines.values())
        metrics["best_baseline_score"] = float(best_baseline)
        metrics["beats_baseline_margin"] = bool(
            metrics["primary_score"] >= best_baseline + args.baseline_margin
        )
        best_scores[task_id] = metrics["primary_score"]
        adapter_label = "no_adapter" if args.temporal_adapter == "none" else args.temporal_adapter
        raw_path = raw_dir / f"{adapter_label}_{args.kernels}k_{task_id}_predictions.npz"
        np.savez_compressed(
            raw_path,
            predictions=predictions,
            targets=test_task.targets,
            mask=test_task.mask,
            inputs=test_task.inputs,
        )
        payload["tasks"][task_id] = {
            "metadata": test_task.metadata,
            "training": training,
            "sleep_metrics": sleep_metrics,
            "evaluation": metrics,
            "baselines": baselines,
            "raw_predictions": raw_path.as_posix(),
            "forgetting_measurements": [],
        }
        print(
            f"  score={metrics['primary_score']:.3f}, "
            f"best_baseline={best_baseline:.3f}, raw={raw_path}"
        )

        for previous_id in payload["task_order"][:task_idx]:
            _, previous_test = split_tasks[previous_id]
            current_predictions = predict_task(network, heads, previous_test)
            current_metrics = evaluate_predictions(current_predictions, previous_test)
            drop = forgetting_drop(
                best_scores[previous_id],
                current_metrics["primary_score"],
                args.learned_threshold,
            )
            payload["tasks"][previous_id]["forgetting_measurements"].append(
                {
                    "measured_after_task": task_id,
                    "primary_score": current_metrics["primary_score"],
                    "forgetting_drop": drop,
                }
            )
            print(f"  previous {previous_id}: score={current_metrics['primary_score']:.3f}, drop={drop:.6f}")

    all_scores = [payload["tasks"][tid]["evaluation"]["primary_score"] for tid in payload["task_order"]]
    all_drops = [
        item["forgetting_drop"]
        for tid in payload["task_order"]
        for item in payload["tasks"][tid]["forgetting_measurements"]
    ]
    payload["summary"] = {
        "mean_primary_score": float(np.mean(all_scores)),
        "min_primary_score": float(np.min(all_scores)),
        "tasks_above_learned_threshold": int(np.sum(np.asarray(all_scores) >= args.learned_threshold)),
        "max_forgetting_drop": float(max(all_drops) if all_drops else 0.0),
        "forgetting_contract_pass": bool((max(all_drops) if all_drops else 0.0) <= args.max_forgetting),
        "all_tasks_beat_baselines": bool(
            all(payload["tasks"][tid]["evaluation"]["beats_baseline_margin"] for tid in payload["task_order"])
        ),
    }
    validate_fresh_payload(payload, args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved strict benchmark results to {output}")
    print(
        f"mean={payload['summary']['mean_primary_score']:.3f}, "
        f"max_forgetting={payload['summary']['max_forgetting_drop']:.6f}, "
        f"beats_baselines={payload['summary']['all_tasks_beat_baselines']}"
    )
    return payload


def parse_args() -> argparse.Namespace:
    """Parse strict runner CLI args."""
    parser = argparse.ArgumentParser(description="Run strict sequence benchmarks")
    parser.add_argument("--kernels", type=int, default=1)
    parser.add_argument("--adapter", dest="temporal_adapter", choices=["none", "helix", "learned-helix"])
    parser.add_argument("--temporal-adapter", choices=["none", "helix", "learned-helix"], default="helix")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--stable-base-lr", type=float, default=0.04)
    parser.add_argument("--feature-mode", choices=["scoreboard", "raw_state"], default="raw_state")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gate-weights", type=str, default=None)
    parser.add_argument("--gate-decay-min", type=float, default=0.0)
    parser.add_argument("--gate-decay-max", type=float, default=0.99)
    parser.add_argument("--learned-threshold", type=float, default=0.98)
    parser.add_argument("--max-forgetting", type=float, default=0.00001)
    parser.add_argument("--baseline-margin", type=float, default=0.05)
    parser.add_argument("--sleep-cycles", type=int, default=0)
    parser.add_argument("--sleep-lr", type=float, default=0.001)
    parser.add_argument("--raw-dir", type=str, default="outputs/raw_predictions")
    parser.add_argument("--output", type=str, default="outputs/strict_sequence_benchmarks.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
