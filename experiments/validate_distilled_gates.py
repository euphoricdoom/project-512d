"""Validate distilled learned gates against Phase 2 criteria."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.constants import Config512D
from experiments.run_classic_benchmarks import CLASSIC_TASKS, generate_classic_task_data
from system.distillation_trainer import DistillationTrainer
from system.learned_gates import LearnedGateNetwork


def _max_forgetting(task: dict) -> float:
    values = []
    for item in task.get("forgetting_measurements", []):
        values.append(abs(item.get("forgetting", item.get("accuracy_forgetting", 0.0))))
        values.append(abs(item.get("loss_drift", 0.0)))
    return max(values) if values else float(task.get("forgetting", 0.0))


def load_benchmark_results(path: str) -> dict:
    """Load learned gate benchmark results."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def measure_dc_correlation(
    gate_network: LearnedGateNetwork,
    task_id: str,
    inputs: np.ndarray,
    input_width: int,
    feature_dim: int,
    n_samples: int = 32,
) -> float:
    """Measure student/teacher DC trajectory correlation on deterministic features."""
    trainer = DistillationTrainer(
        gate_network,
        input_width=input_width,
        feature_dim=feature_dim,
        k_support=min(10, n_samples),
    )
    batch = inputs[:n_samples]
    if batch.shape[2] != input_width:
        padded = np.zeros((batch.shape[0], batch.shape[1], input_width), dtype=np.float32)
        n = min(batch.shape[2], input_width)
        padded[:, :, :n] = batch[:, :, :n]
        batch = padded
    features = trainer.make_feature_stream(batch)
    teacher = trainer.extract_teacher_trajectory(task_id, batch, features)
    support = torch.as_tensor(batch[: min(10, len(batch))], dtype=torch.float32)
    with torch.no_grad():
        config = gate_network.configure_for_task(support)
        student = trainer.extract_student_trajectory(
            config,
            torch.as_tensor(batch, dtype=torch.float32),
            torch.as_tensor(features, dtype=torch.float32),
        )
    s = student.detach().numpy().reshape(-1)
    t = teacher.reshape(-1)
    if np.std(s) < 1e-12 or np.std(t) < 1e-12:
        return 0.0
    return float(np.corrcoef(s, t)[0, 1])


def validate(args: argparse.Namespace) -> dict:
    """Run validation and return report payload."""
    results = load_benchmark_results(args.results)
    gate_net = LearnedGateNetwork(
        input_dim=args.input_width,
        task_embedding_dim=128,
        num_dc_channels=args.feature_dim,
        feature_dim=args.feature_dim,
        decay_min=args.gate_decay_min,
        decay_max=args.gate_decay_max,
    )
    gate_net.load(args.gate_weights)
    report = {
        "accuracy_pass": True,
        "forgetting_pass": True,
        "correlation_pass": True,
        "tasks": [],
        "correlations": {},
    }
    print("=" * 60)
    print("DISTILLED GATE VALIDATION")
    print("=" * 60)
    for config in results.get("configs", []):
        for task in config.get("tasks", []):
            acc = float(task.get("accuracy", 0.0))
            forgetting = _max_forgetting(task)
            acc_pass = acc >= args.target_accuracy
            forget_pass = forgetting < args.max_forgetting
            report["accuracy_pass"] = report["accuracy_pass"] and acc_pass
            report["forgetting_pass"] = report["forgetting_pass"] and forget_pass
            row = {
                "num_kernels": config.get("num_kernels"),
                "task_id": task["task_id"],
                "accuracy": acc,
                "forgetting": forgetting,
                "pass": acc_pass and forget_pass,
            }
            report["tasks"].append(row)
            print(
                f"{row['num_kernels']}K {row['task_id']}: "
                f"acc={acc:.1%}, forgetting={forgetting:.4%}, pass={row['pass']}"
            )

    cfg = Config512D(seed=args.seed)
    for i, task_id in enumerate(CLASSIC_TASKS):
        inputs, _ = generate_classic_task_data(task_id, args.samples, args.seed + i, cfg)
        corr = measure_dc_correlation(
            gate_net,
            task_id,
            inputs,
            input_width=args.input_width,
            feature_dim=args.feature_dim,
            n_samples=min(args.samples, 32),
        )
        report["correlations"][task_id] = corr
        report["correlation_pass"] = report["correlation_pass"] and corr >= args.target_correlation
        print(f"DC correlation {task_id}: {corr:.4f}")

    report["all_pass"] = (
        report["accuracy_pass"] and report["forgetting_pass"] and report["correlation_pass"]
    )
    out = Path("outputs/phase2_validation_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PHASE 2 VALIDATION:", "PASSED" if report["all_pass"] else "FAILED")
    print(f"Saved {out}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate distilled gates")
    parser.add_argument("--gate-weights", default="outputs/distilled_gates.pth")
    parser.add_argument("--results", default="outputs/learned_gates_benchmarks.json")
    parser.add_argument("--input-width", type=int, default=64)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-accuracy", type=float, default=0.98)
    parser.add_argument("--target-correlation", type=float, default=0.95)
    parser.add_argument("--max-forgetting", type=float, default=0.01)
    parser.add_argument("--gate-decay-min", type=float, default=0.0)
    parser.add_argument("--gate-decay-max", type=float, default=0.99)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not Path(args.gate_weights).exists():
        raise SystemExit(f"Gate weights not found: {args.gate_weights}")
    raise SystemExit(0 if validate(args)["all_pass"] else 1)
