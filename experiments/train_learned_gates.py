"""Run learned-gate benchmark experiments.

This script uses the existing classic benchmark runner with
``--temporal-adapter learned-helix`` and saves a Phase 2-shaped results file.
The learned gate network is configured from K support examples per task inside
the adaptive Helix adapter.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_classic_benchmarks import run as run_classic


def _convert_results(classic: dict, k_shot: int) -> dict:
    configs = []
    for config in classic["configs"]:
        tasks = []
        for task_id, task_data in config["tasks"].items():
            evaluation = task_data["evaluation"]
            summary_forgetting = 0.0
            if task_data.get("forgetting_measurements"):
                summary_forgetting = max(
                    abs(
                        item.get(
                            "forgetting",
                            item.get("accuracy_forgetting", item.get("loss_drift", 0.0)),
                        )
                    )
                    for item in task_data["forgetting_measurements"]
                )
            tasks.append(
                {
                    "task_id": task_id,
                    "accuracy": evaluation["accuracy"],
                    "final_loss": evaluation["loss"],
                    "forgetting": summary_forgetting,
                    "gate_specialization": None,
                }
            )
        configs.append(
            {
                "num_kernels": int(config["config_name"].split("_")[0]),
                "k_shot": k_shot,
                "tasks": tasks,
                "summary": config["summary"],
            }
        )
    return {
        "experiment": "learned_gates_benchmarks",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gate_type": "learned",
        "k_shot": k_shot,
        "source_settings": classic["settings"],
        "configs": configs,
        "novel_tasks": [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate learned gates")
    parser.add_argument("--kernels", type=str, default="1,4,16,64")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--projection-from", type=int, default=1)
    parser.add_argument("--stable-base-lr", type=float, default=0.04)
    parser.add_argument("--k-shot", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gate-weights", type=str, default=None)
    parser.add_argument("--gate-decay-min", type=float, default=0.0)
    parser.add_argument("--gate-decay-max", type=float, default=0.99)
    return parser.parse_args()


def main() -> dict:
    args = parse_args()
    classic_args = argparse.Namespace(
        kernels=[int(k.strip()) for k in args.kernels.split(",") if k.strip()],
        epochs=args.epochs,
        samples=args.samples,
        steps=args.steps,
        batch_size=args.batch_size,
        loss_threshold=0.2,
        projection_dim=args.projection_dim,
        projection_from=args.projection_from,
        stable_base_lr=args.stable_base_lr,
        feature_mode="raw_state",
        temporal_adapter="learned-helix",
        gate_weights=args.gate_weights,
        gate_decay_min=args.gate_decay_min,
        gate_decay_max=args.gate_decay_max,
        seed=args.seed,
    )
    classic = run_classic(classic_args)
    payload = _convert_results(classic, args.k_shot)
    out = Path("outputs/learned_gates_benchmarks.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved learned gate results to {out}")
    return payload


if __name__ == "__main__":
    main()
