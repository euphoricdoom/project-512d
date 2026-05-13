"""Ablation cockpit for the single-node 512D field system.

Compares the two highest-leverage architectural switches:
  - routing: sparse top-k winners vs full softmax
  - K_inter: fixed random topology vs learnable Hebbian topology

Outputs:
  outputs/ablation_cockpit.json
  outputs/ablation_cockpit.csv
  outputs/ablation_cockpit.md

Usage:
  python experiments/run_ablation_cockpit.py --epochs 12 --samples 40
  python -m project512d --experiment ablation_cockpit --task all
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from core.constants import Config512D
from system.modular_system import ModularFieldSystem
from system.trainer import ModularTrainer


OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
TASKS = ["tanh", "smooth", "cumsum", "edge"]


def _kernel_factor_norm(system: ModularFieldSystem) -> float:
    U = system.kernel._U
    V = system.kernel._V
    return float(np.linalg.norm(U) + np.linalg.norm(V))


def run_condition(
    task: str,
    seed: int,
    epochs: int,
    samples: int,
    k_winners: int,
    learnable_inter_modules: bool,
) -> Dict[str, Any]:
    cfg = Config512D(
        seed=seed,
        train_epochs=epochs,
        train_samples=samples,
        k_winners=k_winners,
        learnable_inter_modules=learnable_inter_modules,
    )
    system = ModularFieldSystem(cfg)
    before_factor_norm = _kernel_factor_norm(system)
    before_inhibition = system.governance.self_inhibition.copy()
    before_radius = float(system.kernel.radius)

    trainer = ModularTrainer(system)
    result = trainer.train(task, verbose=False)
    system.kernel.renormalize_inter()

    after_factor_norm = _kernel_factor_norm(system)
    after_inhibition = system.governance.self_inhibition.copy()
    after_radius = float(system.kernel.radius)
    losses = result["losses"]
    active_routes = k_winners if 0 < k_winners < cfg.num_modules else cfg.num_modules

    return {
        "task": task,
        "routing": "sparse_top_k" if active_routes < cfg.num_modules else "full_softmax",
        "active_routes": active_routes,
        "learnable_inter_modules": learnable_inter_modules,
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "test_loss": float(result["test_loss"]),
        "improvement_pct": float(
            100.0 * (losses[0] - losses[-1]) / max(losses[0], 1e-12)
        ),
        "kernel_radius_before": before_radius,
        "kernel_radius_after": after_radius,
        "kernel_factor_norm_delta": after_factor_norm - before_factor_norm,
        "governance_params": system.governance.stored_params(),
        "dense_governance_params": cfg.dim_f * cfg.dim_c,
        "governance_inhibition_delta": float(np.linalg.norm(after_inhibition - before_inhibition)),
        "config": {
            "seed": seed,
            "epochs": epochs,
            "samples": samples,
            "k_winners": k_winners,
            "learnable_inter_modules": learnable_inter_modules,
        },
    }


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "task",
        "routing",
        "active_routes",
        "learnable_inter_modules",
        "initial_loss",
        "final_loss",
        "test_loss",
        "improvement_pct",
        "kernel_radius_before",
        "kernel_radius_after",
        "kernel_factor_norm_delta",
        "governance_params",
        "dense_governance_params",
        "governance_inhibition_delta",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def _write_markdown(path: str, payload: Dict[str, Any]) -> None:
    rows = payload["rows"]
    best_by_task = {}
    for task in payload["tasks"]:
        task_rows = [r for r in rows if r["task"] == task]
        best_by_task[task] = min(task_rows, key=lambda r: r["test_loss"])

    lines = [
        "# Ablation Cockpit Report",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Question",
        "",
        "Do sparse routing and learnable inter-module coupling improve the single-node 512D field system?",
        "",
        "## Best Condition By Task",
        "",
        "| task | routing | learnable K_inter | test loss | final train loss | radius after |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for task, row in best_by_task.items():
        lines.append(
            f"| {task} | {row['routing']} | {row['learnable_inter_modules']} | "
            f"{row['test_loss']:.6f} | {row['final_loss']:.6f} | "
            f"{row['kernel_radius_after']:.6f} |"
        )

    lines.extend([
        "",
        "## Full Results",
        "",
        "| task | routing | active routes | learnable K_inter | test loss | improvement | K factor delta |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: |",
    ])
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['routing']} | {row['active_routes']} | "
            f"{row['learnable_inter_modules']} | {row['test_loss']:.6f} | "
            f"{row['improvement_pct']:.1f}% | {row['kernel_factor_norm_delta']:.6f} |"
        )

    lines.extend([
        "",
        "## Governance Footprint",
        "",
        f"- Dense module feedback params: {rows[0]['dense_governance_params']}",
        f"- Compact governance params: {rows[0]['governance_params']}",
        f"- Reduction: {rows[0]['dense_governance_params'] / rows[0]['governance_params']:.2f}x",
        "",
        "## Reading Notes",
        "",
        "- Lower test loss is better.",
        "- Positive K factor delta means the Hebbian inter-module factors moved during training.",
        "- Governance params are the 60 learned self-inhibition scalars.",
        "- Radius after training should remain below the configured target after renormalization.",
    ])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    tasks = TASKS if args.task == "all" else [args.task]
    conditions = [
        {"k_winners": 5, "learnable_inter_modules": True},
        {"k_winners": 5, "learnable_inter_modules": False},
        {"k_winners": 0, "learnable_inter_modules": True},
        {"k_winners": 0, "learnable_inter_modules": False},
    ]

    print("=" * 72)
    print("ABLATION COCKPIT")
    print("=" * 72)
    print(f"tasks:   {', '.join(tasks)}")
    print(f"epochs:  {args.epochs}")
    print(f"samples: {args.samples}")
    print()

    rows: List[Dict[str, Any]] = []
    for task in tasks:
        print(f"Task: {task}")
        for condition in conditions:
            row = run_condition(
                task=task,
                seed=args.seed,
                epochs=args.epochs,
                samples=args.samples,
                **condition,
            )
            rows.append(row)
            print(
                f"  {row['routing']:<12} learnable={str(row['learnable_inter_modules']):<5} "
                f"test={row['test_loss']:.6f} radius={row['kernel_radius_after']:.6f}"
            )
        print()

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "description": "Single-node sparse/full routing x fixed/learnable K_inter ablation.",
        "tasks": tasks,
        "args": vars(args),
        "rows": rows,
    }

    json_path = os.path.join(OUTPUTS_DIR, "ablation_cockpit.json")
    csv_path = os.path.join(OUTPUTS_DIR, "ablation_cockpit.csv")
    md_path = os.path.join(OUTPUTS_DIR, "ablation_cockpit.md")
    _write_json(json_path, payload)
    _write_csv(csv_path, rows)
    _write_markdown(md_path, payload)

    print("Saved:")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print(f"  {md_path}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-node ablation cockpit")
    parser.add_argument("--task", choices=["all", *TASKS], default="all")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    run(parser.parse_args())
