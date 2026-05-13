"""Governance stress chamber for the 512D field system.

The goal is to make the constraint layer earn its keep. This experiment trains
the same stressed single-node system under three governance modes:

  - none:      boundary events are observed, projection is disabled
  - fixed:     selective projection is enabled, inhibition weights are fixed
  - adaptive:  selective projection is enabled, inhibition weights learn

Outputs:
  outputs/governance_stress.json
  outputs/governance_stress.csv
  outputs/governance_stress.md
  outputs/figures/governance_stress.png

Usage:
  python experiments/run_governance_stress.py --epochs 4 --samples 15
  python -m project512d --experiment governance_stress --task all
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.constants import Config512D
from core.kernel import mse
from system.modular_system import ModularFieldSystem
from system.trainer import ModularTrainer, make_dataset


OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
TASKS = ["tanh", "smooth", "cumsum", "edge"]
MODES = ["none", "fixed", "adaptive"]


def stressed_config(args: argparse.Namespace, mode: str) -> Config512D:
    return Config512D(
        seed=args.seed,
        train_epochs=args.epochs,
        train_samples=args.samples,
        input_strength=args.input_strength,
        process_steps=args.process_steps,
        inter_coupling=args.inter_coupling,
        boundary_theta=args.boundary_theta,
        boundary_delta=args.boundary_delta,
        projection_damping=args.projection_damping,
        enable_projection=(mode != "none"),
        learnable_constraint_feedback=(mode == "adaptive"),
        constraint_feedback_lr=args.constraint_feedback_lr,
        learnable_inter_modules=args.learnable_inter_modules,
    )


def feedback_norm(system: ModularFieldSystem) -> float:
    return float(np.linalg.norm(system.governance.self_inhibition))


def feedback_factors(system: ModularFieldSystem) -> tuple[np.ndarray, np.ndarray]:
    weights = system.governance.self_inhibition.copy()
    return weights, weights.copy()


def evaluate(system: ModularFieldSystem, task: str, n_samples: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    X, Y = make_dataset(system.cfg, task, rng, n_samples)
    losses = []
    for x, y in zip(X, Y):
        system.reset_state()
        pred, _ = system.process(x)
        losses.append(mse(pred, y))
    return float(np.mean(losses))


def recovery_probe(system: ModularFieldSystem, args: argparse.Namespace) -> Dict[str, Any]:
    cfg = system.cfg
    rng = np.random.default_rng(args.seed + 777)
    system.reset_state(scale=0.35)
    burst = rng.normal(0.0, args.burst_scale, size=cfg.input_dim)
    system.inject(burst)

    norms = []
    projections_before = system.trace.summary().get("projections", 0)
    recovered_step = None
    for step in range(args.recovery_steps):
        system.step(t=step)
        norm = float(np.linalg.norm(system.state))
        norms.append(norm)
        if recovered_step is None and norm <= cfg.boundary_theta:
            recovered_step = step

    projections_after = system.trace.summary().get("projections", 0)
    return {
        "recovery_steps": args.recovery_steps if recovered_step is None else recovered_step,
        "recovered": recovered_step is not None,
        "recovery_max_norm": float(max(norms)) if norms else 0.0,
        "recovery_final_norm": float(norms[-1]) if norms else 0.0,
        "recovery_projections": int(projections_after - projections_before),
    }


def run_condition(task: str, mode: str, args: argparse.Namespace) -> Dict[str, Any]:
    cfg = stressed_config(args, mode)
    system = ModularFieldSystem(cfg)
    before_feedback = feedback_norm(system)
    A_before, B_before = feedback_factors(system)

    trainer = ModularTrainer(system)
    result = trainer.train(task, verbose=False)
    test_loss = evaluate(system, task, args.eval_samples, args.seed + 9000)
    trace = system.trace.summary()
    recovery = recovery_probe(system, args)
    after_feedback = feedback_norm(system)
    A_after, B_after = feedback_factors(system)
    factor_movement = float(np.linalg.norm(A_after - A_before))

    return {
        "task": task,
        "mode": mode,
        "projection_enabled": cfg.enable_projection,
        "learnable_constraint_feedback": cfg.learnable_constraint_feedback,
        "initial_loss": float(result["losses"][0]),
        "final_loss": float(result["losses"][-1]),
        "test_loss": test_loss,
        "steps": int(trace.get("steps", 0)),
        "max_norm": float(trace.get("max_norm", 0.0)),
        "avg_norm": float(trace.get("avg_norm", 0.0)),
        "final_norm": float(trace.get("final_norm", 0.0)),
        "triggers": int(trace.get("triggers", 0)),
        "projections": int(trace.get("projections", 0)),
        "feedback_norm_delta": after_feedback - before_feedback,
        "feedback_factor_movement": factor_movement,
        "governance_params": system.governance.stored_params(),
        "dense_governance_params": cfg.dim_f * cfg.dim_c,
        "kernel_radius": float(system.kernel.radius),
        **recovery,
    }


def write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "task",
        "mode",
        "projection_enabled",
        "learnable_constraint_feedback",
        "test_loss",
        "final_loss",
        "max_norm",
        "avg_norm",
        "triggers",
        "projections",
        "recovered",
        "recovery_steps",
        "recovery_max_norm",
        "recovery_final_norm",
        "recovery_projections",
        "feedback_norm_delta",
        "feedback_factor_movement",
        "kernel_radius",
        "governance_params",
        "dense_governance_params",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def write_markdown(path: str, payload: Dict[str, Any]) -> None:
    rows = payload["rows"]
    lines = [
        "# Governance Stress Chamber",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Question",
        "",
        "Does compact selective governance improve stability, recovery, or task loss under stress?",
        "",
        "## Stress Settings",
        "",
    ]
    for key, value in payload["stress_settings"].items():
        lines.append(f"- {key}: {value}")

    lines.extend([
        "",
        "## Results",
        "",
        "| task | mode | test loss | max norm | triggers | projections | recovered | recovery steps | feedback movement |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ])
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['mode']} | {row['test_loss']:.6f} | "
            f"{row['max_norm']:.4f} | {row['triggers']} | {row['projections']} | "
            f"{row['recovered']} | {row['recovery_steps']} | "
            f"{row['feedback_factor_movement']:.3e} |"
        )

    dense = rows[0]["dense_governance_params"] if rows else 0
    factored = rows[0]["governance_params"] if rows else 0
    reduction = dense / factored if factored else 0.0
    lines.extend([
        "",
        "## Governance Footprint",
        "",
        f"- Dense module feedback params: {dense}",
        f"- Compact governance params: {factored}",
        f"- Reduction: {reduction:.2f}x",
        "",
        "## Reading Notes",
        "",
        "- `none` keeps boundary detection but disables projection.",
        "- `fixed` enables selective projection without adapting inhibition weights.",
        "- `adaptive` enables selective projection and loss-gated inhibition updates.",
        "- Recovery is measured after a high-amplitude input burst.",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_figure(path: str, rows: List[Dict[str, Any]]) -> None:
    labels = [f"{row['task']}\n{row['mode']}" for row in rows]
    max_norms = [row["max_norm"] for row in rows]
    projections = [row["projections"] for row in rows]
    losses = [row["test_loss"] for row in rows]

    fig, axes = plt.subplots(3, 1, figsize=(max(10, len(rows) * 0.8), 12))
    axes[0].bar(labels, max_norms, color="steelblue")
    axes[0].set_ylabel("Max norm")
    axes[0].set_title("Stress Norm Peak")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(labels, projections, color="tomato")
    axes[1].set_ylabel("Projections")
    axes[1].set_title("Governance Intervention Count")
    axes[1].tick_params(axis="x", rotation=45)

    axes[2].bar(labels, losses, color="seagreen")
    axes[2].set_ylabel("Test loss")
    axes[2].set_title("Task Loss Under Stress")
    axes[2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    tasks = TASKS if args.task == "all" else [args.task]

    print("=" * 72)
    print("GOVERNANCE STRESS CHAMBER")
    print("=" * 72)
    print(f"tasks: {', '.join(tasks)}")
    print(f"modes: {', '.join(MODES)}")
    print()

    rows = []
    for task in tasks:
        print(f"Task: {task}")
        for mode in MODES:
            row = run_condition(task, mode, args)
            rows.append(row)
            print(
                f"  {mode:<8} test={row['test_loss']:.6f} "
                f"max_norm={row['max_norm']:.3f} triggers={row['triggers']} "
                f"projections={row['projections']}"
            )
        print()

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "description": "Stress comparison of no, fixed, and adaptive governance.",
        "tasks": tasks,
        "modes": MODES,
        "stress_settings": {
            "epochs": args.epochs,
            "samples": args.samples,
            "eval_samples": args.eval_samples,
            "input_strength": args.input_strength,
            "process_steps": args.process_steps,
            "inter_coupling": args.inter_coupling,
            "boundary_theta": args.boundary_theta,
            "boundary_delta": args.boundary_delta,
            "projection_damping": args.projection_damping,
            "burst_scale": args.burst_scale,
            "recovery_steps": args.recovery_steps,
        },
        "rows": rows,
    }

    json_path = os.path.join(OUTPUTS_DIR, "governance_stress.json")
    csv_path = os.path.join(OUTPUTS_DIR, "governance_stress.csv")
    md_path = os.path.join(OUTPUTS_DIR, "governance_stress.md")
    fig_path = os.path.join(FIGURES_DIR, "governance_stress.png")
    write_json(json_path, payload)
    write_csv(csv_path, rows)
    write_markdown(md_path, payload)
    write_figure(fig_path, rows)

    print("Saved:")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print(f"  {md_path}")
    print(f"  {fig_path}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Governance stress chamber")
    parser.add_argument("--task", choices=["all", *TASKS], default="all")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--eval-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-strength", type=float, default=0.42)
    parser.add_argument("--process-steps", type=int, default=36)
    parser.add_argument("--inter-coupling", type=float, default=0.018)
    parser.add_argument("--boundary-theta", type=float, default=1.4)
    parser.add_argument("--boundary-delta", type=float, default=0.08)
    parser.add_argument("--projection-damping", type=float, default=0.42)
    parser.add_argument("--constraint-feedback-lr", type=float, default=0.002)
    parser.add_argument("--burst-scale", type=float, default=2.5)
    parser.add_argument("--recovery-steps", type=int, default=48)
    parser.add_argument("--learnable-inter-modules", action="store_true")
    run(parser.parse_args())
