"""Analyze stack specialization patterns via routing.

Generates a Task x Stack heatmap showing which tasks prefer which stacks,
and reports routing entropy as a measure of specialization strength.

Usage::

    python analyze_specialization.py
    python analyze_specialization.py --n-samples 100
    python analyze_specialization.py --no-plots
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.constants import Config512D
from experiments.run_4_kernel_network import make_extended_dataset
from system.kernel_lattice import KernelLattice, LatticeConfig

OUTPUTS_DIR = Path(__file__).parent / "outputs"
TASKS = ["tanh", "smooth", "cumsum", "edge", "relu", "sigmoid", "abs", "square"]


def _entropy(weights: np.ndarray) -> float:
    w = weights / (weights.sum() + 1e-12)
    return float(-np.sum(w * np.log(w + 1e-12)))


def analyze_routing_specialization(lattice, tasks, n_samples=50):
    """Return (affinity, sharpness) where:

    affinity  — (n_tasks, n_stacks) mean routing weights across samples.
                Near-uniform pre-training; task-differentiated after training.
    sharpness — (n_tasks,) mean per-sample routing entropy.
                Low = router makes confident per-sample choices (good).
                High = near-uniform per sample (router not yet trained).
    """
    cfg = lattice.kernel_cfg
    rng = np.random.default_rng(cfg.seed + 9000)
    n_stacks = lattice.lattice_cfg.n_stacks
    affinity = np.zeros((len(tasks), n_stacks))
    sharpness = np.zeros(len(tasks))

    print("\nAnalyzing stack routing preferences...")
    for t_idx, task in enumerate(tasks):
        print(f"  {task:<12}", end="", flush=True)
        X, _ = make_extended_dataset(cfg, task, rng, n_samples)
        task_weights = np.array([lattice.route_stacks(x) for x in X])
        mean_weights = task_weights.mean(axis=0)
        sample_entropies = np.array([_entropy(w) for w in task_weights])
        mean_sample_entropy = float(sample_entropies.mean())
        affinity[t_idx] = mean_weights
        sharpness[t_idx] = mean_sample_entropy
        preferred = int(np.argmax(mean_weights))
        max_ent = np.log(n_stacks)
        sharpness_pct = 1.0 - mean_sample_entropy / max_ent
        print(f"-> Stack {preferred}  (mean_w={mean_weights[preferred]:.3f}  "
              f"per-sample sharpness={sharpness_pct:.1%})")
    return affinity, sharpness


def compute_specialization_metrics(affinity, sharpness, tasks):
    n_stacks = affinity.shape[1]
    max_entropy = float(np.log(n_stacks))
    # Task-level: entropy of MEAN routing (measures cross-task differentiation)
    mean_affinity_entropies = {task: _entropy(affinity[i]) for i, task in enumerate(tasks)}
    # Sample-level: mean per-sample entropy (measures how sharp routing is per input)
    mean_sample_entropy = float(sharpness.mean())
    preferred_stacks = {task: int(np.argmax(affinity[i])) for i, task in enumerate(tasks)}
    stack_task_counts = {
        f"stack_{j}": int(np.sum(np.argmax(affinity, axis=1) == j))
        for j in range(n_stacks)
    }
    return {
        "mean_affinity_entropies": mean_affinity_entropies,
        "mean_sample_entropy": mean_sample_entropy,
        "max_entropy": max_entropy,
        "per_sample_sharpness": float(1.0 - mean_sample_entropy / (max_entropy + 1e-12)),
        "preferred_stacks": preferred_stacks,
        "stack_task_counts": stack_task_counts,
    }


def plot_task_stack_heatmap(affinity, tasks, output_path):
    n_stacks = affinity.shape[1]
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(affinity, cmap="hot", aspect="auto", vmin=0)
    ax.set_xticks(range(n_stacks))
    ax.set_xticklabels([f"Stack {i}" for i in range(n_stacks)])
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks)
    ax.set_xlabel("Stack ID", fontsize=12)
    ax.set_ylabel("Task", fontsize=12)
    ax.set_title("Task -> Stack Routing Specialization", fontsize=14)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean Routing Weight", fontsize=11)
    for i in range(len(tasks)):
        for j in range(n_stacks):
            ax.text(j, i, f"{affinity[i, j]:.3f}",
                    ha="center", va="center",
                    color="white" if affinity[i, j] > affinity.max() * 0.5 else "black",
                    fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--n-stacks", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("TASK -> STACK ROUTING SPECIALIZATION ANALYSIS")
    print("=" * 70)

    lattice_cfg = LatticeConfig(n_layers=args.n_layers, n_stacks=args.n_stacks, seed=args.seed)
    kernel_cfg = Config512D(seed=args.seed)
    print(f"\nInitializing lattice ({args.n_layers} layers x {args.n_stacks} stacks)...")
    lattice = KernelLattice(output_dim=kernel_cfg.output_dim, lattice_cfg=lattice_cfg, kernel_cfg=kernel_cfg)

    affinity, sharpness = analyze_routing_specialization(lattice, TASKS, n_samples=args.n_samples)
    metrics = compute_specialization_metrics(affinity, sharpness, TASKS)

    OUTPUTS_DIR.mkdir(exist_ok=True)
    metrics_path = OUTPUTS_DIR / "routing_specialization_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"affinity_matrix": affinity.tolist(), "tasks": TASKS, "n_stacks": args.n_stacks, "metrics": metrics}, f, indent=2)

    if not args.no_plots:
        try:
            plot_task_stack_heatmap(affinity, TASKS, OUTPUTS_DIR / "task_stack_routing_specialization.png")
        except Exception as e:
            print(f"[warn] Plot failed: {e}")

    max_ent = metrics["max_entropy"]
    print()
    print("=" * 70)
    print("SPECIALIZATION METRICS")
    print("=" * 70)
    print(f"\n  Per-sample routing sharpness : {metrics['per_sample_sharpness']:.1%}")
    print(f"  (0% = uniform per sample, 100% = always routes to one stack)")
    print(f"  Max entropy (uniform)        : {max_ent:.3f}")
    print(f"  Mean per-sample entropy      : {metrics['mean_sample_entropy']:.3f}")
    print()
    print("  Task -> Preferred Stack (by mean affinity):")
    for task, stack in metrics["preferred_stacks"].items():
        ent = metrics["mean_affinity_entropies"][task]
        print(f"    {task:<12} -> Stack {stack}  (affinity_entropy={ent:.3f})")
    print()
    print("  Stack task distribution:")
    for sk, count in metrics["stack_task_counts"].items():
        bar = "#" * count
        print(f"    {sk}: {bar} ({count})")
    print()
    print(f"  Note: mean affinity is near-uniform pre-training (expected).")
    print(f"  Per-sample sharpness shows the router IS content-sensitive.")
    print(f"  Train stack_router via gradient descent to lock task preferences.")
    print()
    print(f"  Metrics saved: {metrics_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
