"""Analyze LSH-based stack routing distribution.

Generates per-task routing histograms showing which stacks each task's inputs
hash to, and reports a balance score as a measure of LSH uniformity.

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
from system.kernel_lattice import KernelLattice, LatticeConfig

OUTPUTS_DIR = Path(__file__).parent / "outputs"
TASKS = ["tanh", "smooth", "cumsum", "edge", "relu", "sigmoid", "abs", "square"]


def plot_routing_distribution(routing_info: dict, output_path: Path) -> None:
    """Bar chart: for each task, show how many samples routed to each stack."""
    tasks = list(routing_info["task_distributions"].keys())
    n_stacks = routing_info["n_stacks"]
    n_tasks = len(tasks)

    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharey=False)
    axes = axes.ravel()
    colors = plt.cm.tab10(np.linspace(0, 1, n_stacks))

    for i, task in enumerate(tasks):
        ax = axes[i]
        counts = routing_info["task_distributions"][task]
        preferred = routing_info["task_preferred_stacks"][task]
        ax.bar(range(n_stacks), counts, color=colors)
        ax.set_title(f"{task}\n(→ Stack {preferred})", fontsize=10)
        ax.set_xlabel("Stack", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_xticks(range(n_stacks))

    for i in range(n_tasks, len(axes)):
        axes[i].set_visible(False)

    balance = routing_info["balance_score"]
    fig.suptitle(
        f"LSH Stack Routing Distribution  (balance={balance:.2f})",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--n-stacks", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-projections", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("LSH STACK ROUTING DISTRIBUTION ANALYSIS")
    print("=" * 70)

    lattice_cfg = LatticeConfig(
        n_layers=args.n_layers, n_stacks=args.n_stacks, seed=args.seed
    )
    kernel_cfg = Config512D(seed=args.seed)
    print(f"\nInitializing lattice ({args.n_layers} layers x {args.n_stacks} stacks)...")
    print(f"  LSH projections: {args.n_projections}")
    lattice = KernelLattice(
        output_dim=kernel_cfg.output_dim,
        lattice_cfg=lattice_cfg,
        kernel_cfg=kernel_cfg,
    )

    print(f"\nRouting {args.n_samples} samples per task ({len(TASKS)} tasks)...\n")
    routing_info = lattice.analyze_routing_distribution(
        TASKS, n_samples=args.n_samples
    )

    OUTPUTS_DIR.mkdir(exist_ok=True)
    metrics_path = OUTPUTS_DIR / "lsh_routing_distribution.json"
    with open(metrics_path, "w") as f:
        json.dump(routing_info, f, indent=2)

    if not args.no_plots:
        try:
            plot_routing_distribution(
                routing_info, OUTPUTS_DIR / "lsh_routing_distribution.png"
            )
        except Exception as e:
            print(f"[warn] Plot failed: {e}")

    print("=" * 70)
    print("LSH ROUTING RESULTS")
    print("=" * 70)
    print(f"\n  Balance score: {routing_info['balance_score']:.3f}")
    print(f"  (1.0 = perfectly uniform, 0.0 = all samples on one stack)\n")
    print(f"  Task -> Preferred Stack (by sample count):")
    for task, preferred in routing_info["task_preferred_stacks"].items():
        counts = routing_info["task_distributions"][task]
        total = sum(counts)
        pct = counts[preferred] / total * 100 if total > 0 else 0.0
        dist_str = "  ".join(f"s{i}={c}" for i, c in enumerate(counts))
        print(f"    {task:<12} -> Stack {preferred}  ({pct:.0f}%)  [{dist_str}]")
    print()
    print("  Stack load distribution (total samples across all tasks):")
    n_stacks = routing_info["n_stacks"]
    totals = [0] * n_stacks
    for counts in routing_info["task_distributions"].values():
        for i, c in enumerate(counts):
            totals[i] += c
    grand_total = sum(totals)
    for i, t in enumerate(totals):
        bar = "#" * int(t / max(totals) * 30) if max(totals) > 0 else ""
        pct = t / grand_total * 100 if grand_total > 0 else 0.0
        print(f"    Stack {i}: {bar} ({t}, {pct:.1f}%)")
    print()
    print(f"  Note: LSH routing is deterministic and requires no training.")
    print(f"  Similar inputs hash to the same stack, creating structural specialization.")
    print()
    print(f"  Metrics saved: {metrics_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
