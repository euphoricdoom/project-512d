"""Compare stable and legacy readouts on the 64-kernel scaling task."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
SCALING_JSON = os.path.join(OUTPUTS_DIR, "scaling_law.json")


def _run_scaling(args: argparse.Namespace, stable: bool) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "run_scaling_law.py"),
        "--kernels",
        str(args.kernels),
        "--num-tasks",
        str(args.num_tasks),
        "--epochs",
        str(args.epochs),
        "--samples",
        str(args.samples),
        "--eval-samples",
        str(args.eval_samples),
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--batch-size",
        str(args.batch_size),
        "--eval-every",
        str(args.eval_every),
        "--forgetting-threshold",
        str(args.forgetting_threshold),
        "--loss-threshold",
        str(args.loss_threshold),
        "--mode",
        args.mode,
        "--architecture",
        args.architecture,
        "--stable-base-lr",
        str(args.stable_base_lr),
        "--stable-readout" if stable else "--no-stable-readout",
    ]
    label = "stable" if stable else "legacy"
    print(f"\nRunning {label} readout comparison leg:")
    print(" ".join(cmd))
    sys.stdout.flush()
    subprocess.run(cmd, check=True)

    with open(SCALING_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def _single_result(payload: Dict[str, Any], kernels: int) -> Dict[str, Any]:
    key = str(kernels)
    if key not in payload["results"]:
        raise KeyError(f"Expected kernel count {key} in {SCALING_JSON}")
    return payload["results"][key]


def _plot(stable: Dict[str, Any], legacy: Dict[str, Any], kernels: int) -> str:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    labels = ["stable", "legacy"]
    metrics = [
        ("capacity", "Strict capacity"),
        ("retained_tasks", "Retained tasks"),
        ("avg_forgetting", "Avg forgetting"),
        ("total_time", "Total time (s)"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, (metric, title) in zip(axes, metrics):
        values = [stable[metric], legacy[metric]]
        colors = ["seagreen", "slateblue"]
        ax.bar(labels, values, color=colors, alpha=0.85)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        if metric == "avg_forgetting":
            ax.set_ylabel("loss delta")
    fig.suptitle(f"Readout Stability Comparison ({kernels} kernels)")
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "readout_stability_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def run(args: argparse.Namespace) -> Dict[str, Any]:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    stable_payload = _run_scaling(args, stable=True)
    legacy_payload = _run_scaling(args, stable=False)

    stable_result = _single_result(stable_payload, args.kernels)
    legacy_result = _single_result(legacy_payload, args.kernels)
    figure_path = _plot(stable_result, legacy_result, args.kernels)

    comparison = {
        "kernels": args.kernels,
        "stable": stable_result,
        "legacy": legacy_result,
        "figure": figure_path,
        "delta": {
            "capacity": stable_result["capacity"] - legacy_result["capacity"],
            "retained_tasks": stable_result["retained_tasks"] - legacy_result["retained_tasks"],
            "avg_forgetting": stable_result["avg_forgetting"] - legacy_result["avg_forgetting"],
            "total_time": stable_result["total_time"] - legacy_result["total_time"],
        },
    }

    print("\nREADOUT STABILITY COMPARISON")
    print("=" * 70)
    print(f"Kernels: {args.kernels}")
    print(
        f"Stable: capacity={stable_result['capacity']}, "
        f"retained={stable_result['retained_tasks']}, "
        f"avg_forgetting={stable_result['avg_forgetting']:.6f}, "
        f"time={stable_result['total_time']:.1f}s"
    )
    print(
        f"Legacy: capacity={legacy_result['capacity']}, "
        f"retained={legacy_result['retained_tasks']}, "
        f"avg_forgetting={legacy_result['avg_forgetting']:.6f}, "
        f"time={legacy_result['total_time']:.1f}s"
    )
    print(
        f"Delta stable-legacy: capacity={comparison['delta']['capacity']:+}, "
        f"retained={comparison['delta']['retained_tasks']:+}, "
        f"avg_forgetting={comparison['delta']['avg_forgetting']:+.6f}, "
        f"time={comparison['delta']['total_time']:+.1f}s"
    )
    print(f"Figure: {figure_path}")
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare stable vs legacy readout stability")
    parser.add_argument("--kernels", type=int, default=64)
    parser.add_argument("--num-tasks", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--eval-samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--forgetting-threshold", type=float, default=1.0)
    parser.add_argument("--loss-threshold", type=float, default=10.0)
    parser.add_argument("--stable-base-lr", type=float, default=0.01)
    parser.add_argument(
        "--architecture",
        choices=["flat", "hierarchical"],
        default="flat",
    )
    parser.add_argument(
        "--mode",
        choices=["accuracy", "retention"],
        default="accuracy",
    )
    run(parser.parse_args())
