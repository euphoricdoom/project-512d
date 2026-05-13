"""Generate figures and validation report for classic benchmarks."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")


def load_results(path: str | None = None) -> dict:
    """Load classic benchmark results."""
    if path is None:
        path = os.path.join(OUTPUTS_DIR, "classic_benchmarks.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_accuracy_comparison(results: dict) -> str:
    """Plot mean and per-task accuracy across benchmark configurations."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    configs = results["configs"]
    names = [config["config_name"] for config in configs]
    tasks = results["task_order"]
    mean_acc = [config["summary"]["mean_accuracy"] for config in configs]

    task_acc = {
        task: [config["tasks"][task]["evaluation"]["accuracy"] for config in configs]
        for task in tasks
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(names))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]

    bars = axes[0].bar(x, mean_acc, color=colors[:len(names)], alpha=0.88)
    axes[0].axhline(0.90, color="#E45756", linestyle="--", linewidth=2, label="90% target")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=35, ha="right")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Mean Accuracy")
    axes[0].set_title("Classic Benchmark Mean Accuracy")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()
    for bar, value in zip(bars, mean_acc):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            min(1.02, value + 0.025),
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    task_x = np.arange(len(tasks))
    width = 0.18
    for idx, name in enumerate(names):
        values = [task_acc[task][idx] for task in tasks]
        axes[1].bar(
            task_x + (idx - (len(names) - 1) / 2) * width,
            values,
            width,
            label=name,
            alpha=0.88,
        )
    axes[1].axhline(0.90, color="#E45756", linestyle="--", linewidth=2)
    axes[1].set_xticks(task_x)
    axes[1].set_xticklabels(tasks, rotation=25, ha="right")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy by Task")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "benchmark_accuracy.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_forgetting_analysis(results: dict) -> str:
    """Plot accuracy forgetting by configuration."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    configs = results["configs"]
    names = [config["config_name"] for config in configs]
    mean_forgetting = [abs(config["summary"]["mean_forgetting"]) * 100 for config in configs]
    max_forgetting = [config["summary"]["max_forgetting"] * 100 for config in configs]

    x = np.arange(len(names))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width / 2, mean_forgetting, width, label="Mean", color="#F58518", alpha=0.85)
    ax.bar(x + width / 2, max_forgetting, width, label="Max", color="#E45756", alpha=0.85)
    ax.axhline(1.0, color="#4C78A8", linestyle="--", linewidth=2, label="1% target")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("Accuracy Forgetting (%)")
    ax.set_title("Classic Benchmark Forgetting")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "benchmark_forgetting.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def write_results_table(results: dict) -> str:
    """Write a markdown summary table."""
    lines = [
        "# Classic Benchmarks Results",
        "",
        "| Configuration | Mean Acc | Tasks >90% | Mean Forgetting | Max Forgetting | Pass |",
        "| --- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for config in results["configs"]:
        summary = config["summary"]
        passed = summary["tasks_above_90pct"] == len(results["task_order"]) and summary["forgetting_below_1pct"]
        lines.append(
            f"| {config['config_name']} | {summary['mean_accuracy']:.1%} | "
            f"{summary['tasks_above_90pct']}/{len(results['task_order'])} | "
            f"{summary['mean_forgetting'] * 100:.4f}% | "
            f"{summary['max_forgetting'] * 100:.4f}% | {'yes' if passed else 'no'} |"
        )
    lines.extend(["", "## Per-Task Accuracy", ""])
    lines.append("| Configuration | " + " | ".join(results["task_order"]) + " |")
    lines.append("| --- | " + " | ".join(["---:" for _ in results["task_order"]]) + " |")
    for config in results["configs"]:
        values = [
            f"{config['tasks'][task]['evaluation']['accuracy']:.1%}"
            for task in results["task_order"]
        ]
        lines.append(f"| {config['config_name']} | " + " | ".join(values) + " |")

    path = os.path.join(OUTPUTS_DIR, "classic_benchmarks_table.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def validate_phase_completion(results: dict) -> bool:
    """Return True when all Phase 1 thresholds are met."""
    required_tasks = len(results["task_order"])
    all_pass = True
    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETION VALIDATION")
    print("=" * 70)
    for config in results["configs"]:
        summary = config["summary"]
        acc_pass = summary["mean_accuracy"] > 0.90
        task_pass = summary["tasks_above_90pct"] == required_tasks
        forget_pass = summary["forgetting_below_1pct"]
        all_pass = all_pass and acc_pass and task_pass and forget_pass
        print(f"{config['config_name']}:")
        print(f"  Mean accuracy >90%: {summary['mean_accuracy']:.1%} {'PASS' if acc_pass else 'FAIL'}")
        print(f"  All tasks >90%: {summary['tasks_above_90pct']}/{required_tasks} {'PASS' if task_pass else 'FAIL'}")
        print(f"  Forgetting <1%: {summary['max_forgetting'] * 100:.4f}% {'PASS' if forget_pass else 'FAIL'}")
    print("=" * 70)
    print("PHASE 1 COMPLETE" if all_pass else "PHASE 1 INCOMPLETE")
    print("=" * 70)
    return all_pass


def main() -> bool:
    results = load_results()
    accuracy_path = plot_accuracy_comparison(results)
    forgetting_path = plot_forgetting_analysis(results)
    table_path = write_results_table(results)
    print(f"Saved accuracy figure: {accuracy_path}")
    print(f"Saved forgetting figure: {forgetting_path}")
    print(f"Saved results table: {table_path}")
    return validate_phase_completion(results)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
