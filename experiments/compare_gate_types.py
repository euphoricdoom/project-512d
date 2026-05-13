"""Generate explicit-vs-learned gate comparison figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: str) -> dict | None:
    """Load JSON if it exists."""
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _mean_task_acc(results: dict) -> dict[str, float]:
    tasks = results.get("task_order") or ["copy_task", "parity_task", "adding_task"]
    acc = {task: [] for task in tasks}
    for config in results["configs"]:
        if isinstance(config["tasks"], dict):
            for task in tasks:
                acc[task].append(config["tasks"][task]["evaluation"]["accuracy"])
        else:
            for task in config["tasks"]:
                acc.setdefault(task["task_id"], []).append(task["accuracy"])
    return {task: float(np.mean(values)) for task, values in acc.items() if values}


def plot_comparison(results_by_label: dict[str, dict], output_file: str) -> None:
    """Plot mean per-task accuracy for all available gate result files."""
    acc_by_label = {
        label: _mean_task_acc(results)
        for label, results in results_by_label.items()
        if results is not None
    }
    tasks = sorted({task for acc in acc_by_label.values() for task in acc})
    x = np.arange(len(tasks))
    width = min(0.8 / max(1, len(acc_by_label)), 0.28)
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(acc_by_label)
    offsets = (np.arange(len(labels)) - (len(labels) - 1) / 2) * width
    for offset, label in zip(offsets, labels):
        ax.bar(
            x + offset,
            [acc_by_label[label].get(t, 0.0) for t in tasks],
            width,
            label=label,
        )
    ax.axhline(0.98, color="red", linestyle="--", linewidth=1.5, label="98% target")
    ax.set_ylabel("Accuracy")
    ax.set_title("Explicit vs Learned Gates")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved comparison to {out}")


def main() -> None:
    results = {
        "Explicit/Latest": load_json("outputs/classic_benchmarks.json"),
        "Learned/Latest": load_json("outputs/learned_gates_benchmarks.json"),
        "Distilled": load_json("outputs/learned_gates_distilled_benchmarks.json"),
    }
    if not any(results.values()):
        raise SystemExit("No benchmark JSON files found in outputs/")
    plot_comparison(results, "outputs/figures/gate_type_comparison.png")


if __name__ == "__main__":
    main()
