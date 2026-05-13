"""Visualize gate distillation training curves."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_distillation_history(
    history_file: str = "outputs/distillation_history.json",
    output_dir: str = "outputs/figures",
) -> None:
    """Generate loss and correlation plots."""
    history = json.loads(Path(history_file).read_text(encoding="utf-8"))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    epochs = history["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, history["total_loss"], linewidth=2)
    axes[0].set_yscale("log")
    axes[0].set_title("Distillation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].grid(alpha=0.3)

    for task_id, values in history["correlations"].items():
        axes[1].plot(epochs, values, label=task_id, linewidth=2)
    axes[1].axhline(0.95, color="red", linestyle="--", label="0.95 target")
    axes[1].set_title("DC Trajectory Correlation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Pearson r")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    path = out / "distillation_training.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    plot_distillation_history()
