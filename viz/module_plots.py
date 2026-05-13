from __future__ import annotations

from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from system.modular_system import ModularFieldSystem


def plot_module_activations(
    system: ModularFieldSystem,
    title: str = "Module Activation Pattern",
) -> plt.Figure:
    """Bar chart of current module activation levels, color-coded by cognitive role."""
    activations = [m.activation(system.state) for m in system.modules]

    color_map = {
        "visual":     "steelblue",
        "audio":      "cornflowerblue",
        "text":       "mediumblue",
        "logical":    "seagreen",
        "analogical": "mediumseagreen",
        "causal":     "limegreen",
        "pattern":    "forestgreen",
        "working":    "darkorange",
        "episodic":   "orange",
        "motor":      "tomato",
        "language":   "crimson",
        "decision":   "firebrick",
        "error":      "darkred",
    }

    colors = []
    for m in system.modules:
        color = "lightgray"
        for key, c in color_map.items():
            if key in m.expertise:
                color = c
                break
        colors.append(color)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(range(system.cfg.num_modules), activations, color=colors, alpha=0.8, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Module ID")
    ax.set_ylabel("Activation Level (||state||)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    legend_elements = [
        Patch(facecolor="steelblue",   label="Perception (visual/audio/text)"),
        Patch(facecolor="seagreen",    label="Reasoning"),
        Patch(facecolor="darkorange",  label="Memory"),
        Patch(facecolor="tomato",      label="Action/Output"),
        Patch(facecolor="lightgray",   label="Other"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    plt.tight_layout()
    return fig


def plot_coupling_heatmap(system: ModularFieldSystem) -> plt.Figure:
    """Heatmap of inter-module coupling strengths."""
    cfg = system.cfg
    K_inter = system.kernel.K_inter[:cfg.dim_f, :cfg.dim_f]
    module_coupling = np.zeros((cfg.num_modules, cfg.num_modules))

    for i in range(cfg.num_modules):
        for j in range(cfg.num_modules):
            i_sl = slice(i * cfg.module_size, (i + 1) * cfg.module_size)
            j_sl = slice(j * cfg.module_size, (j + 1) * cfg.module_size)
            module_coupling[i, j] = float(np.sum(K_inter[i_sl, j_sl]))

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(module_coupling, cmap="YlOrRd", aspect="auto")
    ax.set_xlabel("Target Module")
    ax.set_ylabel("Source Module")
    ax.set_title("Inter-Module Coupling Strength")
    ax.set_xticks(np.arange(0, cfg.num_modules, 5))
    ax.set_yticks(np.arange(0, cfg.num_modules, 5))
    ax.grid(True, alpha=0.3, linewidth=0.5)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Coupling Weight")
    plt.tight_layout()
    return fig


def plot_state_evolution(system: ModularFieldSystem, steps: int = 50) -> plt.Figure:
    """Heatmap of module activation over time (runs the system forward)."""
    cfg = system.cfg
    history = []
    for _ in range(steps):
        system.step()
        activations = [m.activation(system.state) for m in system.modules]
        history.append(activations)

    history_arr = np.array(history).T   # (num_modules, steps)

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(history_arr, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Module ID")
    ax.set_title("Module Activation Evolution")

    sample_ids = [0, 10, 20, 30, 40, 50, 59]
    ax.set_yticks([i for i in sample_ids if i < cfg.num_modules])
    ax.set_yticklabels(
        [f"{i}: {system.modules[i].expertise}" for i in sample_ids if i < cfg.num_modules],
        fontsize=8,
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Activation Level")
    plt.tight_layout()
    return fig


def plot_routing_weights(
    route_means: np.ndarray,
    system: ModularFieldSystem,
    title: str = "Final Mean Routing Weights",
) -> plt.Figure:
    """Bar chart of mean routing weights across the last training epoch."""
    final_routes = route_means[-1]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(np.arange(system.cfg.num_modules), final_routes, edgecolor="black", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Module")
    ax.set_ylabel("Mean route weight")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig
