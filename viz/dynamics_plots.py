from __future__ import annotations

from typing import List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from core.projector import TraceLedger
from core.constants import Config512D


def plot_trace_evolution(trace: TraceLedger, cfg: Config512D) -> plt.Figure:
    """Four-panel dashboard: energy norm, field/constraint split, rate of change,
    and governance events timeline."""
    log = trace.rows
    if not log:
        raise ValueError("Trace is empty — run system.evolve() first.")

    t           = [e["t"]               for e in log]
    norms       = [e["norm"]             for e in log]
    deltas      = [e["delta"]            for e in log]
    f_norms     = [e["field_norm"]       for e in log]
    c_norms     = [e["constraint_norm"]  for e in log]
    triggered   = [e["triggered"]        for e in log]
    projected   = [e["projected"]        for e in log]

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Panel 1: overall norm
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, norms, linewidth=1.5, label="||X||", color="steelblue")
    ax1.axhline(cfg.boundary_theta, color="tomato", linestyle="--", linewidth=2,
                alpha=0.8, label=f"Threshold θ={cfg.boundary_theta}")
    proj_t  = [e["t"]    for e in log if e["projected"]]
    proj_n  = [e["norm"] for e in log if e["projected"]]
    ax1.scatter(proj_t, proj_n, color="tomato", s=80, marker="x", linewidths=2,
                label="Projection events", zorder=5)
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("System norm")
    ax1.set_title("System Energy Evolution", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: field vs constraint
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t, f_norms, label="Field ||X_F||",      linewidth=1.5, alpha=0.9)
    ax2.plot(t, c_norms, label="Constraint ||X_C||", linewidth=1.5, alpha=0.9)
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Norm")
    ax2.set_title("Field vs Constraint Dynamics")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: rate of change
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t, deltas, color="darkorange", linewidth=1.5, alpha=0.9)
    ax3.axhline(cfg.boundary_delta, color="tomato", linestyle="--", linewidth=2,
                alpha=0.8, label=f"Change threshold Δ={cfg.boundary_delta}")
    ax3.set_xlabel("Time step")
    ax3.set_ylabel("Δ||X||")
    ax3.set_title("Rate of Change")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: governance timeline
    ax4 = fig.add_subplot(gs[2, :])
    trig_sig = np.array([1 if e else 0 for e in triggered])
    proj_sig = np.array([1 if e else 0 for e in projected])
    ax4.fill_between(t, 0, trig_sig, step="mid", alpha=0.35, color="darkorange",
                     label="Boundary triggered")
    ax4.fill_between(t, 0, proj_sig, step="mid", alpha=0.6, color="tomato",
                     label="Projection applied")
    ax4.set_xlabel("Time step")
    ax4.set_ylabel("Event")
    ax4.set_title("Governance Events Timeline")
    ax4.set_ylim(-0.1, 1.4)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle("Modular System Evolution Dashboard", fontsize=14, fontweight="bold")
    return fig


def plot_propagation_heatmap(
    history: np.ndarray, title: str = "Propagation Heatmap"
) -> plt.Figure:
    """Heatmap of module activations over time from a propagation test."""
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(history.T, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Module")
    plt.colorbar(im, ax=ax, label="Activation")
    plt.tight_layout()
    return fig


def plot_learning_curve(losses: List[float], title: str = "Training Progress") -> plt.Figure:
    """Log-scale MSE loss curve over training epochs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses, linewidth=2, marker="o", markersize=5, color="steelblue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
