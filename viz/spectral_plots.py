from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from analysis.spectral import EigenmodeAnalyzer
from core.constants import Config512D


def plot_eigenspectrum(analyzer: EigenmodeAnalyzer) -> plt.Figure:
    """Two-panel: eigenvalues in the complex plane + magnitude histogram."""
    eigvals = analyzer.eigenvalues
    assert eigvals is not None, "Call analyzer.decompose() first"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.scatter(eigvals.real, eigvals.imag, alpha=0.5, s=15, color="steelblue")
    theta = np.linspace(0, 2 * np.pi, 200)
    ax1.plot(np.cos(theta), np.sin(theta), "r--", linewidth=2, label="Unit circle")
    ax1.axhline(0, color="k", linewidth=0.5, alpha=0.3)
    ax1.axvline(0, color="k", linewidth=0.5, alpha=0.3)
    ax1.set_xlabel("Real")
    ax1.set_ylabel("Imaginary")
    ax1.set_title("Eigenvalue Distribution (Complex Plane)")
    ax1.set_aspect("equal")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    magnitudes = np.abs(eigvals)
    ax2.hist(magnitudes, bins=50, alpha=0.7, edgecolor="black", color="steelblue")
    ax2.axvline(1.0, color="r", linestyle="--", linewidth=2, label="Stability threshold")
    ax2.set_xlabel("|λ|")
    ax2.set_ylabel("Count")
    ax2.set_title("Eigenvalue Magnitude Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_eigenmode_structure(
    analyzer: EigenmodeAnalyzer,
    n_modes: int = 6,
    cfg: Optional[Config512D] = None,
) -> plt.Figure:
    """Grid of heatmaps showing the spatial structure of the top n_modes."""
    assert analyzer.eigenvectors is not None, "Call analyzer.decompose() first"

    num_modules = 60 if cfg is None else cfg.num_modules
    module_size = 8 if cfg is None else cfg.module_size
    dim_f = num_modules * module_size

    n_cols = 3
    n_rows = (n_modes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes_flat = np.array(axes).flat

    for i in range(n_modes):
        ax = next(axes_flat)
        v = np.abs(analyzer.eigenvectors[:dim_f, i])
        v_blocks = v.reshape(num_modules, module_size)
        im = ax.imshow(v_blocks.T, aspect="auto", cmap="hot", interpolation="nearest")
        ax.set_xlabel("Module ID")
        ax.set_ylabel("Dim within module")
        lam = analyzer.eigenvalues[i]
        ax.set_title(f"Mode {i}: λ={lam:.4f}")
        plt.colorbar(im, ax=ax)

    for ax in axes_flat:
        ax.set_visible(False)

    plt.suptitle("Eigenmode Spatial Structure (Field Space)", fontsize=14)
    plt.tight_layout()
    return fig


def plot_lyapunov_spectrum(
    analyzer: EigenmodeAnalyzer, n_modes: int = 20
) -> plt.Figure:
    """Bar chart of Lyapunov exponents (log|lambda|), colored by stability."""
    lyapunov = analyzer.lyapunov_exponents(n_modes)

    colors = ["seagreen" if L < 0 else "tomato" if L > 0 else "orange" for L in lyapunov]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(lyapunov)), lyapunov, color=colors, alpha=0.8, edgecolor="black")
    ax.axhline(0, color="black", linestyle="--", linewidth=2, label="Stability threshold")
    ax.set_xlabel("Mode index")
    ax.set_ylabel("Lyapunov exponent  log|λ|")
    ax.set_title("Lyapunov Spectrum — Stability Landscape")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


def plot_participation_ratio(analyzer: EigenmodeAnalyzer) -> plt.Figure:
    """Cumulative variance explained curve with 90/95/99% reference lines."""
    data = analyzer.participation_ratio()
    cumsum = data["cumsum"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cumsum[:100], linewidth=2, marker="o", markersize=3, color="steelblue")
    ax.axhline(0.99, color="tomato",   linestyle="--", alpha=0.7, label="99%")
    ax.axhline(0.95, color="darkorange", linestyle="--", alpha=0.7, label="95%")
    ax.axhline(0.90, color="seagreen", linestyle="--", alpha=0.7, label="90%")
    ax.set_xlabel("Number of modes")
    ax.set_ylabel("Cumulative variance explained")
    ax.set_title("Effective Dimensionality")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_mode_simulation(
    analyzer: EigenmodeAnalyzer, mode_idx: int, steps: int = 100
) -> plt.Figure:
    """Simulated vs theoretical decay of a pure eigenmode."""
    simulated, theoretical = analyzer.simulate_mode(mode_idx, steps)
    lam = analyzer.eigenvalues[mode_idx]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(simulated,   linewidth=2, label="Simulated")
    ax.plot(theoretical, "--", linewidth=2, alpha=0.7, label=f"Theoretical |λ|^t")
    ax.set_xlabel("Time steps")
    ax.set_ylabel("||X(t)||")
    ax.set_title(f"Pure Eigenmode {mode_idx} Decay  (λ = {lam:.4f})")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_stability_sweep(
    coupling_values: np.ndarray, spectral_radii: List[float]
) -> plt.Figure:
    """Spectral radius vs coupling strength with stability threshold line."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(coupling_values, spectral_radii, "o-", linewidth=2, markersize=8, color="steelblue")
    ax.axhline(1.0, color="tomato", linestyle="--", linewidth=2, label="Stability threshold")
    ax.set_xlabel("Inter-coupling strength")
    ax.set_ylabel("Spectral radius")
    ax.set_title("Stability Boundary vs Coupling Strength")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
