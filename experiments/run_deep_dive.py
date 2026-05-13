"""Deep mathematical interrogation of the full modular system.

Runs all four analysis classes and saves figures to outputs/figures/.

Usage:
    python experiments/run_deep_dive.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.constants import Config512D
from system.modular_system import ModularFieldSystem
from analysis.spectral import EigenmodeAnalyzer, eigenmode_report
from analysis.flow import InformationFlowAnalysis
from analysis.stability import StabilityAnalysis
from analysis.learning import LearningAnalysis
from viz.spectral_plots import (
    plot_eigenspectrum, plot_eigenmode_structure,
    plot_lyapunov_spectrum, plot_participation_ratio,
    plot_mode_simulation, plot_stability_sweep,
)
from viz.dynamics_plots import plot_propagation_heatmap

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def save(fig: plt.Figure, name: str) -> None:
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


def run_complete_analysis() -> dict:
    print("\n" + "=" * 70)
    print("COMPREHENSIVE DEEP ANALYSIS")
    print("=" * 70)

    cfg = Config512D()
    system = ModularFieldSystem(cfg)

    # ------------------------------------------------------------------
    # 1. Eigenmode analysis
    # ------------------------------------------------------------------
    print("\n[1/4] Eigenmode Analysis")
    print("-" * 40)
    analyzer = EigenmodeAnalyzer(system.kernel.K)
    eigenvalues, eigenvectors = analyzer.decompose()

    print("\n  Top 5 dominant modes:")
    for i in range(5):
        analyzer.analyze_mode(i)

    save(plot_eigenspectrum(analyzer),               "eigenspectrum.png")
    save(plot_eigenmode_structure(analyzer, n_modes=6, cfg=cfg), "eigenmode_structure.png")
    save(plot_lyapunov_spectrum(analyzer, n_modes=20), "lyapunov_spectrum.png")
    save(plot_participation_ratio(analyzer),          "participation_ratio.png")
    save(plot_mode_simulation(analyzer, 0, steps=200), "mode_0_simulation.png")

    print("\n  Energy landscape:")
    energy_info = analyzer.energy_landscape()

    print("\n  Degeneracy analysis:")
    deg_info = analyzer.degeneracy_analysis()

    # ------------------------------------------------------------------
    # 2. Information flow
    # ------------------------------------------------------------------
    print("\n[2/4] Information Flow Analysis")
    print("-" * 40)
    flow = InformationFlowAnalysis(system)

    print("\n  Propagation speed (10 trials):")
    speeds = flow.measure_propagation_speed(n_trials=10)

    print("\n  Propagation wavefront from module 0:")
    wavefront = flow.propagation_wavefront(source_module=0, steps=80)
    print(f"  Reached {wavefront['reached_count']}/{cfg.num_modules} modules")
    print(f"  Avg speed: {wavefront['avg_speed_modules_per_step']:.4f} modules/step")
    save(plot_propagation_heatmap(wavefront["history"], "Propagation Wavefront from Module 0"),
         "propagation_wavefront.png")

    print("\n  Module influence network:")
    influence_matrix = flow.report_influence()

    # ------------------------------------------------------------------
    # 3. Routing strategy comparison
    # ------------------------------------------------------------------
    print("\n[3/4] Routing Strategy Comparison")
    print("-" * 40)
    learning = LearningAnalysis(cfg)
    routing_results = learning.compare_routing_strategies(task="tanh", n_epochs=30, n_samples=30)

    # ------------------------------------------------------------------
    # 4. Stability sweep
    # ------------------------------------------------------------------
    print("\n[4/4] Stability Boundary Analysis")
    print("-" * 40)
    stability = StabilityAnalysis(cfg)
    coupling_vals, radii = stability.sweep_coupling(low=0.001, high=0.020, n_points=10)
    save(plot_stability_sweep(coupling_vals, radii), "stability_boundary.png")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nFigures saved to: {FIGURES_DIR}/")

    return {
        "eigenvalues":      eigenvalues,
        "eigenvectors":     eigenvectors,
        "energy_info":      energy_info,
        "degeneracy":       deg_info,
        "propagation":      wavefront,
        "speeds":           speeds,
        "influence_matrix": influence_matrix,
        "routing":          routing_results,
        "stability":        (coupling_vals, radii),
    }


if __name__ == "__main__":
    run_complete_analysis()
