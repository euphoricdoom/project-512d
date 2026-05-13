"""Sweep inter-module coupling and map the stability boundary.

Usage:
    python experiments/run_stability_sweep.py [--low 0.001] [--high 0.02] [--points 15]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.constants import Config512D
from analysis.stability import StabilityAnalysis
from viz.spectral_plots import plot_stability_sweep

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def main(args: argparse.Namespace) -> None:
    cfg = Config512D()
    stability = StabilityAnalysis(cfg)

    print("=" * 60)
    print("STABILITY SWEEP")
    print("=" * 60)
    print(f"Range: [{args.low}, {args.high}]  n_points={args.points}")

    coupling_vals, radii = stability.sweep_coupling(
        low=args.low, high=args.high, n_points=args.points
    )

    print("\nResults:")
    print(f"  {'inter_coupling':>18}  {'spectral_radius':>16}")
    for c, r in zip(coupling_vals, radii):
        stable = "" if r < 1.0 else "  <<< UNSTABLE"
        print(f"  {c:>18.5f}  {r:>16.6f}{stable}")

    cond = stability.condition_number()
    print(f"\nKernel condition number: {cond:.4f}")

    fig = plot_stability_sweep(coupling_vals, radii)
    path = os.path.join(FIGURES_DIR, "stability_sweep.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stability boundary sweep")
    parser.add_argument("--low",    type=float, default=0.001)
    parser.add_argument("--high",   type=float, default=0.020)
    parser.add_argument("--points", type=int,   default=12)
    main(parser.parse_args())
