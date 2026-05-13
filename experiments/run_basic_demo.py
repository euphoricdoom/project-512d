"""Basic system demo: build, run, and visualize the modular field system.

Usage:
    python experiments/run_basic_demo.py [--task tanh] [--epochs 45] [--seed 42] [--no-plots]
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
from system.modular_system import ModularFieldSystem
from system.trainer import ModularTrainer
from analysis.spectral import eigenmode_report
from analysis.flow import analyze_propagation
from viz.module_plots import plot_module_activations, plot_routing_weights
from viz.dynamics_plots import plot_propagation_heatmap, plot_learning_curve

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def run_demo(args: argparse.Namespace) -> None:
    cfg = Config512D(
        seed=args.seed,
        train_epochs=args.epochs,
        train_samples=args.samples,
    )
    system = ModularFieldSystem(cfg)

    print("=" * 72)
    print("MODULAR FIELD SYSTEM")
    print("=" * 72)
    print(f"Dimensions:        {cfg.dim} = {cfg.dim_f} field + {cfg.dim_c} constraint")
    print(f"Modules:           {cfg.num_modules} x {cfg.module_size}")
    print(f"Kernel radius:     {system.kernel.radius:.6f}")
    print(f"Inter coupling:    {cfg.inter_coupling}")
    print(f"Readout lr:        {cfg.readout_lr}")
    print()

    # Eigenmode summary
    er = eigenmode_report(system)
    print("Eigenmode report:")
    print(f"  spectral radius: {er['spectral_radius']:.6f}")
    print(f"  real modes:      {er['real_modes']}")
    print(f"  complex modes:   {er['complex_modes']}")
    print(f"  top |λ|:         {[round(float(x), 5) for x in er['top_abs'][:5]]}")
    print()

    # Propagation test
    print("Propagation test from module 0...")
    prop = analyze_propagation(system, source_module=0, steps=90)
    print(f"  reached:  {prop['reached_count']}/{cfg.num_modules} modules")
    print(f"  speed:    {prop['avg_speed_modules_per_step']:.4f} modules/step")
    print(f"  max act:  {prop['max_activation']:.6f}")
    print()

    # Training
    print(f"Training task: {args.task}")
    trainer = ModularTrainer(system)
    result = trainer.train(args.task, verbose=True)
    print()
    improvement = 100.0 * (result["losses"][0] - result["losses"][-1]) / max(result["losses"][0], 1e-12)
    print("Training result:")
    print(f"  initial loss:  {result['losses'][0]:.6f}")
    print(f"  final loss:    {result['losses'][-1]:.6f}")
    print(f"  test loss:     {result['test_loss']:.6f}")
    print(f"  improvement:   {improvement:.1f}%")
    print()

    # Module activations
    print("Top 10 active modules after training:")
    for row in system.module_output_table()[:10]:
        print(f"  Module {row['id']:02d} | {row['expertise']:<22} | {row['activation']:.6f}")
    print()

    # Trace summary
    print("Trace summary:")
    print(system.trace.summary())

    if not args.no_plots:
        figs = {
            "propagation.png":   plot_propagation_heatmap(prop["history"], "Propagation Test"),
            "learning_curve.png": plot_learning_curve(result["losses"], f"Readout Learning: {args.task}"),
            "routing_weights.png": plot_routing_weights(result["route_means"], system),
            "module_activations.png": plot_module_activations(system),
        }
        for name, fig in figs.items():
            path = os.path.join(FIGURES_DIR, name)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modular Field System demo")
    parser.add_argument("--task",     choices=["tanh", "smooth", "cumsum", "edge"], default="tanh")
    parser.add_argument("--epochs",   type=int,   default=45)
    parser.add_argument("--samples",  type=int,   default=120)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--no-plots", action="store_true")
    run_demo(parser.parse_args())
