"""Entry point: python -m project512d --experiment <name> [experiment args]

Experiments:
  basic_demo          Single-node system: spectral analysis, training, visualization
  stability_sweep     Sweep inter-module coupling vs spectral radius
  deep_dive           Full analysis pipeline (eigenmodes, flow, routing, stability)
  ablation_cockpit    Compare sparse/full routing and fixed/learnable K_inter
  governance_stress   Stress-test no/fixed/adaptive constraint projection
  kernel_network      Compose atomic kernels into a small-world network
  scaling_law         Measure capacity scaling across kernel counts

Examples:
  python -m project512d --experiment basic_demo --task tanh --epochs 45
  python -m project512d --experiment ablation_cockpit
  python -m project512d --experiment stability_sweep
"""
from __future__ import annotations

import argparse
import runpy
import sys


EXPERIMENTS = {
    "basic_demo":         "experiments.run_basic_demo",
    "stability_sweep":    "experiments.run_stability_sweep",
    "deep_dive":          "experiments.run_deep_dive",
    "ablation_cockpit":   "experiments.run_ablation_cockpit",
    "governance_stress":  "experiments.run_governance_stress",
    "kernel_network":     "experiments.run_4_kernel_network",
    "scaling_law":        "experiments.run_scaling_law",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m project512d",
        description="Project 512D — kernel on kernel dynamical field system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
        add_help=True,
    )
    parser.add_argument(
        "--experiment", "-e",
        choices=list(EXPERIMENTS.keys()),
        required=True,
        metavar="NAME",
        help=f"Experiment to run: {{{', '.join(EXPERIMENTS)}}}"
    )
    # capture --experiment, leave remainder for the experiment's own argparse
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    runpy.run_module(EXPERIMENTS[args.experiment], run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
