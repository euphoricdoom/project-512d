"""Compare direct stable readout with task-specific projection readout."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def _run(cmd: list[str]) -> dict:
    subprocess.run(cmd, check=True)
    with open(os.path.join(OUTPUTS_DIR, "scaling_law.json"), encoding="utf-8") as f:
        return json.load(f)


def run_comparison(args: argparse.Namespace) -> dict:
    common = [
        sys.executable,
        "experiments/run_scaling_law.py",
        "--architecture", args.architecture,
        "--kernels", str(args.kernels),
        "--num-tasks", str(args.tasks),
        "--mode", "retention",
        "--steps", str(args.steps),
        "--batch-size", str(args.batch_size),
        "--eval-every", str(args.eval_every),
        "--stable-base-lr", str(args.stable_base_lr),
    ]

    print("=" * 70)
    print("PROJECTION VS DIRECT READOUT COMPARISON")
    print("=" * 70)

    print("\n--- Direct Stable Readout ---")
    direct = _run(common)
    direct_result = direct["results"][str(args.kernels)]

    print("\n--- Task-Specific Projection Readout ---")
    projection = _run(common + ["--use-projection", "--projection-dim", str(args.projection_dim)])
    projection_result = projection["results"][str(args.kernels)]

    payload = {
        "architecture": args.architecture,
        "kernels": args.kernels,
        "tasks": args.tasks,
        "direct": direct_result,
        "projection": projection_result,
        "accurate_improvement": (
            projection_result["retained_tasks"] - direct_result["retained_tasks"]
        ),
        "forgetting_delta": (
            projection_result["avg_forgetting"] - direct_result["avg_forgetting"]
        ),
    }
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    path = os.path.join(OUTPUTS_DIR, "projection_vs_direct.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print("Direct Readout:")
    print(f"  Capacity: {direct_result['capacity']}/{args.tasks}")
    print(f"  Accurate: {direct_result['retained_tasks']}/{args.tasks}")
    print(f"  Forgetting: {direct_result['avg_forgetting']:.6f}")
    print("\nProjection Readout:")
    print(f"  Capacity: {projection_result['capacity']}/{args.tasks}")
    print(f"  Accurate: {projection_result['retained_tasks']}/{args.tasks}")
    print(f"  Forgetting: {projection_result['avg_forgetting']:.6f}")
    print("\nImprovement:")
    print(f"  Accuracy: {payload['accurate_improvement']:+d} tasks")
    print(f"  Forgetting delta: {payload['forgetting_delta']:+.6f}")
    print(f"Saved: {path}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare projection and direct readouts")
    parser.add_argument("--architecture", choices=["flat", "hierarchical"], default="flat")
    parser.add_argument("--kernels", type=int, default=64)
    parser.add_argument("--tasks", type=int, default=32)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=4)
    parser.add_argument("--stable-base-lr", type=float, default=0.04)
    parser.add_argument("--projection-dim", type=int, default=256)
    run_comparison(parser.parse_args())
