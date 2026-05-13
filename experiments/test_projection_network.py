"""Task-specific projection network experiment.

This is a thin experiment wrapper around ``KernelNetwork`` that swaps the
network readout for ``TaskSpecificProjection``. Kernel dynamics, routing, and
communication stay unchanged; only the feature-to-output head changes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from core.constants import Config512D
from core.kernel import mse
from experiments.run_4_kernel_network import KernelNetwork
from experiments.run_scaling_law import TASK_SUITE, generate_task_data


OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


class ProjectionNetwork(KernelNetwork):
    """Kernel network using task-specific feature projections."""

    def __init__(
        self,
        num_kernels: int,
        cfg: Config512D,
        projection_dim: int = 256,
        base_lr: float = 0.01,
    ) -> None:
        super().__init__(
            num_kernels,
            cfg,
            use_stable_readout=False,
            stable_base_lr=base_lr,
            use_projection=True,
            projection_dim=projection_dim,
        )


def evaluate(network: KernelNetwork, cfg: Config512D, task: str, samples: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    x, y = generate_task_data(cfg, task, samples, rng)
    pred, _ = network.process_batch(x, task)
    return mse(pred, y)


def run_projection_test(args: argparse.Namespace) -> dict:
    cfg = Config512D(
        seed=args.seed,
        train_epochs=args.epochs,
        train_samples=args.samples,
        process_steps=args.steps,
    )
    tasks = TASK_SUITE[:args.tasks]
    network = ProjectionNetwork(
        args.kernels,
        cfg,
        projection_dim=args.projection_dim,
        base_lr=args.base_lr,
    )
    rng = np.random.default_rng(args.seed + 6100)
    baselines: dict[str, float] = {}
    losses_after: dict[str, list[float]] = {task: [] for task in tasks}
    train_losses: dict[str, float] = {}
    seen: list[str] = []
    start = time.time()

    print("=" * 70)
    print("TASK-SPECIFIC PROJECTION NETWORK")
    print("=" * 70)
    print(f"Kernels: {args.kernels}")
    print(f"Projection dim: {args.projection_dim}")
    print(f"Base lr: {args.base_lr}")

    for task_idx, task in enumerate(tasks):
        print(f"\n  [{task_idx + 1}/{len(tasks)}] Training: {task}")
        x, y = generate_task_data(cfg, task, args.samples, rng)
        epoch_losses = []
        for epoch in range(args.epochs):
            order = rng.permutation(args.samples)
            batch_losses = []
            for batch_start in range(0, args.samples, args.batch_size):
                idx = order[batch_start:batch_start + args.batch_size]
                batch_losses.append(network.train_batch(x[idx], y[idx], task))
            epoch_losses.append(float(np.mean(batch_losses)))
            if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
                print(f"    Epoch {epoch + 1}/{args.epochs}: loss={epoch_losses[-1]:.6f}")
        train_losses[task] = epoch_losses[-1]
        seen.append(task)

        eval_all = (task_idx == 0) or ((task_idx + 1) % args.eval_every == 0)
        for prev_task in seen:
            if prev_task == task or eval_all:
                loss = evaluate(network, cfg, prev_task, args.eval_samples, args.seed + 9100)
                losses_after[prev_task].append(loss)
                baselines.setdefault(prev_task, loss)
        current_loss = losses_after[task][-1]
        forgetting_now = [
            losses_after[t][-1] - baselines[t]
            for t in seen[:-1]
        ]
        avg_forgetting_now = float(np.mean(forgetting_now)) if forgetting_now else 0.0
        print(f"    Eval loss: {current_loss:.6f}")
        print(f"    Forgetting check: {avg_forgetting_now:.6f}")

    final_losses = {task: float(losses_after[task][-1]) for task in seen}
    forgetting = {
        task: float(losses_after[task][-1] - baselines[task])
        for task in seen
    }
    accurate = [task for task, loss in final_losses.items() if loss <= args.loss_threshold]
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "architecture": "projection_network",
        "kernels": args.kernels,
        "tasks": tasks,
        "capacity": len(tasks),
        "accurate_tasks": accurate,
        "accurate_count": len(accurate),
        "avg_forgetting": float(np.mean(list(forgetting.values()))) if forgetting else 0.0,
        "final_losses": final_losses,
        "forgetting": forgetting,
        "train_losses": train_losses,
        "total_time": time.time() - start,
        "dynamics_time": float(network.timing["process_seconds"]),
        "resource_usage": network.resource_usage(),
    }

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    path = os.path.join(OUTPUTS_DIR, "projection_network_test.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 70)
    print("PROJECTION NETWORK SUMMARY")
    print("=" * 70)
    print(f"Capacity: {result['capacity']}/{len(tasks)}")
    print(f"Accurate (<{args.loss_threshold}): {len(accurate)}/{len(tasks)}")
    print(f"Avg forgetting: {result['avg_forgetting']:.6f}")
    print(f"Time: {result['total_time']:.1f}s")
    print(f"Dynamics time: {result['dynamics_time']:.1f}s")
    print(f"Saved: {path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task-specific projection network test")
    parser.add_argument("--kernels", type=int, default=64)
    parser.add_argument("--tasks", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--eval-samples", type=int, default=30)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=4)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--base-lr", type=float, default=0.01)
    parser.add_argument("--loss-threshold", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    run_projection_test(parser.parse_args())
