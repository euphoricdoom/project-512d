"""Selective feature experiment for high-kernel networks.

Hypothesis: at 64 kernels, a full concatenated readout dilutes task signal by
normalizing across many irrelevant kernels. This experiment keeps hierarchical
routing/communication, but trains each task readout on only its assigned domain
group features.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from core.constants import Config512D
from core.kernel import mse
from experiments.hierarchical_kernel_network import HierarchicalKernelNetwork
from experiments.run_scaling_law import TASK_SUITE, generate_task_data
from system.multi_readout import MultiTaskReadout
from system.stable_network_readout import StableNetworkReadout


OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


class SelectiveFeatureNetwork(HierarchicalKernelNetwork):
    """Hierarchical network that reads out only the task's assigned group."""

    def __init__(
        self,
        num_kernels: int,
        cfg: Config512D,
        num_domains: int = 4,
        use_stable_readout: bool = True,
        stable_base_lr: float = 0.04,
    ) -> None:
        group_size = max(1, int(np.ceil(num_kernels / num_domains)))
        super().__init__(
            num_kernels,
            cfg,
            group_size=group_size,
            use_stable_readout=use_stable_readout,
            stable_base_lr=stable_base_lr,
        )
        self.num_domains = len(self.groups)
        self.selective_group_size = max(len(group) for group in self.groups)
        self.selective_feature_dim = self.selective_group_size * (
            cfg.num_modules + cfg.dim_c
        )
        self.use_stable_readout = bool(use_stable_readout)
        if use_stable_readout:
            self.network_readout = StableNetworkReadout(
                num_kernels=self.selective_group_size,
                feature_dim=cfg.num_modules + cfg.dim_c,
                output_dim=cfg.output_dim,
                base_lr=stable_base_lr,
                rng=self.rng,
            )
        else:
            self.network_readout = MultiTaskReadout(
                cfg.output_dim, self.selective_feature_dim, self.rng
            )

    def _selected_group(self, task_id: str) -> List[int]:
        group_idx = self.task_group(task_id)
        if group_idx is None:
            group_idx = hash(task_id) % len(self.groups)
        return self.groups[group_idx]

    def _pad_group_features(self, features: List[np.ndarray]) -> np.ndarray:
        if len(features) < self.selective_group_size:
            batch_size = features[0].shape[0]
            width = self.cfg.num_modules + self.cfg.dim_c
            for _ in range(self.selective_group_size - len(features)):
                features.append(np.zeros((batch_size, width)))
        return np.concatenate(features, axis=1)

    def process_batch(
        self, input_vecs: np.ndarray, task_id: str, steps: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        start = time.perf_counter()
        if steps is None:
            steps = self.cfg.process_steps
        input_vecs = np.asarray(input_vecs, dtype=float)
        batch_size = input_vecs.shape[0]
        self.network_readout.set_task(task_id)

        weights = self.route_batch(input_vecs, task_id)
        states = []
        prev_states = []
        for kernel_idx, kernel in enumerate(self.kernels):
            state, prev = kernel.system.reset_state_batch(batch_size)
            state, _ = kernel.system.inject_batch(
                input_vecs * weights[:, kernel_idx, None], state
            )
            states.append(state)
            prev_states.append(prev)

        for t in range(steps):
            for kernel_idx, kernel in enumerate(self.kernels):
                states[kernel_idx], prev_states[kernel_idx] = kernel.system.step_batch(
                    states[kernel_idx], prev_states[kernel_idx]
                )
            if (t + 1) % 5 == 0:
                signals = [
                    kernel.system.predict_batch_states(states[kernel_idx])
                    for kernel_idx, kernel in enumerate(self.kernels)
                ]
                for i, kernel in enumerate(self.kernels):
                    incoming = np.zeros((batch_size, self.cfg.input_dim))
                    for j, signal in enumerate(signals):
                        if i != j:
                            incoming += self.topology[i, j] * signal
                    states[i], _ = kernel.system.inject_batch(incoming, states[i])

        selected = self._selected_group(task_id)
        features = self._pad_group_features(
            [
                self.kernels[kernel_idx].system.features_batch_states(states[kernel_idx])
                for kernel_idx in selected
            ]
        )
        pred = self.network_readout.predict_batch(features, task_id)
        elapsed = time.perf_counter() - start
        self.timing["process_seconds"] += elapsed
        self.timing["batch_process_seconds"] += elapsed
        return pred, features

    def train_batch(self, x: np.ndarray, y: np.ndarray, task_id: str) -> float:
        pred, features = self.process_batch(x, task_id)
        errors = y - pred
        if self.use_stable_readout:
            self.network_readout.update_batch(features, errors, task_id=task_id, as_error=True)
        else:
            self.network_readout.update_batch(features, errors, self.cfg.readout_lr, task_id)
        return float(np.mean((pred - y) ** 2))

    def predict_batch(self, x: np.ndarray, task_id: str) -> np.ndarray:
        pred, _ = self.process_batch(x, task_id)
        return pred

    def predict(self, x: np.ndarray, task_id: str) -> np.ndarray:
        return self.predict_batch(x[None, :], task_id)[0]

    def resource_usage(self) -> Dict[str, int]:
        usage = super().resource_usage()
        usage["selective_feature_dim"] = int(self.selective_feature_dim)
        usage["selective_group_size"] = int(self.selective_group_size)
        usage["num_domains"] = int(self.num_domains)
        usage["network_readout_params"] = int(
            sum(w.size for w in self.network_readout.readouts.values())
        )
        usage["network_readout_total_params"] = int(self.network_readout.stored_params())
        usage["total_params"] = (
            usage["kernel_base_params"]
            + usage["router_params"]
            + usage.get("group_router_params", 0)
            + usage["network_readout_total_params"]
        )
        return usage


def evaluate(
    network: SelectiveFeatureNetwork,
    cfg: Config512D,
    task: str,
    samples: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    x, y = generate_task_data(cfg, task, samples, rng)
    pred = network.predict_batch(x, task)
    return mse(pred, y)


def run_selective_test(args: argparse.Namespace) -> dict:
    cfg = Config512D(
        seed=args.seed,
        train_epochs=args.epochs,
        train_samples=args.samples,
        process_steps=args.steps,
    )
    tasks = TASK_SUITE[:args.tasks]
    network = SelectiveFeatureNetwork(
        args.kernels,
        cfg,
        num_domains=args.domains,
        use_stable_readout=not args.no_stable_readout,
        stable_base_lr=args.stable_base_lr,
    )
    rng = np.random.default_rng(args.seed + 5000)
    baselines: dict[str, float] = {}
    losses_after: dict[str, list[float]] = {task: [] for task in tasks}
    train_losses: dict[str, float] = {}
    seen: list[str] = []
    start = time.time()

    print("=" * 70)
    print("SELECTIVE FEATURE TEST")
    print("=" * 70)
    print(f"Kernels: {args.kernels}")
    print(f"Domains: {network.num_domains}")
    print(f"Selective group size: {network.selective_group_size}")
    print(f"Selective feature dim: {network.selective_feature_dim}")
    print(f"Stable base lr: {args.stable_base_lr}")

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
                loss = evaluate(network, cfg, prev_task, args.eval_samples, args.seed + 8000)
                losses_after[prev_task].append(loss)
                baselines.setdefault(prev_task, loss)
        current_loss = losses_after[task][-1]
        forgetting_values = [
            losses_after[t][-1] - baselines[t]
            for t in seen[:-1]
        ]
        avg_forgetting_now = float(np.mean(forgetting_values)) if forgetting_values else 0.0
        print(f"    Eval loss: {current_loss:.6f}")
        print(f"    Forgetting check: {avg_forgetting_now:.6f}")

    final_forgetting = {
        task: float(losses_after[task][-1] - baselines[task])
        for task in seen
    }
    final_losses = {
        task: float(losses_after[task][-1])
        for task in seen
    }
    accurate = [task for task, loss in final_losses.items() if loss <= args.loss_threshold]
    avg_forgetting = float(np.mean(list(final_forgetting.values()))) if final_forgetting else 0.0
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "architecture": "selective_features",
        "kernels": args.kernels,
        "tasks": tasks,
        "capacity": len(tasks),
        "accurate_tasks": accurate,
        "accurate_count": len(accurate),
        "avg_forgetting": avg_forgetting,
        "final_losses": final_losses,
        "forgetting": final_forgetting,
        "train_losses": train_losses,
        "total_time": time.time() - start,
        "dynamics_time": float(network.timing["process_seconds"]),
        "resource_usage": network.resource_usage(),
    }
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    path = os.path.join(OUTPUTS_DIR, "selective_features_test.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 70)
    print("SELECTIVE FEATURE SUMMARY")
    print("=" * 70)
    print(f"Capacity: {result['capacity']}/{len(tasks)}")
    print(f"Accurate (<{args.loss_threshold}): {len(accurate)}/{len(tasks)}")
    print(f"Avg forgetting: {avg_forgetting:.6f}")
    print(f"Time: {result['total_time']:.1f}s")
    print(f"Dynamics time: {result['dynamics_time']:.1f}s")
    print(f"Saved: {path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Selective feature readout test")
    parser.add_argument("--kernels", type=int, default=64)
    parser.add_argument("--tasks", type=int, default=32)
    parser.add_argument("--domains", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--eval-samples", type=int, default=30)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=4)
    parser.add_argument("--stable-base-lr", type=float, default=0.04)
    parser.add_argument("--loss-threshold", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-stable-readout", action="store_true")
    run_selective_test(parser.parse_args())
