"""4-kernel network experiment.

Builds a small-world network of four atomic modular kernels and tests whether
composition increases continual-learning capacity.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.constants import Config512D
from core.kernel import mse, softmax
from core.readout import module_features
from system.modular_system import ModularFieldSystem
from system.multi_readout import MultiTaskReadout
from system.trainer import make_dataset

try:
    from system.stable_network_readout import StableNetworkReadout
except ImportError:
    StableNetworkReadout = None
try:
    from system.task_specific_projection import TaskSpecificProjection
except ImportError:
    TaskSpecificProjection = None


OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
TASKS_8 = ["tanh", "smooth", "cumsum", "edge", "relu", "sigmoid", "abs", "square"]


def make_extended_dataset(
    cfg: Config512D,
    task: str,
    rng: np.random.Generator,
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if task in {"tanh", "smooth", "cumsum", "edge"}:
        return make_dataset(cfg, task, rng, n)

    X = rng.normal(0.0, 0.55, size=(n, cfg.input_dim))
    base = X[:, :cfg.output_dim]
    if task == "relu":
        Y = np.maximum(base, 0.0)
    elif task == "sigmoid":
        Y = 1.0 / (1.0 + np.exp(-2.0 * base))
    elif task == "abs":
        Y = np.abs(base)
    elif task == "square":
        Y = np.tanh(base * base)
    else:
        raise ValueError(f"Unknown task: {task}")
    return X, Y


class AtomicKernel:
    """Thin network-ready wrapper around ModularFieldSystem."""

    INPUT_DIM = 64
    OUTPUT_DIM = 64
    STATE_DIM = 540

    def __init__(self, kernel_id: int, base_cfg: Config512D) -> None:
        self.kernel_id = kernel_id
        cfg = Config512D(
            seed=base_cfg.seed + kernel_id * 1000,
            train_epochs=base_cfg.train_epochs,
            train_samples=base_cfg.train_samples,
            process_steps=base_cfg.process_steps,
            learnable_inter_modules=base_cfg.learnable_inter_modules,
            learnable_constraint_feedback=base_cfg.learnable_constraint_feedback,
        )
        self.system = ModularFieldSystem(cfg)
        self.cfg = cfg
        self.frozen = False

    def receive(self, signal: np.ndarray) -> None:
        self.system.inject(signal)

    def step(self) -> None:
        self.system.step()

    def broadcast(self) -> np.ndarray:
        return self.system.predict()

    def get_features(self) -> np.ndarray:
        self.system.state[self.cfg.dim_f:] = self.system.current_module_norms()
        return module_features(self.system.state, self.cfg)

    def reset(self) -> None:
        self.system.reset_state()

    def freeze(self) -> None:
        """Freeze shared kernel learning while keeping dynamics active."""
        self.frozen = True
        self.system.freeze()

    def unfreeze(self) -> None:
        """Resume shared kernel learning."""
        self.frozen = False
        self.system.unfreeze()

    def is_frozen(self) -> bool:
        """Return whether this kernel is frozen for consolidation."""
        return self.frozen or self.system.is_frozen()

    def get_spectral_radius(self) -> float:
        return float(self.system.kernel.radius)

    def get_resource_usage(self) -> Dict[str, int]:
        base_params = (
            self.system.router.size
            + sum(m.specialization_weights.size for m in self.system.modules)
            + self.system.kernel._U.size
            + self.system.kernel._V.size
            + self.system.governance.stored_params()
        )
        return {
            "base_params": int(base_params),
            "state_bytes_float32": int(self.cfg.dim * 4),
        }


class KernelNetwork:
    """Network of atomic kernels with small-world signal exchange."""

    def __init__(
        self,
        num_kernels: int,
        cfg: Config512D,
        use_stable_readout: bool = True,
        stable_base_lr: float = 0.01,
        use_projection: bool = False,
        projection_dim: int = 256,
    ) -> None:
        self.cfg = cfg
        self.num_kernels = num_kernels
        self.rng = np.random.default_rng(cfg.seed + 7000)
        self.kernels = [AtomicKernel(i, cfg) for i in range(num_kernels)]
        self.topology = self._build_small_world_topology(num_kernels)
        self.kernel_router = self.rng.normal(0.0, 0.01, size=(num_kernels, cfg.input_dim))
        self.feature_dim = num_kernels * (cfg.num_modules + cfg.dim_c)
        self.use_projection = bool(use_projection and TaskSpecificProjection is not None)
        self.use_stable_readout = bool(
            use_stable_readout and StableNetworkReadout is not None and not self.use_projection
        )
        self.stable_base_lr = float(stable_base_lr)
        self.projection_dim = int(projection_dim)
        if self.use_projection:
            self.network_readout = TaskSpecificProjection(
                num_kernels=num_kernels,
                base_feature_dim=cfg.num_modules + cfg.dim_c,
                projection_dim=projection_dim,
                output_dim=cfg.output_dim,
                base_lr=stable_base_lr,
                rng=self.rng,
            )
        elif use_stable_readout and StableNetworkReadout is not None:
            self.network_readout = StableNetworkReadout(
                num_kernels=num_kernels,
                feature_dim=cfg.num_modules + cfg.dim_c,
                output_dim=cfg.output_dim,
                base_lr=stable_base_lr,
                rng=self.rng,
            )
        else:
            self.network_readout = MultiTaskReadout(cfg.output_dim, self.feature_dim, self.rng)
        self.timing = {"process_seconds": 0.0, "batch_process_seconds": 0.0}

    def _build_small_world_topology(self, n: int) -> np.ndarray:
        W = np.zeros((n, n))
        for i in range(n):
            W[i, (i - 1) % n] = 0.10
            W[i, (i + 1) % n] = 0.10
            if n > 2:
                W[i, (i + 2) % n] = 0.05
        np.fill_diagonal(W, 0.0)
        return W

    def reset_all(self) -> None:
        for kernel in self.kernels:
            kernel.reset()

    def route(self, input_vec: np.ndarray, task_id: str | None = None) -> np.ndarray:
        return softmax(self.kernel_router @ input_vec)

    def process(self, input_vec: np.ndarray, task_id: str, steps: int | None = None) -> np.ndarray:
        start = time.perf_counter()
        if steps is None:
            steps = self.cfg.process_steps
        self.network_readout.set_task(task_id)
        weights = self.route(input_vec, task_id)
        for kernel, weight in zip(self.kernels, weights):
            kernel.receive(input_vec * weight)
        for t in range(steps):
            for kernel in self.kernels:
                kernel.step()
            if (t + 1) % 5 == 0:
                self._exchange_signals()
        pred = self.network_readout.predict(self.get_features(), task_id)
        self.timing["process_seconds"] += time.perf_counter() - start
        return pred

    def _exchange_signals(self) -> None:
        signals = [kernel.broadcast() for kernel in self.kernels]
        for i, kernel in enumerate(self.kernels):
            incoming = np.zeros(self.cfg.input_dim)
            for j, signal in enumerate(signals):
                if i != j:
                    incoming += self.topology[i, j] * signal
            kernel.receive(incoming)

    def get_features(self) -> np.ndarray:
        return np.concatenate([kernel.get_features() for kernel in self.kernels])

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

        features = np.concatenate(
            [
                kernel.system.features_batch_states(states[kernel_idx])
                for kernel_idx, kernel in enumerate(self.kernels)
            ],
            axis=1,
        )
        pred = self.network_readout.predict_batch(features, task_id)
        elapsed = time.perf_counter() - start
        self.timing["process_seconds"] += elapsed
        self.timing["batch_process_seconds"] += elapsed
        return pred, features

    def route_batch(self, input_vecs: np.ndarray, task_id: str | None = None) -> np.ndarray:
        scores = input_vecs @ self.kernel_router.T
        z = scores - np.max(scores, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / (np.sum(exp_z, axis=1, keepdims=True) + 1e-12)

    def train_sample(self, x: np.ndarray, y: np.ndarray, task_id: str) -> float:
        self.reset_all()
        pred = self.process(x, task_id)
        err = y - pred
        loss = mse(pred, y)
        if self.use_projection:
            self.network_readout.update(self.get_features(), y, task_id=task_id)
        elif self.use_stable_readout:
            self.network_readout.update(self.get_features(), err, task_id=task_id, as_error=True)
        else:
            self.network_readout.update(self.get_features(), err, self.cfg.readout_lr, task_id)
        return loss

    def train_batch(self, x: np.ndarray, y: np.ndarray, task_id: str) -> float:
        pred, features = self.process_batch(x, task_id)
        errors = y - pred
        losses = np.mean((pred - y) ** 2, axis=1)
        if self.use_projection:
            self.network_readout.update_batch(features, y, task_id=task_id)
        elif self.use_stable_readout:
            self.network_readout.update_batch(features, errors, task_id=task_id, as_error=True)
        else:
            self.network_readout.update_batch(
                features,
                errors,
                self.cfg.readout_lr,
                task_id,
            )
        return float(np.mean(losses))

    def predict(self, x: np.ndarray, task_id: str) -> np.ndarray:
        self.reset_all()
        return self.process(x, task_id)

    def resource_usage(self) -> Dict[str, int]:
        base = sum(k.get_resource_usage()["base_params"] for k in self.kernels)
        readout_head_params = int(
            sum(w.size for w in getattr(self.network_readout, "readouts", {}).values())
        )
        readout_total_params = int(self.network_readout.stored_params())
        return {
            "kernel_base_params": int(base),
            "router_params": int(self.kernel_router.size),
            "network_readout_params": readout_head_params,
            "network_readout_total_params": readout_total_params,
            "total_params": int(base + self.kernel_router.size + readout_total_params),
            "stable_readout": self.use_stable_readout,
            "stable_base_lr": self.stable_base_lr,
            "projection_readout": self.use_projection,
            "projection_dim": self.projection_dim if self.use_projection else 0,
        }


def evaluate(network: KernelNetwork, task: str, n_samples: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    X, Y = make_extended_dataset(network.cfg, task, rng, n_samples)
    losses = [mse(network.predict(x, task), y) for x, y in zip(X, Y)]
    return float(np.mean(losses))


def benchmark_continual(network: KernelNetwork, tasks: List[str], args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed + 900)
    task_losses_after = {task: [] for task in tasks}
    training_curves = {}
    seen: List[str] = []

    for task in tasks:
        print(f"\n  [network] Training: {task}")
        X, Y = make_extended_dataset(network.cfg, task, rng, args.samples)
        losses = []
        for epoch in range(args.epochs):
            order = rng.permutation(args.samples)
            epoch_losses = []
            for idx in order:
                epoch_losses.append(network.train_sample(X[idx], Y[idx], task))
            losses.append(float(np.mean(epoch_losses)))
        training_curves[task] = losses
        seen.append(task)
        print(f"    final train loss: {losses[-1]:.6f}")
        for seen_task in seen:
            loss = evaluate(network, seen_task, args.eval_samples, args.seed + 10000)
            task_losses_after[seen_task].append(loss)
            print(f"      eval {seen_task:<8} loss={loss:.6f}")

    forgetting = {}
    for task in tasks:
        values = task_losses_after[task]
        forgetting[task] = float(values[-1] - values[0]) if len(values) >= 2 else 0.0
    old_tasks = [task for task in tasks if len(task_losses_after[task]) >= 2]
    avg_forgetting = float(np.mean([forgetting[t] for t in old_tasks])) if old_tasks else 0.0
    learned_tasks = [task for task in tasks if task_losses_after[task][-1] < args.loss_threshold]
    return {
        "tasks": tasks,
        "task_losses_after": task_losses_after,
        "forgetting": forgetting,
        "avg_forgetting": avg_forgetting,
        "learned_tasks": learned_tasks,
        "training_curves": training_curves,
    }


def analyze_kernel_specialization(network: KernelNetwork, tasks: List[str], path: str) -> np.ndarray:
    affinity = np.zeros((len(tasks), network.num_kernels))
    rng = np.random.default_rng(network.cfg.seed + 12000)
    for t_idx, task in enumerate(tasks):
        X, _ = make_extended_dataset(network.cfg, task, rng, 1)
        weights = network.route(X[0])
        for k_idx, kernel in enumerate(network.kernels):
            kernel.reset()
            kernel.receive(X[0] * weights[k_idx])
            for _ in range(network.cfg.process_steps):
                kernel.step()
            affinity[t_idx, k_idx] = float(np.linalg.norm(kernel.get_features()))

    fig, ax = plt.subplots(figsize=(8, max(4, len(tasks) * 0.45)))
    im = ax.imshow(affinity, cmap="hot", aspect="auto")
    ax.set_xlabel("Kernel ID")
    ax.set_ylabel("Task")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks)
    ax.set_title("Kernel Specialization Matrix")
    plt.colorbar(im, ax=ax, label="Activation")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return affinity


def write_outputs(result: dict, network: KernelNetwork, affinity: np.ndarray) -> None:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
        "resource_usage": network.resource_usage(),
        "topology": network.topology.tolist(),
        "affinity": affinity.tolist(),
    }
    with open(os.path.join(OUTPUTS_DIR, "kernel_network.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with open(os.path.join(OUTPUTS_DIR, "kernel_network.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "forgetting", "final_loss"])
        writer.writeheader()
        for task in result["tasks"]:
            writer.writerow({
                "task": task,
                "forgetting": result["forgetting"][task],
                "final_loss": result["task_losses_after"][task][-1],
            })

    lines = [
        "# 4-Kernel Network Report",
        "",
        f"Average forgetting: `{result['avg_forgetting']:.6f}`",
        f"Learned tasks under threshold: `{len(result['learned_tasks'])}/{len(result['tasks'])}`",
        "",
        "## Task Results",
        "",
        "| task | final loss | forgetting |",
        "| --- | ---: | ---: |",
    ]
    for task in result["tasks"]:
        lines.append(
            f"| {task} | {result['task_losses_after'][task][-1]:.6f} | "
            f"{result['forgetting'][task]:+.6f} |"
        )
    usage = network.resource_usage()
    lines.extend([
        "",
        "## Resources",
        "",
        f"- Stable readout: {network.use_stable_readout}",
        f"- Stable base lr: {network.stable_base_lr}",
        f"- Kernel base params: {usage['kernel_base_params']}",
        f"- Router params: {usage['router_params']}",
        f"- Network readout params: {usage['network_readout_params']}",
        f"- Network readout total params: {usage['network_readout_total_params']}",
        f"- Total params: {usage['total_params']}",
    ])
    with open(os.path.join(OUTPUTS_DIR, "kernel_network_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> dict:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    cfg = Config512D(
        seed=args.seed,
        train_epochs=args.epochs,
        train_samples=args.samples,
        process_steps=args.steps,
    )
    tasks = TASKS_8[:args.num_tasks]
    network = KernelNetwork(
        args.kernels,
        cfg,
        use_stable_readout=args.stable_readout,
        stable_base_lr=args.stable_base_lr,
        use_projection=args.use_projection,
        projection_dim=args.projection_dim,
    )
    result = benchmark_continual(network, tasks, args)
    affinity = analyze_kernel_specialization(
        network, tasks, os.path.join(FIGURES_DIR, "kernel_specialization.png")
    )
    write_outputs(result, network, affinity)

    print("\n" + "=" * 72)
    print("4-KERNEL NETWORK SUMMARY")
    print("=" * 72)
    print(f"Capacity under loss<{args.loss_threshold}: {len(result['learned_tasks'])}/{len(tasks)} tasks")
    print(f"Avg forgetting: {result['avg_forgetting']:.6f}")
    print(f"Resources: {network.resource_usage()}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="4-kernel network experiment")
    parser.add_argument("--kernels", type=int, default=4)
    parser.add_argument("--num-tasks", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--eval-samples", type=int, default=30)
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss-threshold", type=float, default=0.2)
    parser.add_argument(
        "--stable-readout",
        dest="stable_readout",
        action="store_true",
        default=True,
        help="use StableNetworkReadout when available (default)",
    )
    parser.add_argument(
        "--no-stable-readout",
        dest="stable_readout",
        action="store_false",
        help="use the legacy MultiTaskReadout",
    )
    parser.add_argument("--stable-base-lr", type=float, default=0.01)
    parser.add_argument("--use-projection", action="store_true")
    parser.add_argument("--projection-dim", type=int, default=256)
    run(parser.parse_args())
