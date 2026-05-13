"""Kernel network scaling law experiment.

Tests capacity versus kernel count for K in {1, 2, 4, 8, 16}.

This script intentionally keeps two metrics separate:
- retention capacity: how many sequential tasks can be retained
- accuracy capacity: how many tasks stay under the configured loss threshold

In accuracy mode, strict capacity stops when either the current task loss
exceeds the threshold or average forgetting exceeds the threshold. In retention
mode, capacity stops only when forgetting exceeds the threshold.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.constants import Config512D
from core.kernel import mse
from experiments.hierarchical_kernel_network import HierarchicalKernelNetwork
from experiments.run_4_kernel_network import KernelNetwork


OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")

TASK_SUITE = [
    "tanh", "smooth", "cumsum", "edge",
    "relu", "sigmoid", "abs", "square",
    "sin", "cos", "exp", "log",
    "sqrt", "cube", "neg", "clip",
    "shift", "scale", "flip", "noise",
    "threshold", "saturate", "normalize", "quantize",
    "sin_abs", "exp_clip", "tanh_square", "relu_shift",
    "sigmoid_scale", "log_abs", "sqrt_clip", "cos_square",
]


def generate_task_function(task_name: str) -> Callable[[np.ndarray], np.ndarray]:
    funcs: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "tanh": lambda x: np.tanh(2.0 * x),
        "smooth": lambda x: np.convolve(x, np.ones(5) / 5, mode="same"),
        "cumsum": lambda x: np.tanh(np.cumsum(x * 0.12)),
        "edge": lambda x: np.tanh(np.r_[0.0, np.abs(x[1:] - x[:-1])]),
        "relu": lambda x: np.maximum(0.0, x),
        "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-2.0 * x)),
        "abs": lambda x: np.abs(x),
        "square": lambda x: np.tanh(x * x),
        "sin": lambda x: np.sin(x),
        "cos": lambda x: np.cos(x),
        "exp": lambda x: np.exp(np.clip(x, -2.0, 1.0)) / np.e,
        "log": lambda x: np.log(np.abs(x) + 1.0),
        "sqrt": lambda x: np.sqrt(np.abs(x)),
        "cube": lambda x: np.tanh(x ** 3),
        "neg": lambda x: -x,
        "clip": lambda x: np.clip(x, -0.5, 0.5),
        "shift": lambda x: x + 0.25,
        "scale": lambda x: x * 1.5,
        "flip": lambda x: -x[::-1],
        "noise": lambda x: x + 0.05 * np.sin(np.arange(x.size)),
        "threshold": lambda x: (x > 0).astype(float),
        "saturate": lambda x: np.tanh(x * 5.0),
        "normalize": lambda x: (x - np.mean(x)) / (np.std(x) + 1e-6),
        "quantize": lambda x: np.round(x * 4.0) / 4.0,
        "sin_abs": lambda x: np.abs(np.sin(x)),
        "exp_clip": lambda x: np.clip(np.exp(x), 0.0, 2.0) / 2.0,
        "tanh_square": lambda x: np.tanh(x) ** 2,
        "relu_shift": lambda x: np.maximum(0.0, x - 0.25),
        "sigmoid_scale": lambda x: 1.0 / (1.0 + np.exp(-3.0 * x)),
        "log_abs": lambda x: np.log(np.abs(x) + 1.0),
        "sqrt_clip": lambda x: np.sqrt(np.clip(np.abs(x), 0.0, 1.0)),
        "cos_square": lambda x: np.cos(x) ** 2,
    }
    if task_name not in funcs:
        raise ValueError(f"Unknown task: {task_name}")
    return funcs[task_name]


def generate_task_data(
    cfg: Config512D,
    task_name: str,
    n_samples: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    X = rng.normal(0.0, 0.55, size=(n_samples, cfg.input_dim))
    fn = generate_task_function(task_name)
    Y = np.array([fn(x[:cfg.output_dim]) for x in X])
    return X, Y


class ScalingExperiment:
    def __init__(
        self,
        kernel_counts: List[int],
        tasks: List[str],
        epochs: int,
        samples: int,
        eval_samples: int,
        forgetting_threshold: float,
        loss_threshold: float,
        seed: int,
        steps: int,
        mode: str,
        batch_size: int,
        eval_every: int,
        architecture: str,
        use_stable_readout: bool = True,
        stable_base_lr: float = 0.01,
        use_projection: bool = False,
        projection_dim: int = 256,
    ) -> None:
        self.kernel_counts = kernel_counts
        self.tasks = tasks
        self.epochs = epochs
        self.samples = samples
        self.eval_samples = eval_samples
        self.forgetting_threshold = forgetting_threshold
        self.loss_threshold = loss_threshold
        self.seed = seed
        self.steps = steps
        self.mode = mode
        self.batch_size = max(1, batch_size)
        self.eval_every = max(1, eval_every)
        self.architecture = architecture
        self.use_stable_readout = use_stable_readout
        self.stable_base_lr = stable_base_lr
        self.use_projection = use_projection
        self.projection_dim = projection_dim
        self.results: dict[int, dict] = {}
        self.scaling_formula: dict = {}

    def run(self) -> dict[int, dict]:
        print("=" * 70)
        print("KERNEL NETWORK SCALING LAW EXPERIMENT")
        print("=" * 70)
        print(f"Kernel counts: {self.kernel_counts}")
        print(f"Architecture: {self.architecture}")
        print(f"Stable readout: {self.use_stable_readout} (base lr={self.stable_base_lr})")
        print(f"Projection readout: {self.use_projection} (dim={self.projection_dim})")
        print(f"Task suite: {len(self.tasks)} tasks")
        print(f"Mode: {self.mode}")
        print(f"Training: {self.epochs} epochs x {self.samples} samples")
        print(f"Batch size: {self.batch_size}")
        print(f"Evaluate previous tasks every {self.eval_every} task(s)")
        if self.mode == "retention":
            print(f"Stopping: forgetting > {self.forgetting_threshold:.4f}")
        else:
            print(
                f"Stopping: forgetting > {self.forgetting_threshold:.4f} "
                f"OR current loss > {self.loss_threshold:.4f}"
            )

        for k in self.kernel_counts:
            print("\n" + "=" * 70)
            print(f"Testing {k}-kernel network")
            print("=" * 70)
            self.results[k] = self._test_kernel_count(k)
            r = self.results[k]
            print(f"\n{k}-kernel network: {r['capacity']} strict-capacity tasks")
            print(f"  Retained tasks: {r['retained_tasks']}")
            print(f"  Avg forgetting: {r['avg_forgetting']:.6f}")
            print(f"  Time: {r['total_time']:.1f}s")
            print(f"  Dynamics time: {r['dynamics_time']:.1f}s")

        self._analyze_results()
        self._save_results()
        self._plot_results()
        return self.results

    def _test_kernel_count(self, k: int) -> dict:
        start = time.time()
        cfg = Config512D(
            seed=self.seed,
            train_epochs=self.epochs,
            train_samples=self.samples,
            process_steps=self.steps,
        )
        network = self._build_network(k, cfg)
        rng = np.random.default_rng(self.seed + 1000 + k)

        strict_capacity = 0
        retained_tasks = 0
        task_losses_after: dict[str, list[float]] = {task: [] for task in self.tasks}
        baseline_losses: dict[str, float] = {}
        task_train_losses: dict[str, float] = {}
        stopped_reason = "completed"
        seen: list[str] = []

        for task_idx, task in enumerate(self.tasks):
            print(f"\n  [{task_idx + 1}/{len(self.tasks)}] Training: {task}")
            X, Y = generate_task_data(cfg, task, self.samples, rng)
            losses = []
            for epoch in range(self.epochs):
                order = rng.permutation(self.samples)
                epoch_losses = []
                for batch_start in range(0, self.samples, self.batch_size):
                    idx = order[batch_start:batch_start + self.batch_size]
                    if self.batch_size == 1:
                        epoch_losses.append(network.train_sample(X[idx[0]], Y[idx[0]], task))
                    else:
                        epoch_losses.append(network.train_batch(X[idx], Y[idx], task))
                losses.append(float(np.mean(epoch_losses)))
                if (epoch + 1) % 5 == 0 or epoch == self.epochs - 1:
                    print(f"    Epoch {epoch + 1}/{self.epochs}: loss={losses[-1]:.6f}")

            final_train_loss = losses[-1]
            task_train_losses[task] = final_train_loss
            seen.append(task)

            eval_previous = (task_idx == 0) or ((task_idx + 1) % self.eval_every == 0)
            for prev_task in seen:
                if prev_task == task or eval_previous:
                    eval_loss = self._evaluate(network, cfg, prev_task, self.seed + 2000)
                    task_losses_after[prev_task].append(eval_loss)
                    if prev_task not in baseline_losses:
                        baseline_losses[prev_task] = eval_loss

            current_eval_loss = task_losses_after[task][-1]
            forgetting_values = [
                task_losses_after[t][-1] - baseline_losses[t]
                for t in seen[:-1]
            ]
            avg_forgetting_now = (
                float(np.mean(forgetting_values)) if forgetting_values else 0.0
            )
            print(f"    Eval loss: {current_eval_loss:.6f}")
            print(f"    Forgetting check: {avg_forgetting_now:.6f}")

            if current_eval_loss <= self.loss_threshold:
                retained_tasks += 1

            if self.mode == "accuracy" and current_eval_loss > self.loss_threshold:
                stopped_reason = (
                    f"loss {current_eval_loss:.6f} > threshold {self.loss_threshold}"
                )
                print(f"    STOP: {stopped_reason}")
                break

            if avg_forgetting_now > self.forgetting_threshold:
                stopped_reason = (
                    f"forgetting {avg_forgetting_now:.6f} > threshold "
                    f"{self.forgetting_threshold}"
                )
                print(f"    STOP: {stopped_reason}")
                break

            strict_capacity += 1
            if self.mode == "retention":
                print(f"    Task retained (retention capacity now {strict_capacity})")
            else:
                print(f"    Task accepted (strict capacity now {strict_capacity})")

        learned = [task for task in seen if task_losses_after[task][-1] <= self.loss_threshold]
        forgetting = {
            task: (
                float(task_losses_after[task][-1] - baseline_losses[task])
                if task in baseline_losses and task_losses_after[task]
                else 0.0
            )
            for task in seen
        }
        avg_forgetting = float(np.mean(list(forgetting.values()))) if forgetting else 0.0

        return {
            "capacity": strict_capacity,
            "retained_tasks": len(learned),
            "seen_tasks": seen,
            "learned_tasks": learned,
            "avg_forgetting": avg_forgetting,
            "task_train_losses": task_train_losses,
            "task_losses_after": {
                task: values for task, values in task_losses_after.items() if values
            },
            "task_forgetting": forgetting,
            "total_time": time.time() - start,
            "dynamics_time": float(network.timing["process_seconds"]),
            "batch_dynamics_time": float(network.timing["batch_process_seconds"]),
            "params": network.resource_usage()["total_params"],
            "resource_usage": network.resource_usage(),
            "stopped_reason": stopped_reason,
        }

    def _build_network(self, k: int, cfg: Config512D) -> KernelNetwork:
        if self.architecture == "hierarchical":
            return HierarchicalKernelNetwork(
                k,
                cfg,
                use_stable_readout=self.use_stable_readout,
                stable_base_lr=self.stable_base_lr,
                use_projection=self.use_projection,
                projection_dim=self.projection_dim,
            )
        return KernelNetwork(
            k,
            cfg,
            use_stable_readout=self.use_stable_readout,
            stable_base_lr=self.stable_base_lr,
            use_projection=self.use_projection,
            projection_dim=self.projection_dim,
        )

    def _evaluate(
        self,
        network: KernelNetwork,
        cfg: Config512D,
        task: str,
        seed: int,
    ) -> float:
        rng = np.random.default_rng(seed)
        X, Y = generate_task_data(cfg, task, self.eval_samples, rng)
        losses = [mse(network.predict(x, task), y) for x, y in zip(X, Y)]
        return float(np.mean(losses))

    def _analyze_results(self) -> None:
        ks = np.array(sorted(self.results.keys()), dtype=float)
        capacities = np.array([self.results[int(k)]["capacity"] for k in ks], dtype=float)
        if len(ks) >= 2:
            alpha, beta = np.polyfit(ks, capacities, 1)
            y_pred = alpha * ks + beta
            ss_res = float(np.sum((capacities - y_pred) ** 2))
            ss_tot = float(np.sum((capacities - np.mean(capacities)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        else:
            alpha, beta, r2 = 0.0, float(capacities[0]) if len(capacities) else 0.0, 0.0

        self.scaling_formula = {
            "alpha": float(alpha),
            "beta": float(beta),
            "r_squared": float(r2),
            "formula": f"C(K) = {alpha:.2f}K + {beta:.2f}",
        }

        print("\n" + "=" * 70)
        print("SCALING LAW ANALYSIS")
        print("=" * 70)
        print(f"Capacity formula: {self.scaling_formula['formula']}")
        print(f"R^2: {r2:.4f}")
        print(
            f"{'Kernels':>8} {'Capacity':>10} {'Retained':>10} {'Forget':>12} "
            f"{'Time':>10} {'Dyn':>10} {'Params':>10}"
        )
        print("-" * 82)
        for k in sorted(self.results):
            r = self.results[k]
            print(
                f"{k:>8} {r['capacity']:>10} {r['retained_tasks']:>10} "
                f"{r['avg_forgetting']:>12.6f} {r['total_time']:>10.1f} "
                f"{r['dynamics_time']:>10.1f} {r['params']:>10,}"
            )

    def _save_results(self) -> None:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "kernel_counts": self.kernel_counts,
            "tasks": self.tasks,
            "epochs": self.epochs,
            "samples": self.samples,
            "eval_samples": self.eval_samples,
            "forgetting_threshold": self.forgetting_threshold,
            "loss_threshold": self.loss_threshold,
            "mode": self.mode,
            "architecture": self.architecture,
            "stable_readout": self.use_stable_readout,
            "stable_base_lr": self.stable_base_lr,
            "projection_readout": self.use_projection,
            "projection_dim": self.projection_dim,
            "batch_size": self.batch_size,
            "eval_every": self.eval_every,
            "results": {str(k): v for k, v in self.results.items()},
            "scaling_formula": self.scaling_formula,
        }
        with open(os.path.join(OUTPUTS_DIR, "scaling_law.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        with open(os.path.join(OUTPUTS_DIR, "scaling_law.csv"), "w", newline="", encoding="utf-8") as f:
            fields = [
                "kernels",
                "capacity",
                "retained_tasks",
                "avg_forgetting",
                "total_time",
                "dynamics_time",
                "batch_dynamics_time",
                "params",
                "stopped_reason",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for k in sorted(self.results):
                r = self.results[k]
                writer.writerow({field: (k if field == "kernels" else r[field]) for field in fields})

        lines = [
            "# Kernel Network Scaling Law",
            "",
            f"Formula: `{self.scaling_formula['formula']}`",
            f"R^2: `{self.scaling_formula['r_squared']:.4f}`",
            "",
            f"Mode: `{self.mode}`",
            f"Architecture: `{self.architecture}`",
            f"Stable readout: `{self.use_stable_readout}`",
            f"Stable base lr: `{self.stable_base_lr}`",
            f"Projection readout: `{self.use_projection}`",
            f"Projection dim: `{self.projection_dim}`",
            f"Steps: `{self.steps}`",
            f"Batch size: `{self.batch_size}`",
            f"Eval every: `{self.eval_every}`",
            "",
            "| kernels | strict capacity | retained tasks | avg forgetting | total time | dynamics time | params | stopped reason |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for k in sorted(self.results):
            r = self.results[k]
            lines.append(
                f"| {k} | {r['capacity']} | {r['retained_tasks']} | "
                f"{r['avg_forgetting']:.6f} | {r['total_time']:.1f} | "
                f"{r['dynamics_time']:.1f} | {r['params']} | {r['stopped_reason']} |"
            )
        with open(os.path.join(OUTPUTS_DIR, "scaling_law.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _plot_results(self) -> None:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        ks = np.array(sorted(self.results.keys()), dtype=float)
        capacities = np.array([self.results[int(k)]["capacity"] for k in ks], dtype=float)
        retained = np.array([self.results[int(k)]["retained_tasks"] for k in ks], dtype=float)
        forgettings = np.array([self.results[int(k)]["avg_forgetting"] for k in ks], dtype=float)
        times = np.array([self.results[int(k)]["total_time"] for k in ks], dtype=float)
        params = np.array([self.results[int(k)]["params"] for k in ks], dtype=float)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax = axes[0, 0]
        ax.plot(ks, capacities, "o-", linewidth=2, label="Strict capacity")
        ax.plot(ks, retained, "s--", linewidth=2, label="Retained under loss threshold")
        if len(ks) >= 2:
            k_line = np.linspace(float(np.min(ks)), float(np.max(ks)), 100)
            ax.plot(
                k_line,
                self.scaling_formula["alpha"] * k_line + self.scaling_formula["beta"],
                ":",
                label=self.scaling_formula["formula"],
            )
        ax.set_xlabel("Kernels")
        ax.set_ylabel("Tasks")
        ax.set_title(f"Capacity Scaling (R^2={self.scaling_formula['r_squared']:.4f})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(ks, np.maximum(forgettings, 1e-9), "o-", color="darkorange")
        ax.axhline(self.forgetting_threshold, color="tomato", linestyle="--")
        ax.set_yscale("log")
        ax.set_xlabel("Kernels")
        ax.set_ylabel("Average forgetting")
        ax.set_title("Forgetting vs Network Size")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(retained, params, "^-", color="seagreen")
        ax.set_xlabel("Retained tasks")
        ax.set_ylabel("Parameters")
        ax.set_title("Parameters vs Retained Tasks")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.plot(ks, times, "D-", color="purple")
        ax.set_xlabel("Kernels")
        ax.set_ylabel("Seconds")
        ax.set_title("Training Time")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "scaling_law.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


def run(args: argparse.Namespace) -> dict[int, dict]:
    kernel_counts = [int(k.strip()) for k in args.kernels.split(",") if k.strip()]
    tasks = TASK_SUITE[:args.num_tasks]
    experiment = ScalingExperiment(
        kernel_counts=kernel_counts,
        tasks=tasks,
        epochs=args.epochs,
        samples=args.samples,
        eval_samples=args.eval_samples,
        forgetting_threshold=args.forgetting_threshold,
        loss_threshold=args.loss_threshold,
        seed=args.seed,
        steps=args.steps,
        mode=args.mode,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        architecture=args.architecture,
        use_stable_readout=args.stable_readout,
        stable_base_lr=args.stable_base_lr,
        use_projection=args.use_projection,
        projection_dim=args.projection_dim,
    )
    return experiment.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kernel network scaling law")
    parser.add_argument("--kernels", type=str, default="1,2,4,8,16")
    parser.add_argument("--num-tasks", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--eval-samples", type=int, default=30)
    parser.add_argument("--forgetting-threshold", type=float, default=0.01)
    parser.add_argument("--loss-threshold", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=1)
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
    parser.add_argument(
        "--architecture",
        choices=["flat", "hierarchical"],
        default="flat",
    )
    parser.add_argument(
        "--mode",
        choices=["accuracy", "retention"],
        default="accuracy",
        help="accuracy stops on loss or forgetting; retention stops only on forgetting",
    )
    run(parser.parse_args())
