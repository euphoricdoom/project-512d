"""2D Kernel Lattice continual-learning benchmark.

Trains all 8 tasks sequentially on the KernelLattice and measures:
  1. Per-task MSE (train/test)
  2. Zero-forgetting: revisit earlier tasks after training later ones
  3. Bridge activation patterns
  4. Specialization heatmap (feature norm per layer × stack)
  5. Local (single stack) vs global (full lattice) readout comparison

Results are saved to outputs/lattice_benchmark_{timestamp}/.

Usage::

    python experiments/run_lattice_benchmark.py
    python experiments/run_lattice_benchmark.py --n-train 300 --n-test 100
    python experiments/run_lattice_benchmark.py --fast          # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.constants import Config512D
from experiments.run_4_kernel_network import TASKS_8, make_extended_dataset
from system.kernel_lattice import KernelLattice, LatticeConfig

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def build_datasets(
    cfg: Config512D,
    rng: np.random.Generator,
    n_train: int,
    n_test: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Return {task_name: {"X_tr": ..., "Y_tr": ..., "X_te": ..., "Y_te": ...}}."""
    datasets: dict[str, dict[str, np.ndarray]] = {}
    for task in TASKS_8:
        X_tr, Y_tr = make_extended_dataset(cfg, task, rng, n_train)
        X_te, Y_te = make_extended_dataset(cfg, task, rng, n_test)
        datasets[task] = {"X_tr": X_tr, "Y_tr": Y_tr, "X_te": X_te, "Y_te": Y_te}
    return datasets


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    n_train: int = 500,
    n_test: int = 150,
    lr: float = 0.005,
    epochs: int = 3,
    seed: int = 42,
    fast: bool = False,
) -> dict:
    if fast:
        n_train, n_test, epochs = 80, 30, 1

    print("=" * 65)
    print("  2D Kernel Lattice — Continual Learning Benchmark")
    print("=" * 65)
    print(f"  Tasks  : {len(TASKS_8)}")
    print(f"  Train  : {n_train} samples / task")
    print(f"  Test   : {n_test} samples / task")
    print(f"  Epochs : {epochs}")
    print(f"  LR     : {lr}")
    print()

    cfg = Config512D(seed=seed)
    rng = np.random.default_rng(seed)

    lattice_cfg = LatticeConfig(
        n_layers=2,
        n_stacks=4,
        projection_dim=256,
        bridge_threshold=0.40,
        bridge_alpha=0.25,
        seed=seed,
    )
    lattice = KernelLattice(
        output_dim=cfg.output_dim,
        lattice_cfg=lattice_cfg,
        kernel_cfg=cfg,
    )

    print(f"  Global feature dim : {lattice_cfg.global_feature_dim:,}")
    print(f"  Readout heads      : {lattice_cfg.n_layers} layers × {lattice_cfg.n_stacks} stacks")
    print()

    datasets = build_datasets(cfg, rng, n_train, n_test)

    # ------------------------------------------------------------------
    # Phase 1: Sequential training
    # ------------------------------------------------------------------
    print("Phase 1: Sequential training")
    print("-" * 40)
    train_results: dict[str, dict] = {}
    task_order = list(TASKS_8)

    for task in task_order:
        d = datasets[task]
        t0 = time.time()
        # fit_task caches features once and runs mini-batch GD — converges faster
        # than train_sequence which recomputes features each epoch
        losses = lattice.fit_task(
            d["X_tr"], d["Y_tr"], task_id=task, lr=lr, epochs=epochs
        )
        elapsed = time.time() - t0
        test_metrics = lattice.evaluate(d["X_te"], d["Y_te"], task_id=task)
        final_train_mse = losses[-1]
        train_results[task] = {
            "train_mse": final_train_mse,
            "test_mse": test_metrics["mse"],
            "test_r2": test_metrics["r2"],
            "elapsed_s": round(elapsed, 2),
        }
        print(
            f"  {task:<10}  train_mse={final_train_mse:.4f}  "
            f"test_mse={test_metrics['mse']:.4f}  r2={test_metrics['r2']:.3f}  "
            f"({elapsed:.1f}s)"
        )

    # ------------------------------------------------------------------
    # Phase 2: Zero-forgetting check — evaluate ALL tasks after full training
    # ------------------------------------------------------------------
    print()
    print("Phase 2: Zero-forgetting check (evaluate all after full training)")
    print("-" * 40)
    forgetting_results: dict[str, dict] = {}
    max_delta = 0.0

    for task in task_order:
        d = datasets[task]
        post_metrics = lattice.evaluate(d["X_te"], d["Y_te"], task_id=task)
        baseline_mse = train_results[task]["test_mse"]
        delta = post_metrics["mse"] - baseline_mse
        max_delta = max(max_delta, abs(delta))
        forgetting_results[task] = {
            "baseline_mse": baseline_mse,
            "post_mse": post_metrics["mse"],
            "delta": round(delta, 6),
        }
        status = "OK" if abs(delta) < 1e-9 else "!!"
        print(
            f"  {status} {task:<10}  baseline={baseline_mse:.4f}  "
            f"post={post_metrics['mse']:.4f}  delta={delta:+.6f}"
        )

    zero_forgetting = max_delta < 1e-9
    print()
    if zero_forgetting:
        print("  PASS: Zero forgetting confirmed (max delta = 0.0)")
    else:
        print(f"  INFO: Max delta = {max_delta:.2e} (expected 0 by construction)")

    # ------------------------------------------------------------------
    # Phase 3: Bridge analysis
    # ------------------------------------------------------------------
    print()
    print("Phase 3: Bridge activation summary")
    print("-" * 40)
    bridge_info = lattice.bridge_summary()
    activations = bridge_info["bridge_activations"]
    hit_rates = bridge_info.get("bridge_hit_rates", {})
    if activations:
        for edge, count in sorted(activations.items(), key=lambda x: -x[1]):
            rate = hit_rates.get(edge, 0.0)
            print(f"  {edge:<30} activations={count}  hit_rate={rate:.2%}")
    else:
        print("  No bridge activations recorded.")

    # ------------------------------------------------------------------
    # Phase 4: Local vs Global comparison (both use fit_task for fair comparison)
    # ------------------------------------------------------------------
    print()
    print("Phase 4: Local (stack 0, layer 0) vs Global readout")
    print("-" * 40)
    from zero_forgetting import ZeroForgetReadout

    local_readout = ZeroForgetReadout(
        output_dim=cfg.output_dim,
        feature_dim=lattice_cfg.feature_dim_per_stack,
        rng=np.random.default_rng(seed + 1),
    )
    stack0 = lattice.layers[0].stacks[0]

    local_results: dict[str, float] = {}
    for task in task_order:
        d = datasets[task]
        n_tr = len(d["X_tr"])

        # Cache stack-0 features once
        F_local = np.zeros((n_tr, lattice_cfg.feature_dim_per_stack))
        for i in range(n_tr):
            F_local[i] = stack0.forward(d["X_tr"][i][None, :], task_id=task)

        # Mini-batch gradient descent on local features
        local_readout.set_task(task)
        head = local_readout._heads[task]
        local_rng = np.random.default_rng(seed + 2)
        for ep in range(epochs):
            idx = local_rng.permutation(n_tr)
            for start in range(0, n_tr, 32):
                b = idx[start:start + 32]
                preds_b = F_local[b] @ head.T
                err_b = preds_b - d["Y_tr"][b]
                head -= lr * (err_b.T @ F_local[b]) / len(b)
                np.clip(head, -local_readout.weight_clip, local_readout.weight_clip, out=head)

        preds = np.array([
            head @ stack0.forward(d["X_te"][i][None, :], task_id=task)
            for i in range(len(d["X_te"]))
        ])
        local_mse = float(np.mean((preds - d["Y_te"]) ** 2))
        local_results[task] = local_mse
        global_mse = train_results[task]["test_mse"]
        delta = global_mse - local_mse
        better = "global" if delta < 0 else "local"
        print(
            f"  {task:<10}  local={local_mse:.4f}  global={global_mse:.4f}  "
            f"better={better}"
        )

    # ------------------------------------------------------------------
    # Collate results
    # ------------------------------------------------------------------
    results = {
        "config": {
            "n_layers": lattice_cfg.n_layers,
            "n_stacks": lattice_cfg.n_stacks,
            "projection_dim": lattice_cfg.projection_dim,
            "global_feature_dim": lattice_cfg.global_feature_dim,
            "n_train": n_train,
            "n_test": n_test,
            "lr": lr,
            "epochs": epochs,
            "seed": seed,
        },
        "training": train_results,
        "forgetting": forgetting_results,
        "zero_forgetting": zero_forgetting,
        "max_forgetting_delta": max_delta,
        "bridges": bridge_info,
        "local_vs_global": {
            task: {
                "local_mse": local_results[task],
                "global_mse": train_results[task]["test_mse"],
            }
            for task in task_order
        },
        "task_summary": lattice.task_summary(),
    }

    return results, lattice, datasets, task_order


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_plots(
    results: dict,
    lattice: KernelLattice,
    datasets: dict,
    task_order: list[str],
    out_dir: Path,
) -> None:
    cfg = Config512D()
    rng = np.random.default_rng(42)
    n_tasks = len(task_order)
    lattice_cfg = lattice.lattice_cfg

    # 1. MSE bar chart
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(n_tasks)
    width = 0.35
    train_mses = [results["training"][t]["train_mse"] for t in task_order]
    test_mses = [results["training"][t]["test_mse"] for t in task_order]
    ax.bar(x - width / 2, train_mses, width, label="Train MSE")
    ax.bar(x + width / 2, test_mses, width, label="Test MSE")
    ax.set_xticks(x)
    ax.set_xticklabels(task_order, rotation=30, ha="right")
    ax.set_ylabel("MSE")
    ax.set_title("KernelLattice: Per-Task MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "mse_per_task.png", dpi=120)
    plt.close(fig)

    # 2. Forgetting delta chart
    fig, ax = plt.subplots(figsize=(10, 3))
    deltas = [results["forgetting"][t]["delta"] for t in task_order]
    colors = ["green" if abs(d) < 1e-9 else "red" for d in deltas]
    ax.bar(task_order, deltas, color=colors)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("MSE delta (post - baseline)")
    ax.set_title("Zero-Forgetting Check (green = 0 forgetting)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / "forgetting_check.png", dpi=120)
    plt.close(fig)

    # 3. Specialization heatmap (read-only — no side effects on readout heads)
    n_small = 30
    X_all = np.zeros((n_tasks, n_small, 1, cfg.input_dim))
    for ti, task in enumerate(task_order):
        d = datasets[task]
        X_all[ti, :, 0, :] = d["X_tr"][:n_small]

    heatmap = lattice.specialization_matrix(X_all, task_order, n_probe=n_small)
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(heatmap, aspect="auto", cmap="viridis")
    ax.set_xlabel("Stack index")
    ax.set_ylabel("Layer index")
    ax.set_title("Specialization Heatmap (mean feature norm)")
    ax.set_xticks(range(lattice_cfg.n_stacks))
    ax.set_yticks(range(lattice_cfg.n_layers))
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "specialization_heatmap.png", dpi=120)
    plt.close(fig)

    # 4. Local vs global bar chart
    fig, ax = plt.subplots(figsize=(10, 4))
    local_mses = [results["local_vs_global"][t]["local_mse"] for t in task_order]
    global_mses = [results["local_vs_global"][t]["global_mse"] for t in task_order]
    x = np.arange(n_tasks)
    ax.bar(x - width / 2, local_mses, width, label="Local (stack 0)")
    ax.bar(x + width / 2, global_mses, width, label="Global (all stacks)")
    ax.set_xticks(x)
    ax.set_xticklabels(task_order, rotation=30, ha="right")
    ax.set_ylabel("Test MSE")
    ax.set_title("Local vs Global Readout")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "local_vs_global.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="KernelLattice benchmark")
    parser.add_argument("--n-train", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true", help="Smoke test (small dataset)")
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib output")
    args = parser.parse_args()

    results, lattice, datasets, task_order = run_benchmark(
        n_train=args.n_train,
        n_test=args.n_test,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        fast=args.fast,
    )

    # Save outputs
    OUTPUTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUTS_DIR / f"lattice_benchmark_{ts}"
    out_dir.mkdir(exist_ok=True)

    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    if not args.no_plots:
        try:
            save_plots(results, lattice, datasets, task_order, out_dir)
            print(f"Plots saved to {out_dir}/")
        except Exception as e:
            print(f"[warn] Plot generation failed: {e}")

    print()
    print("=" * 65)
    zf = results["zero_forgetting"]
    delta = results["max_forgetting_delta"]
    print(f"  Zero forgetting : {'PASS' if zf else f'DELTA={delta:.2e}'}")
    mean_test_mse = np.mean([v["test_mse"] for v in results["training"].values()])
    print(f"  Mean test MSE   : {mean_test_mse:.4f}")
    print(f"  Output dir      : {out_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
