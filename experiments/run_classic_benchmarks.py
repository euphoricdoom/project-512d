"""Classic continual-learning benchmarks.

The benchmark tasks are generated as sequence-shaped data, then encoded into
the fixed 64D input/output interface used by the kernel networks. This keeps
the benchmark standard enough to test memory, parity, and selective addition
while respecting the architecture's current contract.
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
from experiments.run_4_kernel_network import KernelNetwork
from experiments.tasks import (
    generate_adding_task,
    generate_copy_task,
    generate_parity_task,
)
from system.stable_network_readout import StableNetworkReadout
from system.task_specific_projection import TaskSpecificProjection
from system.helix_temporal import HelixTemporalAdapter
from system.adaptive_helix import AdaptiveHelixTemporalAdapter
from system.learned_gates import LearnedGateNetwork
from system.sleep_consolidation import sleep_consolidation


OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
CLASSIC_TASKS = ["copy_task", "parity_task", "adding_task"]


def _fit_width(values: np.ndarray, width: int) -> np.ndarray:
    """Pad or truncate a 2D array to ``width`` columns."""
    values = np.asarray(values, dtype=float)
    if values.shape[1] == width:
        return values
    out = np.zeros((values.shape[0], width), dtype=float)
    n = min(width, values.shape[1])
    out[:, :n] = values[:, :n]
    return out


def _fit_sequence_width(inputs: np.ndarray, width: int) -> np.ndarray:
    """Pad or truncate sequence feature width to the network input width."""
    inputs = np.asarray(inputs, dtype=float)
    if inputs.shape[2] == width:
        return inputs
    out = np.zeros((inputs.shape[0], inputs.shape[1], width), dtype=float)
    n = min(width, inputs.shape[2])
    out[:, :, :n] = inputs[:, :, :n]
    return out


def generate_classic_task_data(
    task_id: str,
    n_samples: int,
    seed: int,
    cfg: Config512D,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one benchmark task and encode it for the kernel network."""
    if task_id == "copy_task":
        x_seq, y_seq = generate_copy_task(
            n_samples=n_samples,
            seq_length=5,
            delay_length=4,
            vocab_size=4,
            seed=seed,
        )
        x = _fit_sequence_width(x_seq, cfg.input_dim)
        y = _fit_width(y_seq.reshape(n_samples, -1), cfg.output_dim)
        return x, y
    elif task_id == "parity_task":
        x_seq, y_seq = generate_parity_task(n_samples=n_samples, seq_length=4, seed=seed)
        x_seq = 2.0 * x_seq - 1.0
        x = _fit_sequence_width(x_seq, cfg.input_dim)
        y = np.zeros((n_samples, cfg.output_dim), dtype=float)
        y[:, :] = y_seq[:, -1, 0, None]
        return x, y
    elif task_id == "adding_task":
        x_seq, y_seq = generate_adding_task(n_samples=n_samples, seq_length=4, seed=seed)
        x = _fit_sequence_width(x_seq, cfg.input_dim)
        y = np.zeros((n_samples, cfg.output_dim), dtype=float)
        y[:, :] = y_seq[:, -1, 0, None]
        return x, y
    else:
        raise ValueError(f"Unknown classic task: {task_id}")


def build_network(
    num_kernels: int,
    cfg: Config512D,
    use_projection: bool,
    projection_dim: int,
    stable_base_lr: float,
    feature_mode: str,
    temporal_adapter: str,
    gate_weights: str | None = None,
    gate_decay_min: float = 0.0,
    gate_decay_max: float = 0.99,
) -> KernelNetwork:
    """Build a benchmark network for one configuration."""
    network = KernelNetwork(
        num_kernels,
        cfg,
        use_stable_readout=not use_projection,
        stable_base_lr=stable_base_lr,
        use_projection=use_projection,
        projection_dim=projection_dim,
    )
    network.classic_feature_mode = feature_mode
    network.temporal_adapter_mode = temporal_adapter
    network.helix_adapter = None
    if temporal_adapter in {"helix", "learned-helix"}:
        raw_dim = num_kernels * cfg.dim
        adapter_cls = (
            AdaptiveHelixTemporalAdapter
            if temporal_adapter == "learned-helix"
            else HelixTemporalAdapter
        )
        gate_network = None
        if temporal_adapter == "learned-helix":
            gate_network = LearnedGateNetwork(
                input_dim=cfg.input_dim,
                task_embedding_dim=128,
                num_dc_channels=projection_dim,
                feature_dim=projection_dim,
                decay_min=gate_decay_min,
                decay_max=gate_decay_max,
            )
            if gate_weights:
                print(f"Loading pretrained gates: {gate_weights}")
                gate_network.load(gate_weights)
            else:
                print("WARNING: using random learned gates; pass --gate-weights for distilled gates")
        adapter_kwargs = {"gate_network": gate_network} if gate_network is not None else {}
        network.helix_adapter = adapter_cls(
            input_dim=raw_dim,
            input_width=cfg.input_dim,
            projection_dim=projection_dim,
            seed=cfg.seed + 17000 + num_kernels,
            diagnostics=False,
            **adapter_kwargs,
        )
        network.feature_dim = network.helix_adapter.feature_dim
        network.network_readout = TaskSpecificProjection(
            num_kernels=1,
            base_feature_dim=network.helix_adapter.feature_dim,
            projection_dim=projection_dim,
            output_dim=cfg.output_dim,
            base_lr=stable_base_lr,
            rng=network.rng,
        )
        network.use_projection = True
        network.use_stable_readout = False
    elif feature_mode == "raw_state":
        network.feature_dim = num_kernels * cfg.dim
        if use_projection:
            network.network_readout = TaskSpecificProjection(
                num_kernels=num_kernels,
                base_feature_dim=cfg.dim,
                projection_dim=projection_dim,
                output_dim=cfg.output_dim,
                base_lr=stable_base_lr,
                rng=network.rng,
            )
            network.use_projection = True
            network.use_stable_readout = False
        else:
            network.network_readout = StableNetworkReadout(
                num_kernels=num_kernels,
                feature_dim=cfg.dim,
                output_dim=cfg.output_dim,
                base_lr=stable_base_lr,
                rng=network.rng,
            )
            network.use_projection = False
            network.use_stable_readout = True
    return network


def task_metric(
    task_id: str,
    predictions: np.ndarray,
    targets: np.ndarray,
    loss_threshold: float,
) -> dict:
    """Compute task-specific accuracy plus generic regression errors."""
    errors = targets - predictions
    loss = float(np.mean(errors**2))
    mae = float(np.mean(np.abs(errors)))

    if task_id == "copy_task":
        sample_loss = np.mean(errors**2, axis=1)
        accuracy = float(np.mean(sample_loss < loss_threshold))
    elif task_id == "parity_task":
        pred_bits = np.mean(predictions, axis=1) >= 0.5
        true_bits = np.mean(targets, axis=1) >= 0.5
        accuracy = float(np.mean(pred_bits == true_bits))
    elif task_id == "adding_task":
        tolerance = 0.15
        accuracy = float(
            np.mean(np.abs(np.mean(predictions, axis=1) - np.mean(targets, axis=1)) <= tolerance)
        )
    else:
        sample_loss = np.mean(errors**2, axis=1)
        accuracy = float(np.mean(sample_loss < loss_threshold))

    return {
        "loss": loss,
        "accuracy": accuracy,
        "mean_absolute_error": mae,
    }


def evaluate_network(
    network: KernelNetwork,
    task_id: str,
    inputs: np.ndarray,
    targets: np.ndarray,
    loss_threshold: float,
) -> dict:
    """Evaluate a network on encoded task data."""
    predictions, _ = process_sequence_batch(network, inputs, task_id)
    return task_metric(task_id, predictions, targets, loss_threshold)


def process_sequence_batch(
    network: KernelNetwork,
    sequences: np.ndarray,
    task_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Process a batch of sequences while preserving state across timesteps."""
    start = time.perf_counter()
    sequences = np.asarray(sequences, dtype=float)
    batch_size, seq_length, input_dim = sequences.shape
    if input_dim != network.cfg.input_dim:
        sequences = _fit_sequence_width(sequences, network.cfg.input_dim)

    network.network_readout.set_task(task_id)
    if getattr(network, "helix_adapter", None) is not None:
        if (
            getattr(network, "temporal_adapter_mode", None) == "learned-helix"
            and getattr(network.helix_adapter, "configured_task_id", None) != task_id
        ):
            support = sequences[: min(10, batch_size)]
            network.helix_adapter.configure_for_task(support, task_id=task_id)
        network.helix_adapter.reset(batch_size)
    states = []
    prev_states = []
    for kernel in network.kernels:
        state, prev = kernel.system.reset_state_batch(batch_size)
        states.append(state)
        prev_states.append(prev)

    for seq_t in range(seq_length):
        input_t = sequences[:, seq_t, :]
        weights = network.route_batch(input_t, task_id)
        for kernel_idx, kernel in enumerate(network.kernels):
            states[kernel_idx], _ = kernel.system.inject_batch(
                input_t * weights[:, kernel_idx, None],
                states[kernel_idx],
            )
        for step_t in range(network.cfg.process_steps):
            for kernel_idx, kernel in enumerate(network.kernels):
                states[kernel_idx], prev_states[kernel_idx] = kernel.system.step_batch(
                    states[kernel_idx], prev_states[kernel_idx]
                )
            if (step_t + 1) % 5 == 0:
                signals = [
                    kernel.system.predict_batch_states(states[kernel_idx])
                    for kernel_idx, kernel in enumerate(network.kernels)
                ]
                for i, kernel in enumerate(network.kernels):
                    incoming = np.zeros((batch_size, network.cfg.input_dim))
                    for j, signal in enumerate(signals):
                        if i != j:
                            incoming += network.topology[i, j] * signal
                    states[i], _ = kernel.system.inject_batch(incoming, states[i])

        if getattr(network, "helix_adapter", None) is not None:
            raw_features_t = np.concatenate(states, axis=1)
            network.helix_adapter.step(
                input_t,
                raw_features_t,
                t=seq_t,
                total_steps=seq_length,
                task_id=task_id,
            )

    if getattr(network, "helix_adapter", None) is not None:
        features = network.helix_adapter.final_features()
    elif getattr(network, "classic_feature_mode", "scoreboard") == "raw_state":
        features = np.concatenate(states, axis=1)
    else:
        features = np.concatenate(
            [
                kernel.system.features_batch_states(states[kernel_idx])
                for kernel_idx, kernel in enumerate(network.kernels)
            ],
            axis=1,
        )
    predictions = network.network_readout.predict_batch(features, task_id)
    elapsed = time.perf_counter() - start
    network.timing["process_seconds"] += elapsed
    network.timing["batch_process_seconds"] += elapsed
    return predictions, features


def train_network_on_task(
    network: KernelNetwork,
    inputs: np.ndarray,
    targets: np.ndarray,
    task_id: str,
    epochs: int,
    batch_size: int,
    rng: np.random.Generator,
) -> dict:
    """Train one task with mini-batched network updates."""
    start = time.time()
    epoch_losses = []
    n_samples = len(inputs)
    for epoch in range(epochs):
        order = rng.permutation(n_samples)
        losses = []
        for batch_start in range(0, n_samples, batch_size):
            idx = order[batch_start:batch_start + batch_size]
            pred, features = process_sequence_batch(network, inputs[idx], task_id)
            errors = targets[idx] - pred
            if network.use_projection:
                loss = network.network_readout.update_batch(
                    features, targets[idx], task_id=task_id
                )
            elif network.use_stable_readout:
                loss = network.network_readout.update_batch(
                    features, errors, task_id=task_id, as_error=True
                )
            else:
                loss = network.network_readout.update_batch(
                    features, errors, network.cfg.readout_lr, task_id
                )
            losses.append(loss)
        epoch_losses.append(float(np.mean(losses)))
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch + 1}/{epochs}: loss={epoch_losses[-1]:.6f}")
    return {
        "final_loss": epoch_losses[-1],
        "training_time": time.time() - start,
        "epoch_losses": epoch_losses,
    }


def run_sequential_learning_experiment(
    network: KernelNetwork,
    datasets: dict[str, tuple[np.ndarray, np.ndarray]],
    task_order: list[str],
    config_name: str,
    args: argparse.Namespace,
) -> dict:
    """Train tasks sequentially and measure post-task forgetting."""
    print("\n" + "=" * 70)
    print(f"Configuration: {config_name}")
    print("=" * 70)
    rng = np.random.default_rng(args.seed + 3100)
    train_data = {}
    test_data = {}
    for task_id, (x, y) in datasets.items():
        split = int(0.8 * len(x))
        train_data[task_id] = (x[:split], y[:split])
        test_data[task_id] = (x[split:], y[split:])

    result = {
        "config_name": config_name,
        "num_kernels": network.num_kernels,
        "use_projection": network.use_projection,
        "tasks": {},
        "task_order": task_order,
    }

    for task_idx, task_id in enumerate(task_order):
        print(f"\n  [{task_idx + 1}/{len(task_order)}] Training: {task_id}")
        train_x, train_y = train_data[task_id]
        training = train_network_on_task(
            network,
            train_x,
            train_y,
            task_id,
            args.epochs,
            args.batch_size,
            rng,
        )
        sleep_metrics = None
        if getattr(args, "sleep_cycles", 0) > 0:
            sleep_metrics = sleep_consolidation(
                network,
                task_id,
                train_x,
                train_y,
                num_cycles=args.sleep_cycles,
                sleep_lr=args.sleep_lr,
                batch_size=args.batch_size,
                verbose=True,
                seed=args.seed + 8100 + task_idx,
            )
        test_x, test_y = test_data[task_id]
        evaluation = evaluate_network(
            network, task_id, test_x, test_y, args.loss_threshold
        )
        result["tasks"][task_id] = {
            "training": training,
            "sleep_metrics": sleep_metrics,
            "evaluation": evaluation,
            "forgetting_measurements": [],
            "learned_at_step": task_idx + 1,
        }
        print(
            f"    Eval loss={evaluation['loss']:.6f}, "
            f"accuracy={evaluation['accuracy']:.1%}"
        )

        for prev_task in task_order[:task_idx]:
            prev_x, prev_y = test_data[prev_task]
            current = evaluate_network(
                network, prev_task, prev_x, prev_y, args.loss_threshold
            )
            original = result["tasks"][prev_task]["evaluation"]
            forgetting = current["accuracy"] - original["accuracy"]
            loss_drift = current["loss"] - original["loss"]
            result["tasks"][prev_task]["forgetting_measurements"].append({
                "measured_after_task": task_id,
                "accuracy": current["accuracy"],
                "loss": current["loss"],
                "accuracy_forgetting": forgetting,
                "loss_drift": loss_drift,
            })
            print(
                f"    Previous {prev_task}: acc={current['accuracy']:.1%}, "
                f"acc_drift={forgetting:+.4f}, loss_drift={loss_drift:+.6f}"
            )

    accuracies = [
        result["tasks"][task]["evaluation"]["accuracy"]
        for task in task_order
    ]
    accuracy_drifts = [
        measurement["accuracy_forgetting"]
        for task in task_order
        for measurement in result["tasks"][task]["forgetting_measurements"]
    ]
    loss_drifts = [
        measurement["loss_drift"]
        for task in task_order
        for measurement in result["tasks"][task]["forgetting_measurements"]
    ]
    result["summary"] = {
        "mean_accuracy": float(np.mean(accuracies)),
        "min_accuracy": float(np.min(accuracies)),
        "tasks_above_90pct": int(np.sum(np.asarray(accuracies) > 0.90)),
        "mean_forgetting": float(np.mean(accuracy_drifts)) if accuracy_drifts else 0.0,
        "max_forgetting": float(abs(np.min(accuracy_drifts))) if accuracy_drifts else 0.0,
        "mean_loss_drift": float(np.mean(loss_drifts)) if loss_drifts else 0.0,
        "max_loss_drift": float(np.max(loss_drifts)) if loss_drifts else 0.0,
        "forgetting_below_1pct": (
            not accuracy_drifts or abs(float(np.min(accuracy_drifts))) < 0.01
        ),
        "dynamics_time": float(network.timing["process_seconds"]),
        "resource_usage": network.resource_usage(),
    }
    print(
        f"\n  Summary: mean_acc={result['summary']['mean_accuracy']:.1%}, "
        f"tasks>90={result['summary']['tasks_above_90pct']}/{len(task_order)}, "
        f"max_forgetting={result['summary']['max_forgetting']:.4f}"
    )
    return result


def run(args: argparse.Namespace) -> dict:
    """Run the full classic benchmark matrix."""
    cfg = Config512D(
        seed=args.seed,
        train_epochs=args.epochs,
        train_samples=args.samples,
        process_steps=args.steps,
    )
    datasets = {
        task: generate_classic_task_data(task, args.samples, args.seed + i, cfg)
        for i, task in enumerate(CLASSIC_TASKS)
    }
    configs = []
    for kernels in args.kernels:
        use_projection = kernels >= args.projection_from
        name = f"{kernels}_kernels" + ("_projection" if use_projection else "")
        network = build_network(
            kernels,
            cfg,
            use_projection=use_projection,
            projection_dim=args.projection_dim,
            stable_base_lr=args.stable_base_lr,
            feature_mode=args.feature_mode,
            temporal_adapter=args.temporal_adapter,
            gate_weights=getattr(args, "gate_weights", None),
            gate_decay_min=getattr(args, "gate_decay_min", 0.0),
            gate_decay_max=getattr(args, "gate_decay_max", 0.99),
        )
        configs.append(
            run_sequential_learning_experiment(
                network, datasets, CLASSIC_TASKS, name, args
            )
        )

    payload = {
        "experiment": "classic_benchmarks",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tasks": CLASSIC_TASKS,
        "task_order": CLASSIC_TASKS,
        "settings": {
            "kernels": args.kernels,
            "epochs": args.epochs,
            "samples": args.samples,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "projection_dim": args.projection_dim,
            "projection_from": args.projection_from,
            "stable_base_lr": args.stable_base_lr,
            "feature_mode": args.feature_mode,
            "temporal_adapter": args.temporal_adapter,
            "sleep_cycles": args.sleep_cycles,
            "sleep_lr": args.sleep_lr,
        },
        "configs": configs,
    }
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    path = getattr(args, "output", None) or os.path.join(OUTPUTS_DIR, "classic_benchmarks.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("\n" + "=" * 70)
    print(f"Saved results to {path}")
    print("=" * 70)
    for config in configs:
        summary = config["summary"]
        print(
            f"{config['config_name']}: mean_acc={summary['mean_accuracy']:.1%}, "
            f"tasks>90={summary['tasks_above_90pct']}/3, "
            f"max_forgetting={summary['max_forgetting']:.4f}"
        )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classic benchmark tasks")
    parser.add_argument("--kernels", type=str, default="1,4,16,64")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--loss-threshold", type=float, default=0.2)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--projection-from", type=int, default=64)
    parser.add_argument("--stable-base-lr", type=float, default=0.04)
    parser.add_argument("--feature-mode", choices=["scoreboard", "raw_state"], default="raw_state")
    parser.add_argument(
        "--adapter",
        dest="temporal_adapter",
        choices=["none", "helix", "learned-helix"],
        help="Alias for --temporal-adapter",
    )
    parser.add_argument(
        "--temporal-adapter",
        choices=["none", "helix", "learned-helix"],
        default="helix",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gate-weights",
        type=str,
        default=None,
        help="Path to pretrained gate weights for --temporal-adapter learned-helix",
    )
    parser.add_argument("--gate-decay-min", type=float, default=0.0)
    parser.add_argument("--gate-decay-max", type=float, default=0.99)
    parser.add_argument("--sleep-cycles", type=int, default=0)
    parser.add_argument("--sleep-lr", type=float, default=0.001)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    args.kernels = [int(k.strip()) for k in args.kernels.split(",") if k.strip()]
    return args


if __name__ == "__main__":
    run(parse_args())
