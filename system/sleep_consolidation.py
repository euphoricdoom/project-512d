"""Sleep consolidation for frozen-feature replay.

The shared feature producer is frozen, a stable feature table is extracted once,
and only the task-owned readout is updated during replay. This keeps sleep
consolidation aligned with the repository's zero-forgetting contract: shared
features are treated as fixed, while each task refines its own head.
"""

from __future__ import annotations

from collections.abc import Callable
import time
from typing import Any

import numpy as np


FeatureExtractor = Callable[[Any, np.ndarray, str], tuple[np.ndarray, np.ndarray]]


def freeze_network(network: Any) -> list[Any]:
    """Freeze all kernel-like objects on a network and return the touched items."""
    kernels = list(getattr(network, "kernels", []))
    for kernel in kernels:
        if hasattr(kernel, "freeze"):
            kernel.freeze()
    return kernels


def unfreeze_network(kernels: list[Any]) -> None:
    """Unfreeze kernel-like objects previously returned by ``freeze_network``."""
    for kernel in kernels:
        if hasattr(kernel, "unfreeze"):
            kernel.unfreeze()


def _default_feature_extractor(network: Any, inputs: np.ndarray, task_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Use the classic benchmark sequence processor to extract features."""
    from experiments.run_classic_benchmarks import process_sequence_batch

    return process_sequence_batch(network, inputs, task_id)


def _readout_for(network: Any, readout: Any | None) -> Any:
    resolved = readout if readout is not None else getattr(network, "network_readout", None)
    if resolved is None:
        raise AttributeError("sleep consolidation needs a readout/head object")
    return resolved


def _predict(readout: Any, features: np.ndarray, task_id: str) -> np.ndarray:
    if hasattr(readout, "predict_batch"):
        return readout.predict_batch(features, task_id)
    if hasattr(readout, "predict"):
        return np.asarray([readout.predict(row, task_id) for row in features])
    raise AttributeError("readout/head has no predict or predict_batch method")


def _update(readout: Any, features: np.ndarray, targets: np.ndarray, task_id: str, lr: float) -> float:
    if hasattr(readout, "update_batch"):
        return float(readout.update_batch(features, targets, task_id=task_id, lr=lr))
    if hasattr(readout, "update"):
        losses = [
            float(readout.update(row, target, task_id=task_id, lr=lr))
            for row, target in zip(features, targets)
        ]
        return float(np.mean(losses))
    raise AttributeError("readout/head has no update or update_batch method")


def _accuracy_like(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Small generic progress metric for sleep logs."""
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        return float(np.mean(np.abs(predictions[:, 0] - targets[:, 0]) < 0.5))
    if predictions.ndim == 2 and predictions.shape[1] > 1:
        return float(np.mean(np.argmax(predictions, axis=1) == np.argmax(targets, axis=1)))
    return float(np.mean(np.abs(predictions - targets) < 0.15))


def sleep_consolidation(
    network: Any,
    task_id: str,
    inputs: np.ndarray,
    targets: np.ndarray,
    num_cycles: int = 1000,
    sleep_lr: float = 0.001,
    batch_size: int = 32,
    verbose: bool = True,
    readout: Any | None = None,
    feature_extractor: FeatureExtractor | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Replay cached frozen features to refine a task-specific readout.

    Args:
        network: Benchmark network with ``kernels`` and usually ``network_readout``.
        task_id: Task identifier whose isolated head should be updated.
        inputs: Sequence inputs shaped ``(samples, seq_len, input_width)``.
        targets: Targets shaped ``(samples, output_dim)``.
        num_cycles: Number of replay mini-batch updates.
        sleep_lr: Learning rate used only for readout/head updates.
        batch_size: Replay mini-batch size.
        verbose: Print progress.
        readout: Optional head/readout object. Defaults to ``network.network_readout``.
        feature_extractor: Optional callable returning ``(predictions, features)``.
        seed: Replay sampling seed.

    Returns:
        Consolidation metrics and short history.
    """
    inputs = np.asarray(inputs, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if num_cycles < 0:
        raise ValueError("num_cycles must be non-negative")
    if len(inputs) != len(targets):
        raise ValueError("inputs and targets must have the same sample count")
    if len(inputs) == 0:
        raise ValueError("sleep consolidation needs at least one sample")

    readout_obj = _readout_for(network, readout)
    extractor = feature_extractor or _default_feature_extractor
    rng = np.random.default_rng(seed)

    start = time.time()
    kernels = freeze_network(network)
    try:
        _, features = extractor(network, inputs, task_id)
        features = np.asarray(features, dtype=float)
        initial_predictions = _predict(readout_obj, features, task_id)
        initial_error = float(np.mean((targets - initial_predictions) ** 2))
        initial_accuracy = _accuracy_like(initial_predictions, targets)
        had_lr_override_flag = hasattr(readout_obj, "honor_external_lr")
        old_lr_override_flag = getattr(readout_obj, "honor_external_lr", False)
        if had_lr_override_flag or readout_obj.__class__.__name__ == "StableNetworkReadout":
            setattr(readout_obj, "honor_external_lr", True)

        if verbose:
            frozen = sum(1 for kernel in kernels if getattr(kernel, "is_frozen", lambda: False)())
            print(f"Sleep consolidation: {task_id}")
            print(f"  frozen kernels: {frozen}/{len(kernels)}")
            print(f"  cached features: {features.shape}")
            print(f"  cycles: {num_cycles}, lr: {sleep_lr}")

        history = {"cycles": [], "errors": [], "accuracies": []}
        effective_batch = min(int(batch_size), len(features))
        for cycle in range(num_cycles):
            idx = rng.choice(len(features), size=effective_batch, replace=False)
            loss = _update(readout_obj, features[idx], targets[idx], task_id, sleep_lr)
            if cycle == 0 or (cycle + 1) == num_cycles or (cycle + 1) % max(1, num_cycles // 10) == 0:
                pred = _predict(readout_obj, features, task_id)
                err = float(np.mean((targets - pred) ** 2))
                acc = _accuracy_like(pred, targets)
                history["cycles"].append(cycle + 1)
                history["errors"].append(err)
                history["accuracies"].append(acc)
                if verbose:
                    print(f"  cycle {cycle + 1:5d}/{num_cycles}: loss={loss:.6f}, error={err:.6f}")

        final_predictions = _predict(readout_obj, features, task_id)
        final_error = float(np.mean((targets - final_predictions) ** 2))
        final_accuracy = _accuracy_like(final_predictions, targets)
    finally:
        if "old_lr_override_flag" in locals():
            if had_lr_override_flag:
                setattr(readout_obj, "honor_external_lr", old_lr_override_flag)
            elif hasattr(readout_obj, "honor_external_lr"):
                delattr(readout_obj, "honor_external_lr")
        unfreeze_network(kernels)

    return {
        "task_id": task_id,
        "num_cycles": int(num_cycles),
        "sleep_lr": float(sleep_lr),
        "batch_size": int(batch_size),
        "feature_shape": list(features.shape) if "features" in locals() else None,
        "initial_error": initial_error,
        "final_error": final_error,
        "error_reduction": initial_error - final_error,
        "initial_accuracy": initial_accuracy,
        "final_accuracy": final_accuracy,
        "total_time": float(time.time() - start),
        "history": history,
    }


def multi_task_sleep_consolidation(
    network: Any,
    tasks: dict[str, tuple[np.ndarray, np.ndarray]],
    num_cycles: int = 1000,
    sleep_lr: float = 0.001,
    batch_size: int = 32,
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    """Run sleep consolidation for each task in order."""
    return {
        task_id: sleep_consolidation(
            network,
            task_id,
            inputs,
            targets,
            num_cycles=num_cycles,
            sleep_lr=sleep_lr,
            batch_size=batch_size,
            verbose=verbose,
        )
        for task_id, (inputs, targets) in tasks.items()
    }


def _self_test() -> None:
    class Kernel:
        def __init__(self) -> None:
            self.frozen = False

        def freeze(self) -> None:
            self.frozen = True

        def unfreeze(self) -> None:
            self.frozen = False

        def is_frozen(self) -> bool:
            return self.frozen

    class Head:
        def __init__(self) -> None:
            self.w = np.zeros((1, 2))

        def predict_batch(self, features, task_id=None):
            return features @ self.w.T

        def update_batch(self, features, targets, task_id=None, lr=0.1):
            pred = self.predict_batch(features)
            err = targets - pred
            self.w += lr * (err.T @ features) / len(features)
            return float(np.mean(err**2))

    class Net:
        def __init__(self) -> None:
            self.kernels = [Kernel(), Kernel()]
            self.network_readout = Head()

    def extractor(network, inputs, task_id):
        return np.zeros((len(inputs), 1)), inputs[:, 0, :]

    x = np.array([[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]])
    y = np.array([[1.0], [2.0], [3.0]])
    net = Net()
    metrics = sleep_consolidation(net, "demo", x, y, 50, 0.2, 2, False, feature_extractor=extractor)
    assert metrics["final_error"] < metrics["initial_error"]
    assert all(not kernel.is_frozen() for kernel in net.kernels)
    print("Sleep consolidation tests passed")


if __name__ == "__main__":
    _self_test()
