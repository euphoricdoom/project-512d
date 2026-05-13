"""Evaluation helpers for weightless model outputs."""

from __future__ import annotations

from typing import Optional

import numpy as np


def accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    """Compute top-1 accuracy for one-hot encoded predictions and targets.

    Args:
        preds:   Logit or probability matrix ``(n, n_classes)``.
        targets: One-hot label matrix ``(n, n_classes)``.

    Returns:
        Accuracy in ``[0, 1]``.
    """
    pred_labels = np.argmax(preds, axis=1)
    true_labels = np.argmax(targets, axis=1)
    return float(np.mean(pred_labels == true_labels))


def evaluate_model(
    model,
    X: np.ndarray,
    Y: np.ndarray,
    task_id: Optional[str] = None,
) -> float:
    """Run a full forward pass and return accuracy.

    Args:
        model:   ``WeightlessModel`` instance.
        X:       Input sequences ``(n, timesteps, input_dim)``.
        Y:       One-hot labels ``(n, n_classes)``.
        task_id: Task to evaluate (uses model's current task if None).

    Returns:
        Accuracy in ``[0, 1]``.
    """
    features = model.encode_sequence(X)
    preds = model.readout.predict_batch(features, task_id=task_id)
    return accuracy(preds, Y)


def classification_report(
    model,
    X: np.ndarray,
    Y: np.ndarray,
    label_names: list[str],
    task_id: Optional[str] = None,
) -> dict[str, object]:
    """Compute per-class precision, recall, and F1.

    Args:
        model:       ``WeightlessModel`` instance.
        X:           Input sequences ``(n, timesteps, input_dim)``.
        Y:           One-hot labels ``(n, n_classes)``.
        label_names: Class name strings in class-index order.
        task_id:     Task to evaluate.

    Returns:
        Dict with keys ``"accuracy"``, ``"per_class"`` (list of per-class dicts),
        and ``"macro_f1"``.
    """
    features = model.encode_sequence(X)
    preds = model.readout.predict_batch(features, task_id=task_id)

    pred_labels = np.argmax(preds, axis=1)
    true_labels = np.argmax(Y, axis=1)
    n_classes = Y.shape[1]

    per_class = []
    for c in range(n_classes):
        tp = int(np.sum((pred_labels == c) & (true_labels == c)))
        fp = int(np.sum((pred_labels == c) & (true_labels != c)))
        fn = int(np.sum((pred_labels != c) & (true_labels == c)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        per_class.append({
            "class":     label_names[c] if c < len(label_names) else str(c),
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "support":   int(np.sum(true_labels == c)),
        })

    macro_f1 = float(np.mean([d["f1"] for d in per_class]))
    acc       = float(np.mean(pred_labels == true_labels))

    return {
        "accuracy":  round(acc, 4),
        "macro_f1":  round(macro_f1, 4),
        "per_class": per_class,
    }
