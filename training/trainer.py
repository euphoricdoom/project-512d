"""Core training loop for weightless continual learning models.

Wraps WeightlessModel with a clean epoch/batch loop. No dependencies on
train_hf.py — this module is self-contained given a fitted adapter.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from config.model_config import TrainingConfig
from training.evaluation import accuracy


class WeightlessTrainer:
    """Epoch-based trainer for ``WeightlessModel``.

    Handles mini-batch shuffling, progress reporting, and optional sleep
    consolidation. Pre-computes features once before training because the
    encoder is fixed — a significant speedup for multi-epoch runs.

    Args:
        model:   A ``WeightlessModel`` instance.
        config:  ``TrainingConfig`` with epochs, batch_size, learning_rate etc.
        verbose: Print epoch-level progress to stdout.

    Usage::

        trainer = WeightlessTrainer(model, config)
        result  = trainer.train(X_tr, Y_tr, X_te, Y_te, task_id="sentiment")
        print(f"Accuracy: {result['final_accuracy']:.1%}")
    """

    def __init__(
        self,
        model,
        config: TrainingConfig,
        verbose: bool = True,
    ) -> None:
        self.model   = model
        self.config  = config
        self.verbose = verbose

    def train(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        task_id: str,
    ) -> Dict[str, Any]:
        """Train on one task and return history + final metrics.

        Features are pre-computed once from X_train (encoder is fixed, so
        this is lossless and saves recomputing them every epoch).

        Args:
            X_train:  ``(n_train, T, input_dim)`` sequence array.
            Y_train:  ``(n_train, n_classes)`` one-hot array.
            X_test:   ``(n_test, T, input_dim)`` sequence array.
            Y_test:   ``(n_test, n_classes)`` one-hot array.
            task_id:  Task identifier — creates a new isolated head if needed.

        Returns:
            Dict with ``"task_id"``, ``"history"`` (list of epoch dicts),
            and ``"final_accuracy"``.
        """
        self.model.set_task(task_id)
        cfg = self.config

        if self.verbose:
            print(f"\n[ Training task: {task_id!r} ]")
            print(f"  Epochs={cfg.epochs}, batch={cfg.batch_size}, lr={cfg.learning_rate}")

        # Pre-compute features once — encoder is frozen, so this is exact
        if self.verbose:
            print("  Pre-computing features (encoder is fixed) …")
        feats_train = self.model.encode_sequence(X_train)
        feats_test  = self.model.encode_sequence(X_test)

        n        = len(feats_train)
        rng      = np.random.default_rng(42)
        history: List[Dict[str, float]] = []

        for epoch in range(cfg.epochs):
            idx        = rng.permutation(n)
            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, n, cfg.batch_size):
                batch      = idx[start : start + cfg.batch_size]
                loss       = self.model.update(
                    feats_train[batch], Y_train[batch], lr=cfg.learning_rate,
                    task_id=task_id,
                )
                epoch_loss += loss
                n_batches  += 1

            avg_loss = epoch_loss / max(1, n_batches)
            test_acc = accuracy(
                self.model.readout.predict_batch(feats_test, task_id), Y_test
            )
            history.append({
                "epoch":          epoch + 1,
                "train_loss":     round(avg_loss, 6),
                "test_accuracy":  round(test_acc, 4),
            })
            if self.verbose:
                print(f"  Epoch {epoch+1}/{cfg.epochs}  loss={avg_loss:.4f}  "
                      f"test_acc={test_acc:.1%}")

        if cfg.sleep_cycles > 0:
            self._sleep(feats_train, Y_train, task_id)

        return {
            "task_id":         task_id,
            "history":         history,
            "final_accuracy":  history[-1]["test_accuracy"],
        }

    def _sleep(
        self,
        feats_train: np.ndarray,
        Y_train: np.ndarray,
        task_id: str,
    ) -> None:
        """Run sleep consolidation using pre-computed features."""
        from zero_forgetting import sleep_consolidation

        cfg = self.config
        if self.verbose:
            print(f"\n  Sleep consolidation ({cfg.sleep_cycles} cycles, "
                  f"lr={cfg.sleep_lr}) …")

        result = sleep_consolidation(
            freeze_fn=lambda: None,
            unfreeze_fn=lambda: None,
            extract_fn=lambda _: feats_train,
            readout=self.model.readout,
            task_id=task_id,
            inputs=np.zeros((len(feats_train), 1)),  # dummy — extract_fn ignores it
            targets=Y_train,
            num_cycles=cfg.sleep_cycles,
            sleep_lr=cfg.sleep_lr,
            batch_size=32,
            seed=42,
        )

        if self.verbose:
            print(f"  Sleep done — CE: {result['initial_error']:.4f} → "
                  f"{result['final_error']:.4f}")
