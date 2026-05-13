"""Train a 3-head perception model: sentiment + intent + urgency.

All three tasks share a single fixed encoder (zero forgetting by construction).
No HuggingFace downloads required — all data is generated synthetically.

Usage
-----
    python train_multihead.py
    python train_multihead.py --epochs 10 --out multihead.npz
    python train_multihead.py --reservoir-dim 512 --tfidf-dim 3000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from hf_adapter import HFAdapter
from multihead_model import MultiHeadModel
from synthetic_datasets import (
    make_intent_dataset,
    make_sentiment_dataset,
    make_urgency_dataset,
)
from weightless_model import WeightlessModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.argmax(preds, axis=1) == np.argmax(targets, axis=1)))


def _bar(step: int, total: int, loss: float, acc: float) -> None:
    filled = int(30 * step / max(1, total))
    bar = "█" * filled + "░" * (30 - filled)
    print(f"\r  [{bar}] {step:>3}/{total}  loss={loss:.4f}  acc={acc:.1%}",
          end="", flush=True)


def _train_head(
    model: WeightlessModel,
    task_id: str,
    feats_tr: np.ndarray,
    Y_tr: np.ndarray,
    feats_te: np.ndarray,
    Y_te: np.ndarray,
    label_names: list[str],
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int = 42,
) -> None:
    """Train one isolated task head, leaving all others untouched."""
    feature_dim = feats_tr.shape[1]
    n_classes   = Y_tr.shape[1]
    rng         = np.random.default_rng(seed)
    n           = len(feats_tr)

    # Initialize head directly — bypasses the fixed output_dim of ZeroForgetReadout
    model.readout._heads[task_id] = rng.normal(
        0.0, 0.02, size=(n_classes, feature_dim)
    ).astype(np.float32)

    for epoch in range(epochs):
        idx        = rng.permutation(n)
        epoch_loss = 0.0
        steps      = 0
        for start in range(0, n, batch_size):
            batch     = idx[start : start + batch_size]
            loss      = model.update(feats_tr[batch], Y_tr[batch],
                                     lr=lr, task_id=task_id)
            epoch_loss += loss
            steps      += 1
        avg_loss  = epoch_loss / max(1, steps)
        train_acc = _accuracy(model.readout.predict_batch(feats_tr, task_id), Y_tr)
        _bar(epoch + 1, epochs, avg_loss, train_acc)

    test_acc = _accuracy(model.readout.predict_batch(feats_te, task_id), Y_te)
    print(f"\n  Classes : {label_names}")
    print(f"  Test acc: {test_acc:.1%}\n")


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(
    out: str,
    epochs: int,
    batch_size: int,
    lr: float,
    tfidf_dim: int,
    reservoir_dim: int,
    max_sentences: int,
    n_train: int,
    n_test: int,
) -> None:
    t_start = time.time()
    print(f"\n{'='*60}")
    print("  Multi-head Perception Model Training")
    print(f"  Tasks    : sentiment + intent + urgency")
    print(f"  Epochs   : {epochs}   Batch: {batch_size}   LR: {lr}")
    print(f"{'='*60}\n")

    # ── 1. Generate datasets ──────────────────────────────────────────
    print("[ 1/4 ] Generating synthetic datasets …")
    s_tr_tx, s_tr_lb, s_te_tx, s_te_lb = make_sentiment_dataset(n_train, n_test)
    i_tr_tx, i_tr_lb, i_te_tx, i_te_lb = make_intent_dataset(n_train, n_test)
    u_tr_tx, u_tr_lb, u_te_tx, u_te_lb = make_urgency_dataset(n_train, n_test)
    print(f"  sentiment : {len(s_tr_tx)} train / {len(s_te_tx)} test")
    print(f"  intent    : {len(i_tr_tx)} train / {len(i_te_tx)} test")
    print(f"  urgency   : {len(u_tr_tx)} train / {len(u_te_tx)} test\n")

    # ── 2. Fit shared TF-IDF on the combined training corpus ──────────
    print("[ 2/4 ] Fitting shared TF-IDF vocabulary …")
    adapter = HFAdapter(
        tfidf_dim    = tfidf_dim,
        reservoir_dim= reservoir_dim,
        max_sentences= max_sentences,
        seed         = 42,
    )
    all_train_texts = s_tr_tx + i_tr_tx + u_tr_tx
    adapter.fit(all_train_texts)
    print(f"  Vocabulary: {adapter.input_dim:,} tokens  "
          f"encoder: ({reservoir_dim} × {adapter.input_dim:,})\n")

    # ── 3. Encode all datasets (one-time, fixed encoder) ─────────────
    print("[ 3/4 ] Encoding datasets (TF-IDF → fixed encoder → memory) …")
    t1 = time.time()
    X_s_tr, Y_s_tr, s_labels = adapter.encode_texts(s_tr_tx, s_tr_lb)
    X_s_te, Y_s_te, _        = adapter.encode_texts(s_te_tx, s_te_lb)
    X_i_tr, Y_i_tr, i_labels = adapter.encode_texts(i_tr_tx, i_tr_lb)
    X_i_te, Y_i_te, _        = adapter.encode_texts(i_te_tx, i_te_lb)
    X_u_tr, Y_u_tr, u_labels = adapter.encode_texts(u_tr_tx, u_tr_lb)
    X_u_te, Y_u_te, _        = adapter.encode_texts(u_te_tx, u_te_lb)

    # Build model — encoder replaced with adapter's fixed W
    model = WeightlessModel(
        input_dim    = adapter.input_dim,
        reservoir_dim= adapter.reservoir_dim,
        output_dim   = 4,   # placeholder; heads are sized per-task below
        seed         = 42,
    )
    model.encoder._W = adapter._W   # share the same frozen projection

    # Pre-compute memory features (encoder is fixed — do it once)
    feats_s_tr = model.encode_sequence(X_s_tr)
    feats_s_te = model.encode_sequence(X_s_te)
    feats_i_tr = model.encode_sequence(X_i_tr)
    feats_i_te = model.encode_sequence(X_i_te)
    feats_u_tr = model.encode_sequence(X_u_tr)
    feats_u_te = model.encode_sequence(X_u_te)
    print(f"  Feature dim : {model.feature_dim}  |  encode time: {time.time()-t1:.1f}s\n")

    # ── 4. Train each head (others stay untouched) ────────────────────
    print("[ 4/4 ] Training heads …\n")

    TASKS = [
        ("sentiment", feats_s_tr, Y_s_tr, feats_s_te, Y_s_te, s_labels),
        ("intent",    feats_i_tr, Y_i_tr, feats_i_te, Y_i_te, i_labels),
        ("urgency",   feats_u_tr, Y_u_tr, feats_u_te, Y_u_te, u_labels),
    ]
    for task_id, f_tr, y_tr, f_te, y_te, labels in TASKS:
        print(f"  ── {task_id} ──")
        _train_head(model, task_id, f_tr, y_tr, f_te, y_te,
                    labels, epochs, batch_size, lr)

    # Summary
    print(f"{'─'*60}")
    print(f"  Fixed params   : {model.total_fixed_params:,}")
    print(f"  Learned params : {model.total_learned_params:,}")
    print(f"  Tasks stored   : {model.task_ids}")
    print(f"  Total time     : {time.time()-t_start:.1f}s")
    print(f"{'─'*60}")

    # ── 5. Save ───────────────────────────────────────────────────────
    task_map = {
        "sentiment": s_labels,
        "intent":    i_labels,
        "urgency":   u_labels,
    }
    mm = MultiHeadModel(adapter=adapter, model=model, task_map=task_map)
    mm.save(out)
    print(f"\n  Saved → {out}  (load with MultiHeadModel.load('{out}'))")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Train a 3-head weightless perception model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out",           default="multihead.npz",
                   help="Output path (default: multihead.npz)")
    p.add_argument("--epochs",        type=int,   default=20,
                   help="Training epochs per head (default: 20)")
    p.add_argument("--batch-size",    type=int,   default=32,
                   help="Batch size (default: 32)")
    p.add_argument("--lr",            type=float, default=0.05,
                   help="Learning rate (default: 0.05)")
    p.add_argument("--tfidf-dim",     type=int,   default=2000,
                   help="TF-IDF vocabulary size (default: 2000)")
    p.add_argument("--reservoir-dim", type=int,   default=256,
                   help="Fixed encoder width (default: 256)")
    p.add_argument("--max-sentences", type=int,   default=6,
                   help="Sentence chunks per doc (default: 6)")
    p.add_argument("--n-train",       type=int,   default=40,
                   help="Train samples per class (default: 40)")
    p.add_argument("--n-test",        type=int,   default=10,
                   help="Test samples per class (default: 10)")
    args = p.parse_args()

    train(
        out          = args.out,
        epochs       = args.epochs,
        batch_size   = args.batch_size,
        lr           = args.lr,
        tfidf_dim    = args.tfidf_dim,
        reservoir_dim= args.reservoir_dim,
        max_sentences= args.max_sentences,
        n_train      = args.n_train,
        n_test       = args.n_test,
    )


if __name__ == "__main__":
    main()
