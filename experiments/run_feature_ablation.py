"""Feature ablation sweep for reservoir-based text classification.

Systematically tests which components of the pipeline matter:
  1. max_sentences (sequence length): 1 vs 3 vs 5
  2. reservoir_dim: 256 vs 512 vs 1024
  3. tfidf_dim: 3k vs 10k vs 30k
  4. Feature block ablation (last / mean / max / ac / dc / ac*dc / all)
  5. Classifier head: softmax+CE / ridge / logistic regression

Locked defaults (post-ablation):
  max_sentences = 3
  tfidf_dim     = 3000
  reservoir_dim = 1024
  head          = softmax + cross-entropy (ZeroForgetReadout)

Usage
-----
    python -m experiments.run_feature_ablation
    python -m experiments.run_feature_ablation --max-train 10000 --epochs 30
    python -m experiments.run_feature_ablation --full   # use entire train split
"""

from __future__ import annotations

import argparse
import time
from typing import Callable

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.metrics import accuracy_score
except ImportError as e:
    raise ImportError("pip install scikit-learn") from e

from hf_adapter import HFAdapter
from weightless_model import SystemMemory
from zero_forgetting import ZeroForgetReadout


# ---------------------------------------------------------------------------
# Feature-slice helpers
# ---------------------------------------------------------------------------

def _feature_slices(reservoir_dim: int) -> dict[str, list[tuple[int, int]]]:
    """Return index slices for each named feature block."""
    d = reservoir_dim
    return {
        "last":    [(0,   d)],
        "mean":    [(d,   2*d)],
        "max":     [(2*d, 3*d)],
        "ac":      [(3*d, 4*d)],
        "dc":      [(4*d, 5*d)],
        "ac*dc":   [(5*d, 6*d)],
        "phase":   [(6*d, 6*d+3)],
        "all":     [(0,   6*d+3)],
    }


def select_features(F: np.ndarray, slices: list[tuple[int, int]]) -> np.ndarray:
    parts = [F[:, s:e] for s, e in slices]
    return np.concatenate(parts, axis=1)


# ---------------------------------------------------------------------------
# Classifier wrappers — unified interface
# ---------------------------------------------------------------------------

def eval_logistic(F_tr: np.ndarray, y_tr: np.ndarray,
                  F_te: np.ndarray, y_te: np.ndarray) -> float:
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(F_tr, y_tr)
    return float(accuracy_score(y_te, clf.predict(F_te)))


def eval_ridge(F_tr: np.ndarray, y_tr: np.ndarray,
               F_te: np.ndarray, y_te: np.ndarray) -> float:
    clf = RidgeClassifier(alpha=1.0)
    clf.fit(F_tr, y_tr)
    return float(accuracy_score(y_te, clf.predict(F_te)))


def eval_softmax_ce(F_tr: np.ndarray, y_tr: np.ndarray,
                    F_te: np.ndarray, y_te: np.ndarray,
                    n_classes: int = 2, epochs: int = 200,
                    lr: float = 0.1) -> float:
    """ZeroForgetReadout with softmax + cross-entropy gradient."""
    rng = np.random.default_rng(42)
    head = ZeroForgetReadout(feature_dim=F_tr.shape[1], output_dim=n_classes, rng=rng)
    head.set_task("t0")
    # one-hot targets
    Y_tr = np.eye(n_classes)[y_tr]
    for _ in range(epochs):
        idx = np.random.permutation(len(F_tr))
        for i in range(0, len(idx), 32):
            batch_idx = idx[i:i+32]
            head.update_batch(F_tr[batch_idx], Y_tr[batch_idx], lr=lr, task_id="t0")
    preds = np.argmax(head.predict_batch(F_te, task_id="t0"), axis=1)
    return float(accuracy_score(y_te, preds))


# ---------------------------------------------------------------------------
# Core: build features for one config
# ---------------------------------------------------------------------------

def build_features(
    tfidf_dim: int,
    reservoir_dim: int,
    max_sentences: int,
    max_train: int | None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (F_tr, y_tr, F_te, y_te) for SST-2."""
    adapter = HFAdapter(
        tfidf_dim=tfidf_dim,
        reservoir_dim=reservoir_dim,
        max_sentences=max_sentences,
        seed=seed,
    )
    X_tr, Y_tr, _ = adapter.fit_transform("sst2", split="train",
                                           max_samples=max_train)
    X_te, Y_te, _ = adapter.transform("sst2", split="validation")
    mem = SystemMemory(reservoir_dim=reservoir_dim)
    F_tr = mem.process_sequence(X_tr)
    F_te = mem.process_sequence(X_te)
    y_tr = np.argmax(Y_tr, axis=1)
    y_te = np.argmax(Y_te, axis=1)
    return F_tr, y_tr, F_te, y_te


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _hdr(title: str) -> None:
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def _row(label: str, acc: float, t: float) -> None:
    bar_w = int(acc * 30)
    bar   = "#" * bar_w + "." * (30 - bar_w)
    print(f"  {label:<30s}  {acc:.1%}  [{bar}]  {t:.1f}s")


# ---------------------------------------------------------------------------
# Main sweeps
# ---------------------------------------------------------------------------

def sweep_max_sentences(max_train: int | None) -> None:
    _hdr("Sweep 1: max_sentences (tfidf=3k, res=256, LogReg)")
    for ms in [1, 3, 5]:
        t0 = time.perf_counter()
        F_tr, y_tr, F_te, y_te = build_features(3000, 256, ms, max_train)
        acc = eval_logistic(F_tr, y_tr, F_te, y_te)
        _row(f"max_sentences={ms}", acc, time.perf_counter() - t0)


def sweep_reservoir_dim(max_train: int | None) -> None:
    _hdr("Sweep 2: reservoir_dim (tfidf=3k, max_sent=5, LogReg)")
    for rd in [256, 512, 1024]:
        t0 = time.perf_counter()
        F_tr, y_tr, F_te, y_te = build_features(3000, rd, 5, max_train)
        acc = eval_logistic(F_tr, y_tr, F_te, y_te)
        _row(f"reservoir_dim={rd}", acc, time.perf_counter() - t0)


def sweep_tfidf_dim(max_train: int | None) -> None:
    _hdr("Sweep 3: tfidf_dim (res=256, max_sent=5, LogReg)")
    for td in [3000, 10000, 30000]:
        t0 = time.perf_counter()
        F_tr, y_tr, F_te, y_te = build_features(td, 256, 5, max_train)
        acc = eval_logistic(F_tr, y_tr, F_te, y_te)
        _row(f"tfidf_dim={td}", acc, time.perf_counter() - t0)


def sweep_feature_blocks(max_train: int | None) -> None:
    _hdr("Sweep 4: feature block ablation (tfidf=3k, res=256, max_sent=5, LogReg)")
    F_tr, y_tr, F_te, y_te = build_features(3000, 256, 5, max_train)
    slices = _feature_slices(256)
    for name, sl in slices.items():
        t0 = time.perf_counter()
        Ftr_s = select_features(F_tr, sl)
        Fte_s = select_features(F_te, sl)
        acc = eval_logistic(Ftr_s, y_tr, Fte_s, y_te)
        dim = Ftr_s.shape[1]
        _row(f"{name:<8s}  (dim={dim:>4d})", acc, time.perf_counter() - t0)


def sweep_classifiers(max_train: int | None, epochs: int, lr: float = 0.1) -> None:
    _hdr("Sweep 5: classifier head (tfidf=3k, res=1024, max_sent=3, all features)")
    F_tr, y_tr, F_te, y_te = build_features(3000, 1024, 3, max_train)
    n_classes = len(np.unique(y_tr))

    for name, fn in [
        ("logistic regression", lambda: eval_logistic(F_tr, y_tr, F_te, y_te)),
        ("ridge classifier",    lambda: eval_ridge(F_tr, y_tr, F_te, y_te)),
        (f"softmax+CE ({epochs} ep, lr={lr})", lambda: eval_softmax_ce(
            F_tr, y_tr, F_te, y_te, n_classes=n_classes, epochs=epochs, lr=lr)),
    ]:
        t0 = time.perf_counter()
        acc = fn()
        _row(name, acc, time.perf_counter() - t0)


def raw_tfidf_baseline(max_train: int | None) -> None:
    """TF-IDF with no reservoir — pure upper bound reference."""
    _hdr("Baseline: raw TF-IDF (no reservoir), LogReg")
    from sklearn.feature_extraction.text import TfidfVectorizer
    try:
        from datasets import load_dataset
    except ImportError:
        print("  (skipped — datasets not installed)")
        return

    for td in [3000, 10000]:
        t0 = time.perf_counter()
        ds_tr = load_dataset("sst2", split="train")
        ds_te = load_dataset("sst2", split="validation")

        texts_tr = ds_tr["sentence"][:max_train] if max_train else ds_tr["sentence"]
        labels_tr = ds_tr["label"][:max_train] if max_train else ds_tr["label"]
        texts_te  = ds_te["sentence"]
        labels_te = ds_te["label"]

        vec = TfidfVectorizer(max_features=td)
        Xtr = vec.fit_transform(texts_tr)
        Xte = vec.transform(texts_te)
        y_tr_arr = np.array(labels_tr)
        y_te_arr = np.array(labels_te)

        acc = eval_logistic(Xtr, y_tr_arr, Xte, y_te_arr)
        _row(f"raw tfidf (dim={td})", acc, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Feature ablation sweep")
    parser.add_argument("--max-train", type=int, default=2000,
                        help="Max training samples per config (default 2000). "
                             "Use 0 for full dataset.")
    parser.add_argument("--full", action="store_true",
                        help="Run on entire training split (~67k). Overrides --max-train.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Epochs for softmax+CE classifier (default 200)")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Learning rate for softmax+CE classifier (default 0.1)")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip the raw TF-IDF baseline")
    args = parser.parse_args()

    max_train: int | None = None if args.full else (args.max_train or None)

    n_samples_label = "all" if max_train is None else str(max_train)
    print()
    print(f"Feature Ablation Sweep  |  dataset: SST-2  |  train samples: {n_samples_label}")
    print("Accuracy metric: LogisticRegression on 872 validation samples")
    print("(unless noted in sweep 5)")

    if not args.skip_baseline:
        raw_tfidf_baseline(max_train)

    sweep_max_sentences(max_train)
    sweep_reservoir_dim(max_train)
    sweep_tfidf_dim(max_train)
    sweep_feature_blocks(max_train)
    sweep_classifiers(max_train, args.epochs, args.lr)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
