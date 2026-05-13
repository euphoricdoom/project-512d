"""Real-data forgetting experiment: SST-2 -> AG News -> Emotion

This is the first serious empirical test of the weightless model.

The proof suite (proof/claims.py) verified zero forgetting at the *parameter*
level -- i.e. that head A's weight matrix is byte-identical before and after
training head B.  That is a necessary but not sufficient condition.  What
matters scientifically is whether the *task accuracy* on A drops after
training B and C on a completely different distribution.

Protocol
--------
1. Fit TF-IDF + fixed encoder on SST-2 training split
2. Train head A on SST-2 -> measure accuracy A_before
3. Train head B on AG News (same encoder; new TF-IDF)
4. Train head C on Emotion (same encoder; new TF-IDF)
5. Re-evaluate head A on SST-2 test split -> measure A_after
6. Forgetting = A_before - A_after

Key design constraint:
- All three tasks share the SAME fixed encoder _W matrix.
- Each task has its own TF-IDF vectorizer (fitted on its own training vocabulary).
- Only the output heads differ.  The encoder never changes.

If forgetting ≈ 0 across real distributions, the scientific claim is serious.

Run
---
    pip install datasets scikit-learn
    python experiments/real_forgetting_experiment.py

    # With sample limits for quick smoke-test
    python experiments/real_forgetting_experiment.py --max-train 1000 --max-test 500

Results are printed to stdout and written to outputs/real_forgetting_result.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hf_adapter import HFAdapter, _split_sentences
from weightless_model import WeightlessModel, FixedEncoder


# ---------------------------------------------------------------------------
# Shared-encoder multi-task adapter
# ---------------------------------------------------------------------------

class MultiTaskAdapter:
    """Manages per-task TF-IDF vectorizers that all feed the same fixed encoder.

    Each task gets its own TF-IDF fitted on its training corpus, but ALL tasks
    project through the same frozen W matrix.

    The shared encoder W is built lazily after the first task is fitted, using
    the actual vocabulary size (may be smaller than tfidf_dim on small corpora).
    Subsequent tasks are fitted with max_features set to that actual size so
    all TF-IDF outputs are the same width.
    """

    def __init__(
        self,
        tfidf_dim: int = 3000,
        reservoir_dim: int = 256,
        max_sentences: int = 6,
        target_radius: float = 0.94,
        seed: int = 42,
    ) -> None:
        self.tfidf_dim         = tfidf_dim
        self.actual_tfidf_dim: int | None = None  # locked after first fit
        self.reservoir_dim     = reservoir_dim
        self.max_sentences     = max_sentences
        self.target_radius     = target_radius
        self.seed              = seed
        self._W: np.ndarray | None = None
        self._adapters: dict[str, HFAdapter] = {}

    def _build_encoder(self, vocab_size: int) -> None:
        rng = np.random.default_rng(self.seed)
        W   = rng.normal(0.0, 1.0 / np.sqrt(vocab_size),
                         size=(self.reservoir_dim, vocab_size))
        sv  = np.linalg.svd(W, compute_uv=False)
        if sv[0] > self.target_radius:
            W *= self.target_radius / sv[0]
        self._W               = W
        self.actual_tfidf_dim = vocab_size

    def fit_transform(
        self,
        task_id: str,
        dataset_name: str,
        split: str = "train",
        subset: str | None = None,
        text_field: str | None = None,
        label_field: str | None = None,
        max_samples: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Fit TF-IDF on this task, inject shared W, return features."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import LabelEncoder

        # 1. Load dataset
        adapter = HFAdapter(
            tfidf_dim=self.tfidf_dim,
            reservoir_dim=self.reservoir_dim,
            max_sentences=self.max_sentences,
            seed=self.seed,
        )
        ds      = adapter._load_hf(dataset_name, subset, split, max_samples)
        t_field = text_field  or adapter._detect_fields(ds[0])[0]
        l_field = label_field or adapter._detect_fields(ds[0])[1]
        texts   = [str(row[t_field]) for row in ds]
        labels  = [row[l_field]       for row in ds]

        # Get the FULL class set from dataset metadata (not just the subset)
        # This prevents LabelEncoder failing on unseen classes in the test split
        all_classes = None
        try:
            feat = ds.features[l_field]
            if hasattr(feat, "names"):
                # HuggingFace ClassLabel -- use canonical integer range
                all_classes = list(range(len(feat.names)))
        except Exception:
            pass
        if all_classes is None:
            all_classes = sorted(set(labels))

        # 2. Fit TF-IDF -- use the already-locked vocab size for subsequent tasks
        max_features = self.actual_tfidf_dim if self.actual_tfidf_dim else self.tfidf_dim
        all_sents = []
        for text in texts:
            all_sents.extend(_split_sentences(text, self.max_sentences))
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"(?u)\b\w\w+\b",
            ngram_range=(1, 2),
        )
        vectorizer.fit(all_sents)
        actual_vocab = len(vectorizer.vocabulary_)

        # 3. Build shared encoder from first task's actual vocab size
        if self._W is None:
            self._build_encoder(actual_vocab)
            print(f"    shared W built: shape={self._W.shape}  "
                  f"(actual vocab={actual_vocab})")

        # 4. Inject everything into adapter so _encode_all works
        le = LabelEncoder()
        le.fit(all_classes)  # fit on FULL class set, not just subset
        adapter._vectorizer = vectorizer
        adapter._label_enc  = le
        adapter._W          = self._W  # shared, frozen

        X, Y, label_names = adapter._encode_all(texts, labels)
        self._adapters[task_id] = adapter
        return X, Y, label_names

    def transform(
        self,
        task_id: str,
        dataset_name: str,
        split: str = "test",
        subset: str | None = None,
        text_field: str | None = None,
        label_field: str | None = None,
        max_samples: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Transform a new split using the fitted TF-IDF for this task."""
        if task_id not in self._adapters:
            raise KeyError(f"Task '{task_id}' not fitted. Call fit_transform first.")
        return self._adapters[task_id].transform(
            dataset_name, split=split, subset=subset,
            text_field=text_field, label_field=label_field,
            max_samples=max_samples,
        )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.argmax(preds, axis=1) == np.argmax(targets, axis=1)))


def train_task(
    model: WeightlessModel,
    feats: np.ndarray,
    targets: np.ndarray,
    task_id: str,
    epochs: int,
    batch_size: int,
    lr: float,
    verbose: bool = True,
) -> list[float]:
    """Train one head and return per-epoch loss."""
    model.set_task(task_id)
    n   = len(feats)
    rng = np.random.default_rng(0)
    losses = []
    for epoch in range(epochs):
        idx        = rng.permutation(n)
        epoch_loss = 0.0
        steps      = 0
        for start in range(0, n, batch_size):
            batch = idx[start:start + batch_size]
            epoch_loss += model.update(feats[batch], targets[batch], lr=lr)
            steps += 1
        avg = epoch_loss / max(1, steps)
        losses.append(avg)
        if verbose and (epoch == 0 or (epoch + 1) % max(1, epochs // 5) == 0):
            acc = accuracy(model.readout.predict_batch(feats, task_id), targets)
            print(f"    epoch {epoch+1:>3}/{epochs}  loss={avg:.4f}  train_acc={acc:.1%}")
    return losses


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

TASKS = [
    {
        "id":      "sst2",
        "dataset": "sst2",
        "subset":  None,
        "split_train": "train",
        "split_test":  "validation",   # SST-2 test labels not public; use validation
        "text_field":  "sentence",
        "label_field": "label",
        "description": "SST-2 binary sentiment",
    },
    {
        "id":      "ag_news",
        "dataset": "ag_news",
        "subset":  None,
        "split_train": "train",
        "split_test":  "test",
        "text_field":  "text",
        "label_field": "label",
        "description": "AG News 4-class topic",
    },
    {
        "id":      "emotion",
        "dataset": "dair-ai/emotion",
        "subset":  None,
        "split_train": "train",
        "split_test":  "test",
        "text_field":  "text",
        "label_field": "label",
        "description": "Emotion 6-class",
    },
]


def run_experiment(
    max_train: int | None,
    max_test:  int | None,
    epochs:    int,
    batch_size: int,
    lr:        float,
    tfidf_dim: int,
    reservoir_dim: int,
    max_sentences: int,
) -> dict:
    print("\n" + "=" * 65)
    print("  REAL-DATA FORGETTING EXPERIMENT")
    print("  SST-2 -> AG News -> Emotion | Measure A_before vs A_after")
    print("=" * 65)

    t_start = time.perf_counter()
    results = {}

    # ── 1. Build shared encoder + per-task TF-IDF ─────────────────────────
    print("\n[ SETUP ] Building shared encoder + per-task TF-IDF vectorizers …")
    mt_adapter = MultiTaskAdapter(
        tfidf_dim=tfidf_dim,
        reservoir_dim=reservoir_dim,
        max_sentences=max_sentences,
        seed=42,
    )

    # ── 2. Extract features for all tasks ─────────────────────────────────
    task_data = {}
    for t in TASKS:
        print(f"\n  Extracting: {t['description']} ({t['dataset']}) …")
        t0 = time.perf_counter()
        X_tr, Y_tr, label_names = mt_adapter.fit_transform(
            task_id=t["id"],
            dataset_name=t["dataset"],
            split=t["split_train"],
            subset=t["subset"],
            text_field=t["text_field"],
            label_field=t["label_field"],
            max_samples=max_train,
        )
        X_te, Y_te, _ = mt_adapter.transform(
            task_id=t["id"],
            dataset_name=t["dataset"],
            split=t["split_test"],
            subset=t["subset"],
            text_field=t["text_field"],
            label_field=t["label_field"],
            max_samples=max_test,
        )
        elapsed = time.perf_counter() - t0
        print(f"    train={len(X_tr):,}  test={len(X_te):,}  "
              f"classes={label_names}  time={elapsed:.1f}s")
        task_data[t["id"]] = dict(
            X_tr=X_tr, Y_tr=Y_tr, X_te=X_te, Y_te=Y_te,
            label_names=label_names,
        )

    # ── 3. Build model ─────────────────────────────────────────────────────
    print("\n[ MODEL ] Assembling shared SystemMemory + per-task readouts ...")
    from weightless_model import SystemMemory
    from zero_forgetting import ZeroForgetReadout

    memory      = SystemMemory(reservoir_dim=reservoir_dim)
    feature_dim = memory.feature_dim
    rng         = np.random.default_rng(42)

    # One readout per task with the correct output_dim for that task
    readouts: dict[str, ZeroForgetReadout] = {}
    for t in TASKS:
        n_classes = len(task_data[t["id"]]["label_names"])
        readouts[t["id"]] = ZeroForgetReadout(
            output_dim=n_classes,
            feature_dim=feature_dim,
            rng=rng,
        )

    fixed_params = mt_adapter._W.size
    print(f"  feature_dim={feature_dim}")
    print(f"  fixed encoder params: {fixed_params:,}")
    print(f"  per-task heads: " +
          ", ".join(f"{t['id']}({len(task_data[t['id']]['label_names'])})" for t in TASKS))

    # ── 4. Pre-compute all sequence features (encoder already applied by adapter) ─
    print("\n[ FEATURES ] Pre-computing SystemMemory features for all splits ...")
    all_feats: dict[str, dict] = {}
    for t in TASKS:
        td = task_data[t["id"]]
        t0 = time.perf_counter()
        # X_tr shape: (n, max_sentences, reservoir_dim) -- already encoded by adapter
        # Run only the SystemMemory pass (encoder step is done inside the adapter)
        feats_tr = memory.process_sequence(td["X_tr"])
        feats_te = memory.process_sequence(td["X_te"])
        print(f"  {t['id']:10s}  train={feats_tr.shape}  test={feats_te.shape}  "
              f"({time.perf_counter()-t0:.1f}s)")
        all_feats[t["id"]] = {"train": feats_tr, "test": feats_te}

    # Helper: train one readout head
    def _train_head(tid: str, epochs: int) -> None:
        rd  = readouts[tid]
        td  = task_data[tid]
        rd.set_task(tid)
        n   = len(all_feats[tid]["train"])
        rng_t = np.random.default_rng(0)
        for epoch in range(epochs):
            idx = rng_t.permutation(n)
            ep_loss = 0.0
            steps = 0
            for start in range(0, n, batch_size):
                batch   = idx[start:start + batch_size]
                ep_loss += rd.update_batch(all_feats[tid]["train"][batch],
                                           td["Y_tr"][batch], lr=lr)
                steps  += 1
            if epoch == 0 or (epoch + 1) % max(1, epochs // 5) == 0:
                acc = accuracy(rd.predict_batch(all_feats[tid]["train"], tid),
                               td["Y_tr"])
                print(f"    epoch {epoch+1:>3}/{epochs}  "
                      f"loss={ep_loss/max(1,steps):.4f}  train_acc={acc:.1%}")

    # ── 5. Train SST-2 (task A) -- record A_before ─────────────────────────
    print(f"\n{'─'*65}")
    print(f"  PHASE 1: Train SST-2 (head A)")
    print(f"{'─'*65}")
    td_a = task_data["sst2"]
    _train_head("sst2", epochs)

    preds_a_before = readouts["sst2"].predict_batch(all_feats["sst2"]["test"], "sst2")
    a_before = accuracy(preds_a_before, td_a["Y_te"])
    print(f"\n  OK SST-2 test accuracy (A_before) = {a_before:.4f}  ({a_before:.1%})")
    results["A_before"] = a_before

    # Snapshot head A weight matrix for parameter-level verification
    head_a_before = readouts["sst2"]._heads["sst2"].copy()

    # ── 6. Train AG News (task B) ──────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  PHASE 2: Train AG News (head B) -- head A must be untouched")
    print(f"{'─'*65}")
    td_b = task_data["ag_news"]
    _train_head("ag_news", epochs)
    assert np.array_equal(head_a_before, readouts["sst2"]._heads["sst2"]), \
        "FATAL: SST-2 head was modified while training AG News!"
    print("  OK Head A parameter check after B: array-equal (confirmed)")

    preds_a_after_b = readouts["sst2"].predict_batch(all_feats["sst2"]["test"], "sst2")
    a_after_b = accuracy(preds_a_after_b, td_a["Y_te"])
    results["A_after_B"] = a_after_b
    print(f"  SST-2 accuracy after B (A_after_B) = {a_after_b:.4f}  ({a_after_b:.1%})")

    # ── 7. Train Emotion (task C) ──────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  PHASE 3: Train Emotion (head C) -- head A and B must be untouched")
    print(f"{'─'*65}")
    td_c = task_data["emotion"]
    _train_head("emotion", epochs)
    assert np.array_equal(head_a_before, readouts["sst2"]._heads["sst2"]), \
        "FATAL: SST-2 head was modified while training Emotion!"
    print("  OK Head A parameter check after C: array-equal (confirmed)")

    # ── 8. Re-evaluate all three tasks ────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  FINAL EVALUATION")
    print(f"{'─'*65}")

    preds_a_after = readouts["sst2"].predict_batch(all_feats["sst2"]["test"], "sst2")
    a_after = accuracy(preds_a_after, td_a["Y_te"])
    results["A_after"] = a_after

    preds_b = readouts["ag_news"].predict_batch(all_feats["ag_news"]["test"], "ag_news")
    b_acc = accuracy(preds_b, td_b["Y_te"])
    results["B_acc"] = b_acc

    preds_c = readouts["emotion"].predict_batch(all_feats["emotion"]["test"], "emotion")
    c_acc = accuracy(preds_c, td_c["Y_te"])
    results["C_acc"] = c_acc

    results["A_after_C"] = a_after
    forgetting = a_before - a_after
    results["forgetting"] = forgetting
    results["total_time_s"] = time.perf_counter() - t_start
    results["config"] = {
        "epochs": epochs,
        "lr": lr,
        "tfidf_dim": tfidf_dim,
        "reservoir_dim": reservoir_dim,
        "max_sentences": max_sentences,
        "max_train": max_train,
        "max_test": max_test,
    }
    results["learned_params"] = sum(
        rd._heads[tid].size for tid, rd in readouts.items() if tid in rd._heads
    )
    results["fixed_params"] = mt_adapter._W.size

    # ── 9. Print summary ───────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*65}")
    print(f"  SST-2 accuracy before  (A_before) : {a_before:.4f}  ({a_before:.1%})")
    print(f"  SST-2 accuracy after B (A_after_B): {results.get('A_after_B', float('nan')):.4f}  ({results.get('A_after_B', float('nan')):.1%})")
    print(f"  SST-2 accuracy after C (A_after_C): {a_after:.4f}  ({a_after:.1%})")
    print(f"  AG News accuracy       (B_acc)    : {b_acc:.4f}  ({b_acc:.1%})")
    print(f"  Emotion accuracy       (C_acc)    : {c_acc:.4f}  ({c_acc:.1%})")
    print(f"{'─'*65}")
    forgetting_pct = forgetting * 100
    verdict = "[PASS] NEAR-ZERO FORGETTING" if abs(forgetting_pct) < 0.5 \
              else "[WARN] FORGETTING DETECTED" if forgetting_pct > 0 \
              else "[INFO] SLIGHT IMPROVEMENT (transfer)"
    print(f"  Forgetting (A_before - A_after)   : {forgetting_pct:+.4f}%  {verdict}")
    print(f"{'─'*65}")
    print(f"  Fixed params (shared encoder)  : {results['fixed_params']:,}")
    print(f"  Learned params (all heads)     : {results['learned_params']:,}")
    print(f"  Tasks stored                   : {list(readouts.keys())}")
    print(f"  Total time                     : {results['total_time_s']:.1f}s")
    print(f"{'='*65}")

    # ── 10. Interpretation ────────────────────────────────────────────────
    print("\n  INTERPRETATION:")
    print(f"  The proof suite guarantees head A's weight matrix is byte-identical")
    print(f"  before and after training B/C.  This experiment tests whether that")
    print(f"  invariant translates to accuracy preservation on real distributions.")
    if abs(forgetting_pct) < 0.5:
        print(f"\n  [PASS] Result: Forgetting of {forgetting_pct:+.4f}% is within")
        print(f"     measurement noise.  The structural isolation claim holds on")
        print(f"     real data.")
    elif forgetting_pct > 0:
        print(f"\n  [WARN] Result: {forgetting_pct:.4f}% forgetting detected.")
        print(f"     Head weights are intact -- the drop is from shared TF-IDF")
        print(f"     vocabulary interaction.  Consider per-task vocabulary isolation.")
    else:
        print(f"\n  [INFO] Result: Slight negative forgetting (improvement) -- the model")
        print(f"     may have benefited from broader vocabulary exposure.")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Real-data forgetting experiment: SST-2 -> AG News -> Emotion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--max-train",     type=int,   default=None,
                   help="Max training samples per task (default: all)")
    p.add_argument("--max-test",      type=int,   default=None,
                   help="Max test samples per task (default: all)")
    p.add_argument("--epochs",        type=int,   default=200,
                   help="Training epochs per task (default: 200)")
    p.add_argument("--batch-size",    type=int,   default=64,
                   help="Mini-batch size (default: 64)")
    p.add_argument("--lr",            type=float, default=0.1,
                   help="Learning rate (default: 0.1)")
    p.add_argument("--tfidf-dim",     type=int,   default=3000,
                   help="TF-IDF vocabulary size (default: 3000)")
    p.add_argument("--reservoir-dim", type=int,   default=1024,
                   help="Fixed encoder width (default: 1024)")
    p.add_argument("--max-sentences", type=int,   default=3,
                   help="Sentence chunks per document (default: 3)")
    p.add_argument("--output",        type=str,   default="outputs/real_forgetting_result.json",
                   help="Output JSON path")
    args = p.parse_args()

    results = run_experiment(
        max_train=args.max_train,
        max_test=args.max_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        tfidf_dim=args.tfidf_dim,
        reservoir_dim=args.reservoir_dim,
        max_sentences=args.max_sentences,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  Results saved -> {out_path}")


if __name__ == "__main__":
    main()
