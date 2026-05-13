"""Train your own AI from HuggingFace datasets on a consumer laptop.

No GPU required.  No hours of training.  No catastrophic forgetting.

The model encodes text with a fixed random projection (weightless — never
updated) and trains only a tiny linear output head.  Adding a new task adds
one small head; every previous task is permanently intact.

Quick start
-----------
    # Sentiment analysis on IMDB (~25k samples, ~30 seconds on CPU)
    python train_hf.py --dataset imdb --task sentiment

    # Add a 4-class topic classifier WITHOUT forgetting sentiment
    python train_hf.py --dataset ag_news --task topic --model sentiment.npz

    # Test that sentiment is still intact after adding topic
    python train_hf.py --evaluate --model topic.npz --dataset imdb --task sentiment

Saved model files are tiny — just the frozen encoder config + one small head
matrix per task.  A model with 10 tasks might be 200KB total.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from weightless_model import WeightlessModel
from hf_adapter import HFAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.argmax(preds, axis=1) == np.argmax(targets, axis=1)))


def _progress(step: int, total: int, loss: float, acc: float) -> None:
    bar_len = 30
    filled  = int(bar_len * step / max(1, total))
    bar     = "#" * filled + "-" * (bar_len - filled)
    print(f"\r  [{bar}] {step:>4}/{total}  loss={loss:.4f}  acc={acc:.1%}", end="", flush=True)


def _quantize_int8(W: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize encoder weights to int8 using symmetric quantization."""
    W = W.astype(np.float32)
    abs_max = max(abs(W.min()), abs(W.max()))
    scale = abs_max / 127.0
    W_int8 = np.clip(np.round(W / scale), -128, 127).astype(np.int8)
    return W_int8, scale


def _dequantize_int8(W_int8: np.ndarray, scale: float) -> np.ndarray:
    """Dequantize int8 encoder weights back to float32."""
    return W_int8.astype(np.float32) * scale


def load_encoder_weights(data: np.lib.npyio.NpzFile) -> np.ndarray:
    """Load encoder weights, always returning float32.

    Dequantizes int8 models on load. For zero-copy int8 compute use
    _install_encoder() instead, which installs a QuantizedEncoder.
    """
    if "encoder_W_int8" in data:
        W_int8 = data["encoder_W_int8"]
        scale = float(data["encoder_scale"])
        return _dequantize_int8(W_int8, scale)
    elif "encoder_W" in data:
        return data["encoder_W"].astype(np.float32)
    else:
        raise KeyError("No encoder weights found in model file")


def _install_encoder(model: "WeightlessModel", data: np.lib.npyio.NpzFile) -> str:
    """Attach the right encoder to model based on which weights are in data.

    Int8 models get a QuantizedEncoder so inference runs int8×int8 matmul
    natively.  Float32 models keep the existing FixedEncoder._W path.

    Returns 'int8' or 'float32' to indicate the active compute path.
    """
    if "encoder_W_int8" in data:
        from int8_matmul import QuantizedEncoder
        model.encoder = QuantizedEncoder(
            data["encoder_W_int8"], float(data["encoder_scale"])
        )
        return "int8"
    elif "encoder_W" in data:
        model.encoder._W = data["encoder_W"].astype(np.float32)
        return "float32"
    else:
        raise KeyError("No encoder weights found in model file")


def save_model(model: WeightlessModel, adapter: HFAdapter,
               path: str, task_map: dict[str, list[str]]) -> None:
    """Save model weights and metadata to a .npz file.

    Encoder is automatically quantized to int8 for efficient storage.
    """
    # Quantize encoder to int8 (skip if already QuantizedEncoder to avoid round-trip loss)
    from int8_matmul import QuantizedEncoder as _QE
    if isinstance(model.encoder, _QE):
        W_int8, scale = model.encoder._W_int8, model.encoder._scale_w
    else:
        W_int8, scale = _quantize_int8(model.encoder._W)

    arrays = {
        "encoder_W_int8": W_int8,
        "encoder_scale": np.array(scale, dtype=np.float32),
    }

    # Save task heads (keep as float32 for now)
    for task_id, head in model.readout._heads.items():
        arrays[f"head_{task_id}"] = head.astype(np.float32)

    # Convert numpy types to Python types for JSON serialization
    task_map_clean = {
        k: [int(v) if isinstance(v, (int, np.integer, np.int32, np.int64)) else str(v)
            for v in vals]
        for k, vals in task_map.items()
    }

    meta = {
        "input_dim":    int(model.input_dim),
        "reservoir_dim": int(model.reservoir_dim),
        "output_dim":   int(model.output_dim),
        "task_ids":     list(model.task_ids),
        "task_map":     task_map_clean,
        "tfidf_dim":    int(adapter.tfidf_dim),
        "max_sentences": int(adapter.max_sentences),
        "quantized":    True,  # Flag for int8 encoding
    }
    np.savez_compressed(path, **arrays)
    Path(path).with_suffix(".json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    # Save the vectorizer so all tasks use the same vocabulary
    adapter.save_vectorizer(str(Path(path).with_suffix(".vocab")))

    file_size_kb = Path(path).stat().st_size // 1024
    print(f"\n  Saved -> {path}  ({file_size_kb}KB, int8 quantized)")


def load_model_meta(path: str) -> dict:
    meta_path = Path(path).with_suffix(".json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Train one task
# ---------------------------------------------------------------------------

def train(
    dataset: str,
    task_id: str,
    model_in: str | None,
    model_out: str,
    subset: str | None,
    text_field: str | None,
    label_field: str | None,
    epochs: int,
    batch_size: int,
    lr: float,
    tfidf_dim: int,
    reservoir_dim: int,
    max_sentences: int,
    max_train: int | None,
    max_test: int | None,
    sleep_cycles: int,
) -> None:
    print(f"\n{'='*60}")
    print(f"  Dataset  : {dataset}")
    print(f"  Task     : {task_id}")
    print(f"  Epochs   : {epochs}   Batch: {batch_size}   LR: {lr}")
    print(f"{'='*60}\n")

    # 1 ── Feature extraction (weightless, one-time)
    print("[ 1/4 ] Extracting features (TF-IDF + fixed encoder) …")
    t0       = time.time()
    adapter  = HFAdapter(
        tfidf_dim=tfidf_dim,
        reservoir_dim=reservoir_dim,
        max_sentences=max_sentences,
        seed=42,
    )

    # If loading existing model, use its saved vocabulary
    vocab_path = Path(model_in).with_suffix(".vocab") if model_in else None
    if vocab_path and vocab_path.exists():
        from sklearn.preprocessing import LabelEncoder
        print(f"  Loading shared vocabulary from {vocab_path}")
        adapter.load_vectorizer(str(vocab_path))
        # Transform with existing vocab (don't fit)
        ds = adapter._load_hf(dataset, subset, "train", max_train)
        t_field = text_field or adapter._detect_fields(ds[0])[0]
        l_field = label_field or adapter._detect_fields(ds[0])[1]
        texts = [str(row[t_field]) for row in ds]
        labels = [row[l_field] for row in ds]
        # Fit label encoder on THIS dataset's labels (each task has its own labels)
        adapter._label_enc = LabelEncoder()
        adapter._label_enc.fit(labels)
        X_tr, Y_tr, label_names = adapter._encode_all(texts, labels)
    else:
        # New model - fit vocabulary on this dataset
        X_tr, Y_tr, label_names = adapter.fit_transform(
            dataset, split="train", subset=subset,
            text_field=text_field, label_field=label_field,
            max_samples=max_train,
        )
    print(f"  Train : {len(X_tr):,} samples  |  classes: {label_names}")

    # Try 'test' first; fall back to 'validation' for datasets whose test
    # split is unlabeled (e.g. GLUE SST-2 returns label=-1 for all test rows).
    for test_split in ("test", "validation"):
        try:
            X_te, Y_te, _ = adapter.transform(
                dataset, split=test_split, subset=subset,
                text_field=text_field, label_field=label_field,
                max_samples=max_test,
            )
            break
        except ValueError:
            if test_split == "validation":
                raise  # both splits failed — propagate
            print(f"  test split has no labeled rows, trying validation split …")
    print(f"  Test  : {len(X_te):,} samples ({test_split})  |  extract time: {time.time()-t0:.1f}s\n")

    # 2 ── Build or load model
    print("[ 2/4 ] Building model …")
    output_dim  = adapter.output_dim
    feature_dim = adapter.max_sentences  # placeholder — set properly below
    task_map: dict[str, list[str]] = {}

    if model_in and Path(model_in).exists():
        meta    = load_model_meta(model_in)
        data    = np.load(model_in)
        model   = WeightlessModel(
            input_dim=adapter.input_dim,
            reservoir_dim=adapter.reservoir_dim,
            output_dim=output_dim,
            seed=42,
        )
        # Restore encoder — int8 models get QuantizedEncoder for native int8 compute
        compute = _install_encoder(model, data)
        # Restore all existing heads
        for key in data.files:
            if key.startswith("head_"):
                existing_tid = key[5:]
                model.readout._heads[existing_tid] = data[key]
        task_map = meta.get("task_map", {})
        compute_str = " (int8 compute)" if compute == "int8" else " (float32)"
        print(f"  Loaded model from {model_in}{compute_str}")
        print(f"  Existing tasks: {list(model.readout._heads.keys())}")
    else:
        # Pre-extract features to discover feature_dim
        probe    = adapter._encode_text("probe sentence")  # (T, vocab_size)
        from weightless_model import SystemMemory
        mem      = SystemMemory(reservoir_dim=adapter.reservoir_dim)
        feat_dim = mem.feature_dim
        model    = WeightlessModel(
            input_dim=adapter.input_dim,
            reservoir_dim=adapter.reservoir_dim,
            output_dim=output_dim,
            seed=42,
        )
        # Replace encoder W with adapter's frozen W (same spectral property)
        model.encoder._W = adapter._W
        print(f"  New model — feature dim: {feat_dim}  |  encoder params: {model.total_fixed_params:,}")

    task_map[task_id] = label_names
    model.set_task(task_id)

    # 3 ── Pre-compute all features (encoder is fixed — do it once)
    print("[ 3/4 ] Pre-computing sequence features (one-time, cached) …")
    t1          = time.time()
    feats_train = model.encode_sequence(X_tr)  # (N, feature_dim)
    feats_test  = model.encode_sequence(X_te)
    print(f"  Done in {time.time()-t1:.1f}s  |  feature shape: {feats_train.shape}\n")

    # 4 ── Train only the output head
    print("[ 4/4 ] Training output head …")
    n      = len(feats_train)
    rng    = np.random.default_rng(42)
    t2     = time.time()
    losses: list[float] = []

    for epoch in range(epochs):
        idx   = rng.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            batch = idx[start:start + batch_size]
            loss  = model.update(feats_train[batch], Y_tr[batch], lr=lr)
            epoch_loss += loss
        avg_loss = epoch_loss / max(1, n // batch_size)
        losses.append(avg_loss)
        train_acc = _accuracy(model.readout.predict_batch(feats_train, task_id), Y_tr)
        _progress(epoch + 1, epochs, avg_loss, train_acc)

    train_time = time.time() - t2
    print(f"\n  Train time: {train_time:.1f}s")

    # Evaluate
    test_preds = model.readout.predict_batch(feats_test, task_id)
    test_acc   = _accuracy(test_preds, Y_te)
    train_acc  = _accuracy(model.readout.predict_batch(feats_train, task_id), Y_tr)
    print(f"\n  Train accuracy : {train_acc:.1%}")
    print(f"  Test  accuracy : {test_acc:.1%}")

    # Optional sleep consolidation
    if sleep_cycles > 0:
        print(f"\n  Sleep consolidation ({sleep_cycles} cycles) …")
        result = model.sleep(task_id, X_tr, Y_tr, num_cycles=sleep_cycles, sleep_lr=lr * 0.1)
        test_preds_post = model.readout.predict_batch(feats_test, task_id)
        test_acc_post   = _accuracy(test_preds_post, Y_te)
        print(f"  Post-sleep test accuracy: {test_acc_post:.1%}  "
              f"(error: {result['initial_error']:.4f} -> {result['final_error']:.4f})")

    # Summary
    print(f"\n{'-'*60}")
    print(f"  Model params  : {model.total_fixed_params:,} fixed  +  "
          f"{model.total_learned_params:,} learned")
    print(f"  Tasks stored  : {model.task_ids}")
    print(f"  Memory        : ~{(model.total_fixed_params + model.total_learned_params) * 4 // 1024}KB")
    print(f"{'-'*60}")

    save_model(model, adapter, model_out, task_map)


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(
    model_path: str,
    dataset: str,
    task_id: str,
    subset: str | None,
    text_field: str | None,
    label_field: str | None,
    max_test: int | None,
) -> float:
    """Re-evaluate a saved model on a dataset test split.

    Loads the shared vocabulary from <model_path>.vocab (saved at train time),
    encodes the test split without re-fitting, and reports accuracy.

    Returns
    -------
    acc : float  Test accuracy in [0, 1].
    """
    from sklearn.preprocessing import LabelEncoder

    meta = load_model_meta(model_path)
    data = np.load(model_path)

    if task_id not in meta.get("task_map", {}):
        raise ValueError(
            f"Task '{task_id}' not found in model.  "
            f"Available tasks: {list(meta.get('task_map', {}).keys())}"
        )

    adapter = HFAdapter(
        tfidf_dim=meta["tfidf_dim"],
        reservoir_dim=meta["reservoir_dim"],
        max_sentences=meta["max_sentences"],
        seed=42,
    )

    # Load shared vocabulary saved alongside the model
    vocab_path = str(Path(model_path).with_suffix(".vocab"))
    if not Path(vocab_path).exists():
        raise FileNotFoundError(
            f"Vocabulary file not found: {vocab_path}\n"
            "The model must have been saved with save_model() which writes a .vocab file."
        )
    adapter.load_vectorizer(vocab_path)

    # Reconstruct label encoder for this task from saved task_map
    task_classes = meta["task_map"][task_id]  # e.g. [0, 1] or ["neg", "pos"]
    adapter._label_enc = LabelEncoder()
    adapter._label_enc.fit(task_classes)

    # Load test split (fall back to validation if test has no labeled rows)
    known = set(adapter._label_enc.classes_)
    # Also build an int-coerced version of known, for models saved before the
    # Python-int → str serialisation bug was fixed (classes stored as "0","1").
    known_int: set = set()
    if all(isinstance(c, str) for c in known):
        try:
            known_int = {int(c) for c in known}
        except ValueError:
            pass

    X_te = Y_te = None
    for eval_split in ("test", "validation"):
        try:
            ds = adapter._load_hf(dataset, subset, eval_split, max_test)
        except Exception:
            continue
        t_field = text_field or adapter._detect_fields(ds[0])[0]
        l_field = label_field or adapter._detect_fields(ds[0])[1]
        texts  = [str(row[t_field]) for row in ds]
        labels = [row[l_field] for row in ds]
        pairs = [(t, l) for t, l in zip(texts, labels) if l in known]
        # Fallback: match int labels against string classes ("0","1" bug)
        if not pairs and known_int:
            pairs = [(t, int(l)) for t, l in zip(texts, labels)
                     if isinstance(l, (int, np.integer)) and int(l) in known_int]
            # Remap classes to int so _encode_all can fit them
            if pairs:
                adapter._label_enc.fit(sorted(known_int))
                known = set(adapter._label_enc.classes_)
        if pairs:
            texts, labels = zip(*pairs)
            X_te, Y_te, _ = adapter._encode_all(list(texts), list(labels))
            break
    if X_te is None:
        raise ValueError(f"No rows with known labels in test or validation split of '{dataset}'.")

    # Build model and restore weights
    model = WeightlessModel(
        input_dim=adapter.input_dim,
        reservoir_dim=adapter.reservoir_dim,
        output_dim=len(task_classes),
        seed=42,
    )
    _install_encoder(model, data)
    head_key = f"head_{task_id}"
    if head_key not in data.files:
        raise KeyError(f"Head '{head_key}' not found in {model_path}")
    model.readout._heads[task_id] = data[head_key]
    model.set_task(task_id)

    feats_test = model.encode_sequence(X_te)
    preds = model.readout.predict_batch(feats_test, task_id)
    acc = _accuracy(preds, Y_te)

    print(f"\n{'='*60}")
    print(f"  Evaluate  : {task_id}  on  {dataset}")
    print(f"  Test samples : {len(X_te):,}")
    print(f"  Accuracy     : {acc:.1%}")
    print(f"{'='*60}\n")
    return acc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Train a weightless AI on any HuggingFace dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_hf.py --dataset imdb --task sentiment
  python train_hf.py --dataset ag_news --task topic --model-in sentiment.npz
  python train_hf.py --dataset yelp_review_full --task rating --epochs 3
        """,
    )
    p.add_argument("--dataset",     required=True,  help="HuggingFace dataset name")
    p.add_argument("--task",        required=True,  help="Task identifier (your label)")
    p.add_argument("--subset",      default=None,   help="Dataset subset/config name")
    p.add_argument("--text-field",  default=None,   help="Column name for text (auto-detected)")
    p.add_argument("--label-field", default=None,   help="Column name for label (auto-detected)")
    p.add_argument("--model-in",    default=None,   help=".npz model file to load (for adding tasks)")
    p.add_argument("--model-out",   default=None,   help=".npz output path (default: <task>.npz)")
    p.add_argument("--epochs",      type=int, default=5,    help="Training epochs (default: 5)")
    p.add_argument("--batch-size",  type=int, default=64,   help="Batch size (default: 64)")
    p.add_argument("--lr",          type=float, default=0.05, help="Learning rate (default: 0.05)")
    p.add_argument("--tfidf-dim",   type=int, default=2000, help="TF-IDF vocab size (default: 2000)")
    p.add_argument("--reservoir-dim", type=int, default=256, help="Encoder width (default: 256)")
    p.add_argument("--max-sentences", type=int, default=6,  help="Sentence chunks per doc (default: 6)")
    p.add_argument("--max-train",   type=int, default=None, help="Limit training samples")
    p.add_argument("--max-test",    type=int, default=None, help="Limit test samples")
    p.add_argument("--sleep-cycles", type=int, default=0,  help="Sleep consolidation cycles (default: 0)")
    p.add_argument("--evaluate",    action="store_true",    help="Evaluate only, don't train")

    args = p.parse_args()
    model_out = args.model_out or f"{args.task}.npz"

    if args.evaluate:
        if not args.model_in:
            p.error("--evaluate requires --model-in <path.npz>")
        evaluate(
            model_path=args.model_in,
            dataset=args.dataset,
            task_id=args.task,
            subset=args.subset,
            text_field=args.text_field,
            label_field=args.label_field,
            max_test=args.max_test,
        )
    else:
        train(
            dataset=args.dataset,
            task_id=args.task,
            model_in=args.model_in,
            model_out=model_out,
            subset=args.subset,
            text_field=args.text_field,
            label_field=args.label_field,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            tfidf_dim=args.tfidf_dim,
            reservoir_dim=args.reservoir_dim,
            max_sentences=args.max_sentences,
            max_train=args.max_train,
            max_test=args.max_test,
            sleep_cycles=args.sleep_cycles,
        )


if __name__ == "__main__":
    main()
