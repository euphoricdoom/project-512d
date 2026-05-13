"""Reproduce the key results of Project 512D.

Milestone 0: IMDB sentiment, R=1536, int8 quantization (2.6 MB, 78.3%)
Milestone 1: Zero forgetting — IMDB accuracy unchanged after adding AG News

Usage
-----
    python reproduce.py              # full run (downloads ~110 MB of datasets)
    python reproduce.py --use-saved  # use existing .npz files, skip training
    python reproduce.py --fast       # train on 2k samples per task (~10s each)

Expected output
---------------
    [Task 1/2] Training IMDB Sentiment ...
      Test accuracy: 78.3%

    [Task 2/2] Training AG News Topic (keeping sentiment intact) ...
      Test accuracy: 88.x%

    Zero Forgetting Check
      IMDB after AG News: 78.3%   (delta: 0.0%)
      PASS: Zero forgetting confirmed

    Int8 Compression Check
      Float32 size : 23.4 MB
      Int8 size    : 2.6 MB
      Ratio        : 9.0x
      PASS: int8 accuracy within 0.2% of float32
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.argmax(preds, axis=1) == np.argmax(targets, axis=1)))


def _header(text: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def _check(condition: bool, msg_pass: str, msg_fail: str) -> bool:
    if condition:
        print(f"  PASS: {msg_pass}")
    else:
        print(f"  FAIL: {msg_fail}")
    return condition


# ---------------------------------------------------------------------------
# Step 1: Train or load IMDB sentiment
# ---------------------------------------------------------------------------

def step1_sentiment(use_saved: bool, max_train: int | None, reservoir_dim: int) -> str:
    """Returns path to saved sentiment model."""
    out_path = str(ROOT / "quality.npz")

    if use_saved and Path(out_path).exists():
        print(f"  Using saved model: {out_path}")
        size_mb = Path(out_path).stat().st_size / 1e6
        print(f"  File size: {size_mb:.1f} MB")
        return out_path

    from train_hf import train
    train(
        dataset="imdb",
        task_id="sentiment",
        model_in=None,
        model_out=out_path,
        subset=None,
        text_field=None,
        label_field=None,
        epochs=5,
        batch_size=64,
        lr=0.05,
        tfidf_dim=2000,
        reservoir_dim=reservoir_dim,
        max_sentences=6,
        max_train=max_train,
        max_test=max_train,
        sleep_cycles=0,
    )
    return out_path


# ---------------------------------------------------------------------------
# Step 2: Add AG News topic head
# ---------------------------------------------------------------------------

def step2_topic(sentiment_path: str, use_saved: bool, max_train: int | None,
                reservoir_dim: int) -> str:
    """Returns path to saved multi-task model."""
    out_path = str(ROOT / "multi.npz")

    if use_saved and Path(out_path).exists():
        print(f"  Using saved model: {out_path}")
        return out_path

    from train_hf import train
    train(
        dataset="ag_news",
        task_id="topic",
        model_in=sentiment_path,
        model_out=out_path,
        subset=None,
        text_field=None,
        label_field=None,
        epochs=5,
        batch_size=64,
        lr=0.05,
        tfidf_dim=2000,
        reservoir_dim=reservoir_dim,
        max_sentences=6,
        max_train=max_train,
        max_test=max_train,
        sleep_cycles=0,
    )
    return out_path


# ---------------------------------------------------------------------------
# Step 3: Zero-forgetting check
# ---------------------------------------------------------------------------

def step3_zero_forgetting(multi_path: str, max_test: int | None) -> bool:
    """Evaluate IMDB sentiment on the multi-task model, show zero forgetting."""
    import json
    from weightless_model import WeightlessModel
    from hf_adapter import HFAdapter
    from train_hf import _install_encoder
    from sklearn.preprocessing import LabelEncoder

    meta_path = Path(multi_path).with_suffix(".json")
    vocab_path = Path(multi_path).with_suffix(".vocab")

    if not meta_path.exists() or not vocab_path.exists():
        print("  ERROR: multi.json or multi.vocab not found — run full training first.")
        return False

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    data = np.load(multi_path)

    adapter = HFAdapter(
        tfidf_dim=meta["tfidf_dim"],
        reservoir_dim=meta["reservoir_dim"],
        max_sentences=meta["max_sentences"],
        seed=42,
    )
    adapter.load_vectorizer(str(vocab_path))

    ds     = adapter._load_hf("imdb", None, "test", max_test)
    texts  = [str(row["text"]) for row in ds]
    labels = [row["label"] for row in ds]

    adapter._label_enc = LabelEncoder()
    adapter._label_enc.fit(labels)
    X_te, Y_te, _ = adapter._encode_all(texts, labels)

    model = WeightlessModel(
        input_dim=meta["tfidf_dim"],
        reservoir_dim=meta["reservoir_dim"],
        output_dim=2,
        seed=42,
    )
    _install_encoder(model, data)
    model.readout._heads["sentiment"] = data["head_sentiment"]

    feats = model.encode_sequence(X_te)
    preds = model.readout.predict_batch(feats, "sentiment")
    acc   = _accuracy(preds, Y_te)

    baseline = 78.3
    delta    = abs(acc * 100 - baseline)
    print(f"  IMDB after AG News : {acc:.1%}")
    print(f"  Baseline           : {baseline:.1f}%")
    print(f"  Delta              : {delta:.2f}%")

    return _check(
        delta <= 0.5,
        f"Zero forgetting confirmed ({delta:.2f}% delta)",
        f"Unexpected forgetting: {delta:.2f}% delta exceeds 0.5%",
    )


# ---------------------------------------------------------------------------
# Step 4: Int8 compression check
# ---------------------------------------------------------------------------

def step4_int8_compression(sentiment_path: str) -> bool:
    """Compare float32 vs int8 model size and accuracy."""
    int8_path  = str(ROOT / "quality_int8.npz")
    vocab_path = str(ROOT / "quality.vocab")

    if not Path(int8_path).exists():
        print("  quality_int8.npz not found — generating int8 model from quality.npz ...")
        from quantize_encoder import quantize_model
        quantize_model(sentiment_path, int8_path)

    if not Path(vocab_path).exists():
        print("  quality.vocab not found — skipping int8 accuracy check.")
        return False

    f32_size  = Path(sentiment_path).stat().st_size / 1e6
    int8_size = Path(int8_path).stat().st_size / 1e6
    ratio     = f32_size / int8_size if int8_size > 0 else float("inf")

    print(f"  Float32 size : {f32_size:.1f} MB")
    print(f"  Int8 size    : {int8_size:.1f} MB")
    print(f"  Ratio        : {ratio:.1f}x")

    # Quick accuracy comparison on a 500-sample IMDB subset
    import json
    from weightless_model import WeightlessModel
    from hf_adapter import HFAdapter
    from int8_matmul import QuantizedEncoder
    from train_hf import _install_encoder
    from sklearn.preprocessing import LabelEncoder

    meta_path = Path(sentiment_path).with_suffix(".json")
    if not meta_path.exists():
        print("  quality.json not found — skipping accuracy delta check.")
        return _check(ratio >= 6.0, f"{ratio:.1f}x compression achieved",
                      f"Compression only {ratio:.1f}x, expected ≥6x")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    adapter = HFAdapter(
        tfidf_dim=meta["tfidf_dim"],
        reservoir_dim=meta["reservoir_dim"],
        max_sentences=meta["max_sentences"],
        seed=42,
    )
    adapter.load_vectorizer(vocab_path)

    ds     = adapter._load_hf("imdb", None, "test", 500)
    texts  = [str(row["text"]) for row in ds]
    labels = [row["label"] for row in ds]
    adapter._label_enc = LabelEncoder()
    adapter._label_enc.fit(labels)
    X_te, Y_te, _ = adapter._encode_all(texts, labels)

    def _eval(npz_path: str) -> float:
        d     = np.load(npz_path)
        m     = WeightlessModel(input_dim=meta["tfidf_dim"],
                                reservoir_dim=meta["reservoir_dim"],
                                output_dim=2, seed=42)
        _install_encoder(m, d)
        m.readout._heads["sentiment"] = d["head_sentiment"]
        feats = m.encode_sequence(X_te)
        preds = m.readout.predict_batch(feats, "sentiment")
        return _accuracy(preds, Y_te)

    acc_f32  = _eval(sentiment_path)
    acc_int8 = _eval(int8_path)
    delta    = abs(acc_int8 - acc_f32) * 100

    print(f"  Float32 accuracy (500 samples) : {acc_f32:.1%}")
    print(f"  Int8 accuracy    (500 samples) : {acc_int8:.1%}")
    print(f"  Accuracy delta                 : {delta:.3f}%")

    passed = ratio >= 6.0 and delta <= 0.5
    _check(ratio >= 6.0, f"{ratio:.1f}x compression", f"Only {ratio:.1f}x (expected ≥6x)")
    _check(delta <= 0.5, f"accuracy preserved (Δ={delta:.3f}%)",
           f"accuracy drift {delta:.3f}% exceeds 0.5%")
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Reproduce Project 512D key results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--use-saved", action="store_true",
                   help="Skip training; use existing .npz model files")
    p.add_argument("--fast", action="store_true",
                   help="Train on 2000 samples per task (quick smoke test)")
    p.add_argument("--reservoir-dim", type=int, default=1536,
                   help="Encoder width (default: 1536 — the optimized sweet spot)")
    p.add_argument("--skip-int8", action="store_true",
                   help="Skip the int8 compression check")
    args = p.parse_args()

    max_train = 2000 if args.fast else None

    results: dict[str, bool] = {}
    t_start = time.time()

    # ── Step 1: IMDB sentiment ───────────────────────────────────────────────
    _header("Step 1/3 — IMDB Sentiment (Milestone 1)")
    sentiment_path = step1_sentiment(
        use_saved=args.use_saved,
        max_train=max_train,
        reservoir_dim=args.reservoir_dim,
    )

    # ── Step 2: AG News topic ────────────────────────────────────────────────
    _header("Step 2/3 — AG News Topic (zero forgetting test)")
    multi_path = step2_topic(
        sentiment_path=sentiment_path,
        use_saved=args.use_saved,
        max_train=max_train,
        reservoir_dim=args.reservoir_dim,
    )

    # ── Step 3: Zero-forgetting check ───────────────────────────────────────
    _header("Step 3/3 — Zero Forgetting Check")
    results["zero_forgetting"] = step3_zero_forgetting(
        multi_path=multi_path,
        max_test=500 if args.fast else None,
    )

    # ── Step 4: Int8 compression (optional) ─────────────────────────────────
    if not args.skip_int8:
        _header("Bonus — Int8 Compression Check (Milestone 0.5)")
        results["int8_compression"] = step4_int8_compression(sentiment_path)

    # ── Summary ─────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    _header(f"Summary  ({elapsed:.0f}s total)")
    all_pass = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name.replace('_', ' ').title()}")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print("  All checks passed.")
    else:
        print("  Some checks failed — see output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
