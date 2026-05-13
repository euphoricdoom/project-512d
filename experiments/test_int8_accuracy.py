"""Compare QuantizedEncoder (int8 compute) vs FixedEncoder (float32 compute).

Loads quality.npz (float32) and quality_int8.npz (int8 weights + scale),
runs both through the full WeightlessModel pipeline on IMDB test split,
and reports the accuracy delta.

Usage
-----
    python experiments/test_int8_accuracy.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from weightless_model import WeightlessModel
from hf_adapter import HFAdapter
from int8_matmul import QuantizedEncoder

FLOAT_MODEL = str(ROOT / "quality.npz")
INT8_MODEL  = str(ROOT / "quality_int8.npz")
VOCAB_FILE  = str(ROOT / "quality.vocab")
BATCH_SIZE  = 256
N_SAMPLES   = None  # None = full 25k IMDB test set

for path in [FLOAT_MODEL, INT8_MODEL, VOCAB_FILE]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Required file not found: {path}\nRun: python reproduce.py")

print("Loading model weights...")
data_f32  = np.load(FLOAT_MODEL)
data_int8 = np.load(INT8_MODEL)

W_float32     = data_f32["encoder_W"]
head_f32      = data_f32["head_sentiment"]
W_int8        = data_int8["encoder_W_int8"]
encoder_scale = float(data_int8["encoder_scale"])
head_int8     = data_int8["head_sentiment"]

print(f"  Float32 encoder:  {W_float32.shape}  {W_float32.dtype}")
print(f"  Int8 encoder:     {W_int8.shape}  {W_int8.dtype}  scale={encoder_scale:.6f}")
print(f"  Head identical:   {np.allclose(head_f32, head_int8)}")

print("\nLoading IMDB test features (TF-IDF)...")
adapter = HFAdapter(tfidf_dim=2000, reservoir_dim=1536, max_sentences=6, seed=42)
adapter.load_vectorizer(VOCAB_FILE)

ds     = adapter._load_hf("imdb", None, "test", N_SAMPLES)
texts  = [str(row["text"]) for row in ds]
labels = [row["label"] for row in ds]

adapter._label_enc = LabelEncoder()
adapter._label_enc.fit(labels)
X_te, Y_te, label_names = adapter._encode_all(texts, labels)

print(f"  Test samples: {len(X_te)}  shape: {X_te.shape}")
print(f"  Labels: {label_names}")


def run_inference(model: WeightlessModel, X: np.ndarray, batch_size: int) -> np.ndarray:
    n, all_preds = len(X), []
    for start in range(0, n, batch_size):
        end   = min(start + batch_size, n)
        feats = model.encode_sequence(X[start:end])
        preds = model.readout.predict_batch(feats, "sentiment")
        all_preds.append(preds)
        print(f"\r  {end}/{n}", end="", flush=True)
    print()
    return np.concatenate(all_preds, axis=0)


def accuracy(preds: np.ndarray, Y: np.ndarray) -> float:
    return float(np.mean(np.argmax(preds, axis=1) == np.argmax(Y, axis=1)))


print("\nRunning float32 inference...")
model_f32 = WeightlessModel(input_dim=2000, reservoir_dim=1536, output_dim=2, seed=42)
model_f32.encoder._W             = W_float32
model_f32.readout._heads["sentiment"] = head_f32
preds_f32 = run_inference(model_f32, X_te, BATCH_SIZE)
acc_f32   = accuracy(preds_f32, Y_te)

print("Running int8 inference...")
model_int8 = WeightlessModel(input_dim=2000, reservoir_dim=1536, output_dim=2, seed=42)
model_int8.encoder               = QuantizedEncoder(W_int8, encoder_scale)
model_int8.readout._heads["sentiment"] = head_int8
preds_int8 = run_inference(model_int8, X_te, BATCH_SIZE)
acc_int8   = accuracy(preds_int8, Y_te)

delta_pp = (acc_int8 - acc_f32) * 100

print()
print("=" * 60)
print("INT8 COMPUTE ACCURACY TEST — IMDB SENTIMENT")
print("=" * 60)
print(f"  Test samples:      {len(X_te):,}")
print(f"  Float32 accuracy:  {acc_f32:.4%}")
print(f"  Int8 accuracy:     {acc_int8:.4%}")
print(f"  Delta:             {delta_pp:+.4f}%")

n_differ = int(np.sum(np.argmax(preds_f32, axis=1) != np.argmax(preds_int8, axis=1)))
print(f"  Predictions that changed: {n_differ}/{len(X_te)}  ({100*n_differ/len(X_te):.2f}%)")
print()

if abs(delta_pp) <= 0.2:
    print("PASS: Int8 compute preserves accuracy within +/-0.2%")
elif abs(delta_pp) <= 0.5:
    print(f"BORDERLINE: delta {abs(delta_pp):.3f}% is within +/-0.5%")
else:
    print(f"FAIL: delta {abs(delta_pp):.3f}% exceeds +/-0.5% threshold")
    sys.exit(1)
