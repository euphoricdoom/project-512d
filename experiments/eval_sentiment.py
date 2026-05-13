"""Evaluate IMDB sentiment accuracy on a multi-task model after AG News training.

Demonstrates zero forgetting: the sentiment head is intact despite training a
second task (AG News topic classification) on the same model.

Usage
-----
    python experiments/eval_sentiment.py
"""
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).parent.parent

# Add root to path so root-level modules are importable
import sys
sys.path.insert(0, str(ROOT))

from hf_adapter import HFAdapter
from weightless_model import WeightlessModel

data    = np.load(ROOT / "multi.npz")
meta_path = ROOT / "multi.json"

if not meta_path.exists():
    raise FileNotFoundError("multi.json not found — run: python train_hf.py --dataset ag_news --task topic --model-in sentiment.npz --model-out multi.npz")

import json
meta = json.loads(meta_path.read_text(encoding="utf-8"))

model = WeightlessModel(
    input_dim=meta["tfidf_dim"],
    reservoir_dim=meta["reservoir_dim"],
    output_dim=2,
    seed=42,
)
model.encoder._W              = data["encoder_W"].astype(np.float32) if "encoder_W" in data else None
model.readout._heads["sentiment"] = data["head_sentiment"]

adapter = HFAdapter(tfidf_dim=meta["tfidf_dim"], reservoir_dim=meta["reservoir_dim"],
                    max_sentences=meta["max_sentences"], seed=42)
adapter.load_vectorizer(str(ROOT / "multi.vocab"))

ds     = adapter._load_hf("imdb", None, "test", None)
texts  = [str(row["text"]) for row in ds]
labels = [row["label"] for row in ds]

adapter._label_enc = LabelEncoder()
adapter._label_enc.fit(labels)

X_te, Y_te, _ = adapter._encode_all(texts, labels)
feats = model.encode_sequence(X_te)
preds = model.readout.predict_batch(feats, "sentiment")
acc   = float(np.mean(np.argmax(preds, axis=1) == np.argmax(Y_te, axis=1)))

print(f"IMDB after AG News : {acc:.1%}")
print(f"Baseline           : 78.3%")
print(f"Forgetting         : {78.3 - acc*100:.1f}%")
