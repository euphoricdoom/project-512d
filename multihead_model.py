"""Multi-head perception + coordination model.

Wraps a WeightlessModel with multiple isolated task heads and a shared
fixed encoder, providing a unified perceive() interface that runs all
heads in one forward pass over the same feature vector.

Architecture:
    text → TF-IDF → FixedEncoder → SystemMemory features → [head_1, head_2, head_3]

Zero forgetting is structural: each head is a completely isolated weight
matrix; no gradient from head B can touch head A.

Usage
-----
    from multihead_model import MultiHeadModel

    model = MultiHeadModel.load("multihead.npz")

    outputs  = model.perceive("URGENT: server is completely down!")
    decision = MultiHeadModel.decide(outputs)

    print(outputs["urgency"]["label"])   # "critical"
    print(outputs["sentiment"]["label"]) # "negative"
    print(decision)                      # "escalate"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from hf_adapter import HFAdapter
from weightless_model import WeightlessModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# ---------------------------------------------------------------------------
# MultiHeadModel
# ---------------------------------------------------------------------------

class MultiHeadModel:
    """Weightless multi-head perception model.

    Args:
        adapter:  Fitted HFAdapter providing text → TF-IDF encoding.
        model:    WeightlessModel whose readout holds all task heads.
        task_map: Dict mapping task_id → list of class label strings
                  in the order corresponding to head output indices
                  (i.e. alphabetical — matches LabelEncoder default).
    """

    def __init__(
        self,
        adapter: HFAdapter,
        model: WeightlessModel,
        task_map: dict[str, list[str]],
    ) -> None:
        self.adapter  = adapter
        self.model    = model
        self.task_map = task_map   # {task_id: [label_name, ...]}

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def perceive(self, text: str) -> dict[str, dict[str, Any]]:
        """Encode text once, run all heads, return structured outputs.

        Args:
            text: Raw input string.

        Returns:
            Dict keyed by task_id, each value a dict with:
              ``label``      — winning class name string.
              ``confidence`` — softmax probability of winning class (0–1).
              ``probs``      — dict mapping every class name → probability.
        """
        seq   = self.adapter._encode_text(text)          # (T, vocab)
        X_seq = seq[np.newaxis, :, :]                    # (1, T, vocab)
        feats = self.model.encode_sequence(X_seq)        # (1, feature_dim)

        result: dict[str, dict[str, Any]] = {}
        for task_id, label_names in self.task_map.items():
            logits     = self.model.readout.predict_batch(feats, task_id)  # (1, C)
            probs      = _softmax(logits[0])                               # (C,)
            pred_idx   = int(np.argmax(probs))
            result[task_id] = {
                "label":      label_names[pred_idx],
                "confidence": float(probs[pred_idx]),
                "probs":      {name: float(p)
                               for name, p in zip(label_names, probs)},
            }
        return result

    @staticmethod
    def decide(outputs: dict[str, dict[str, Any]]) -> str:
        """Coordinate head outputs into a single action decision.

        Rules (in priority order):
          1. escalate — combined high+critical urgency probability > 0.8
          2. watch    — sentiment is negative
          3. normal   — everything else

        Args:
            outputs: Return value of :meth:`perceive`.

        Returns:
            One of ``"escalate"``, ``"watch"``, ``"normal"``.
        """
        urgency_probs = outputs.get("urgency", {}).get("probs", {})
        p_urgent      = urgency_probs.get("high", 0.0) + urgency_probs.get("critical", 0.0)

        sentiment_label = outputs.get("sentiment", {}).get("label", "neutral")

        if p_urgent > 0.8:
            return "escalate"
        if sentiment_label == "negative":
            return "watch"
        return "normal"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model to <path>.npz + <path>.json + <path>.vocab."""
        from train_hf import save_model
        save_model(self.model, self.adapter, path, self.task_map)

    @classmethod
    def load(cls, path: str) -> "MultiHeadModel":
        """Load model saved by :meth:`save` or ``train_hf.save_model``.

        Expects ``<path>.npz``, ``<path>.json``, and ``<path>.vocab``.
        """
        from train_hf import _install_encoder

        path_p = Path(path)
        meta   = json.loads(path_p.with_suffix(".json").read_text(encoding="utf-8"))
        data   = np.load(path_p)

        adapter = HFAdapter(
            tfidf_dim    = meta["tfidf_dim"],
            reservoir_dim= meta["reservoir_dim"],
            max_sentences= meta["max_sentences"],
            seed         = 42,
        )
        adapter.load_vectorizer(str(path_p.with_suffix(".vocab")))

        model = WeightlessModel(
            input_dim    = adapter.input_dim,
            reservoir_dim= adapter.reservoir_dim,
            output_dim   = 4,   # placeholder — overridden by loaded heads
            seed         = 42,
        )
        _install_encoder(model, data)

        for key in data.files:
            if key.startswith("head_"):
                task_id = key[5:]
                model.readout._heads[task_id] = data[key]

        task_map = meta.get("task_map", {})
        return cls(adapter=adapter, model=model, task_map=task_map)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        tasks = list(self.task_map.keys())
        return (f"MultiHeadModel(tasks={tasks}, "
                f"reservoir_dim={self.model.reservoir_dim}, "
                f"feature_dim={self.model.feature_dim})")
