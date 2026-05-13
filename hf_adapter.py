"""HuggingFace dataset adapter for WeightlessModel.

Converts any text classification dataset from HuggingFace into sequence
feature batches the WeightlessModel can train on directly.

Pipeline
--------
  Text  →  Sentence chunks  →  TF-IDF vectors  →  Fixed encoder  →  SystemMemory features

The feature extraction step (TF-IDF + fixed encoder) is completely weightless:
once fitted on the training vocabulary it is frozen forever.  The only learned
parameters are the per-task output heads in WeightlessModel.

Minimum install:
    pip install datasets scikit-learn

Usage
-----
    from hf_adapter import HFAdapter, load_and_prep

    adapter = HFAdapter(input_dim=1000, reservoir_dim=256)
    X_train, Y_train, label_names = adapter.fit_transform("imdb", split="train")
    X_test,  Y_test,  _           = adapter.transform("imdb", split="test")
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder
except ImportError as e:
    raise ImportError(
        "scikit-learn is required: pip install scikit-learn"
    ) from e


# ---------------------------------------------------------------------------
# Text → sentences
# ---------------------------------------------------------------------------

def _split_sentences(text: str, max_sentences: int = 8) -> list[str]:
    """Split text into up to max_sentences chunks."""
    text = str(text).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    # Merge very short fragments
    merged: list[str] = []
    buf = ""
    for part in parts:
        buf = (buf + " " + part).strip()
        if len(buf) >= 40:
            merged.append(buf)
            buf = ""
    if buf:
        merged.append(buf)
    if not merged:
        merged = [text[:500]]
    # Pad by repeating the last real sentence (NOT empty strings which destroy reservoir signal)
    while len(merged) < max_sentences:
        merged.append(merged[-1])
    return merged[:max_sentences]


# ---------------------------------------------------------------------------
# HFAdapter
# ---------------------------------------------------------------------------

class HFAdapter:
    """Adapter from HuggingFace text datasets to WeightlessModel sequences.

    Args:
        tfidf_dim:       TF-IDF vocabulary size (number of input features).
        reservoir_dim:   Fixed encoder output width.
        max_sentences:   Max sentence chunks per document (= sequence length T).
        target_radius:   Spectral bound for the fixed encoder.
        seed:            Reproducibility seed.
    """

    def __init__(
        self,
        tfidf_dim: int = 2000,
        reservoir_dim: int = 256,
        max_sentences: int = 6,
        target_radius: float = 0.94,
        seed: int = 42,
    ) -> None:
        self.tfidf_dim      = int(tfidf_dim)
        self.reservoir_dim  = int(reservoir_dim)
        self.max_sentences  = int(max_sentences)
        self.target_radius  = float(target_radius)
        self.seed           = int(seed)
        self._vectorizer: TfidfVectorizer | None = None
        self._label_enc: LabelEncoder | None = None
        self._W: np.ndarray | None = None   # frozen encoder matrix

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_encoder(self) -> None:
        # Use the actual fitted vocabulary size, not the requested tfidf_dim
        actual_vocab = (len(self._vectorizer.vocabulary_)
                        if self._vectorizer is not None else self.tfidf_dim)
        rng  = np.random.default_rng(self.seed)
        W    = rng.normal(0.0, 1.0 / np.sqrt(actual_vocab),
                          size=(self.reservoir_dim, actual_vocab))
        sv   = np.linalg.svd(W, compute_uv=False)
        if sv[0] > self.target_radius:
            W *= self.target_radius / sv[0]
        self._W = W

    def _encode_text(self, text: str) -> np.ndarray:
        """Text → (max_sentences, vocab_size) TF-IDF features."""
        sentences = _split_sentences(text, self.max_sentences)
        tfidf_mat = self._vectorizer.transform(sentences).toarray()  # (T, vocab_size)
        # Pad or truncate to exactly max_sentences
        T = tfidf_mat.shape[0]
        vocab_size = tfidf_mat.shape[1]
        result = np.zeros((self.max_sentences, vocab_size), dtype=np.float32)
        result[:min(T, self.max_sentences)] = tfidf_mat[:self.max_sentences]
        return result

    @staticmethod
    def _load_hf(dataset_name: str, subset: str | None, split: str,
                  max_samples: int | None) -> Any:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "HuggingFace datasets required: pip install datasets"
            ) from exc
        ds = load_dataset(dataset_name, subset, split=split, trust_remote_code=False)
        if max_samples and len(ds) > max_samples:
            ds = ds.shuffle(seed=42).select(range(max_samples))
        return ds

    def _detect_fields(self, ds_row: dict) -> tuple[str, str]:
        """Auto-detect text and label fields from the first row."""
        text_candidates  = ["text", "sentence", "review", "content", "abstract", "body"]
        label_candidates = ["label", "labels", "category", "sentiment", "target"]
        keys = list(ds_row.keys())
        text_field  = next((k for k in text_candidates if k in keys), None)
        label_field = next((k for k in label_candidates if k in keys), None)
        if text_field is None:
            # Pick first string field
            text_field = next((k for k in keys if isinstance(ds_row[k], str)), keys[0])
        if label_field is None:
            label_field = next((k for k in keys if k != text_field), keys[-1])
        return text_field, label_field

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        dataset_name: str,
        split: str = "train",
        subset: str | None = None,
        text_field: str | None = None,
        label_field: str | None = None,
        max_samples: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Load, fit TF-IDF on this split, and return (X_seq, Y_onehot, label_names).

        Call this once on your training split, then use `transform()` for all
        subsequent splits.  The TF-IDF vocabulary is never updated again after
        this call.

        Returns
        -------
        X_seq : (n_samples, max_sentences, vocab_size)  TF-IDF features
        Y_onehot : (n_samples, n_classes)  one-hot encoded labels
        label_names : list of class name strings
        """
        ds = self._load_hf(dataset_name, subset, split, max_samples)

        # Detect fields
        t_field = text_field or self._detect_fields(ds[0])[0]
        l_field = label_field or self._detect_fields(ds[0])[1]

        texts  = [str(row[t_field]) for row in ds]
        labels = [row[l_field] for row in ds]

        # Fit TF-IDF on all sentences from all documents
        all_sents = []
        for text in texts:
            all_sents.extend(_split_sentences(text, self.max_sentences))
        self._vectorizer = TfidfVectorizer(
            max_features=self.tfidf_dim,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"(?u)\b\w\w+\b",
            ngram_range=(1, 2),
        )
        self._vectorizer.fit(all_sents)

        # Fit label encoder
        self._label_enc = LabelEncoder()
        self._label_enc.fit(labels)

        # Build fixed encoder once
        self._build_encoder()

        return self._encode_all(texts, labels)

    def transform(
        self,
        dataset_name: str,
        split: str = "test",
        subset: str | None = None,
        text_field: str | None = None,
        label_field: str | None = None,
        max_samples: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Transform a new split using the already-fitted TF-IDF vocabulary."""
        if self._vectorizer is None or self._W is None:
            raise RuntimeError("Call fit_transform() before transform()")

        ds = self._load_hf(dataset_name, subset, split, max_samples)
        t_field = text_field or self._detect_fields(ds[0])[0]
        l_field = label_field or self._detect_fields(ds[0])[1]

        texts  = [str(row[t_field]) for row in ds]
        labels = [row[l_field] for row in ds]

        # Filter rows whose label was not seen during fit_transform (e.g. SST-2
        # test split contains label=-1 for unlabeled examples).
        known = set(self._label_enc.classes_)
        pairs = [(t, l) for t, l in zip(texts, labels) if l in known]
        if not pairs:
            raise ValueError(
                f"No rows with known labels found in split '{split}' of "
                f"'{dataset_name}'.  Known classes: {sorted(known)}"
            )
        texts, labels = zip(*pairs)
        texts, labels = list(texts), list(labels)

        return self._encode_all(texts, labels)

    def _encode_all(
        self, texts: list[str], labels: list
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        n   = len(texts)
        vocab_size = len(self._vectorizer.vocabulary_)
        X   = np.zeros((n, self.max_sentences, vocab_size), dtype=np.float32)
        for i, text in enumerate(texts):
            X[i] = self._encode_text(text)

        int_labels = self._label_enc.transform(labels)
        n_classes  = len(self._label_enc.classes_)
        Y          = np.eye(n_classes, dtype=np.float32)[int_labels]

        return X, Y, list(self._label_enc.classes_)

    @property
    def output_dim(self) -> int:
        if self._label_enc is None:
            raise RuntimeError("Call fit_transform() first")
        return len(self._label_enc.classes_)

    @property
    def input_dim(self) -> int:
        if self._vectorizer is None:
            raise RuntimeError("Call fit_transform() first")
        return len(self._vectorizer.vocabulary_)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def fit(self, texts: list[str]) -> None:
        """Fit TF-IDF vocabulary on raw text strings (no labels, no HF download).

        Call this once on the combined training corpus of all tasks, then use
        ``encode_texts()`` per task.  The vocabulary is frozen after this call.
        """
        all_sents: list[str] = []
        for text in texts:
            all_sents.extend(_split_sentences(text, self.max_sentences))
        self._vectorizer = TfidfVectorizer(
            max_features=self.tfidf_dim,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"(?u)\b\w\w+\b",
            ngram_range=(1, 2),
        )
        self._vectorizer.fit(all_sents)
        self._build_encoder()

    def encode_texts(
        self,
        texts: list[str],
        labels: list[str],
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Encode raw texts and string labels using the fitted vocabulary.

        Args:
            texts:  List of raw text strings.
            labels: Parallel list of string class labels.

        Returns:
            X_seq      : ``(n, max_sentences, vocab_size)`` TF-IDF features.
            Y_onehot   : ``(n, n_classes)`` one-hot labels (alphabetical class order).
            label_names: Sorted list of unique class name strings.
        """
        if self._vectorizer is None or self._W is None:
            raise RuntimeError("Call fit() before encode_texts()")
        try:
            from sklearn.preprocessing import LabelEncoder
        except ImportError as exc:
            raise ImportError("scikit-learn required: pip install scikit-learn") from exc

        le = LabelEncoder()
        le.fit(sorted(set(labels)))
        int_labels = le.transform(labels)
        n_classes = len(le.classes_)
        Y = np.eye(n_classes, dtype=np.float32)[int_labels]

        n = len(texts)
        vocab_size = len(self._vectorizer.vocabulary_)
        X = np.zeros((n, self.max_sentences, vocab_size), dtype=np.float32)
        for i, text in enumerate(texts):
            X[i] = self._encode_text(text)

        return X, Y, list(le.classes_)

    def save_vectorizer(self, path: str) -> None:
        """Save the fitted TF-IDF vectorizer to disk."""
        import pickle
        if self._vectorizer is None:
            raise RuntimeError("No vectorizer to save - call fit_transform() first")
        with open(path, "wb") as f:
            pickle.dump(self._vectorizer, f)

    def load_vectorizer(self, path: str) -> None:
        """Load a fitted TF-IDF vectorizer from disk."""
        import pickle
        with open(path, "rb") as f:
            self._vectorizer = pickle.load(f)
        # Rebuild encoder with the loaded vocabulary
        self._build_encoder()
