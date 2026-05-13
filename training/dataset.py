"""Dataset loading and preprocessing for HuggingFace text classification datasets.

Provides a thin, typed wrapper around the HFAdapter so callers only need to
specify what they want (dataset name, limits) rather than how to load it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class DatasetConfig:
    """Specifies a HuggingFace text classification dataset.

    Args:
        name:        HuggingFace dataset identifier, e.g. ``"imdb"``.
        subset:      Optional subset/config name, e.g. ``"sst2"`` for GLUE.
        text_field:  Column name for the text (auto-detected if not given).
        label_field: Column name for the label (auto-detected if not given).
        max_train:   Cap on training samples (None = use all).
        max_test:    Cap on test/validation samples (None = use all).
    """

    name: str
    subset: Optional[str] = None
    text_field: Optional[str] = None
    label_field: Optional[str] = None
    max_train: Optional[int] = None
    max_test: Optional[int] = None


class DatasetLoader:
    """Load and encode a HuggingFace dataset using a fitted HFAdapter.

    Args:
        config:  Dataset specification.
        adapter: A fitted ``HFAdapter`` instance (or None to create one).

    Usage::

        config = DatasetConfig(name="imdb", max_train=5000)
        loader = DatasetLoader(config)
        X_tr, Y_tr, X_te, Y_te, label_names = loader.load()
    """

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    def load(
        self,
        adapter=None,
        reservoir_dim: int = 1536,
        tfidf_dim: int = 2000,
        max_sentences: int = 6,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        """Load and encode train + test splits.

        If *adapter* is provided and already fitted, it is used directly (shared
        vocabulary mode — required when adding a second task to an existing model).
        Otherwise a new adapter is created and fitted on the training split.

        Args:
            adapter:       Pre-fitted ``HFAdapter``, or None to create and fit one.
            reservoir_dim: Reservoir dimension (ignored if *adapter* provided).
            tfidf_dim:     TF-IDF vocab size (ignored if *adapter* provided).
            max_sentences: Sentence chunks per document (ignored if *adapter* provided).

        Returns:
            Tuple ``(X_train, Y_train, X_test, Y_test, label_names)`` where:
              - ``X_train``: ``(n_train, max_sentences, vocab_size)`` float32
              - ``Y_train``: ``(n_train, n_classes)`` one-hot float32
              - ``X_test``:  ``(n_test, max_sentences, vocab_size)`` float32
              - ``Y_test``:  ``(n_test, n_classes)`` one-hot float32
              - ``label_names``: list of class name strings
        """
        from hf_adapter import HFAdapter

        cfg = self.config

        if adapter is None:
            adapter = HFAdapter(
                tfidf_dim=tfidf_dim,
                reservoir_dim=reservoir_dim,
                max_sentences=max_sentences,
                seed=42,
            )
            X_tr, Y_tr, label_names = adapter.fit_transform(
                cfg.name,
                split="train",
                subset=cfg.subset,
                text_field=cfg.text_field,
                label_field=cfg.label_field,
                max_samples=cfg.max_train,
            )
        else:
            from sklearn.preprocessing import LabelEncoder

            ds = adapter._load_hf(cfg.name, cfg.subset, "train", cfg.max_train)
            t_field = cfg.text_field or adapter._detect_fields(ds[0])[0]
            l_field = cfg.label_field or adapter._detect_fields(ds[0])[1]
            texts = [str(row[t_field]) for row in ds]
            labels = [row[l_field] for row in ds]
            adapter._label_enc = LabelEncoder()
            adapter._label_enc.fit(labels)
            X_tr, Y_tr, label_names = adapter._encode_all(texts, labels)

        # Test split — fall back to validation for unlabelled test splits
        X_te = Y_te = None
        for split in ("test", "validation"):
            try:
                X_te, Y_te, _ = adapter.transform(
                    cfg.name,
                    split=split,
                    subset=cfg.subset,
                    text_field=cfg.text_field,
                    label_field=cfg.label_field,
                    max_samples=cfg.max_test,
                )
                break
            except ValueError:
                if split == "validation":
                    raise

        return X_tr, Y_tr, X_te, Y_te, label_names
