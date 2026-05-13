"""Quickstart: train an IMDB sentiment model in under 60 seconds.

Demonstrates the core API end-to-end using the new modular training package.
No GPU required. No PyTorch. No catastrophic forgetting.

Run with:
    python examples/quickstart.py

Expected output:
    [1/3] Configuring...
    [2/3] Loading data (2 000 train / 500 test)...
    [3/3] Training (5 epochs)...
    ...
    Final accuracy: ~75-78%
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make root importable when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from config.model_config import ModelConfig
from hf_adapter import HFAdapter
from training.dataset import DatasetConfig, DatasetLoader
from training.evaluation import accuracy, classification_report
from training.trainer import WeightlessTrainer
from weightless_model import WeightlessModel


def main() -> None:
    print("=" * 60)
    print("  Project 512D Quickstart — IMDB Sentiment")
    print("  Zero forgetting · No GPU · numpy only")
    print("=" * 60)

    # 1. Configure
    print("\n[1/3] Configuring...")
    config = ModelConfig.default_imdb()
    print(f"  reservoir_dim : {config.encoder.reservoir_dim:,}")
    print(f"  tfidf_dim     : {config.tfidf.tfidf_dim:,}")
    print(f"  feature_dim   : {config.memory.feature_dim:,}")

    # 2. Load dataset (small subset for speed)
    print("\n[2/3] Loading IMDB (2 000 train / 500 test)...")
    dataset_cfg = DatasetConfig(name="imdb", max_train=2000, max_test=500)
    adapter = HFAdapter(
        tfidf_dim=config.tfidf.tfidf_dim,
        reservoir_dim=config.encoder.reservoir_dim,
        max_sentences=config.tfidf.max_sentences,
        seed=42,
    )
    loader = DatasetLoader(dataset_cfg)
    X_tr, Y_tr, X_te, Y_te, label_names = loader.load(
        adapter=None,
        reservoir_dim=config.encoder.reservoir_dim,
        tfidf_dim=config.tfidf.tfidf_dim,
        max_sentences=config.tfidf.max_sentences,
    )
    print(f"  Train: {len(X_tr):,}  Test: {len(X_te):,}  Classes: {label_names}")

    # 3. Build model
    model = WeightlessModel(
        input_dim=X_tr.shape[2],
        reservoir_dim=config.encoder.reservoir_dim,
        output_dim=len(label_names),
        target_radius=config.encoder.target_radius,
        ac_decay=config.memory.ac_decay,
        dc_decay=config.memory.dc_decay,
        lr=config.training.learning_rate,
        seed=42,
    )

    # 4. Train
    print("\n[3/3] Training (5 epochs)...")
    trainer = WeightlessTrainer(model, config.training, verbose=True)
    result  = trainer.train(X_tr, Y_tr, X_te, Y_te, task_id="sentiment")

    # 5. Summary
    report = classification_report(model, X_te, Y_te, label_names, "sentiment")
    print("\n" + "=" * 60)
    print(f"  Final accuracy : {result['final_accuracy']:.1%}")
    print(f"  Macro F1       : {report['macro_f1']:.4f}")
    print(f"  Fixed params   : {model.total_fixed_params:,}  (encoder, never updated)")
    print(f"  Learned params : {model.total_learned_params:,}  (readout head only)")
    print("=" * 60)


if __name__ == "__main__":
    main()
