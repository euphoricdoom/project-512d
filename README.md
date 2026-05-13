# Project 512D — Weightless Continual Learning

![Tests](https://github.com/euphoricdoom/project-512d/workflows/Tests/badge.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**v0.2**

A zero-forgetting AI substrate for continual learning on consumer hardware.

**No GPU. No PyTorch. No catastrophic forgetting.**

The model encodes text with a fixed random projection (never updated), trains only a tiny linear head per task, and achieves zero forgetting by construction. Adding a new task adds one small weight matrix; all previous tasks are permanently intact.

## Key Results

| Metric | Value |
|---|---|
| IMDB sentiment accuracy | 78.3% |
| Zero forgetting (IMDB after AG News) | 0.0% degradation |
| Model size (int8, R=1536) | **2.6 MB** |
| Compression vs float64 baseline | **9x** |
| Training hardware | CPU only |
| Training time (IMDB, 25k samples) | ~30 seconds |

## Quick Start

```bash
# 1. Install dependencies (Python 3.11+ required)
pip install -r requirements.txt

# 2. Train sentiment on IMDB (~30s)
python train_hf.py --dataset imdb --task sentiment

# 3. Add topic classification without forgetting
python train_hf.py --dataset ag_news --task topic --model-in sentiment.npz --model-out multi.npz

# 4. Reproduce the key results end-to-end
python reproduce.py
```

## What This Is

The core claim: **catastrophic forgetting is eliminated by construction, not approximation.**

Three immutable rules:
1. The encoder (fixed random projection) is **never updated** after initialization.
2. Each task gets its **own isolated linear head** — task B's gradient cannot touch task A's weights.
3. Sleep consolidation (optional) replays cached features to tighten each head independently.

This gives you a model where forgetting is structurally impossible, not just unlikely. See `zero_forgetting.py` for the two-class API that implements this.

### The Nintendo Principle

Milestone 0.5 applied aggressive optimization to find the minimal edge-runtime config:

- Started: R=2048, float64, 78.9% accuracy, 47 MB
- Optimized: R=1536, int8, **78.3% accuracy, 2.6 MB** — 9x smaller, 0.6% accuracy cost

Int8 symmetric quantization of the fixed encoder (`int8_matmul.py`) runs inference natively in int8, with dequantization only at the final accumulation step.

## Reproducing Main Results

```bash
python reproduce.py
```

This script:
1. Trains IMDB sentiment (downloads dataset ~80 MB, trains ~30s)
2. Adds AG News topic without forgetting (downloads ~30 MB, trains ~30s)
3. Evaluates IMDB accuracy on the multi-task model
4. Reports zero forgetting and int8 compression

Expected output:
```
[Task 1/2] IMDB Sentiment
  Test accuracy: 78.3%

[Task 2/2] AG News Topic
  Test accuracy: 88.x%

Zero Forgetting Check
  IMDB after AG News: 78.3%   (delta: 0.0%)
  PASS: Zero forgetting confirmed
```

If you already have the saved models (`quality.npz`, `quality_int8.npz`, `multi.npz`), `reproduce.py` will use them directly without retraining.

## Project Structure

```
project-512d/
├── train_hf.py          # Main training script (CLI + importable API)
├── weightless_model.py  # FixedEncoder + SystemMemory + ZeroForgetReadout
├── zero_forgetting.py   # Isolated readout heads + sleep consolidation
├── hf_adapter.py        # HuggingFace → WeightlessModel feature pipeline
├── int8_matmul.py       # Int8 × Int8 → Int32 matmul with scaling
├── quantize_encoder.py  # Symmetric int8 quantization of encoder weights
├── reproduce.py         # End-to-end reproduction script
│
├── config/              # Configuration system
│   ├── model_config.py  # Typed dataclasses for all model hyperparameters
│   ├── constants.py     # Named project-wide constants (no magic numbers)
│   └── default.py       # Kernel network config (Config512D)
├── training/            # Modular training package
│   ├── dataset.py       # Dataset loading (DatasetLoader, DatasetConfig)
│   ├── trainer.py       # Training loop (WeightlessTrainer)
│   └── evaluation.py    # Metrics (accuracy, classification_report)
├── examples/            # Runnable quickstart examples
│   └── quickstart.py    # IMDB sentiment in <60 seconds
│
├── core/                # Kernel network research (spectral, governance)
├── system/              # Modular field system, helix temporal, gates
├── analysis/            # Spectral, flow, stability, learning dynamics
├── viz/                 # Visualization modules
├── experiments/         # Experiment runners and benchmark scripts
├── tests/               # Pytest test suite (25 test files, 143 tests)
└── docs/                # Research ledger and project direction
    ├── RESEARCH_LEDGER.md
    └── PROJECT_DIRECTION.md
```

## Using the API

### Zero-forgetting readout (numpy only)

```python
import numpy as np
from zero_forgetting import ZeroForgetReadout, sleep_consolidation

rng  = np.random.default_rng(0)
head = ZeroForgetReadout(output_dim=4, feature_dim=64, rng=rng)

head.set_task("task_A")
for X_batch, Y_batch in task_a_data:
    features = encoder(X_batch)
    head.update_batch(features, Y_batch, lr=0.01)

head.set_task("task_B")
for X_batch, Y_batch in task_b_data:
    features = encoder(X_batch)
    head.update_batch(features, Y_batch, lr=0.01)

# task_A is completely intact
preds_a = head.predict_batch(encoder(X_test_a), task_id="task_A")
```

### Training on any HuggingFace dataset

```python
from train_hf import train

# Train task 1
train(dataset="imdb", task_id="sentiment", model_in=None, model_out="sentiment.npz",
      subset=None, text_field=None, label_field=None,
      epochs=5, batch_size=64, lr=0.05,
      tfidf_dim=2000, reservoir_dim=1536, max_sentences=6,
      max_train=None, max_test=None, sleep_cycles=0)

# Train task 2, loading task 1's model — no forgetting
train(dataset="ag_news", task_id="topic", model_in="sentiment.npz", model_out="multi.npz",
      subset=None, text_field=None, label_field=None,
      epochs=5, batch_size=64, lr=0.05,
      tfidf_dim=2000, reservoir_dim=1536, max_sentences=6,
      max_train=None, max_test=None, sleep_cycles=0)
```

### Int8 inference

```python
import numpy as np
from int8_matmul import QuantizedEncoder
from weightless_model import WeightlessModel

data = np.load("quality_int8.npz")
model = WeightlessModel(input_dim=2000, reservoir_dim=1536, output_dim=2, seed=42)
model.encoder = QuantizedEncoder(data["encoder_W_int8"], float(data["encoder_scale"]))
model.readout._heads["sentiment"] = data["head_sentiment"]
```

## Running Tests

```bash
pytest tests/ -v
```

The test suite covers stability invariants (spectral radius, routing sums), governance mechanisms, task projection, int8 pipeline, continual learning behavior, phase encoding correctness, and the configuration system.

## Using the Modular Training API

The `training/` package provides a cleaner interface than calling `train_hf.train()` directly:

```python
from config.model_config import ModelConfig
from training.dataset import DatasetConfig, DatasetLoader
from training.trainer import WeightlessTrainer
from training.evaluation import accuracy, classification_report
from hf_adapter import HFAdapter
from weightless_model import WeightlessModel

# Configure
config = ModelConfig.default_imdb()

# Load data
loader = DatasetLoader(DatasetConfig(name="imdb", max_train=5000))
X_tr, Y_tr, X_te, Y_te, labels = loader.load()

# Build model
model = WeightlessModel(
    input_dim=X_tr.shape[2],
    reservoir_dim=config.encoder.reservoir_dim,
    output_dim=len(labels),
    seed=42,
)

# Train
trainer = WeightlessTrainer(model, config.training)
result  = trainer.train(X_tr, Y_tr, X_te, Y_te, task_id="sentiment")
print(f"Accuracy: {result['final_accuracy']:.1%}")
```

See `examples/quickstart.py` for a complete runnable example.

## Dependencies

- **numpy** — everything
- **scikit-learn** — TF-IDF vectorization
- **datasets** — HuggingFace dataset loading (training only)
- **scipy, networkx, matplotlib** — analysis and visualization

No PyTorch, no CUDA, no deep learning framework required for inference.

## Research Context

This project explores kernel-based neural network architectures with:
- Spectral analysis and stability guarantees (spectral radius ≤ 0.94)
- Governance mechanisms (attention, metabolic, oscillatory, thalamic)
- Hierarchical network structures
- Multi-readout continual learning
- Projection-based task adaptation

See `docs/RESEARCH_LEDGER.md` for detailed experimental results and `docs/PROJECT_DIRECTION.md` for current research direction.
