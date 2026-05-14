# Dependency Tiers

Project 512D now spans both a lightweight product-oriented continual-learning core and several research modules. This document defines which dependencies belong to which layer.

## Tier 1 — Core runtime (PyTorch-free)

Goal:
- small edge runtime
- CPU-first
- reproducible local inference/training
- no heavyweight deep-learning framework required

Core files:
- `weightless_model.py`
- `zero_forgetting.py`
- `train_hf.py`
- `hf_adapter.py`
- `int8_matmul.py`
- `quantize_encoder.py`

Primary dependencies:
- numpy
- scikit-learn
- datasets (training only)
- scipy
- networkx
- matplotlib

This is the path described by the README claim:

> No GPU required for the core path.

## Tier 2 — Research field system

Goal:
- governance experiments
- kernel topology experiments
- spectral analysis
- lattice studies

Primary locations:
- `core/`
- `analysis/`
- `viz/`
- `experiments/`
- `system/kernel_lattice.py`

Additional dependencies may include:
- matplotlib
- networkx
- profiling tools

## Tier 3 — Learned gate research (optional PyTorch)

Goal:
- replace explicit Helix task-aware channels with learned temporal gating
- task-conditioned gate generation
- support-set-driven temporal adaptation

Primary files:
- `system/adaptive_helix.py`
- `system/learned_gates.py`
- `system/task_embedder.py`
- `system/gate_generator.py`

This layer currently imports:
- torch

Important:
- PyTorch is optional research infrastructure.
- The core continual-learning path does not require PyTorch.
- README and installation instructions should distinguish the core path from learned-gate research.

## Recommended future requirement split

```text
requirements-core.txt
requirements-research.txt
requirements-gates.txt
requirements-dev.txt
```

## Installation examples

### Core runtime only

```bash
pip install -r requirements-core.txt
```

### Full research environment

```bash
pip install -r requirements-core.txt
pip install -r requirements-research.txt
pip install -r requirements-gates.txt
```
