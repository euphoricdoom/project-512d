# 512D Project - System Brain

> This is not documentation. This is how Claude thinks inside your project.

## Mission

Two parallel workstreams:

1. **Weightless Continual Learning** — production-ready AI substrate with zero forgetting by construction, int8 efficiency. Current focus.
2. **Kernel Network Research** — spectral analysis, governance mechanisms, hierarchical structures. Ongoing research.

**Core Achievement (Milestone 1):** Zero catastrophic forgetting on IMDB → AG News. Forgetting = 0.0% by construction (isolated heads, frozen encoder). 78.3% IMDB accuracy in 2.6 MB (int8 quantized, R=1536).

## Mental Model

### Repository Structure

```
project-512d/
├── CLAUDE.md                 # You are here — System brain
├── README.md                 # User-facing documentation and quick start
├── reproduce.py              # Single-command reproduction of key results
├── train_hf.py               # Main training script (CLI + API)
├── weightless_model.py       # FixedEncoder + SystemMemory + ZeroForgetReadout
├── zero_forgetting.py        # Isolated heads + sleep consolidation API
├── hf_adapter.py             # HuggingFace → WeightlessModel feature pipeline
├── int8_matmul.py            # Int8 × Int8 matmul with scaling (no dependencies)
├── quantize_encoder.py       # Symmetric int8 quantization utility
├── conftest.py               # pytest path setup (root modules discoverable)
├── .claude/
│   ├── memory/              # Experimental insights & patterns
│   ├── skills/              # Research workflows
│   └── workflows/           # Experiment pipelines
├── core/                    # Kernel network core (constants, kernel, governance, projector)
├── system/                  # ModularFieldSystem, trainer, helix temporal, gates, distillation
├── analysis/                # Spectral, flow, stability, learning dynamics
├── viz/                     # Visualization modules
├── experiments/             # Runnable experiment scripts (27+) + benchmark/eval scripts
├── tests/                   # 23 pytest test files
├── config/                  # Experiment configurations
├── proof/                   # Mathematical claims validation
└── docs/
    ├── RESEARCH_LEDGER.md   # Detailed milestone log — update after experiments
    └── PROJECT_DIRECTION.md # Current research direction and open questions
```

### Saved Model Artifacts (root level)
- `quality.npz / .json / .vocab` — R=1536 float32 sentiment model (IMDB 78.3%)
- `quality_int8.npz / .json` — int8 quantized version (2.6 MB, same accuracy)
- `multi.npz / .json / .vocab` — multi-task model: sentiment + topic (zero forgetting)
- `sentiment.npz / .json / .vocab` — R=2048 baseline sentiment model

### Key Concepts

**Weightless learning layer (production):**
- **FixedEncoder**: random projection, spectral radius ≤ target, never updated
- **SystemMemory**: AC/DC temporal state (fast+slow streams, phase code, cross-frequency binding)
- **ZeroForgetReadout**: one head per task, structural isolation — forgetting is impossible by construction
- **Int8 quantization**: symmetric quantization of fixed encoder weights (9x compression, <0.01% accuracy delta)

**Kernel network research layer:**
- **Kernel Networks**: recurrent networks using kernel functions instead of standard activations
- **Governance**: top-down control mechanisms (attention, metabolic, oscillatory, thalamic)
- **Spectral Properties**: eigenvalue analysis for stability (target_radius=0.94)
- **Task Projection**: low-rank projections for multi-task learning
- **Readout Stability**: stable linear readout despite complex dynamics

## Execution Rules

### Code Organization

- **Root-level `.py` files** — the public API: `weightless_model`, `zero_forgetting`, `hf_adapter`, `int8_matmul`, `quantize_encoder`, `train_hf`
- **Core modules** (`core/`) — reusable kernel network components
- **System modules** (`system/`) — modular field system, training, helix, gates, distillation
- **Experiments** (`experiments/`) — runnable scripts; also holds benchmark/eval scripts
- **Tests** (`tests/`) — validates correctness via pytest
- **Analysis** (`analysis/`) — post-hoc analysis tools

### Dependency Rule

The weightless learning layer depends only on **numpy + scikit-learn + datasets**.
No PyTorch. No GPU. The kernel research layer uses scipy, networkx, matplotlib.
PyTorch appears only in `system/pytorch_helix.py` and related helix experiments.

### Testing Philosophy

- Integration tests > unit tests for complex dynamics
- Test invariants (routing sums to 1, dynamics bounded, spectral radius ≤ target)
- Validate theoretical properties
- `conftest.py` at root adds root to `sys.path` so all root-level modules are importable

### Experiment Workflow

1. **Configure** — define experiment in `config/experiments/`
2. **Implement** — create runner in `experiments/`
3. **Validate** — add tests for new components
4. **Analyze** — use `analysis/` tools
5. **Visualize** — generate plots with `viz/`
6. **Document** — update `docs/RESEARCH_LEDGER.md`

### Memory Guidelines

**Save to memory when you discover:**
- Stability patterns across governance modes
- Performance characteristics of different architectures
- Common failure modes or numerical issues
- Effective hyperparameter ranges
- Surprising emergent behaviors
- Quantization sensitivity (which layers tolerate int8)

## Project-Specific Constraints

- **Python 3.11+** required
- **numpy** is the primary framework for the weightless layer; scipy/sklearn for analysis
- **PyTorch** only in `system/pytorch_helix.py` and related — not in core weightless API
- **Reproducibility**: All experiments use fixed seeds (seed=42 everywhere)
- **Clean outputs**: Results go in `outputs/` (gitignored)
- **No `__pycache__` in git**: Covered by `.gitignore`
- **Paths**: All scripts use `Path(__file__).parent` relative paths — no hardcoded absolutes

## Common Tasks

### Reproduce key results
```bash
python reproduce.py                    # Full run (downloads ~110 MB of datasets)
python reproduce.py --use-saved        # Use existing .npz files, skip training
python reproduce.py --fast             # Quick smoke test (2k samples)
```

### Train from scratch
```bash
python train_hf.py --dataset imdb --task sentiment
python train_hf.py --dataset ag_news --task topic --model-in sentiment.npz --model-out multi.npz
```

### Running Tests
```bash
pytest tests/
pytest tests/test_kernel_stability.py -v
pytest tests/test_int8_pipeline.py -v
```

### Running Kernel Experiments
```bash
python -m experiments.run_stability_sweep
python -m experiments.run_governance_comparison
```

### Benchmarks and analysis
```bash
python experiments/benchmark_int8.py
python experiments/eval_sentiment.py
python experiments/test_int8_accuracy.py
```

### Analyzing Results
```python
from analysis.spectral import compute_spectrum
from viz.spectral_plots import plot_eigenvalues
```

## Key Insight

Zero forgetting is not a regularization trick — it is a structural guarantee. Shared weights are never updated. Each task's gradient lives in its own isolated weight matrix. This gives you a model that accumulates capability without limit, in constant memory.

The int8 work (Milestone 0.5) shows the same principle applies to inference efficiency: the fixed encoder's weights are quantized once and never change, making the compression lossless in practice.
