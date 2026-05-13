# 2D Kernel Lattice Architecture

## Overview

The Kernel Lattice (`system/kernel_lattice.py`) is a 2D grid of coupled neural processing units designed for continual multi-task learning. It combines:

- **Vertical temporal hierarchy** — abstract layers build on concrete ones
- **Horizontal functional specialization** — stacks within a layer develop distinct response profiles
- **Sparse resonance bridges** — feature-level coupling between adjacent stacks

Zero forgetting is structural, not approximate: shared dynamics (ModularFieldSystem, HelixTemporalAdapter) are fixed after construction. Only isolated per-task readout weights are ever trained.

---

## Grid Layout

```
Layer 1 (abstract)   [Stack 0] --bridge-- [Stack 1] --bridge-- [Stack 2] --bridge-- [Stack 3]
        ^
        | inter-layer projection (prev_features @ W_mix -> 64-dim -> added to input)
        |
Layer 0 (concrete)   [Stack 0] --bridge-- [Stack 1] --bridge-- [Stack 2] --bridge-- [Stack 3]
             |           |           |           |
             +-----+-----+-----+-----+
                   input sequence (T x 64)
```

Default configuration: `n_layers=2`, `n_stacks=4`.

---

## KernelStack

Each stack is an independent (ModularFieldSystem, HelixTemporalAdapter) pair.

**ModularFieldSystem** (540-dim):
- 60 modules x 8 dims field space + 60 constraint dims
- Coupled modular kernel with spectral radius <= 0.94
- Softmax router selects top-k active modules per timestep
- Governance: lateral inhibition + constraint projection

**HelixTemporalAdapter** (output: 1539-dim):
- Projects 540-dim reservoir states to 256-dim space
- Tracks fast (AC) and slow (DC) temporal streams
- Final features: [last, mean, max, AC, DC, AC*DC] x 256 + 3 phase = 1539

**Forward pass** (sequence of T timesteps):
1. `reset_state(scale=0.0)` — zero-init kernel for determinism
2. For each timestep t: `inject(x_t)` → `step(t)` → `helix.step(...)`
3. Return `helix.final_features()` — shape (1539,)

Each forward call is fully deterministic given the same input sequence.

---

## Horizontal Bridges

After all stacks in a layer complete their forward passes, adjacent stacks may exchange information via cosine resonance:

```python
sim = cosine_similarity(features_i, features_j)
if sim > threshold:
    blend = (features_i + features_j) / 2
    features_i = (1 - alpha) * features_i + alpha * blend
    features_j = (1 - alpha) * features_j + alpha * blend
```

**Why feature-level only:** Injecting into kernel state after `process()` would be immediately overwritten by `reset_state()` at the next call. Feature blending is the only safe and effective mechanism.

Default: `bridge_threshold=0.15`, `bridge_alpha=0.25`.

---

## Vertical Connections

Layer l+1 receives a projection of layer l's aggregate features:

```python
mixed_signal = tanh(prev_layer_features @ W_mix)   # (64,)
layer_inputs = original_inputs + mixed_signal       # broadcast over T
```

`W_mix` shape: `(layer_feature_dim, input_dim)` — initialized with scale `1/sqrt(layer_feature_dim)`.

Layer 0 has no mixing (direct input only).

---

## Global Readout

```python
global_features = concat([layer_0_features, layer_1_features])
# shape: (n_layers x n_stacks x 1539,) = (2 x 4 x 1539,) = (12312,)

output = readout._heads[task_id] @ global_features
# shape: (output_dim,)
```

`ZeroForgetReadout` maintains one isolated `(output_dim, 12312)` weight matrix per task. Delta rule (MSE gradient) update:

```python
error = pred - target
grad = outer(error, features)
head -= lr * grad
```

Task B's gradient has zero mathematical access to task A's weight matrix.

---

## Forgetting Analysis

| Source of forgetting | Eliminated? | Mechanism |
|---|---|---|
| Shared readout weights | Yes | Isolated heads per task |
| Shared kernel dynamics | Yes | Never updated during training |
| Bridge signal bleed | Yes | Bridges are read-only feature blends |
| Inter-layer projection | Yes | W_mix is fixed at construction |

Benchmark result (8 tasks, sequential training):
- Max forgetting delta: 0.0 (floating point exact)
- Global readout beats single-stack local readout on 7/8 tasks

---

## Configuration

```python
from system.kernel_lattice import KernelLattice, LatticeConfig

cfg = LatticeConfig(
    n_layers=2,           # vertical depth
    n_stacks=4,           # horizontal width
    projection_dim=256,   # AC/DC projection width (feature_dim = 256*6+3 = 1539)
    bridge_threshold=0.15,  # min cosine sim to fire a bridge
    bridge_alpha=0.25,    # blend weight when bridge fires
    seed=42,
)

lattice = KernelLattice(output_dim=64, lattice_cfg=cfg)
```

---

## Benchmark

```bash
# Full benchmark (8 tasks, 500 train / 150 test each, 3 epochs)
python experiments/run_lattice_benchmark.py

# Fast smoke test (~10s)
python experiments/run_lattice_benchmark.py --fast

# Custom
python experiments/run_lattice_benchmark.py --n-train 1000 --n-test 300 --lr 0.003 --epochs 5
```

Output: JSON results + 4 plots (MSE, forgetting, specialization heatmap, local vs global).

---

## Complexity

| Component | Parameters | Notes |
|---|---|---|
| ModularFieldSystem (per stack) | ~300K | Fixed at construction |
| HelixTemporalAdapter (per stack) | ~140K | Fixed at construction |
| Global readout head (per task) | 12,312 x output_dim | ~787K for output_dim=64 |
| Total shared (8 stacks) | ~3.5M | Never updated |
| Per-task overhead | ~787K | Grows with task count |

Training cost: O(T x n_layers x n_stacks) per sample, where T = sequence length.
