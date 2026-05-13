# Learned Task-Aware Gates

## Overview

Phase 2 adds an adaptive gate path for the Helix AC/DC temporal adapter. The
Phase 1 explicit `HelixTemporalAdapter` remains the trusted baseline. The new
`AdaptiveHelixTemporalAdapter` preserves the same final feature contract while
replacing hand-written task channels with learned gate responses.

## Components

### TaskEmbedder

`system/task_embedder.py` implements a PyTorch LSTM plus attention encoder. It
accepts support examples shaped `(K, seq_length, input_dim)` and emits one task
embedding.

### GateGenerator

`system/gate_generator.py` maps the task embedding to:

- `write_gates`: `(num_dc_channels, feature_dim)`
- `decay_rates`: `(num_dc_channels,)`, clamped to `[0.9, 0.99]`
- `update_weights`: `(num_dc_channels, feature_dim)`

### LearnedGateNetwork

`system/learned_gates.py` composes `TaskEmbedder` and `GateGenerator`. It also
contains a compact NumPy `LearnedGateController` used for fast supervised gate
diagnostics and unit tests.

### AdaptiveHelixTemporalAdapter

`system/adaptive_helix.py` subclasses the Phase 1 adapter but overrides the
timestep update. It does not call the explicit copy/parity/adding channel
logic. DC updates use either generated PyTorch gate configs or the lightweight
controller:

```python
write_gate = sigmoid(projected @ write_weights.T)
candidate = tanh(projected @ update_weights.T)
dc = decay * dc + (1 - decay) * write_gate * candidate
```

Final features remain:

```text
concat(last_projected, mean_projected, max_projected,
       ac_final, dc_final, ac_final * dc_final, phase_final)
```

## Benchmark Usage

Run the learned-gate classic benchmark path:

```bash
python experiments/train_learned_gates.py --kernels 1 --epochs 5 --samples 128 --steps 3 --projection-from 1
```

This delegates to `experiments/run_classic_benchmarks.py` with:

```bash
--temporal-adapter learned-helix
```

and saves `outputs/learned_gates_benchmarks.json`.

## Validation

Targeted tests:

```bash
python -m pytest tests/test_learned_gates.py tests/test_adaptive_helix.py -q
```

Phase 2 validation:

```bash
python experiments/validate_learned_gates.py
```

The strict research gate remains:

- classic copy, parity, and adding accuracy at or above 98%;
- forgetting below 1%;
- at least two novel tasks above 90%;
- all tests passing.

## Current Caveat

The implementation establishes the learned-gate architecture and adaptive
benchmark path. The generated gates are not yet meta-trained across task
families, so benchmark validation should be treated as an experimental result,
not assumed equivalent to the Phase 1 explicit-channel baseline.

## Escalation Path

If learned gates miss the strict gate:

1. Train the `LearnedGateNetwork` by distilling explicit Helix gate trajectories.
2. Add marker-supervised auxiliary gate loss for adding-style tasks.
3. Add parity-oriented auxiliary state targets for XOR accumulation.
4. Add timestep-level running-sum and running-parity losses while preserving
   task-specific readout heads.
