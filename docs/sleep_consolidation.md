# Sleep Consolidation

Sleep consolidation replays a task against a frozen shared feature table and
updates only the task-owned readout/head. In this repo that is the important
contract: retention comes from fixed shared features plus isolated task heads.

## What Freezes

- `CoupledModularKernel.freeze()` blocks Hebbian `K_inter` updates.
- `ModularFieldSystem.freeze()` forwards the freeze to its kernel.
- `experiments.run_4_kernel_network.AtomicKernel.freeze()` exposes the same
  control at benchmark-network level.

Dynamics still run when frozen. The freeze is about shared parameter updates,
not about stopping the reservoir.

## What Sleep Does

`system.sleep_consolidation.sleep_consolidation()`:

1. Freezes all `network.kernels`.
2. Extracts the task feature table once.
3. Replays mini-batches from that cached table for many cycles.
4. Updates only the supplied readout/head for the current task.
5. Unfreezes the kernels.

Caching the feature table is deliberate. The current benchmark processors use
randomized batch resets, so extracting features repeatedly can make replay a
moving target even if shared weights are frozen.

## Usage

Classic benchmark:

```bash
python experiments/run_classic_benchmarks.py --kernels 4 --temporal-adapter helix --sleep-cycles 1000 --sleep-lr 0.001
```

Strict benchmark:

```bash
python experiments/run_strict_sequence_benchmarks.py --kernels 4 --temporal-adapter helix --sleep-cycles 1000 --sleep-lr 0.001
```

Gate distillation:

```bash
python experiments/distill_gates.py --epochs 30 --sleep-cycles 1000 --sleep-lr 0.001
```

## Validation

```bash
python -m pytest tests/test_sleep_consolidation.py -q
```

Sleep metrics are stored with each task under `sleep_metrics` in the benchmark
JSON payloads.
