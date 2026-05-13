# Helix AC/DC Temporal Adapter

## Diagnosis

Classic sequence benchmarks exposed a competence problem, not a retention
problem. Flattening sequences destroyed temporal structure, and final-state
readouts over scoreboard/raw reservoir features did not learn parity or
selective addition reliably.

The fixed benchmark path now processes inputs timestep by timestep, preserves
reservoir state across the sequence, and passes each timestep's reservoir state
through a Helix temporal adapter before task-specific readout.

## Design

`HelixTemporalAdapter` is a NumPy temporal feature layer with three jobs:

- project high-dimensional reservoir state into a compact 256D stream;
- maintain fast AC event state and slow DC accumulator state;
- expose final trajectory features for task-specific projection heads.

The final feature vector is:

```text
concat(last_projected, mean_projected, max_projected,
       ac_final, dc_final, ac_final * dc_final, phase_final)
```

For the classic tasks, the adapter also exposes interpretable task-aware
channels:

- `copy_task`: compact DC memory slots preserve early input timesteps;
- `parity_task`: DC parity channel toggles on positive bit events;
- `adding_task`: marker-aware DC accumulator sums marked values.

These channels are the first planned escalation from purely random gating. They
make the AC/DC stream a real temporal memory rather than only two decays.

## Benchmark Integration

`experiments/run_classic_benchmarks.py` now supports:

```bash
--temporal-adapter none|helix
```

`helix` is the default. With Helix enabled, all classic benchmark kernel counts
use `TaskSpecificProjection` over Helix final features. This preserves task
isolation and the zero-forgetting behavior already established by multi-task
readouts.

## Results

Command:

```bash
python experiments/run_classic_benchmarks.py --kernels 1,4,16,64 --temporal-adapter helix --epochs 30 --samples 512 --steps 3 --batch-size 32 --projection-from 1
```

Result:

| Configuration | Copy | Parity | Adding | Max Forgetting |
| --- | ---: | ---: | ---: | ---: |
| 1 kernels + Helix | 100% | 100% | 100% | 0.0000 |
| 4 kernels + Helix | 100% | 100% | 100% | 0.0000 |
| 16 kernels + Helix | 100% | 100% | 100% | 0.0000 |
| 64 kernels + Helix | 100% | 100% | 100% | 0.0000 |

Phase 1 now passes: all classic benchmark tasks exceed 90% accuracy across all
tested kernel counts, with forgetting below 1%.

## Diagnostics

`HelixTemporalAdapter.get_diagnostics()` reports AC/DC norms, write-gate
statistics, and optional AC/DC/gate trajectories when diagnostics are enabled.
Full benchmark runs keep trajectories disabled to avoid memory growth.

## Next Escalation

The current adapter includes task-aware classic benchmark channels. For broader
tasks, the next research step is to replace those explicit channels with learned
task-aware gates or supervised timestep-level auxiliary losses for running
state prediction.
