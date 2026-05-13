# Gate Distillation

## Purpose

Gate distillation is the escalation path from explicit Helix task channels to
learned gates. The explicit `HelixTemporalAdapter` remains the teacher because
its classic-task channels are interpretable and already satisfy the Phase 1
gate. The learned path should imitate those timestep-level write gates while
preserving the same final feature contract.

## API Contract

The torch-native adapter should use the same dimension names as the existing
NumPy adapters:

```python
PyTorchHelix(input_dim, input_width=64, projection_dim=256)
```

Tests should not assume separate `ac_dim` or `dc_dim` constructor arguments
unless the class exposes them as optional compatibility aliases. The public
feature contract is:

```text
concat(last_projected, mean_projected, max_projected,
       ac_final, dc_final, ac_final * dc_final, phase_final)
```

so the final width is `projection_dim * 6 + 3`.

## Distillation Loop

1. Run the explicit `HelixTemporalAdapter` on a support trajectory with the
   relevant `task_id`.
2. Record `last_write_gate` after every timestep.
3. Run `AdaptiveHelixTemporalAdapter` on the same inputs without explicit task
   channels.
4. Update the learned gate controller or gate network against the recorded
   teacher gate for that timestep.
5. Re-evaluate on the same deterministic support sequence and require lower
   gate-target loss.

The unit tests keep this loop tiny: two examples, three timesteps, four
projection channels, and fixed seeds.

## Validation

Targeted command:

```bash
python -m pytest tests/test_pytorch_helix.py tests/test_distillation.py -q
```

Expected behavior today:

- `tests/test_distillation.py` runs against the current adaptive/controller
  implementation.
- `tests/test_pytorch_helix.py` runs when `system.pytorch_helix` is present and
  otherwise skips cleanly for branches that have not added the torch-native
  adapter yet.

Promotion criteria:

- distillation loss decreases on deterministic support trajectories;
- learned/adaptive final features remain finite and shaped
  `(batch, projection_dim * 6 + 3)`;
- the torch-native adapter accepts `input_dim`, `input_width`, and
  `projection_dim` directly;
- classic benchmark gates remain bounded in `[0, 1]`;
- full benchmark validation does not regress copy, parity, adding, or
  forgetting targets documented in `docs/learned_gates.md`.

## V2 Fixed-Feature Trainer

`system/distillation_trainer_v2.py` adds the current default distillation path.
It fixes the moving-target problem by caching a deterministic feature stream per
task/batch/sequence shape. The cached stream is not purely random: the leading
channels contain the current input and cumulative input so the learned gates can
see parity bits, copy symbols, and adding markers.

Run it with:

```bash
python experiments/distill_gates.py --trainer v2 --epochs 100 --batch-size 16
```

The CLI also exposes a decay range:

```bash
--gate-decay-min 0.0 --gate-decay-max 0.99
```

This matters because the explicit teacher writes some DC channels directly.
The original learned-gate decay range of `0.9-0.99` limits the student to very
small per-step writes and makes direct-channel distillation unnecessarily hard.

Current compact diagnostic runs show V2 improves correlation substantially but
does not yet reach the strict `0.95` gate by itself. The next likely escalation
is a weighted active-channel loss or direct supervision of teacher DC deltas in
addition to raw DC-state MSE.
