# Project 512D Direction

## North Star

Project 512D is becoming an experimental cognitive substrate: a bounded,
modular dynamical field that can be trained, inspected, ablated, and scaled
from one 512-dimensional node into a hierarchy of coupled nodes.

The creative direction I would take is not "make another neural net." It is:

1. Treat the 512D state as a living field with measurable dynamics.
2. Treat modules as routed cognitive territories, not opaque hidden units.
3. Treat constraints and spectral bounds as architectural citizens.
4. Make every new idea falsifiable through ablations.
5. Preserve experiment artifacts so the system can tell its own history.

## Current Research Questions

- Does sparse top-k routing reduce interference without starving useful modules?
- Does learnable inter-module coupling improve task learning over a fixed random
  reservoir?
- Does hierarchy reduce forgetting compared with a single bounded field?
- Which modules become influential for each synthetic task?
- Can spectral containment coexist with useful plasticity?

## Next Milestone

The next milestone is an evidence cockpit:

- single-node ablations for routing and K_inter plasticity
- hierarchy-vs-single comparisons for continual learning
- saved JSON, CSV, Markdown, and figures for each experiment
- a short research ledger for every change that affects architecture or results

The project should feel like a small lab instrument: run it, get traces, compare
conditions, and learn what the architecture is doing.

## Phase 2D: Nintendo Efficiency Pass

Principle: do not store what can be derived cheaply.

The governance feedback layer now follows this rule. Instead of storing a dense
480 x 32 constraint-to-field matrix, the system stores two small factors:

- A: 480 x 4
- B: 4 x 32
- W_cf is derived as A @ B only for diagnostics

The projector applies feedback as A @ (B @ constraint_state), which keeps the
same functional shape while reducing governance parameters from 15,360 to 2,048.

The feedback factors are also optionally learnable. A scalar loss-gated local
update teaches the constraint space which active field patterns to damp, while
the main spectral kernel remains independently bounded.

## Phase 2E: Governance Stress Chamber

The next question is whether governance helps when the field is actually under
pressure. Phase 2E adds a stress experiment with three modes:

- none: boundary detector logs stress, but projection is disabled
- fixed: low-rank projection is enabled with fixed feedback factors
- adaptive: low-rank projection is enabled and feedback factors learn from loss

The stress chamber increases input strength, processing depth, inter-module
coupling, and burst amplitude, then records:

- task loss
- max and average state norm
- boundary triggers
- projection count
- burst recovery speed
- feedback factor movement

This gives the constraint space a scoreboard. It can no longer be merely neat;
it has to stabilize the field, improve recovery, or justify its footprint in
the experiment report.

## Phase 2F: Selective Governance

The constraint space is now the module scoreboard itself:

- field space: 480 dims
- constraint space: 60 dims, one activation norm per module
- total state: 540 dims

Dense constraint-to-field feedback has been removed from the active design.
Governance is now derived from module activations:

- compact mode stores 60 self-inhibition scalars
- lateral inhibition is computed from functional group boundaries
- binary mode keeps top-k winners per group as an extreme baseline

This is the Nintendo version of governance: store one scalar per module, derive
the rest from the scoreboard.
