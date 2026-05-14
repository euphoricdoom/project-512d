# Project 512D Architecture Ledger

Status key: **implemented** means code exists in the repo; **validated** means the repo docs or ledger record passing tests/benchmarks; **experimental** means code exists but is not yet proven as a replacement for the baseline; **planned** means design direction exists but code was not confirmed in this pass.

## North Star

Project 512D is a local continual-learning substrate built around frozen shared dynamics and isolated task-owned readouts. The system should remain small, inspectable, testable, and edge-friendly while growing into a modular cognitive field.

## Core invariants

1. Shared encoders, reservoirs, kernels, bridges, and projection dynamics are fixed during task training.
2. Each task owns an isolated readout/head.
3. New task training must not mutate prior task heads.
4. Any claim of zero forgetting must name the feature space that stayed fixed.
5. Any experimental module must be benchmarked against the simpler baseline it intends to replace.

## Subsystem ledger

| Subsystem | Primary files | Status | Proof / evidence | Risks | Next action |
|---|---|---:|---|---|---|
| Zero-forgetting task heads | `zero_forgetting.py` | implemented / validated | `ZeroForgetReadout` owns one matrix per task; CE batch update writes only active task | Claims can be overstated if shared features drift | Keep as the core retention mechanism |
| Sleep consolidation | `zero_forgetting.py` | implemented | Frozen feature replay tightens one head | Optional; not the primary forgetting solution | Keep as optional quality pass |
| Weightless model | `weightless_model.py` | implemented | `FixedEncoder` + `SystemMemory` + `ZeroForgetReadout` | Simple product path may diverge from research path | Treat as the clean product/MVP base |
| HuggingFace trainer | `train_hf.py` | implemented | Saves `.npz`, `.json`, `.vocab`; supports adding tasks | Shared vocabulary can cap new-task accuracy | Add clearer dataset/accuracy caveat docs |
| Int8 encoder path | `train_hf.py`, `int8_matmul.py`, `quantize_encoder.py` | implemented / validated | Research ledger reports 2.6 MB R=1536 model with preserved IMDB accuracy | Quantization only covers encoder by default, not full readout storage | Extend later to readout compression |
| Research ledger | `docs/RESEARCH_LEDGER.md` | implemented | Records milestones, tests, findings, failures | Can become hard to scan as it grows | Keep append-only; add summary links here |
| Project direction | `docs/PROJECT_DIRECTION.md` | implemented | Defines field-system north star | May lag implementation | Update at phase gates |
| Modular field system | `core/`, `system/modular_system.py` | implemented / experimental | Governance/routing/stability experiments recorded | Not the direct forgetting solution | Keep as fixed feature/dynamics substrate |
| Governance modes | `core/governance*.py`, governance experiments | implemented / experimental | Ledger shows stable but modest impact | Easy to overvalue; did not solve forgetting alone | Retain for stability/interpretability, not retention claims |
| Multi-readout path | `system/multi_readout.py` | implemented / validated | Ledger says shared readout was overwrite site and multi-readout fixed forgetting | Needs consistent usage across all research runners | Prefer task-owned readouts by default |
| Kernel network | `experiments/run_4_kernel_network.py` | implemented / validated prototype | Ledger reports near-zero forgetting across 8 tasks | Some task losses remain high | Compare retention and competence separately |
| Classic benchmarks | `experiments/run_classic_benchmarks.py` | implemented / validated | Supports copy/parity/adding and adapter modes | Learned adapter not yet proven equivalent to explicit Helix | Run none vs helix vs learned-helix matrix |
| Helix AC/DC adapter | `system/helix_temporal.py`, `docs/helix_temporal_adapter.md` | implemented / validated | Docs report 100% copy/parity/adding across 1/4/16/64 kernels with 0.0000 forgetting | Uses explicit task-aware channels for classic tasks | Keep explicit Helix as honest baseline |
| Adaptive Helix | `system/adaptive_helix.py` | implemented / experimental | Preserves Helix feature contract while replacing explicit task writes | Requires gate training/proof; imports PyTorch | Benchmark as `learned-helix` before claiming replacement |
| Learned gates | `system/learned_gates.py`, `system/task_embedder.py`, `system/gate_generator.py` | implemented / experimental | Task embedder + gate generator + NumPy controller exist | `torch` is optional but not listed in base requirements | Split dependency tiers and add gate benchmark |
| 2D Kernel Lattice | `system/kernel_lattice.py`, `docs/LATTICE_ARCHITECTURE.md` | implemented / experimental | Lattice composes stacks, bridges, vertical mixing, global task readout | Docs/code bridge threshold mismatch; high per-task head size | Add reproducible lattice comparison table |
| Hybrid readout storage | planned | planned | Prior design: disk cache + int8 + sparse + low-rank | Needed for task growth; not confirmed in code | Implement disk-backed + int8 first |
| Novelty memory field | planned | planned | Design rule: novelty grows, resonance routes, stability blocks noise | Not confirmed in code | Add small tested allocator module |
| Product shell / Arbiter Local | partial / planned | planned | README has CLI paths, not full product shell | Research/product paths are tangled | Build after ledger/storage cleanup |

## Claims that are safe today

- Task-owned readout heads eliminate direct readout-parameter interference.
- With fixed features, adding a new task does not mutate old task heads.
- The documented real-data and classic-benchmark runs show zero measured forgetting under their stated settings.
- Helix solves the classic temporal benchmark gate by adding explicit task-aware temporal channels.
- The lattice is implemented as a fixed-dynamics, feature-bridge, global-readout architecture.

## Claims to avoid without extra evidence

- Do not claim universal zero forgetting if the shared feature extractor changes.
- Do not claim learned gates replace explicit Helix until the learned-helix benchmark passes.
- Do not claim governance modes solve forgetting; the ledger says the shared readout was the main overwrite site.
- Do not claim the lattice improves all competence metrics without side-by-side benchmark tables.
- Do not claim the entire repo is PyTorch-free; the core path is PyTorch-free, but learned gates use optional PyTorch.

## Immediate execution roadmap

1. Align documentation and dependencies.
2. Register or document all experiment entrypoints.
3. Run a clean benchmark matrix: `none`, `helix`, `learned-helix`.
4. Implement `storage/` with disk-backed int8 readouts.
5. Implement `memory/novelty_field.py` with deterministic tests.
6. Promote the smallest reliable product path into an Arbiter Local shell.
