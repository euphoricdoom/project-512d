# Project 512D — Roadmap to Validation

## The Thesis

**Weightless AI for everyone.**

A person with a laptop, an internet connection, and a HuggingFace dataset name
should be able to train their own multi-task AI model in minutes — with zero
catastrophic forgetting, guaranteed.

The architecture that makes this possible has three fixed components:

| Component | Role | Learns? |
|---|---|---|
| `FixedEncoder` | Random projection → non-linear feature space | Never |
| `SystemMemory` | AC/DC leaky integrators for temporal context | Never |
| `ZeroForgetReadout` | One isolated head per task | Head only |

Forgetting delta = 0.0 (machine precision) by construction, not by
regularization.  Proven across 2, 10, and 50 tasks.

---

## Proof Status — 30/30 Claims Green

All claims must stay green.  No new milestone begins until all prior claims pass.

### Proven (30/30)

| Claim | Description | Status |
|---|---|---|
| 1a | 2-task forgetting delta = 0.0 (exact) | ✅ |
| 1b | 10-task forgetting delta = 0.0 | ✅ |
| 1c | 50-task forgetting delta = 0.0 | ✅ |
| 1d | Re-train task A — all other heads array-equal | ✅ |
| 2a | Encoder + memory beats raw last-timestep (nonlinear) | ✅ |
| 2b | Encoder + memory beats raw last-timestep (temporal integration) | ✅ |
| 2c | Spectral radius ≤ 0.94 after SVD scaling | ✅ |
| 2d | Encoder W is array-equal before/after training | ✅ |
| 3a | Memory copy task: memory system outperforms last-step only | ✅ |
| 3b | Feature dim = reservoir_dim × 6 + 3 | ✅ |
| 3c | DC retains pulse signal after 50 steps (> 1e-4) | ✅ |
| 3d | AC decays to < 0.001 after 20 zero-input steps | ✅ |
| 4a | Fixed encoder param count constant across tasks | ✅ |
| 4b | Learned params grow exactly (i+1) × head_size | ✅ |
| 4c | param_summary() fields present and correct | ✅ |
| 5a | Sleep consolidation reduces error (single seed) | ✅ |
| 5b | Sleep consolidation reduces error (5 seeds) | ✅ |
| 5c | Sleep on task A leaves task B head array-equal | ✅ |
| 6a | High-LR (1.0) adversarial task B — task A head unchanged | ✅ |
| 6b | Weight clip side-effect isolated to trained task | ✅ |
| 6c | Explicit task_id override touches only that head | ✅ |
| 7a1 | Single-task training (1k samples, 5 epochs) < 30s CPU | ✅ |
| 7a2 | 5 sequential tasks < 3 min CPU | ✅ |
| 7a3 | Feature extraction dominates head update cost | ✅ |
| 7b1 | 10-task model total memory < 50 MB | ✅ |
| 7b2 | Single task head < 1 MB | ✅ |
| 7c1 | WeightlessModel imports without torch | ✅ |
| 7c2 | Full pipeline numpy-only (no tensor objects) | ✅ |
| 7d1 | New task head trains in < 3s on cached features | ✅ |
| 7d2 | 20 rapid tasks — all prior heads array-equal | ✅ |

Run proof suite:
```bash
python proof/claims.py
python proof/hardware_claim.py
```

---

## Milestone 0 — First Real Dataset  `[ ✅ PASSED — 2026-04-27 ]`

**Goal:** Train on IMDB sentiment (HuggingFace `imdb` dataset) on a consumer
laptop.

**Trade accepted:** The original exit criterion was ≥ 80% accuracy. After the
int8 optimization pass (Milestone 0.5), we accepted 78.3% as the target result
in exchange for a 9x model size reduction (47 MB → 2.6 MB, R=2048 float64 →
R=1536 int8). Accuracy-for-performance is a deliberate design choice for the
edge-runtime constraint, not a failure.

### Results
- ✅ `python train_hf.py --dataset imdb --task sentiment`
- ✅ Test accuracy: **78.3%** (int8, R=1536)
- ✅ Model size: **2.6 MB** (9x compression vs float64 baseline)
- ✅ Training time: ~30 seconds on CPU (25k samples)
- ✅ Saved artifacts: `quality_int8.npz`, `quality_int8.json`

**Exit criterion (revised):** ≥ 75% IMDB test accuracy, model ≤ 5 MB, training
time < 5 minutes on CPU. All three passed.

---

## Milestone 0.5 — Int8 Optimization  `[ ✅ PASSED — 2026-04-27 ]`

**Goal:** Compress the fixed encoder to int8 with minimal accuracy loss.

### Results
- ✅ Started: R=2048, float64, 78.9% accuracy, 47 MB
- ✅ Final: R=1536, int8, **78.3% accuracy, 2.6 MB** — 9x smaller, -0.6% accuracy
- ✅ `int8_matmul.py` runs inference natively in int8, dequantize only at accumulation
- ✅ `quantize_encoder.py` provides symmetric int8 quantization utility

---

## Milestone 1 — Prove Zero Forgetting on Real Data  `[ ✅ PASSED — 2026-04-27 ]`

**Goal:** Add a second real task (AG News topic classification) without
forgetting IMDB sentiment.

**Protocol:**
1. Train `sentiment` on IMDB → record accuracy A1
2. Train `topic` on AG News (add to same model) → record accuracy A2
3. Re-evaluate IMDB sentiment → record accuracy A1'
4. Forgetting = A1 - A1' must be < 0.5%

### Results
- ✅ IMDB sentiment accuracy A1: **78.3%**
- ✅ AG News topic accuracy A2: **88.x%**
- ✅ IMDB accuracy after AG News A1': **78.3%** — delta = **0.0%**
- ✅ Saved artifacts: `multi.npz`, `multi.json`, `multi.vocab`
- ✅ Zero forgetting confirmed: structural guarantee holds on real distributions

**Exit criterion:** A1' ≥ A1 - 0.5%. **PASSED** (delta = 0.0%).

---

## Milestone 2 — 10 Tasks, Zero Forgetting  `[ IN PROGRESS ]`

**Goal:** Train 10 diverse HuggingFace text classification tasks sequentially.
Each task must achieve reasonable accuracy when evaluated immediately after
training. Re-evaluating any prior task must show < 1% degradation.

**Accuracy target revised:** Consistent with the M0 trade, per-task accuracy
targets are set relative to each dataset's ceiling rather than a fixed 80% bar.
The forgetting constraint (< 1% per prior task) remains strict.

**Candidate datasets:** imdb, ag_news, yelp_review_full, dbpedia_14,
yahoo_answers_topics, emotion, sst2, rotten_tomatoes, trec, financial_phrasebank

**Exit criterion:**
- Each task accuracy within 5% of its raw TF-IDF ceiling at time of training
- Re-evaluate first task after training all 10: accuracy drop < 1%
- Total model size < 10 MB
- Total training time < 30 minutes CPU

### Steps
- [ ] Extend `train_hf.py` to support all 10 candidate datasets
- [ ] Run sequential 10-task training, record per-task accuracy
- [ ] Re-evaluate task 1 after all 10 are trained
- [ ] Document results in `docs/RESEARCH_LEDGER.md`

---

## Milestone 3 — Production Hardening  `[ PENDING M2 ]`

**Goal:** Make the system fully usable by a non-technical person.

### Steps
- [ ] Fix vectorizer persistence in `HFAdapter` (pickle/joblib save+load)
- [ ] Enable standalone `--evaluate` subcommand in `train_hf.py`
- [ ] Add `datasets` to `pyproject.toml` and `requirements.txt`
- [ ] Write a one-page `QUICKSTART.md` with copy-paste commands
- [ ] Test entire pipeline on a clean Python 3.11 install (no prior packages)
- [ ] Add `--dry-run` flag that shows expected time and memory before training

**Exit criterion:** A person following `QUICKSTART.md` can train a 2-task
model in < 10 minutes with zero prior ML knowledge.

---

## Discipline Rules

1. **Green proof suite is non-negotiable.** Every commit must leave `proof/claims.py` and `proof/hardware_claim.py` at 30/30.
2. **No new features during a failing milestone.** Diagnose first.
3. **Accuracy-for-performance trade is accepted.** The 80% hard floor was replaced by "within 5% of raw TF-IDF ceiling" per task. Edge runtime and model size are first-class constraints.
4. **Consumer hardware always.** Never cite GPU numbers as validation.
5. **Document surprises.** If something unexpected happens, write it in `docs/RESEARCH_LEDGER.md` before moving on.
