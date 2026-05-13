# Changelog

## v0.2 (In Development)

This version is under active development

## v0.1 — TF-IDF Edition (2026-04-27)

First stable release. Proves the core claim: **zero catastrophic forgetting by construction**, not approximation, running on CPU with no deep-learning framework.

### What's in v0.1

**Zero catastrophic forgetting**
- Structural isolation: each task gets its own independent linear head; task B's gradient cannot reach task A's weights.
- Proven on IMDB → AG News sequential training. Forgetting = 0.0% by construction.
- `zero_forgetting.py` — two-class public API (`ZeroForgetReadout`, `sleep_consolidation`).

**TF-IDF vectorization**
- scikit-learn `TfidfVectorizer` as the front-end feature extractor (2 000 dimensions by default).
- Vocabulary saved alongside model weights so inference requires no retraining.
- `hf_adapter.py` handles HuggingFace dataset → TF-IDF → FixedEncoder pipeline.

**Fixed random encoder**
- `FixedEncoder` in `weightless_model.py`: random projection initialized once, spectral radius ≤ 0.94, never updated.
- `SystemMemory`: AC/DC temporal state with fast + slow streams, phase code, cross-frequency binding.
- R=1536 default (reservoir dimension).

**Int8 quantization (NPU-ready)**
- `int8_matmul.py`: symmetric int8 × int8 matmul with per-tensor scale, no external dependencies.
- `quantize_encoder.py`: one-shot quantization of the fixed encoder weights.
- 9× compression vs float32 baseline (23.4 MB → 2.6 MB).
- Accuracy delta < 0.01% on IMDB.

**Multi-head perception system**
- `multihead_model.py`: wraps a `WeightlessModel` with N isolated task heads behind a single `perceive()` call.
- `train_multihead.py`: training script for multi-head models with coordination decisions.
- `MultiHeadModel.decide()`: rule-based coordination across head outputs (e.g. urgency + sentiment → action).

**End-to-end reproduction**
- `reproduce.py --use-saved`: verify results without redownloading datasets.
- `reproduce.py --fast`: 2 000-sample smoke test (~10s per task).
- Full run reproduces 78.3% IMDB accuracy and confirmed 0.0% forgetting.

**Saved model artifacts**
- `quality.npz / .json / .vocab` — R=1536 float32 sentiment model (IMDB 78.3%, 23.4 MB).
- `quality_int8.npz / .json` — int8 quantized version (2.6 MB, same accuracy).
- `multi.npz / .json / .vocab` — sequential multi-task model: sentiment + topic (zero forgetting).

### What's NOT in v0.1

These are explicitly deferred to v0.2 or later:

- **Streaming / online learning interface** — batch training only; no token-by-token update API.
- **Sub-word tokenization** — TF-IDF bag-of-words only; no BPE, WordPiece, or SentencePiece.
- **Transformer-based encoder** — fixed random projection only; no pre-trained language model backbone.
- **GPU / NPU execution path** — int8 matmul is written for CPU; ONNX/CoreML export not yet implemented.
- **Automatic hyperparameter selection** — R, TF-IDF dim, learning rate require manual tuning per task.
- **Multi-GPU / distributed training** — single-process only.
- **Structured prediction** — classification heads only; no sequence labeling, span extraction, or generation.
- **Kernel network integration into production path** — kernel/governance/spectral research code lives in `core/` and `system/` but is not wired into the TF-IDF pipeline.
- **Web / REST API** — library interface only; no serving layer.
- **Model versioning / registry** — saved `.npz` files only; no MLflow, DVC, or similar.
