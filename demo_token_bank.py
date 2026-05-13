"""demo_token_bank.py — end-to-end walkthrough of the v0.2 vocab evolution system.

Shows:
  1. Core vocab loaded (frozen)
  2. Text observed -> unknown tokens tracked
  3. Expansion pack frozen from top candidates
  4. Pack loaded into VocabManager -> combined features
  5. Char n-gram fallback for residual unknowns
  6. Manifest saved / round-trip reload

Run:
    python demo_token_bank.py
"""

from pathlib import Path

import numpy as np

from token_bank import TokenBank
from vocab_manager import VocabManager
from char_fallback import CharNGramFallback

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
CORE_VOCAB = ROOT / "v0.2" / "vocabs" / "core.vocab"
VOCABS_DIR = ROOT / "v0.2" / "vocabs"
MANIFEST = ROOT / "v0.2" / "manifest.json"
BANK_STATE = ROOT / "v0.2" / "token_bank.json"

# ---------------------------------------------------------------------------
# Demo corpus — domain-shifted text with tokens unlikely in the IMDB corpus
# ---------------------------------------------------------------------------

TECH_DOCS = [
    "The Kubernetes deployment uses etcd as the backing datastore for cluster state.",
    "Serverless functions on AWS Lambda scale to zero automatically between invocations.",
    "GraphQL subscriptions enable real-time websocket push from the Apollo server.",
    "Terraform provider plugins are written in Golang and communicate via gRPC.",
    "The CUDA kernel launch overhead dominates for microbatch sizes below 16.",
    "Tokenisation in sentencepiece uses unigram language model pruning.",
    "Quantization-aware training (QAT) calibrates int8 scales during backprop.",
    "LoRA adapters inject low-rank matrices into frozen pretrained transformer layers.",
    "RLHF relies on a reward model trained from human preference comparisons.",
    "FlashAttention fuses the attention softmax with the matmul to avoid HBM round-trips.",
    "The tokenizer uses byte-pair encoding (BPE) with a 50k vocabulary.",
    "Mixtral uses sparse MoE routing where each token activates two experts.",
    "DeepSpeed ZeRO-3 shards optimizer states, gradients, and parameters across GPUs.",
    "PEFT methods like prefix-tuning and prompt-tuning keep backbone weights frozen.",
    "vLLM's PagedAttention allocates KV cache in non-contiguous memory blocks.",
]

# ---------------------------------------------------------------------------
# Step 1 — Load core vocab
# ---------------------------------------------------------------------------

print("=" * 60)
print("STEP 1 — Core vocab (frozen)")
print("=" * 60)

bank = TokenBank(str(CORE_VOCAB))
print(f"  Core vocab size : {bank.core_vocab_size:,} tokens")
print(f"  Vocab file      : {CORE_VOCAB.name}")

# ---------------------------------------------------------------------------
# Step 2 — Observe domain-shifted text
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 2 — Observing tech-domain corpus")
print("=" * 60)

for i, doc in enumerate(TECH_DOCS):
    unknowns = bank.observe(doc)
    if unknowns:
        top = sorted(unknowns.items(), key=lambda x: x[1], reverse=True)[:3]
        top_str = ", ".join(f"'{t}' ({c})" for t, c in top)
        print(f"  doc {i+1:2d}: {len(unknowns):3d} unknowns  (top: {top_str})")

print(f"\n  Observations : {bank.n_observations}")
print(f"  Candidates   : {bank.n_candidates}")

# ---------------------------------------------------------------------------
# Step 3 — Score and inspect top candidates
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 3 — Top candidates by frequency")
print("=" * 60)

top_candidates = bank.score_candidates(top_k=20)
for rank, (token, count) in enumerate(top_candidates, 1):
    bar = "#" * count
    print(f"  {rank:2d}. {token:<30s} {count:3d} {bar}")

# ---------------------------------------------------------------------------
# Step 4 — Freeze expansion pack 001
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 4 — Freezing expansion_pack_001")
print("=" * 60)

pack_path = bank.freeze_expansion_pack(
    pack_id="001",
    output_dir=str(VOCABS_DIR),
    min_count=1,   # low threshold for demo; production would use 2+
    top_k=200,
)
print(f"  Frozen to : {pack_path}")
print(f"  File size : {pack_path.stat().st_size / 1024:.1f} KB")

bank.save_state(str(BANK_STATE))
print(f"  Bank state saved -> {BANK_STATE.name}")

# ---------------------------------------------------------------------------
# Step 5 — Load into VocabManager, inspect combined vocab
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 5 — VocabManager: core + expansion_pack_001")
print("=" * 60)

vm = VocabManager(str(CORE_VOCAB))
new_size = vm.load_expansion_pack(str(pack_path))

print(f"  Core vocab size      : {vm.core_vocab_size:,}")
print(f"  Expansion vocab size : {vm.expansion_vocab_size:,}")
print(f"  Combined vocab size  : {vm.total_vocab_size:,}")
print(f"  Loaded packs         : {vm.loaded_pack_ids}")

# Transform a sample sentence and show feature vector stats
sample = ["LoRA adapters inject low-rank matrices into frozen transformer layers."]
features = vm.transform(sample)
print(f"\n  Sample transform shape : {features.shape}")
print(f"  Non-zero features      : {int((features != 0).sum())}")
print(f"  Core-range non-zeros   : {int((features[:, :vm.core_vocab_size] != 0).sum())}")
print(f"  Expansion-range n.z.   : {int((features[:, vm.core_vocab_size:] != 0).sum())}")

# Save manifest
vm.save_manifest(str(MANIFEST))
print(f"\n  Manifest saved -> {MANIFEST}")

# ---------------------------------------------------------------------------
# Step 6 — Char n-gram fallback for residual unknowns
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 6 — CharNGramFallback for residual unknowns")
print("=" * 60)

fallback = CharNGramFallback(ngram_range=(3, 6), max_features=512)

residual_tokens = ["FlashAttention", "PagedAttention", "ZeRO3", "sentencepiece"]
encodings = fallback.encode_batch(residual_tokens)

print(f"  Encoding shape : {encodings.shape}")
for token, vec in zip(residual_tokens, encodings):
    nnz = int((vec != 0).sum())
    print(f"  '{token}': {nnz} non-zero dims, norm={np.linalg.norm(vec):.4f}")

# Similarity between morphologically similar tokens
pairs = [
    ("FlashAttention", "PagedAttention"),
    ("FlashAttention", "ZeRO3"),
    ("sentencepiece", "sentence"),
]
print()
for a, b in pairs:
    sim = fallback.similarity(a, b)
    print(f"  sim('{a}', '{b}') = {sim:.4f}")

# ---------------------------------------------------------------------------
# Step 7 — Round-trip: reload from manifest
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 7 — Round-trip reload from manifest")
print("=" * 60)

vm2 = VocabManager.from_manifest(str(MANIFEST))
print(f"  Reloaded total vocab size : {vm2.total_vocab_size:,}")
assert vm2.total_vocab_size == vm.total_vocab_size, "Manifest round-trip mismatch!"
print("  Round-trip OK")

bank2 = TokenBank.load_state(str(BANK_STATE))
print(f"  Reloaded bank candidates  : {bank2.n_candidates}")
assert bank2.n_candidates == bank.n_candidates, "Bank state round-trip mismatch!"
print("  Bank state round-trip OK")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Core vocab (frozen)  : {vm.core_vocab_size:,} tokens")
print(f"  Expansion pack 001   : {vm.expansion_vocab_size:,} tokens")
print(f"  Combined             : {vm.total_vocab_size:,} tokens")
print(f"  Char fallback dims   : {fallback.max_features}")
print()
print("  Files created:")
for f in sorted(Path(ROOT / "v0.2").rglob("*")):
    if f.is_file():
        print(f"    {f.relative_to(ROOT)}  ({f.stat().st_size / 1024:.1f} KB)")
print()
print("  Vocabulary can evolve without breaking reproducibility.")
print("  Existing models using core.vocab are unaffected.")
