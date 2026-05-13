"""Project-wide constants for the weightless learning layer.

All magic numbers extracted here with documentation explaining their
origin and validated tradeoffs.
"""

# =============================================================================
# Model Architecture
# =============================================================================

# Reservoir dimension — empirically validated across three sizes (Milestone 0.5):
#   R=2048: 78.9% accuracy, 47 MB float32
#   R=1536: 78.3% accuracy, 23.4 MB float32  ← optimal tradeoff
#   R=1024: 76.1% accuracy, 15 MB float32
DEFAULT_RESERVOIR_DIM: int = 1536

# TF-IDF vocabulary — 2000 features captures the long tail while staying
# tractable on CPU; diminishing returns observed above ~3000
DEFAULT_TFIDF_DIM: int = 2000

# Sequence chunking — 6 sentences per document gives the memory enough
# temporal structure to build AC/DC contrast without padding overhead
DEFAULT_MAX_SENTENCES: int = 6

# Spectral radius for FixedEncoder — must be strictly below 1.0 (stable dynamics)
# 0.94 balances expressiveness vs. contraction; tested range was 0.85–0.99
TARGET_SPECTRAL_RADIUS: float = 0.94

# AC stream (fast): ~4-timestep effective memory window (1/(1-0.25) = 1.33 steps)
# DC stream (slow): ~50-timestep effective memory window (1/(1-0.98) = 50 steps)
AC_DECAY_RATE: float = 0.25
DC_DECAY_RATE: float = 0.98

# =============================================================================
# Training
# =============================================================================

DEFAULT_EPOCHS: int = 5
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_LEARNING_RATE: float = 0.05

# Sleep consolidation — optional post-training replay pass
DEFAULT_SLEEP_LR: float = 0.001
DEFAULT_SLEEP_CYCLES: int = 500
DEFAULT_SLEEP_BATCH_SIZE: int = 32

# =============================================================================
# Readout Head
# =============================================================================

# Xavier-style init scale: keeps pre-activations O(1) at init
WEIGHT_INIT_SCALE: float = 0.02

# Absolute weight clamp — prevents divergence in long online learning runs
# where gradients accumulate without the natural stabilisation of SGD batching
WEIGHT_CLIP_THRESHOLD: float = 3.0

# =============================================================================
# Phase Encoding
# =============================================================================

# Phase vector for single-timestep sequences: no temporal structure to encode
PHASE_SINGLE_TIMESTEP: tuple[float, float, float] = (0.0, 0.0, 1.0)

# =============================================================================
# Reproducibility
# =============================================================================

DEFAULT_SEED: int = 42

# =============================================================================
# Persistence
# =============================================================================

MODEL_FORMAT_VERSION: str = "1.0"
