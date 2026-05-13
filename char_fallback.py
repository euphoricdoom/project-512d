"""Character n-gram fallback encoder for completely unknown tokens.

Hash-trick based: no vocabulary needed, fixed output dimension,
stable across Python sessions (uses MD5 instead of built-in hash).
"""

import hashlib
import numpy as np
from typing import List


class CharNGramFallback:
    """Encode unknown tokens via character n-grams (default 3–6 chars).

    Uses MD5 hash modulo max_features for bucket assignment, so results
    are deterministic regardless of PYTHONHASHSEED.
    """

    def __init__(self, ngram_range: tuple = (3, 6), max_features: int = 512):
        self.ngram_range = ngram_range
        self.max_features = max_features

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_ngrams(self, token: str) -> List[str]:
        token = token.lower()
        ngrams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(token) - n + 1):
                ngrams.append(token[i : i + n])
        return ngrams

    @staticmethod
    def _stable_hash(s: str) -> int:
        return int(hashlib.md5(s.encode()).hexdigest(), 16)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, unknown_token: str) -> np.ndarray:
        """Return L2-normalised char n-gram feature vector for one token."""
        ngrams = self._extract_ngrams(unknown_token)
        vec = np.zeros(self.max_features, dtype=np.float32)
        for ng in ngrams:
            idx = self._stable_hash(ng) % self.max_features
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def encode_batch(self, tokens: List[str]) -> np.ndarray:
        """Encode multiple tokens → (n_tokens, max_features)."""
        if not tokens:
            return np.zeros((0, self.max_features), dtype=np.float32)
        return np.stack([self.encode(t) for t in tokens])

    def similarity(self, token_a: str, token_b: str) -> float:
        """Cosine similarity between two token encodings (0–1)."""
        va = self.encode(token_a)
        vb = self.encode(token_b)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(np.dot(va, vb) / denom) if denom > 0 else 0.0
