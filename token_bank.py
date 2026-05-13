"""Token Bank — observes text, tracks unknown tokens, proposes expansion packs.

Core contract: never mutates existing vocabs. The bank only observes and
proposes; callers decide when to freeze candidates into a new pack.
"""

import json
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from char_fallback import CharNGramFallback

# Same tokenisation pattern used by the core TfidfVectorizer.
_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")

# Fallback stopword list used when sklearn is unavailable.
_FALLBACK_STOPWORDS: frozenset = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "as", "is", "was", "are",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "can",
    "this", "that", "these", "those", "it", "its", "not", "no",
    "so", "if", "then", "than", "such", "into", "through", "during",
    "about", "up", "down", "out", "off", "over", "under", "again",
    "once", "here", "there", "when", "where", "which", "who", "how",
    "all", "both", "each", "few", "more", "most", "other", "some",
    "what", "your", "my", "our", "their", "use", "uses", "used",
    "also", "just", "only", "any", "between", "after", "before",
})

# Max observed sentences to buffer for IDF estimation during pack freeze.
_MAX_BUFFER = 10_000


class TokenBank:
    """Observe text streams and track tokens absent from core_vocab.

    Candidates accumulate with frequency counts.  Call ``freeze_expansion_pack``
    to snapshot the top candidates into a frozen ``.vocab`` file that can be
    loaded by :class:`VocabManager` without touching any existing pack.
    """

    def __init__(self, core_vocab_path: str):
        self.core_vocab_path = Path(core_vocab_path)
        self._core_tokens: set = self._load_core_tokens()
        self.candidates: Dict[str, int] = defaultdict(int)
        self.char_ngrams = CharNGramFallback()
        self._observations: int = 0
        self._text_buffer: List[str] = []  # sampled sentences for IDF fitting

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_core_tokens(self) -> set:
        with open(self.core_vocab_path, "rb") as f:
            vectorizer = pickle.load(f)
        return set(vectorizer.vocabulary_.keys())

    def _tokenize(self, text: str) -> List[str]:
        return _TOKEN_RE.findall(text.lower())

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe(self, text: str) -> Dict[str, int]:
        """Process one text document; return unknown tokens seen this call.

        Unknown unigrams *and* bigrams are tracked.  Tokens already in
        core_vocab are silently skipped.
        """
        tokens = self._tokenize(text)
        new_this_call: Dict[str, int] = {}

        for tok in tokens:
            if tok not in self._core_tokens:
                self.candidates[tok] += 1
                new_this_call[tok] = self.candidates[tok]

        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i + 1]}"
            if bigram not in self._core_tokens:
                self.candidates[bigram] += 1
                new_this_call[bigram] = self.candidates[bigram]

        # Buffer a sentence sample for later IDF fitting.
        if len(self._text_buffer) < _MAX_BUFFER:
            self._text_buffer.append(text)

        self._observations += 1
        return new_this_call

    def observe_batch(self, texts: List[str]) -> int:
        """Observe multiple texts; returns total unknown token types seen."""
        before = len(self.candidates)
        for text in texts:
            self.observe(text)
        return len(self.candidates) - before

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_candidates(self, top_k: int = 200) -> List[Tuple[str, int]]:
        """Return top-k candidates filtered for quality, unigrams ranked first.

        Filters applied before ranking:
        - Unigrams that are stopwords are dropped.
        - Bigrams that start or end with a stopword are dropped.
        - Bigrams with frequency < 3 are dropped (higher bar than unigrams).

        Result order: unigrams by frequency (desc), then bigrams by frequency (desc).
        This surfaces single content words ahead of phrases, which keeps the
        expansion pack readable and avoids promoting glue-word combinations like
        "the kubernetes" or "uses etcd".
        """
        try:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            stopwords: set = set(ENGLISH_STOP_WORDS)
        except ImportError:
            stopwords = set(_FALLBACK_STOPWORDS)

        unigrams: List[Tuple[str, int]] = []
        bigrams: List[Tuple[str, int]] = []

        for token, count in self.candidates.items():
            parts = token.split(" ", 1)
            if len(parts) == 1:
                if token not in stopwords:
                    unigrams.append((token, count))
            else:
                first, last = parts[0], parts[1]
                if first in stopwords or last in stopwords:
                    continue
                if count < 3:
                    continue
                bigrams.append((token, count))

        unigrams.sort(key=lambda x: x[1], reverse=True)
        bigrams.sort(key=lambda x: x[1], reverse=True)
        return (unigrams + bigrams)[:top_k]

    def get_char_encoding(self, token: str) -> np.ndarray:
        """Return char n-gram fallback encoding for any unknown token."""
        return self.char_ngrams.encode(token)

    # ------------------------------------------------------------------
    # Pack freezing
    # ------------------------------------------------------------------

    def freeze_expansion_pack(
        self,
        pack_id: str,
        output_dir: str,
        min_count: int = 2,
        top_k: int = 500,
    ) -> Path:
        """Freeze top candidates into ``expansion_pack_{pack_id}.vocab``.

        Selects candidates that appear at least *min_count* times, takes the
        top-*top_k* by frequency, fits a TfidfVectorizer with a fixed
        vocabulary on the buffered observed text, and pickles the result.

        Does NOT touch core_vocab or any previously frozen pack.

        Returns the path of the created file.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        eligible = [t for t, c in self.candidates.items() if c >= min_count]
        if not eligible:
            raise ValueError(
                f"No candidates meet min_count={min_count}. "
                f"Total candidates: {len(self.candidates)}, "
                f"max count: {max(self.candidates.values(), default=0)}."
            )

        eligible.sort(key=lambda t: self.candidates[t], reverse=True)
        eligible = eligible[:top_k]

        vocab_map = {token: i for i, token in enumerate(eligible)}

        # Use buffered texts for IDF; fall back to synthetic single-doc corpus.
        fit_corpus = self._text_buffer if self._text_buffer else [" ".join(eligible)]

        vectorizer = TfidfVectorizer(
            vocabulary=vocab_map,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"(?u)\b\w\w+\b",
        )
        vectorizer.fit(fit_corpus)

        pack_path = output_dir / f"expansion_pack_{pack_id}.vocab"
        with open(pack_path, "wb") as f:
            pickle.dump(vectorizer, f)

        return pack_path

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path: str) -> None:
        """Persist candidate counts to JSON (text buffer not saved)."""
        state = {
            "core_vocab_path": str(self.core_vocab_path),
            "observations": self._observations,
            "candidates": dict(self.candidates),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_state(cls, path: str) -> "TokenBank":
        """Restore a TokenBank from a previously saved JSON state."""
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        bank = cls(state["core_vocab_path"])
        bank._observations = state["observations"]
        bank.candidates = defaultdict(int, state["candidates"])
        return bank

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)

    @property
    def n_observations(self) -> int:
        return self._observations

    @property
    def core_vocab_size(self) -> int:
        return len(self._core_tokens)
