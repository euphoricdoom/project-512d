"""VocabManager — combines a frozen core vocab with zero or more expansion packs.

Core vocab is always immutable.  Expansion packs are loaded at runtime and
contribute additional features appended after the core features.  If the same
token appears in both core and an expansion pack, core always wins (its index
is used and the expansion entry is ignored in the combined vocab).
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp


def load_vocab(path: str):
    """Deserialise a pickled TfidfVectorizer."""
    with open(path, "rb") as f:
        return pickle.load(f)


class VocabManager:
    """Manages core vocab + zero or more frozen expansion packs.

    Usage::

        vm = VocabManager("v0.2/vocabs/core.vocab")
        vm.load_expansion_pack("v0.2/vocabs/expansion_pack_001.vocab")
        features = vm.transform(["some new text"])   # (1, core+exp size)
        vm.save_manifest("v0.2/manifest.json")
    """

    def __init__(self, core_vocab_path: str):
        self.core_vocab_path = Path(core_vocab_path)
        self.core = load_vocab(core_vocab_path)
        self._expansions: List[Tuple[str, object]] = []   # (pack_id, vectorizer)
        self._expansion_paths: List[str] = []

    # ------------------------------------------------------------------
    # Pack management
    # ------------------------------------------------------------------

    def load_expansion_pack(self, pack_path: str) -> int:
        """Load a frozen expansion pack.  Returns new total vocab size."""
        pack_path = Path(pack_path)
        if not pack_path.exists():
            raise FileNotFoundError(f"Expansion pack not found: {pack_path}")

        pack_id = pack_path.stem
        if any(pid == pack_id for pid, _ in self._expansions):
            raise ValueError(f"Pack '{pack_id}' already loaded.")

        vectorizer = load_vocab(pack_path)
        self._expansions.append((pack_id, vectorizer))
        self._expansion_paths.append(str(pack_path))
        return self.total_vocab_size

    def unload_expansion_pack(self, pack_id: str) -> None:
        """Remove a loaded expansion pack from memory (does not delete file)."""
        before = len(self._expansions)
        self._expansions = [(pid, v) for pid, v in self._expansions if pid != pack_id]
        self._expansion_paths = [
            p for p in self._expansion_paths if Path(p).stem != pack_id
        ]
        if len(self._expansions) == before:
            raise KeyError(f"Pack '{pack_id}' is not loaded.")

    # ------------------------------------------------------------------
    # Vocab inspection
    # ------------------------------------------------------------------

    def get_combined_vocab(self) -> Dict[str, int]:
        """Return merged token→index dict (core first, expansions appended).

        Core always wins: if a token exists in core *and* an expansion pack,
        its core index is used and the expansion entry is skipped.
        """
        combined: Dict[str, int] = dict(self.core.vocabulary_)
        offset = len(combined)
        for pack_id, vectorizer in self._expansions:
            for token, local_idx in vectorizer.vocabulary_.items():
                if token not in combined:
                    combined[token] = offset + local_idx
        return combined

    # ------------------------------------------------------------------
    # Transformation
    # ------------------------------------------------------------------

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform text list → (n_docs, total_vocab_size) float32 array.

        Core features come first; expansion features are appended in load order.
        """
        core_mat = self.core.transform(texts)

        if not self._expansions:
            return core_mat.toarray().astype(np.float32)

        parts = [core_mat]
        for pack_id, vectorizer in self._expansions:
            parts.append(vectorizer.transform(texts))

        combined = sp.hstack(parts, format="csr")
        return combined.toarray().astype(np.float32)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def core_vocab_size(self) -> int:
        return len(self.core.vocabulary_)

    @property
    def expansion_vocab_size(self) -> int:
        return sum(len(v.vocabulary_) for _, v in self._expansions)

    @property
    def total_vocab_size(self) -> int:
        return self.core_vocab_size + self.expansion_vocab_size

    @property
    def n_expansion_packs(self) -> int:
        return len(self._expansions)

    @property
    def loaded_pack_ids(self) -> List[str]:
        return [pid for pid, _ in self._expansions]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_manifest(self, path: str) -> None:
        """Write a JSON manifest of the active configuration."""
        manifest = {
            "core_vocab_path": str(self.core_vocab_path),
            "core_vocab_size": self.core_vocab_size,
            "total_vocab_size": self.total_vocab_size,
            "expansion_packs": [
                {"id": pid, "path": p, "size": len(v.vocabulary_)}
                for (pid, v), p in zip(self._expansions, self._expansion_paths)
            ],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    @classmethod
    def from_manifest(cls, manifest_path: str) -> "VocabManager":
        """Reconstruct a VocabManager from a saved manifest."""
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        manager = cls(manifest["core_vocab_path"])
        for pack in manifest.get("expansion_packs", []):
            manager.load_expansion_pack(pack["path"])
        return manager
