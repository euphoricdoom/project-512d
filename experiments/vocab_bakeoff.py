"""experiments/vocab_bakeoff.py — Vocabulary system ablation study.

Three vocabulary conditions × two datasets.

Conditions
----------
1. Core only        — 2,000 TF-IDF features (baseline)
2. Core + expansion — 2,000 + up to 200 domain-specific tokens
3. Core + exp + char— condition 2 features + 512 char n-gram fallback features

Datasets
--------
- IMDB (general domain) — must not regress
- 20newsgroups comp.*  — tech-domain, should improve with expansion

Metrics
-------
- Accuracy           (LogisticRegression on features)
- Feature dim        (input dimensionality)
- Vocab files KB     (disk size of loaded vocab files)
- Unknown-token rate (% of test tokens absent from the vocab)
- Expansion nonzeros (mean expansion features that fire per test doc; n/a for core-only)

Success gates
-------------
- Expansion improves domain task accuracy
- Does NOT hurt IMDB accuracy (within 0.5%)
- Expansion features are actually being used (nonzero expansion usage > 0)

Run
---
    python experiments/vocab_bakeoff.py
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from char_fallback import CharNGramFallback
from token_bank import TokenBank
from vocab_manager import VocabManager

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CORE_VOCAB = ROOT / "v0.2" / "vocabs" / "core.vocab"
VOCABS_DIR = ROOT / "v0.2" / "vocabs"
_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")

N_IMDB_TRAIN = 3_000    # subset for speed
N_IMDB_TEST  = 1_000
CHAR_DIM     = 512
EXPANSION_TOP_K = 200


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_imdb(n_train: int, n_test: int) -> Tuple[List[str], List[int], List[str], List[int]]:
    """Load a subset of IMDB via HuggingFace datasets."""
    from datasets import load_dataset
    rng = np.random.default_rng(42)

    ds_train = load_dataset("imdb", split="train")
    ds_test  = load_dataset("imdb", split="test")

    idx_tr = rng.choice(len(ds_train), min(n_train, len(ds_train)), replace=False)
    idx_te = rng.choice(len(ds_test),  min(n_test,  len(ds_test)),  replace=False)

    train_texts  = [str(ds_train[int(i)]["text"])  for i in idx_tr]
    train_labels = [int(ds_train[int(i)]["label"]) for i in idx_tr]
    test_texts   = [str(ds_test[int(i)]["text"])   for i in idx_te]
    test_labels  = [int(ds_test[int(i)]["label"])  for i in idx_te]
    return train_texts, train_labels, test_texts, test_labels


def _load_tech() -> Tuple[List[str], List[int], List[str], List[int]]:
    """Load 20newsgroups comp.* categories (5-class tech classification)."""
    cats = [
        "comp.graphics",
        "comp.os.ms-windows.misc",
        "comp.sys.ibm.pc.hardware",
        "comp.sys.mac.hardware",
        "comp.windows.x",
    ]
    train = fetch_20newsgroups(subset="train", categories=cats, remove=("headers", "footers", "quotes"))
    test  = fetch_20newsgroups(subset="test",  categories=cats, remove=("headers", "footers", "quotes"))
    return train.data, train.target.tolist(), test.data, test.target.tolist()


# ---------------------------------------------------------------------------
# Expansion pack helpers
# ---------------------------------------------------------------------------

def _build_expansion_pack(train_texts: List[str], pack_path: Path) -> Path:
    """Observe training texts, filter with new score_candidates, freeze pack."""
    bank = TokenBank(str(CORE_VOCAB))
    bank.observe_batch(train_texts)

    # score_candidates now applies stopword + bigram-frequency filters
    candidates = bank.score_candidates(top_k=EXPANSION_TOP_K)
    if not candidates:
        raise RuntimeError("No candidates survived filtering.")

    n_unigrams = sum(1 for t, _ in candidates if " " not in t)
    n_bigrams  = sum(1 for t, _ in candidates if " " in t)
    print(f"    Candidates: {len(candidates)} ({n_unigrams} unigrams, {n_bigrams} bigrams)")
    print(f"    Top 10: {[t for t, _ in candidates[:10]]}")

    # Freeze only the filtered candidates into the pack
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pickle

    vocab_map = {token: i for i, token in enumerate(t for t, _ in candidates)}
    fit_corpus = bank._text_buffer if bank._text_buffer else [" ".join(vocab_map)]
    vec = TfidfVectorizer(
        vocabulary=vocab_map,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"(?u)\b\w\w+\b",
    )
    vec.fit(fit_corpus)
    with open(pack_path, "wb") as f:
        pickle.dump(vec, f)
    return pack_path


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _transform(
    vm: VocabManager,
    texts: List[str],
    use_char: bool = False,
) -> np.ndarray:
    """VocabManager transform + optional char fallback columns."""
    X = vm.transform(texts)                     # (n, core + expansion)
    if not use_char:
        return X

    char_enc = CharNGramFallback(ngram_range=(3, 6), max_features=CHAR_DIM)
    core_tokens: set = set(vm.core.vocabulary_.keys())
    exp_tokens:  set = set()
    for _, vec in vm._expansions:
        exp_tokens.update(vec.vocabulary_.keys())
    known = core_tokens | exp_tokens

    char_rows = []
    for doc in texts:
        toks = _TOKEN_RE.findall(doc.lower())
        unknown_toks = [t for t in toks if t not in known]
        if unknown_toks:
            char_vecs = char_enc.encode_batch(unknown_toks)
            char_rows.append(char_vecs.mean(axis=0))
        else:
            char_rows.append(np.zeros(CHAR_DIM, dtype=np.float32))

    return np.hstack([X, np.stack(char_rows)])


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _unknown_rate(texts: List[str], known_tokens: set) -> float:
    """Fraction of token occurrences in texts not covered by known_tokens."""
    total = unknown = 0
    for doc in texts:
        toks = _TOKEN_RE.findall(doc.lower())
        total += len(toks)
        unknown += sum(1 for t in toks if t not in known_tokens)
    return 100.0 * unknown / max(total, 1)


def _expansion_nonzeros(X: np.ndarray, core_dim: int) -> float:
    """Mean number of non-zero expansion features per document."""
    if X.shape[1] <= core_dim:
        return float("nan")
    exp_cols = X[:, core_dim:]
    return float((exp_cols != 0).sum(axis=1).mean())


def _vocab_kb(vm: VocabManager) -> float:
    """Total disk size of loaded vocab files in KB."""
    paths = [str(vm.core_vocab_path)] + vm._expansion_paths
    total = sum(Path(p).stat().st_size for p in paths if Path(p).exists())
    return total / 1024.0


# ---------------------------------------------------------------------------
# Single condition evaluation
# ---------------------------------------------------------------------------

def _run_condition(
    label: str,
    train_X: np.ndarray,
    train_y: List[int],
    test_X: np.ndarray,
    test_y: List[int],
    test_texts: List[str],
    vm: VocabManager,
    use_char: bool,
) -> Dict:
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
    clf.fit(train_X, train_y)
    acc = float(np.mean(clf.predict(test_X) == test_y))

    core_dim = vm.core_vocab_size
    known = set(vm.core.vocabulary_.keys())
    for _, vec in vm._expansions:
        known.update(vec.vocabulary_.keys())

    unk_rate = _unknown_rate(test_texts, known)
    exp_nz   = _expansion_nonzeros(test_X, core_dim)
    disk_kb  = _vocab_kb(vm)

    return {
        "label": label,
        "accuracy": acc,
        "feat_dim": test_X.shape[1],
        "vocab_kb": disk_kb,
        "unk_rate": unk_rate,
        "exp_nonzeros": exp_nz,
    }


# ---------------------------------------------------------------------------
# Main bakeoff
# ---------------------------------------------------------------------------

def run_bakeoff():
    print("=" * 70)
    print("VOCAB BAKEOFF — Core / Core+Expansion / Core+Expansion+Char")
    print("=" * 70)

    if not CORE_VOCAB.exists():
        raise FileNotFoundError(
            f"Core vocab not found: {CORE_VOCAB}\n"
            "Run:  python demo_token_bank.py  to create v0.2/vocabs/"
        )

    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------
    print("\n[1/4] Loading datasets...")
    print("  IMDB (HuggingFace)...")
    imdb_tr_texts, imdb_tr_y, imdb_te_texts, imdb_te_y = _load_imdb(N_IMDB_TRAIN, N_IMDB_TEST)
    print(f"  IMDB: {len(imdb_tr_texts)} train / {len(imdb_te_texts)} test")

    print("  20newsgroups (comp.*)...")
    tech_tr_texts, tech_tr_y, tech_te_texts, tech_te_y = _load_tech()
    print(f"  20news-tech: {len(tech_tr_texts)} train / {len(tech_te_texts)} test")

    # ------------------------------------------------------------------
    # Build expansion pack from TECH training data
    # ------------------------------------------------------------------
    print("\n[2/4] Building expansion pack from tech training data...")
    pack_path = VOCABS_DIR / "expansion_pack_bakeoff.vocab"
    _build_expansion_pack(tech_tr_texts, pack_path)
    print(f"  Saved: {pack_path.name}  ({pack_path.stat().st_size / 1024:.1f} KB)")

    # ------------------------------------------------------------------
    # Build feature matrices for each condition × dataset
    # ------------------------------------------------------------------
    print("\n[3/4] Extracting features (3 conditions × 2 datasets)...")

    # Condition 1: core only
    vm_core = VocabManager(str(CORE_VOCAB))

    # Condition 2: core + expansion
    vm_exp = VocabManager(str(CORE_VOCAB))
    vm_exp.load_expansion_pack(str(pack_path))

    # Condition 3: same vm but with char fallback columns added in _transform

    datasets = {
        "IMDB":      (imdb_tr_texts, imdb_tr_y, imdb_te_texts, imdb_te_y),
        "20news-tech": (tech_tr_texts, tech_tr_y, tech_te_texts, tech_te_y),
    }

    conditions = [
        ("Core only",         vm_core, False),
        ("Core + Expansion",  vm_exp,  False),
        ("Core + Exp + Char", vm_exp,  True),
    ]

    results = []
    for ds_name, (tr_texts, tr_y, te_texts, te_y) in datasets.items():
        print(f"\n  Dataset: {ds_name}")
        for cond_label, vm, use_char in conditions:
            sys.stdout.write(f"    {cond_label:<22} ... ")
            sys.stdout.flush()
            tr_X = _transform(vm, tr_texts, use_char)
            te_X = _transform(vm, te_texts, use_char)
            row = _run_condition(cond_label, tr_X, tr_y, te_X, te_y, te_texts, vm, use_char)
            row["dataset"] = ds_name
            results.append(row)
            print(f"acc={row['accuracy']:.1%}")

    # ------------------------------------------------------------------
    # Build comparison table (ASCII-safe)
    # ------------------------------------------------------------------
    print("\n[4/4] Results\n")

    col_w = [22, 8, 9, 9, 11, 14]
    header = ["Condition", "Acc", "Feat dim", "Vocab KB", "Unk rate %", "Exp nonzeros"]
    _row = lambda cells: "  ".join(str(c).ljust(w) for c, w in zip(cells, col_w))

    sep    = "-" * (sum(col_w) + 2 * (len(col_w) - 1))
    hline  = "-" * 70

    lines: List[str] = []

    def _get(ds, label):
        for r in results:
            if r["dataset"] == ds and r["label"] == label:
                return r
        return None

    for ds_name in ("IMDB", "20news-tech"):
        lines.append(hline)
        lines.append(f"  {ds_name}")
        lines.append(hline)
        lines.append("  " + _row(header))
        lines.append("  " + sep)
        ds_rows = [r for r in results if r["dataset"] == ds_name]
        for r in ds_rows:
            exp_nz = f"{r['exp_nonzeros']:.1f}" if not (r['exp_nonzeros'] != r['exp_nonzeros']) else "n/a"
            cells = [
                r["label"],
                f"{r['accuracy']:.1%}",
                str(r["feat_dim"]),
                f"{r['vocab_kb']:.0f}",
                f"{r['unk_rate']:.1f}%",
                exp_nz,
            ]
            lines.append("  " + _row(cells))
        lines.append("")

    # ------------------------------------------------------------------
    # Success gate evaluation
    # ------------------------------------------------------------------
    core_imdb  = _get("IMDB", "Core only")
    exp_imdb   = _get("IMDB", "Core + Expansion")
    core_tech  = _get("20news-tech", "Core only")
    exp_tech   = _get("20news-tech", "Core + Expansion")

    gate1 = exp_tech["accuracy"] > core_tech["accuracy"]
    gate2 = abs(exp_imdb["accuracy"] - core_imdb["accuracy"]) <= 0.005
    gate3 = exp_tech["exp_nonzeros"] > 0

    def _tick(ok): return "PASS" if ok else "FAIL"

    lines.append(hline)
    lines.append("  SUCCESS GATES")
    lines.append(hline)
    lines.append(
        f"  [{_tick(gate1)}] Expansion improves 20news-tech: "
        f"{core_tech['accuracy']:.1%} -> {exp_tech['accuracy']:.1%} "
        f"({'+' if gate1 else ''}{(exp_tech['accuracy'] - core_tech['accuracy'])*100:.1f}pp)"
    )
    lines.append(
        f"  [{_tick(gate2)}] Expansion does not hurt IMDB (<=0.5pp): "
        f"{core_imdb['accuracy']:.1%} -> {exp_imdb['accuracy']:.1%} "
        f"(delta {(exp_imdb['accuracy'] - core_imdb['accuracy'])*100:+.1f}pp)"
    )
    lines.append(
        f"  [{_tick(gate3)}] Expansion features fire on tech test set: "
        f"{exp_tech['exp_nonzeros']:.1f} mean nonzeros/doc"
    )
    lines.append("")

    all_pass = gate1 and gate2 and gate3
    status = "ALL GATES PASS" if all_pass else "SOME GATES FAILED"
    lines.append(f"  Result: {status}")
    lines.append("")

    # Print to console
    for line in lines:
        print(line)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    txt_path  = out_dir / "bakeoff_results.txt"
    json_path = out_dir / "bakeoff_results.json"

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved table  -> {txt_path}")

    # Sanitise NaN before serialising (JSON doesn't allow NaN literals)
    json_results = []
    for r in results:
        row = dict(r)
        if row.get("exp_nonzeros") != row.get("exp_nonzeros"):  # NaN check
            row["exp_nonzeros"] = None
        json_results.append(row)

    json_path.write_text(
        json.dumps({"gates": {"gate1_expansion_improves_tech": gate1,
                               "gate2_no_imdb_regression": gate2,
                               "gate3_expansion_fires": gate3,
                               "all_pass": all_pass},
                    "results": json_results},
                   indent=2),
        encoding="utf-8",
    )
    print(f"  Saved JSON   -> {json_path}")

    return results


if __name__ == "__main__":
    run_bakeoff()
