"""Vectorizer comparison: TF-IDF vs HashingVectorizer for edge deployment.

Goal (Nintendo principles): same accuracy, less machine.
- Eliminate .vocab file (65 KB pickled sklearn object)
- No vocabulary to store = vocabulary-free deployment

Strategies tested:
  A. TF-IDF word 1-2grams     (current, requires .vocab)
  B. Hashing word 1-2grams    (vocabulary-free drop-in)
  C. Hashing char 3-6grams    (vocabulary-free, subword)

Evaluation:
  - Accuracy on IMDB (baseline reference: 78.3% full pipeline)
  - Vectorizer pickle size on disk
  - Memory footprint (sys.getsizeof approximation)
  - Vocabulary-free deployment: yes/no

Note: this script uses a fixed encoder + logistic head (no WeightlessModel reservoir)
to isolate vectorizer effects. Absolute accuracy won't match 78.3% but deltas are
representative of relative vectorizer quality.

Usage:
    python -m experiments.compare_vectorizers
    python -m experiments.compare_vectorizers --n-train 25000 --n-test 25000  # full IMDB
    python -m experiments.compare_vectorizers --n-features 4000               # wider hash space
"""

from __future__ import annotations

import argparse
import io
import pickle
import re
import sys
import time
from dataclasses import dataclass, field

import numpy as np

try:
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MaxAbsScaler
except ImportError as exc:
    raise ImportError("pip install scikit-learn") from exc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-train",    type=int, default=10_000, help="Training samples (default 10k for speed)")
    p.add_argument("--n-test",     type=int, default=5_000,  help="Test samples (default 5k for speed)")
    p.add_argument("--n-features", type=int, default=2_000,  help="Hash space / TF-IDF vocab size (default 2000)")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--full",       action="store_true",      help="Use full IMDB (25k train / 25k test)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_imdb(split: str, max_samples: int | None) -> tuple[list[str], list[int]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("pip install datasets") from exc

    ds = load_dataset("imdb", split=split, trust_remote_code=False)
    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))
    texts  = [str(row["text"])  for row in ds]
    labels = [int(row["label"]) for row in ds]
    return texts, labels


# ---------------------------------------------------------------------------
# Vectorizer strategies
# ---------------------------------------------------------------------------

def _tfidf_pipeline(n_features: int) -> Pipeline:
    vec = TfidfVectorizer(
        max_features=n_features,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 2),
    )
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    return Pipeline([("vec", vec), ("clf", clf)])


def _hash_word_pipeline(n_features: int) -> Pipeline:
    # HashingVectorizer has no sublinear_tf; norm="l2" compensates for doc-length bias.
    # Uses TfidfTransformer for IDF reweighting — stores only n_features IDF floats, no vocab strings.
    from sklearn.feature_extraction.text import TfidfTransformer
    vec = HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 2),
        norm=None,              # let TfidfTransformer handle normalization
    )
    idf = TfidfTransformer(sublinear_tf=True, norm="l2")
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    return Pipeline([("vec", vec), ("idf", idf), ("clf", clf)])


def _hash_char_pipeline(n_features: int) -> Pipeline:
    from sklearn.feature_extraction.text import TfidfTransformer
    vec = HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        strip_accents="unicode",
        analyzer="char_wb",     # char n-grams within word boundaries
        ngram_range=(3, 6),
        norm=None,
    )
    idf = TfidfTransformer(sublinear_tf=True, norm="l2")
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    return Pipeline([("vec", vec), ("idf", idf), ("clf", clf)])


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def _pickle_size_bytes(obj) -> int:
    buf = io.BytesIO()
    pickle.dump(obj, buf)
    return buf.tell()


def _vocab_size(pipe: Pipeline) -> int | None:
    vec = pipe.named_steps["vec"]
    if hasattr(vec, "vocabulary_"):
        return len(vec.vocabulary_)
    return None  # HashingVectorizer has no vocabulary string mapping


def _vectorizer_pickle_bytes(pipe: Pipeline) -> int:
    """Bytes needed to persist the feature-extraction portion (no clf head)."""
    parts = {}
    for key in ("vec", "idf"):
        if key in pipe.named_steps:
            parts[key] = pipe.named_steps[key]
    return _pickle_size_bytes(parts)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class Result:
    name: str
    accuracy: float
    train_time_s: float
    vectorizer_pickle_bytes: int
    vocab_size: int | None
    vocab_free: bool
    notes: str = ""


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_strategy(name: str, pipe: Pipeline, X_train, y_train, X_test, y_test) -> Result:
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    acc = pipe.score(X_test, y_test)

    vec_bytes = _vectorizer_pickle_bytes(pipe)
    vocab_sz  = _vocab_size(pipe)
    vocab_free = vocab_sz is None

    return Result(
        name=name,
        accuracy=acc,
        train_time_s=train_time,
        vectorizer_pickle_bytes=vec_bytes,
        vocab_size=vocab_sz,
        vocab_free=vocab_free,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

_COL = 28

def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n/1024:.1f} KB"
    return f"{n/1024**2:.1f} MB"


def _print_report(results: list[Result], n_features: int) -> None:
    baseline_acc = results[0].accuracy
    baseline_bytes = results[0].vectorizer_pickle_bytes

    header = (
        f"\n{'Strategy':<{_COL}}  {'Accuracy':>9}  {'dAcc':>7}  "
        f"{'Vec size':>10}  {'dSize':>8}  {'Vocab-free':>11}  {'Train(s)':>9}"
    )
    print("\n" + "=" * (len(header) - 1))
    print(f"  Vectorizer Comparison — IMDB Sentiment  (n_features={n_features})")
    print("=" * (len(header) - 1))
    print(header)
    print("-" * (len(header) - 1))

    for r in results:
        delta_acc   = r.accuracy - baseline_acc
        delta_bytes = r.vectorizer_pickle_bytes - baseline_bytes
        sign_acc    = "+" if delta_acc >= 0 else ""
        sign_bytes  = "+" if delta_bytes >= 0 else ""
        vf          = "YES" if r.vocab_free else "no"
        vocab_str   = f"  (vocab={r.vocab_size:,})" if r.vocab_size else ""
        delta_acc_str   = f"{sign_acc}{delta_acc*100:.2f}%"
        delta_bytes_str = f"{sign_bytes}{_fmt_bytes(abs(delta_bytes))}" if delta_bytes != 0 else "---"
        if delta_bytes < 0:
            delta_bytes_str = f"-{_fmt_bytes(abs(delta_bytes))}"
        print(
            f"  {r.name:<{_COL-2}}  {r.accuracy*100:>8.2f}%  {delta_acc_str:>7}  "
            f"{_fmt_bytes(r.vectorizer_pickle_bytes):>10}  {delta_bytes_str:>8}  "
            f"{vf:>11}  {r.train_time_s:>8.1f}s{vocab_str}"
        )

    print("-" * (len(header) - 1))
    print()

    # Recommendation
    winner = None
    for r in results[1:]:  # skip baseline
        delta = r.accuracy - baseline_acc
        if delta >= -0.01:  # within 1% accuracy loss
            winner = r
            break

    if winner:
        savings = baseline_bytes - winner.vectorizer_pickle_bytes
        print(f"  RECOMMENDATION: '{winner.name}'")
        print(f"    Accuracy delta   : {(winner.accuracy - baseline_acc)*100:+.2f}%  (threshold: -1.00%)")
        print(f"    Storage savings  : {_fmt_bytes(savings)} vectorizer pickle")
        print(f"    Vocab-free       : {'YES - no .vocab file needed at deployment' if winner.vocab_free else 'no'}")
    else:
        print("  All alternatives exceed -1% accuracy threshold vs TF-IDF baseline.")
        print("  Consider increasing --n-features or using TF-IDF + compression.")

    print()
    print("  Note: accuracy measured with logistic regression head, not WeightlessModel.")
    print("  Reference point for full pipeline: 78.3% (quality_int8 config).")
    print("  Relative deltas between strategies transfer across architectures.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    n_train = 25_000 if args.full else args.n_train
    n_test  = 25_000 if args.full else args.n_test
    n_feat  = args.n_features

    print(f"\nLoading IMDB (train={n_train:,}, test={n_test:,}) …", flush=True)
    X_train, y_train = _load_imdb("train", n_train)
    X_test,  y_test  = _load_imdb("test",  n_test)
    print(f"  Loaded {len(X_train):,} train / {len(X_test):,} test samples.")

    strategies = [
        ("A. TF-IDF word 1-2grams (current)",  _tfidf_pipeline(n_feat)),
        ("B. Hashing word 1-2grams",            _hash_word_pipeline(n_feat)),
        ("C. Hashing char 3-6grams (char_wb)",  _hash_char_pipeline(n_feat)),
    ]

    results: list[Result] = []
    for name, pipe in strategies:
        print(f"\nRunning: {name} …", flush=True)
        r = _run_strategy(name, pipe, X_train, y_train, X_test, y_test)
        results.append(r)
        print(f"  accuracy={r.accuracy*100:.2f}%  vec_size={_fmt_bytes(r.vectorizer_pickle_bytes)}  "
              f"vocab_free={r.vocab_free}  time={r.train_time_s:.1f}s")

    _print_report(results, n_feat)


if __name__ == "__main__":
    main()
