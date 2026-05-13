"""Reservoir dim sweep: 512, 1024, 2048 across all three tasks.

At fixed 10k training samples and locked head config (ep=500, lr=0.3/0.2),
does increasing reservoir_dim close the gap to the TF-IDF ceiling?

Each dim uses an independent random encoder (same seed), so results are
directly comparable.  Feature dimension scales as 6*res_dim + 3.

Usage
-----
    python -m experiments.run_reservoir_sweep
    python -m experiments.run_reservoir_sweep --dims 256 512 1024 2048
    python -m experiments.run_reservoir_sweep --max-train 5000 --max-test 872
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from weightless_model import SystemMemory
from zero_forgetting import ZeroForgetReadout
from experiments.real_forgetting_experiment import MultiTaskAdapter, accuracy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TFIDF_DIM     = 3000
MAX_SENTENCES = 3
BATCH_SIZE    = 64
SEED          = 42

BEST_CONFIG = {
    "sst2":    {"epochs": 500, "lr": 0.3},
    "ag_news": {"epochs": 500, "lr": 0.2},
    "emotion": {"epochs": 500, "lr": 0.3},
}

TASKS = [
    {
        "id":          "sst2",
        "dataset":     "sst2",
        "subset":      None,
        "split_train": "train",
        "split_test":  "validation",
        "text_field":  "sentence",
        "label_field": "label",
        "label":       "SST-2",
    },
    {
        "id":          "ag_news",
        "dataset":     "ag_news",
        "subset":      None,
        "split_train": "train",
        "split_test":  "test",
        "text_field":  "text",
        "label_field": "label",
        "label":       "AG News",
    },
    {
        "id":          "emotion",
        "dataset":     "dair-ai/emotion",
        "subset":      None,
        "split_train": "train",
        "split_test":  "test",
        "text_field":  "text",
        "label_field": "label",
        "label":       "Emotion",
    },
]


# ---------------------------------------------------------------------------
# Build features for a given reservoir dim
# ---------------------------------------------------------------------------

def build_features_for_dim(
    reservoir_dim: int, max_train: int | None, max_test: int | None
) -> dict:
    mt = MultiTaskAdapter(
        tfidf_dim=TFIDF_DIM,
        reservoir_dim=reservoir_dim,
        max_sentences=MAX_SENTENCES,
        seed=SEED,
    )
    mem = SystemMemory(reservoir_dim=reservoir_dim)
    result: dict = {}
    for t in TASKS:
        X_tr, Y_tr, lnames = mt.fit_transform(
            task_id=t["id"],
            dataset_name=t["dataset"],
            split=t["split_train"],
            subset=t["subset"],
            text_field=t["text_field"],
            label_field=t["label_field"],
            max_samples=max_train,
        )
        X_te, Y_te, _ = mt.transform(
            task_id=t["id"],
            dataset_name=t["dataset"],
            split=t["split_test"],
            subset=t["subset"],
            text_field=t["text_field"],
            label_field=t["label_field"],
            max_samples=max_test,
        )
        F_tr = mem.process_sequence(X_tr)
        F_te = mem.process_sequence(X_te)
        result[t["id"]] = {
            "F_tr": F_tr, "F_te": F_te,
            "Y_tr": Y_tr, "Y_te": Y_te,
            "n_classes": len(lnames),
            "label": t["label"],
        }
    return result


# ---------------------------------------------------------------------------
# Train one task head
# ---------------------------------------------------------------------------

def train_task(tid: str, task_data: dict, epochs: int, lr: float) -> dict:
    F_tr = task_data["F_tr"]
    Y_tr = task_data["Y_tr"]
    F_te = task_data["F_te"]
    Y_te = task_data["Y_te"]
    n    = len(F_tr)

    rng  = np.random.default_rng(SEED)
    head = ZeroForgetReadout(
        output_dim=task_data["n_classes"],
        feature_dim=F_tr.shape[1],
        rng=rng,
    )
    head.set_task(tid)
    rng_t = np.random.default_rng(0)
    t0 = time.perf_counter()

    for _ in range(epochs):
        idx = rng_t.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            batch = idx[start : start + BATCH_SIZE]
            head.update_batch(F_tr[batch], Y_tr[batch], lr=lr, task_id=tid)

    elapsed = time.perf_counter() - t0
    return {
        "task":       tid,
        "label":      task_data["label"],
        "epochs":     epochs,
        "lr":         lr,
        "feature_dim": F_tr.shape[1],
        "val_acc":    accuracy(head.predict_batch(F_te, tid), Y_te),
        "train_acc":  accuracy(head.predict_batch(F_tr, tid), Y_tr),
        "time_s":     elapsed,
    }


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def _print_table(rows: list[dict], dims: list[int]) -> None:
    print()
    print("=" * 80)
    print("  RESERVOIR DIM SWEEP")
    print("=" * 80)
    print(f"  {'task':<10s}", end="")
    for d in dims:
        print(f"  dim={d:>4d}", end="")
    print(f"  {'feat_dim(1024)':>14s}")
    print(f"  {'-'*76}")

    for t in TASKS:
        tid = t["id"]
        print(f"  {t['label']:<10s}", end="")
        for d in dims:
            row = next((r for r in rows if r["task"] == tid
                        and r["feature_dim"] == d * 6 + 3), None)
            if row:
                print(f"  {row['val_acc']:>8.1%}", end="")
            else:
                print(f"  {'N/A':>8s}", end="")
        # show feature dim for largest
        print(f"  {1024*6+3:>14d}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Reservoir dim sweep")
    p.add_argument("--dims", type=int, nargs="+", default=[512, 1024, 2048],
                   help="Reservoir dims to test (default: 512 1024 2048)")
    p.add_argument("--max-train", type=int, default=10000)
    p.add_argument("--max-test",  type=int, default=None)
    p.add_argument("--output", default="outputs/reservoir_sweep_results.json")
    args = p.parse_args()

    Path("outputs").mkdir(exist_ok=True)

    print()
    print("Reservoir Dim Sweep")
    print(f"  Dims      : {args.dims}")
    print(f"  max_train : {args.max_train}")
    print(f"  Config    : tfidf=3000  max_sent=3  ep=500  lr=0.3/0.2")

    all_rows: list[dict] = []
    all_results: dict = {}

    for res_dim in args.dims:
        feat_dim = res_dim * 6 + 3
        print(f"\n{'='*60}")
        print(f"  reservoir_dim={res_dim}  feature_dim={feat_dim}")
        print("=" * 60)

        t0 = time.perf_counter()
        feature_dict = build_features_for_dim(res_dim, args.max_train, args.max_test)
        print(f"  Features built in {time.perf_counter()-t0:.1f}s")

        dim_rows: list[dict] = []
        for t in TASKS:
            tid = t["id"]
            cfg = BEST_CONFIG[tid]
            print(f"  [{tid}]  ep={cfg['epochs']}  lr={cfg['lr']}  ... ",
                  end="", flush=True)
            row = train_task(tid, feature_dict[tid], cfg["epochs"], cfg["lr"])
            dim_rows.append(row)
            all_rows.append(row)
            print(f"val={row['val_acc']:.1%}  train={row['train_acc']:.1%}  {row['time_s']:.1f}s")

        all_results[str(res_dim)] = dim_rows

    # Summary table
    _print_table(all_rows, args.dims)

    print("\n  Accuracy delta vs dim=1024 (baseline):")
    baseline = {r["task"]: r for r in all_rows if r["feature_dim"] == 1024 * 6 + 3}
    for d in args.dims:
        if d == 1024:
            continue
        print(f"  dim={d}:")
        for t in TASKS:
            tid = t["id"]
            cmp = next((r for r in all_rows if r["task"] == tid
                        and r["feature_dim"] == d * 6 + 3), None)
            base = baseline.get(tid)
            if cmp and base:
                delta = cmp["val_acc"] - base["val_acc"]
                print(f"    {t['label']:<10s}  {base['val_acc']:.1%} -> {cmp['val_acc']:.1%}"
                      f"  ({delta:+.1%})")

    output_path = Path(args.output)
    with output_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
