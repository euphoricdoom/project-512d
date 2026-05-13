"""Full-scale experiment: scale all 3 tasks to maximum available data.

Locked config (from Phase 2 sweep):
  tfidf_dim     = 3000
  reservoir_dim = 1024
  max_sentences = 3
  SST-2/Emotion : ep=500, lr=0.3
  AG News       : ep=500, lr=0.2

Runs at multiple data scales (10k, 30k, full) and prints a cumulative
accuracy table.  Forgetting check is run at the final scale.

Usage
-----
    python -m experiments.run_full_scale
    python -m experiments.run_full_scale --scales 10000 30000 0
    python -m experiments.run_full_scale --max-test 872
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
RESERVOIR_DIM = 1024
MAX_SENTENCES = 3
BATCH_SIZE    = 64
SEED          = 42

# Per-task best configs from Phase 2 sweep
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
# Feature builder
# ---------------------------------------------------------------------------

def build_all_features(max_train: int | None, max_test: int | None) -> dict:
    label = "full" if max_train is None else str(max_train)
    print(f"\n  Building features  max_train={label} ...")
    mt = MultiTaskAdapter(
        tfidf_dim=TFIDF_DIM,
        reservoir_dim=RESERVOIR_DIM,
        max_sentences=MAX_SENTENCES,
        seed=SEED,
    )
    mem = SystemMemory(reservoir_dim=RESERVOIR_DIM)
    result: dict = {}
    for t in TASKS:
        t0 = time.perf_counter()
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
        elapsed = time.perf_counter() - t0
        n_classes = len(lnames)
        print(f"    {t['id']:10s}  train={F_tr.shape[0]}  test={F_te.shape[0]}"
              f"  classes={n_classes}  ({elapsed:.1f}s)")
        result[t["id"]] = {
            "F_tr": F_tr, "F_te": F_te,
            "Y_tr": Y_tr, "Y_te": Y_te,
            "n_classes": n_classes,
            "label": t["label"],
        }
    return result


# ---------------------------------------------------------------------------
# Train + evaluate one task
# ---------------------------------------------------------------------------

def train_task(tid: str, task_data: dict, epochs: int, lr: float) -> dict:
    F_tr      = task_data["F_tr"]
    Y_tr      = task_data["Y_tr"]
    F_te      = task_data["F_te"]
    Y_te      = task_data["Y_te"]
    n_classes = task_data["n_classes"]
    n         = len(F_tr)

    rng  = np.random.default_rng(SEED)
    head = ZeroForgetReadout(
        output_dim=n_classes, feature_dim=F_tr.shape[1], rng=rng
    )
    head.set_task(tid)
    rng_t = np.random.default_rng(0)
    t0 = time.perf_counter()

    for _ in range(epochs):
        idx = rng_t.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            batch = idx[start : start + BATCH_SIZE]
            head.update_batch(F_tr[batch], Y_tr[batch], lr=lr, task_id=tid)

    elapsed    = time.perf_counter() - t0
    train_acc  = accuracy(head.predict_batch(F_tr, tid), Y_tr)
    val_acc    = accuracy(head.predict_batch(F_te, tid), Y_te)
    weight_norm = float(np.linalg.norm(head._heads[tid]))
    return {
        "task":        tid,
        "label":       task_data["label"],
        "epochs":      epochs,
        "lr":          lr,
        "train_samples": n,
        "train_acc":   train_acc,
        "val_acc":     val_acc,
        "time_s":      elapsed,
        "weight_norm": weight_norm,
    }


# ---------------------------------------------------------------------------
# Multi-task forgetting check
# ---------------------------------------------------------------------------

def run_forgetting_check(feature_dict: dict) -> float:
    """Train A -> B -> C, verify A is unaffected.  Returns forgetting."""
    feature_dim = feature_dict["sst2"]["F_tr"].shape[1]
    rng = np.random.default_rng(SEED)
    heads: dict[str, ZeroForgetReadout] = {}
    for t in TASKS:
        tid = t["id"]
        heads[tid] = ZeroForgetReadout(
            output_dim=feature_dict[tid]["n_classes"],
            feature_dim=feature_dim,
            rng=rng,
        )

    def _train(tid: str) -> None:
        cfg = BEST_CONFIG[tid]
        h   = heads[tid]
        fd  = feature_dict[tid]
        h.set_task(tid)
        n   = len(fd["F_tr"])
        rng_t = np.random.default_rng(0)
        for _ in range(cfg["epochs"]):
            idx = rng_t.permutation(n)
            for start in range(0, n, BATCH_SIZE):
                batch = idx[start : start + BATCH_SIZE]
                h.update_batch(fd["F_tr"][batch], fd["Y_tr"][batch],
                               lr=cfg["lr"], task_id=tid)

    def _eval(tid: str) -> float:
        fd = feature_dict[tid]
        return accuracy(heads[tid].predict_batch(fd["F_te"], tid), fd["Y_te"])

    print("\n  Forgetting check  (A->B->C) ...")

    _train("sst2")
    a_before = _eval("sst2")
    snap = heads["sst2"]._heads["sst2"].copy()
    print(f"    A_before  = {a_before:.1%}")

    _train("ag_news")
    assert np.array_equal(snap, heads["sst2"]._heads["sst2"]), \
        "FATAL: head A modified while training B"
    a_after_b = _eval("sst2")
    print(f"    A_after_B = {a_after_b:.1%}  delta={a_before-a_after_b:+.4f}")
    print(f"    B_acc     = {_eval('ag_news'):.1%}")

    _train("emotion")
    assert np.array_equal(snap, heads["sst2"]._heads["sst2"]), \
        "FATAL: head A modified while training C"
    a_after_c = _eval("sst2")
    forgetting = a_before - a_after_c
    print(f"    A_after_C = {a_after_c:.1%}  delta={forgetting:+.4f}")
    print(f"    C_acc     = {_eval('emotion'):.1%}")
    verdict = "[PASS]" if abs(forgetting) < 1e-6 else "[FAIL]"
    print(f"    Forgetting = {forgetting*100:+.6f}%  {verdict}")
    return forgetting


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def _print_table(rows_by_scale: list[tuple[str, list[dict]]]) -> None:
    print()
    print("=" * 80)
    print("  FULL-SCALE RESULTS")
    print("=" * 80)
    print(f"  {'task':<10s}  {'n_train':>8s}  {'ep':>5s}  {'lr':>4s}  "
          f"{'val_acc':>8s}  {'train_acc':>9s}  {'|W|':>6s}  {'time':>6s}")
    print(f"  {'-'*76}")
    prev_task = None
    for scale_label, rows in rows_by_scale:
        for r in rows:
            if r["task"] != prev_task and prev_task is not None:
                print()
            prev_task = r["task"]
            print(f"  {r['label']:<10s}  {r['train_samples']:>8d}  "
                  f"{r['epochs']:>5d}  {r['lr']:>4.1f}  "
                  f"  {r['val_acc']:>7.1%}  {r['train_acc']:>8.1%}  "
                  f"{r['weight_norm']:>6.2f}  {r['time_s']:>5.1f}s")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Full-data scaling experiment")
    p.add_argument(
        "--scales", type=int, nargs="+", default=[10000, 30000, 0],
        help="Training sample counts to test (0 = full dataset). Default: 10000 30000 0",
    )
    p.add_argument("--max-test", type=int, default=None)
    p.add_argument(
        "--skip-forgetting-check", action="store_true",
        help="Skip the sequential forgetting check at the final scale",
    )
    p.add_argument(
        "--output", type=str, default="outputs/full_scale_results.json",
    )
    args = p.parse_args()

    Path("outputs").mkdir(exist_ok=True)

    # Replace 0 with None (full dataset)
    scales = [s if s != 0 else None for s in args.scales]

    print()
    print("Full-Scale Experiment")
    print(f"  Scales    : {[str(s) if s else 'full' for s in scales]}")
    print(f"  Config    : tfidf=3000  res=1024  max_sent=3")
    print(f"  Best cfg  : SST-2/Emotion ep=500,lr=0.3  |  AG News ep=500,lr=0.2")

    all_rows: list[tuple[str, list[dict]]] = []
    all_results: dict = {}

    for scale in scales:
        scale_label = "full" if scale is None else str(scale)
        print(f"\n{'='*60}")
        print(f"  Scale: {scale_label} samples")
        print("=" * 60)

        t_build = time.perf_counter()
        feature_dict = build_all_features(scale, args.max_test)
        print(f"  Features built in {time.perf_counter()-t_build:.1f}s")

        rows: list[dict] = []
        for t in TASKS:
            tid = t["id"]
            cfg = BEST_CONFIG[tid]
            print(f"\n  [{tid}]  ep={cfg['epochs']}  lr={cfg['lr']}  ... ",
                  end="", flush=True)
            row = train_task(tid, feature_dict[tid], cfg["epochs"], cfg["lr"])
            rows.append(row)
            print(f"val={row['val_acc']:.1%}  train={row['train_acc']:.1%}  "
                  f"|W|={row['weight_norm']:.2f}  {row['time_s']:.1f}s")

        all_rows.append((scale_label, rows))
        all_results[scale_label] = rows

        # Forgetting check only at the last scale
        if scale == scales[-1] and not args.skip_forgetting_check:
            forgetting = run_forgetting_check(feature_dict)
            all_results["forgetting_final"] = float(forgetting)

    _print_table(all_rows)

    # Summary: accuracy deltas across scales
    print("\n  Accuracy delta  (final scale vs 10k):")
    rows_10k = {r["task"]: r for _, rows in all_rows
                for r in rows if r["train_samples"] <= 10000}
    rows_final = {r["task"]: r for r in all_rows[-1][1]}
    for t in TASKS:
        tid = t["id"]
        if tid in rows_10k and tid in rows_final:
            delta = rows_final[tid]["val_acc"] - rows_10k[tid]["val_acc"]
            print(f"    {t['label']:<10s}  10k={rows_10k[tid]['val_acc']:.1%}"
                  f"  final={rows_final[tid]['val_acc']:.1%}"
                  f"  delta={delta:+.1%}")

    # Save JSON
    output_path = Path(args.output)
    with output_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
