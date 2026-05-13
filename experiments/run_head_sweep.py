"""CE head hyperparameter sweep: epochs x learning_rate.

Sweeps the ZeroForgetReadout (softmax+CE) head across all three classification
tasks with two data-scale phases.

Phase 1 -- Low data (2 000 samples)
    epochs : 200, 500, 1000
    lr     : 0.1, 0.2, 0.3

Phase 2 -- Medium data (10 000 samples)
    epochs : 200, 500
    lr     : best 2 from Phase 1

For every (task, epochs, lr) cell the script records:
    task_name, epochs, lr, train_acc, val_acc, CE_loss (final epoch),
    training_time_s, weight_norm, forgetting

The forgetting column is always 0.0000 for single-task runs (structural
guarantee).  A multi-task forgetting check is run once for the best config
found in Phase 1 to validate the guarantee on real distributions.

Usage
-----
    python -m experiments.run_head_sweep
    python -m experiments.run_head_sweep --skip-phase2
    python -m experiments.run_head_sweep --max-test 500
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

TASKS = [
    {
        "id":          "sst2",
        "dataset":     "sst2",
        "subset":      None,
        "split_train": "train",
        "split_test":  "validation",
        "text_field":  "sentence",
        "label_field": "label",
        "label":       "SST-2 (binary)",
    },
    {
        "id":          "ag_news",
        "dataset":     "ag_news",
        "subset":      None,
        "split_train": "train",
        "split_test":  "test",
        "text_field":  "text",
        "label_field": "label",
        "label":       "AG News (4-class)",
    },
    {
        "id":          "emotion",
        "dataset":     "dair-ai/emotion",
        "subset":      None,
        "split_train": "train",
        "split_test":  "test",
        "text_field":  "text",
        "label_field": "label",
        "label":       "Emotion (6-class)",
    },
]

TFIDF_DIM     = 3000
RESERVOIR_DIM = 1024
MAX_SENTENCES = 3
BATCH_SIZE    = 64
SEED          = 42

# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------

def build_all_features(
    max_train: int,
    max_test: int | None,
) -> dict[str, dict]:
    """Returns {task_id: {train, test, Y_tr, Y_te, n_classes}} for all tasks."""
    print(f"\n  Building shared encoder + TF-IDF (max_train={max_train}) ...")
    mt = MultiTaskAdapter(
        tfidf_dim=TFIDF_DIM,
        reservoir_dim=RESERVOIR_DIM,
        max_sentences=MAX_SENTENCES,
        seed=SEED,
    )
    mem = SystemMemory(reservoir_dim=RESERVOIR_DIM)

    result: dict[str, dict] = {}
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
        print(f"    {t['id']:10s}  train={F_tr.shape}  test={F_te.shape}  "
              f"classes={n_classes}  ({elapsed:.1f}s)")
        result[t["id"]] = {
            "F_tr":      F_tr,
            "F_te":      F_te,
            "Y_tr":      Y_tr,
            "Y_te":      Y_te,
            "n_classes": n_classes,
            "label":     t["label"],
        }
    return result


# ---------------------------------------------------------------------------
# Single head trainer + metric collector
# ---------------------------------------------------------------------------

def _ce_loss(F: np.ndarray, Y: np.ndarray, W: np.ndarray) -> float:
    """Mean cross-entropy on a feature/label pair given weight matrix W."""
    logits = F @ W.T
    logits -= logits.max(axis=1, keepdims=True)
    log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    return float(-np.mean(np.sum(Y * log_probs, axis=1)))


def train_and_measure(
    task_id:   str,
    task_data: dict,
    epochs:    int,
    lr:        float,
) -> dict:
    """Train a fresh ZeroForgetReadout head and return a metrics row."""
    n_classes  = task_data["n_classes"]
    F_tr       = task_data["F_tr"]
    Y_tr       = task_data["Y_tr"]
    F_te       = task_data["F_te"]
    Y_te       = task_data["Y_te"]
    feature_dim = F_tr.shape[1]
    n           = len(F_tr)

    rng  = np.random.default_rng(SEED)
    head = ZeroForgetReadout(output_dim=n_classes, feature_dim=feature_dim, rng=rng)
    head.set_task(task_id)

    rng_t      = np.random.default_rng(0)
    last_loss  = 0.0
    t_start    = time.perf_counter()

    for epoch in range(epochs):
        idx   = rng_t.permutation(n)
        ep_loss = 0.0
        steps = 0
        for start in range(0, n, BATCH_SIZE):
            batch = idx[start:start + BATCH_SIZE]
            ep_loss += head.update_batch(F_tr[batch], Y_tr[batch], lr=lr,
                                         task_id=task_id)
            steps += 1
        last_loss = ep_loss / max(1, steps)

    elapsed = time.perf_counter() - t_start

    # Metrics
    train_preds = head.predict_batch(F_tr, task_id)
    val_preds   = head.predict_batch(F_te, task_id)
    train_acc   = accuracy(train_preds, Y_tr)
    val_acc     = accuracy(val_preds,   Y_te)

    W           = head._heads[task_id]
    weight_norm = float(np.linalg.norm(W))
    # Final CE on validation set
    val_loss    = _ce_loss(F_te, Y_te, W)

    return {
        "task":        task_id,
        "task_label":  task_data["label"],
        "epochs":      epochs,
        "lr":          lr,
        "train_acc":   train_acc,
        "val_acc":     val_acc,
        "CE_loss":     val_loss,
        "train_loss":  last_loss,
        "time_s":      elapsed,
        "weight_norm": weight_norm,
        "forgetting":  0.0,   # structural guarantee; checked separately
    }


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------

def run_phase(
    phase_name: str,
    feature_dict: dict[str, dict],
    epochs_list: list[int],
    lr_list: list[float],
    max_train: int,
) -> list[dict]:
    rows: list[dict] = []
    total_combos = len(TASKS) * len(epochs_list) * len(lr_list)
    done = 0
    t_phase = time.perf_counter()

    for t in TASKS:
        tid = t["id"]
        for epochs in epochs_list:
            for lr in lr_list:
                done += 1
                print(f"  [{done:>2}/{total_combos}]  {tid:10s}  "
                      f"ep={epochs:>4d}  lr={lr:.1f}  ... ", end="", flush=True)
                row = train_and_measure(tid, feature_dict[tid], epochs, lr)
                rows.append(row)
                print(f"val={row['val_acc']:.1%}  "
                      f"train={row['train_acc']:.1%}  "
                      f"loss={row['CE_loss']:.4f}  "
                      f"|W|={row['weight_norm']:.2f}  "
                      f"{row['time_s']:.1f}s")

    print(f"\n  Phase '{phase_name}' complete in "
          f"{time.perf_counter()-t_phase:.1f}s")
    return rows


# ---------------------------------------------------------------------------
# Forgetting validation for best config
# ---------------------------------------------------------------------------

def run_forgetting_check(
    feature_dict: dict[str, dict],
    best_epochs: int,
    best_lr: float,
) -> float:
    """Train tasks sequentially, confirm A_before == A_after_B == A_after_C.

    Returns forgetting (should be exactly 0.0).
    """
    feature_dim = feature_dict["sst2"]["F_tr"].shape[1]
    rng = np.random.default_rng(SEED)
    readouts: dict[str, ZeroForgetReadout] = {}
    for t in TASKS:
        tid = t["id"]
        n_classes = feature_dict[tid]["n_classes"]
        readouts[tid] = ZeroForgetReadout(
            output_dim=n_classes, feature_dim=feature_dim, rng=rng
        )

    def _train(tid: str) -> None:
        rd  = readouts[tid]
        fd  = feature_dict[tid]
        rd.set_task(tid)
        n   = len(fd["F_tr"])
        rng_t = np.random.default_rng(0)
        for epoch in range(best_epochs):
            idx = rng_t.permutation(n)
            for start in range(0, n, BATCH_SIZE):
                batch = idx[start:start + BATCH_SIZE]
                rd.update_batch(fd["F_tr"][batch], fd["Y_tr"][batch],
                                lr=best_lr, task_id=tid)

    def _eval(tid: str) -> float:
        fd = feature_dict[tid]
        return accuracy(readouts[tid].predict_batch(fd["F_te"], tid), fd["Y_te"])

    print(f"\n  Forgetting check  ep={best_epochs}  lr={best_lr} ...")

    _train("sst2")
    a_before       = _eval("sst2")
    head_a_snap    = readouts["sst2"]._heads["sst2"].copy()
    print(f"    A_before      = {a_before:.4f}  ({a_before:.1%})")

    _train("ag_news")
    assert np.array_equal(head_a_snap, readouts["sst2"]._heads["sst2"]), \
        "FATAL: head A was modified while training B"
    a_after_b = _eval("sst2")
    b_acc     = _eval("ag_news")
    print(f"    A_after_B     = {a_after_b:.4f}  ({a_after_b:.1%})  "
          f"delta={a_before-a_after_b:+.4f}")
    print(f"    B_acc         = {b_acc:.4f}  ({b_acc:.1%})")

    _train("emotion")
    assert np.array_equal(head_a_snap, readouts["sst2"]._heads["sst2"]), \
        "FATAL: head A was modified while training C"
    a_after_c = _eval("sst2")
    c_acc     = _eval("emotion")
    forgetting = a_before - a_after_c
    print(f"    A_after_C     = {a_after_c:.4f}  ({a_after_c:.1%})  "
          f"delta={forgetting:+.4f}")
    print(f"    C_acc         = {c_acc:.4f}  ({c_acc:.1%})")
    verdict = "[PASS]" if abs(forgetting) < 1e-6 else "[FAIL]"
    print(f"    Forgetting    = {forgetting*100:+.6f}%  {verdict}")
    return forgetting


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def _print_table(rows: list[dict], phase_label: str) -> None:
    print(f"\n{'='*90}")
    print(f"  {phase_label}")
    print(f"{'='*90}")
    header = (f"  {'task':<12s}  {'ep':>5s}  {'lr':>4s}  "
              f"{'val_acc':>8s}  {'train_acc':>9s}  {'CE_loss':>8s}  "
              f"{'|W|':>6s}  {'time':>6s}  {'forget':>8s}")
    print(header)
    print(f"  {'-'*86}")
    prev_task = None
    for r in rows:
        if r["task"] != prev_task and prev_task is not None:
            print()
        prev_task = r["task"]
        print(f"  {r['task_label']:<12s}  {r['epochs']:>5d}  {r['lr']:>4.1f}  "
              f"  {r['val_acc']:>7.1%}  {r['train_acc']:>8.1%}  {r['CE_loss']:>8.4f}  "
              f"{r['weight_norm']:>6.2f}  {r['time_s']:>5.1f}s  {r['forgetting']:>+8.4f}")
    print(f"{'='*90}")


def _summarize(rows: list[dict], phase_label: str) -> dict[str, dict]:
    """Per-task best config summary."""
    print(f"\n  Summary for {phase_label}:")
    best: dict[str, dict] = {}
    for t in TASKS:
        tid    = t["id"]
        t_rows = [r for r in rows if r["task"] == tid]
        if not t_rows:
            continue
        best_r = max(t_rows, key=lambda x: x["val_acc"])
        # baseline: ep=200, lr=0.1
        baseline = next(
            (r for r in t_rows if r["epochs"] == 200 and r["lr"] == 0.1),
            None,
        )
        gain = (best_r["val_acc"] - baseline["val_acc"]) if baseline else float("nan")
        print(f"    {t['label']:<20s}  best=ep={best_r['epochs']}/lr={best_r['lr']:.1f}"
              f"  val={best_r['val_acc']:.1%}"
              f"  gain_vs_baseline={gain:+.1%}"
              f"  time={best_r['time_s']:.1f}s")
        best[tid] = best_r
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="CE head sweep: epochs x lr",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--max-train-p1", type=int, default=2000,
                   help="Training samples for Phase 1 (default: 2000)")
    p.add_argument("--max-train-p2", type=int, default=10000,
                   help="Training samples for Phase 2 (default: 10000)")
    p.add_argument("--max-test",     type=int, default=None,
                   help="Test samples cap (default: all)")
    p.add_argument("--skip-phase2",  action="store_true",
                   help="Run Phase 1 only")
    p.add_argument("--skip-forgetting-check", action="store_true",
                   help="Skip the multi-task forgetting validation")
    p.add_argument("--output", type=str,
                   default="outputs/head_sweep_results.json",
                   help="Output JSON path")
    args = p.parse_args()

    Path("outputs").mkdir(exist_ok=True)
    all_results: dict = {}

    # ======================================================================
    # PHASE 1 — 2k samples
    # ======================================================================
    print("\n" + "#" * 70)
    print(f"  PHASE 1 — Low Data (max_train={args.max_train_p1})")
    print(f"  epochs x lr: {{200,500,1000}} x {{0.1,0.2,0.3}}")
    print("#" * 70)

    feats_p1 = build_all_features(args.max_train_p1, args.max_test)

    rows_p1 = run_phase(
        phase_name   = "Phase 1 (2k)",
        feature_dict = feats_p1,
        epochs_list  = [200, 500, 1000],
        lr_list      = [0.1, 0.2, 0.3],
        max_train    = args.max_train_p1,
    )
    _print_table(rows_p1, f"PHASE 1 RESULTS (train={args.max_train_p1})")
    best_p1 = _summarize(rows_p1, "Phase 1")
    all_results["phase1"] = rows_p1

    # Pick best 2 lr values from Phase 1 (avg across tasks)
    lr_scores: dict[float, list[float]] = {}
    for r in rows_p1:
        lr_scores.setdefault(r["lr"], []).append(r["val_acc"])
    lr_ranked = sorted(lr_scores.items(), key=lambda kv: np.mean(kv[1]), reverse=True)
    best_2_lr = [lr for lr, _ in lr_ranked[:2]]
    print(f"\n  Best 2 lr values for Phase 2: {best_2_lr}")

    # Forgetting check with overall best config from Phase 1
    if not args.skip_forgetting_check:
        all_sst2 = [r for r in rows_p1 if r["task"] == "sst2"]
        top      = max(all_sst2, key=lambda x: x["val_acc"])
        print(f"\n  Running forgetting check for best Phase 1 config: "
              f"ep={top['epochs']}  lr={top['lr']}")
        f_p1 = run_forgetting_check(feats_p1, top["epochs"], top["lr"])
        all_results["forgetting_check_p1"] = {
            "epochs": top["epochs"], "lr": top["lr"], "forgetting": f_p1
        }

    if args.skip_phase2:
        _save_and_exit(args.output, all_results)
        return

    # ======================================================================
    # PHASE 2 — 10k samples
    # ======================================================================
    print("\n" + "#" * 70)
    print(f"  PHASE 2 — Medium Data (max_train={args.max_train_p2})")
    print(f"  epochs x lr: {{200,500}} x {best_2_lr}")
    print("#" * 70)

    feats_p2 = build_all_features(args.max_train_p2, args.max_test)

    rows_p2 = run_phase(
        phase_name   = "Phase 2 (10k)",
        feature_dict = feats_p2,
        epochs_list  = [200, 500],
        lr_list      = best_2_lr,
        max_train    = args.max_train_p2,
    )
    _print_table(rows_p2, f"PHASE 2 RESULTS (train={args.max_train_p2})")
    best_p2 = _summarize(rows_p2, "Phase 2")
    all_results["phase2"] = rows_p2

    # Forgetting check with best Phase 2 config
    if not args.skip_forgetting_check:
        all_sst2_p2 = [r for r in rows_p2 if r["task"] == "sst2"]
        top2        = max(all_sst2_p2, key=lambda x: x["val_acc"])
        print(f"\n  Running forgetting check for best Phase 2 config: "
              f"ep={top2['epochs']}  lr={top2['lr']}")
        f_p2 = run_forgetting_check(feats_p2, top2["epochs"], top2["lr"])
        all_results["forgetting_check_p2"] = {
            "epochs": top2["epochs"], "lr": top2["lr"], "forgetting": f_p2
        }

    # ======================================================================
    # Final recommendation
    # ======================================================================
    _print_final_recommendation(rows_p1, rows_p2, best_2_lr)

    _save_and_exit(args.output, all_results)


def _save_and_exit(path: str, results: dict) -> None:
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved -> {path}")


def _print_final_recommendation(
    rows_p1: list[dict],
    rows_p2: list[dict],
    best_2_lr: list[float],
) -> None:
    print("\n" + "=" * 70)
    print("  FINAL RECOMMENDATION")
    print("=" * 70)

    for t in TASKS:
        tid   = t["id"]
        p1    = [r for r in rows_p1 if r["task"] == tid]
        p2    = [r for r in rows_p2 if r["task"] == tid]
        if not p1:
            continue
        base  = next((r for r in p1 if r["epochs"] == 200 and r["lr"] == 0.1), None)
        best1 = max(p1, key=lambda x: x["val_acc"])
        best2 = max(p2, key=lambda x: x["val_acc"]) if p2 else None

        print(f"\n  {t['label']}")
        print(f"    Baseline  ep=200  lr=0.1  ->  val={base['val_acc']:.1%}" if base else "")
        print(f"    Best 2k   ep={best1['epochs']}  lr={best1['lr']:.1f}  ->  val={best1['val_acc']:.1%}"
              f"  ({best1['val_acc']-base['val_acc']:+.1%} vs baseline)" if base else "")
        if best2:
            print(f"    Best 10k  ep={best2['epochs']}  lr={best2['lr']:.1f}  ->  val={best2['val_acc']:.1%}"
                  f"  ({best2['val_acc']-base['val_acc']:+.1%} vs baseline)")

        # Overfitting check: is train_acc >> val_acc for best config?
        gap = best1["train_acc"] - best1["val_acc"]
        if gap > 0.10:
            print(f"    [WARN] train-val gap {gap:.1%} at best 2k config -- overfitting likely")
        elif best1["val_acc"] <= (base["val_acc"] if base else 0) + 0.005:
            print(f"    [INFO] Accuracy saturated -- encoder/data is bottleneck, not epochs")
        else:
            print(f"    [OK]   Increasing epochs helped")

        # Data vs epochs verdict
        if best2 and p2:
            data_gain  = best2["val_acc"] - best1["val_acc"]
            epoch_gain = best1["val_acc"] - (base["val_acc"] if base else best1["val_acc"])
            if data_gain > epoch_gain:
                print(f"    [REC]  Prioritize data scaling (data gain {data_gain:+.1%} > epoch gain {epoch_gain:+.1%})")
            else:
                print(f"    [REC]  Prioritize epoch tuning (epoch gain {epoch_gain:+.1%} >= data gain {data_gain:+.1%})")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
