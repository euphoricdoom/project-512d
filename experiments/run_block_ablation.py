"""Feature block ablation across all three tasks.

Systematically removes each reservoir feature block to measure its
contribution to classification accuracy:

  Blocks:
    last   - final reservoir state  (dim: res_dim)
    mean   - mean across time       (dim: res_dim)
    max    - max across time        (dim: res_dim)
    ac     - AC stream (fast decay) (dim: res_dim)
    dc     - DC stream (slow decay) (dim: res_dim)
    ac*dc  - cross-freq product     (dim: res_dim)
    phase  - temporal phase code    (dim: 3)

  Conditions:
    all      - all blocks (baseline)
    no_X     - drop block X, keep others
    only_X   - only block X

Locked config: tfidf=3000, res=1024, max_sent=3, ep=200, lr=0.3/0.2
(ep=200 for speed; use --epochs 500 for final-quality runs)

Usage
-----
    python -m experiments.run_block_ablation
    python -m experiments.run_block_ablation --max-train 5000 --epochs 200
    python -m experiments.run_block_ablation --epochs 500 --max-train 10000
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

BEST_LR = {
    "sst2":    0.3,
    "ag_news": 0.2,
    "emotion": 0.3,
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

# Feature block slice definitions (for res_dim=1024)
def _block_slices(res_dim: int) -> dict[str, tuple[int, int]]:
    d = res_dim
    return {
        "last":  (0,     d),
        "mean":  (d,     2 * d),
        "max":   (2 * d, 3 * d),
        "ac":    (3 * d, 4 * d),
        "dc":    (4 * d, 5 * d),
        "ac*dc": (5 * d, 6 * d),
        "phase": (6 * d, 6 * d + 3),
    }


def _select(F: np.ndarray, keep: list[tuple[int, int]]) -> np.ndarray:
    return np.concatenate([F[:, s:e] for s, e in keep], axis=1)


def build_ablation_configs(res_dim: int) -> dict[str, list[tuple[int, int]]]:
    """Returns {config_name: list_of_slices}."""
    slices = _block_slices(res_dim)
    all_slices = list(slices.values())
    configs: dict[str, list[tuple[int, int]]] = {}

    # Baseline: all blocks
    configs["all"] = all_slices

    # Leave-one-out (no_X)
    for name, sl in slices.items():
        configs[f"no_{name}"] = [s for s in all_slices if s != sl]

    # Single-block (only_X)
    for name, sl in slices.items():
        configs[f"only_{name}"] = [sl]

    return configs


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------

def build_all_features(max_train: int | None, max_test: int | None) -> dict:
    print(f"  Building features  max_train={max_train} ...")
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
        print(f"    {t['id']:10s}  train={F_tr.shape[0]}  test={F_te.shape[0]}"
              f"  ({elapsed:.1f}s)")
        result[t["id"]] = {
            "F_tr": F_tr, "F_te": F_te,
            "Y_tr": Y_tr, "Y_te": Y_te,
            "n_classes": len(lnames),
            "label": t["label"],
        }
    return result


# ---------------------------------------------------------------------------
# Train one ablation config
# ---------------------------------------------------------------------------

def train_ablation(
    tid: str, task_data: dict,
    feature_slices: list[tuple[int, int]],
    epochs: int, lr: float,
) -> dict:
    F_tr_raw = task_data["F_tr"]
    F_te_raw = task_data["F_te"]
    Y_tr     = task_data["Y_tr"]
    Y_te     = task_data["Y_te"]

    F_tr = _select(F_tr_raw, feature_slices)
    F_te = _select(F_te_raw, feature_slices)
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

    elapsed  = time.perf_counter() - t0
    val_acc  = accuracy(head.predict_batch(F_te, tid), Y_te)
    return {
        "val_acc":   val_acc,
        "feat_dim":  F_tr.shape[1],
        "time_s":    elapsed,
    }


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def _print_table(
    results: dict[str, dict[str, dict]],
    configs: list[str],
    epochs: int,
) -> None:
    """results[config_name][task_id] = {val_acc, feat_dim, time_s}"""
    print()
    print("=" * 80)
    print(f"  FEATURE BLOCK ABLATION  (ep={epochs})")
    print("=" * 80)
    col_w = 10
    header = f"  {'config':<14s}"
    for t in TASKS:
        header += f"  {t['label']:>{col_w}s}"
    header += f"  {'feat_dim':>8s}  note"
    print(header)
    print(f"  {'-'*76}")

    baseline_accs = {t["id"]: results.get("all", {}).get(t["id"], {}).get("val_acc", 0.0)
                     for t in TASKS}

    for cfg in configs:
        if cfg not in results:
            continue
        row = results[cfg]
        line = f"  {cfg:<14s}"
        deltas = []
        feat_dim = None
        for t in TASKS:
            tid = t["id"]
            if tid in row:
                acc = row[tid]["val_acc"]
                feat_dim = row[tid]["feat_dim"]
                delta = acc - baseline_accs[tid]
                deltas.append(delta)
                marker = ""
                if cfg != "all" and abs(delta) >= 0.005:
                    marker = "v" if delta < 0 else "^"
                line += f"  {acc:>{col_w}.1%}"
            else:
                line += f"  {'N/A':>{col_w}s}"
        if feat_dim is not None:
            line += f"  {feat_dim:>8d}"
        if cfg != "all" and deltas:
            avg_delta = sum(deltas) / len(deltas)
            line += f"  avg={avg_delta:+.1%}"
        print(line)

        # separator between sections
        if cfg == "all" or cfg.startswith("no_") and not any(
            c.startswith("no_") for c in configs[:configs.index(cfg)]
        ):
            pass

    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Feature block ablation")
    p.add_argument("--max-train", type=int, default=10000)
    p.add_argument("--max-test",  type=int, default=None)
    p.add_argument("--epochs",    type=int, default=200,
                   help="Training epochs (default 200; use 500 for higher quality)")
    p.add_argument("--output", default="outputs/block_ablation_results.json")
    p.add_argument(
        "--only-baseline", action="store_true",
        help="Only run the 'all' baseline + leave-one-out (skip single-block)",
    )
    args = p.parse_args()

    Path("outputs").mkdir(exist_ok=True)

    configs = build_ablation_configs(RESERVOIR_DIM)
    if args.only_baseline:
        # Keep 'all' + 'no_X' only
        configs = {k: v for k, v in configs.items()
                   if k == "all" or k.startswith("no_")}

    cfg_names = list(configs.keys())
    n_total = len(cfg_names) * len(TASKS)

    print()
    print("Feature Block Ablation")
    print(f"  max_train : {args.max_train}")
    print(f"  epochs    : {args.epochs}")
    print(f"  configs   : {len(cfg_names)}")
    print(f"  total runs: {n_total}")

    feature_dict = build_all_features(args.max_train, args.max_test)

    results: dict[str, dict[str, dict]] = {}
    done = 0

    for cfg_name, slices in configs.items():
        results[cfg_name] = {}
        for t in TASKS:
            tid = t["id"]
            done += 1
            lr  = BEST_LR[tid]
            print(f"  [{done:>2}/{n_total}]  {cfg_name:<14s}  {tid:<10s}  "
                  f"ep={args.epochs}  lr={lr}  ... ", end="", flush=True)
            row = train_ablation(tid, feature_dict[tid], slices, args.epochs, lr)
            results[cfg_name][tid] = row
            print(f"val={row['val_acc']:.1%}  dim={row['feat_dim']}  {row['time_s']:.1f}s")

    _print_table(results, cfg_names, args.epochs)

    # Highlight most important blocks
    print("\n  Block importance ranking (by avg accuracy drop when removed):")
    block_names = ["last", "mean", "max", "ac", "dc", "ac*dc", "phase"]
    drops: list[tuple[str, float]] = []
    baseline_accs = {t["id"]: results["all"][t["id"]]["val_acc"] for t in TASKS}
    for block in block_names:
        cfg = f"no_{block}"
        if cfg not in results:
            continue
        task_drops = []
        for t in TASKS:
            tid = t["id"]
            if tid in results[cfg]:
                drop = baseline_accs[tid] - results[cfg][tid]["val_acc"]
                task_drops.append(drop)
        if task_drops:
            avg_drop = sum(task_drops) / len(task_drops)
            drops.append((block, avg_drop))
    drops.sort(key=lambda x: x[1], reverse=True)
    for rank, (block, drop) in enumerate(drops, 1):
        bar = "#" * int(max(0, drop) * 200)
        print(f"    {rank}. {block:<8s}  avg_drop={drop:+.2%}  {bar}")

    output_path = Path(args.output)
    with output_path.open("w") as f:
        json.dump({k: {tid: {kk: float(vv) if isinstance(vv, (int, float, np.floating)) else vv
                              for kk, vv in v.items()}
                        for tid, v in v2.items()}
                   for k, v2 in results.items()},
                  f, indent=2)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
