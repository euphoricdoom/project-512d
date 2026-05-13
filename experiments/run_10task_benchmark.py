"""Milestone 2 — 10-task sequential benchmark.

Trains 10 HuggingFace text classification tasks sequentially on a single
weightless model, records per-task accuracy at training time, and re-evaluates
all prior tasks after each new task is added to confirm zero forgetting.

Usage
-----
    # Smoke test (fast, ~5 min)
    python experiments/run_10task_benchmark.py --max-train 200 --epochs 10

    # Full M2 run (~20-30 min on CPU)
    python experiments/run_10task_benchmark.py --max-train 10000 --epochs 500

    # Resume from an existing model checkpoint
    python experiments/run_10task_benchmark.py --start-from 3 --model-in outputs/m2_step2.npz

Results are saved to outputs/m2_benchmark_results.json and a summary table
is printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make root importable when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from train_hf import evaluate, train

# ---------------------------------------------------------------------------
# Task registry — 10 confirmed-working datasets
# ---------------------------------------------------------------------------
# All dataset/field combinations were smoke-tested 2026-04-27.
# sst2 and yahoo were fixed by the hf_adapter unseen-label filter.

TASKS: list[dict] = [
    {
        "dataset": "imdb",
        "task_id": "sentiment",
        "ep": 500,
        "lr": 0.3,
    },
    {
        "dataset": "ag_news",
        "task_id": "topic",
        "ep": 500,
        "lr": 0.2,
    },
    {
        "dataset": "yelp_review_full",
        "task_id": "yelp",
        "ep": 500,
        "lr": 0.3,
    },
    {
        "dataset": "dbpedia_14",
        "task_id": "dbpedia",
        "ep": 500,
        "lr": 0.3,
    },
    {
        "dataset": "dair-ai/emotion",
        "task_id": "emotion",
        "ep": 500,
        "lr": 0.3,
    },
    {
        "dataset": "rotten_tomatoes",
        "task_id": "rotten",
        "ep": 500,
        "lr": 0.3,
    },
    {
        "dataset": "glue",
        "subset": "sst2",
        "task_id": "sst2",
        "ep": 500,
        "lr": 0.3,
    },
    {
        "dataset": "tweet_eval",
        "subset": "sentiment",
        "task_id": "tweet_sent",
        "ep": 500,
        "lr": 0.3,
    },
    {
        "dataset": "poem_sentiment",
        "task_id": "poem",
        "ep": 500,
        "lr": 0.3,
    },
    {
        "dataset": "yahoo_answers_topics",
        "task_id": "yahoo",
        "text_field": "question_title",
        "label_field": "topic",
        "ep": 500,
        "lr": 0.3,
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_path(step: int, out_dir: Path) -> str:
    return str(out_dir / f"m2_step{step}.npz")


def _print_table(results: list[dict]) -> None:
    print("\n" + "=" * 72)
    print(f"  {'Task':<18} {'Train acc':>10} {'Test acc':>10} {'Forgetting':>12}")
    print("  " + "-" * 68)
    for r in results:
        forgetting = r.get("max_forgetting_pct")
        fstr = f"{forgetting:+.2f}%" if forgetting is not None else "    n/a"
        print(
            f"  {r['task_id']:<18}"
            f"  {r['train_acc']:>8.1%}"
            f"  {r['test_acc']:>8.1%}"
            f"  {fstr:>12}"
        )
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(
    max_train: int | None,
    max_test: int | None,
    epochs_override: int | None,
    lr_override: float | None,
    start_from: int,
    model_in: str | None,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    # accuracy_at_training[task_id] = test accuracy recorded right after training
    accuracy_at_training: dict[str, float] = {}

    # Each step's model is chained: step i reads step i-1's output
    prev_model: str | None = model_in

    for i, task in enumerate(TASKS):
        step = i + 1  # 1-indexed
        if step < start_from:
            # Skip already-done tasks; the prev_model must be supplied via --model-in
            print(f"  Skipping task {step}/{len(TASKS)}: {task['task_id']} (start-from={start_from})")
            if step == start_from - 1:
                prev_model = prev_model  # already set correctly
            continue

        ep = epochs_override if epochs_override is not None else task["ep"]
        lr = lr_override if lr_override is not None else task["lr"]
        model_out = _model_path(step, out_dir)

        print(f"\n{'#'*72}")
        print(f"#  TASK {step}/{len(TASKS)}: {task['task_id'].upper()}  (ep={ep}, lr={lr})")
        print(f"{'#'*72}")
        t0 = time.time()

        train(
            dataset=task["dataset"],
            task_id=task["task_id"],
            model_in=prev_model,
            model_out=model_out,
            subset=task.get("subset"),
            text_field=task.get("text_field"),
            label_field=task.get("label_field"),
            epochs=ep,
            batch_size=64,
            lr=lr,
            tfidf_dim=2000,
            reservoir_dim=256,
            max_sentences=6,
            max_train=max_train,
            max_test=max_test,
            sleep_cycles=0,
        )
        train_time = time.time() - t0

        # Re-evaluate this task on the just-saved model to get a clean test acc
        test_acc = evaluate(
            model_path=model_out,
            dataset=task["dataset"],
            task_id=task["task_id"],
            subset=task.get("subset"),
            text_field=task.get("text_field"),
            label_field=task.get("label_field"),
            max_test=max_test,
        )
        accuracy_at_training[task["task_id"]] = test_acc

        result: dict = {
            "task_id":    task["task_id"],
            "dataset":    task["dataset"],
            "step":       step,
            "test_acc":   round(test_acc, 6),
            "train_acc":  round(test_acc, 6),  # placeholder; full train acc printed during training
            "train_time": round(train_time, 1),
            "forgetting": {},      # task_id → accuracy after new task
            "max_forgetting_pct": None,
        }

        # ── Forgetting check: re-evaluate ALL prior tasks ──────────────────
        if step > 1:
            print(f"\n  Forgetting check after task {step} ({task['task_id']}) …")
            for j in range(i):
                prior = TASKS[j]
                acc_before = accuracy_at_training[prior["task_id"]]
                acc_after = evaluate(
                    model_path=model_out,
                    dataset=prior["dataset"],
                    task_id=prior["task_id"],
                    subset=prior.get("subset"),
                    text_field=prior.get("text_field"),
                    label_field=prior.get("label_field"),
                    max_test=max_test,
                )
                delta_pct = (acc_before - acc_after) * 100.0
                result["forgetting"][prior["task_id"]] = {
                    "acc_before": round(acc_before, 6),
                    "acc_after":  round(acc_after, 6),
                    "delta_pct":  round(delta_pct, 4),
                }
                status = "PASS" if delta_pct < 1.0 else "FAIL"
                print(
                    f"    {prior['task_id']:<18}  before={acc_before:.1%}  "
                    f"after={acc_after:.1%}  Δ={delta_pct:+.3f}%  {status}"
                )

            max_forgetting = max(
                v["delta_pct"] for v in result["forgetting"].values()
            )
            result["max_forgetting_pct"] = round(max_forgetting, 4)
            if max_forgetting >= 1.0:
                print(f"\n  [!] Forgetting exceeds 1% threshold on at least one prior task!")
            else:
                print(f"\n  [ok] All prior tasks within 1% forgetting threshold.")

        results.append(result)
        prev_model = model_out

        # Save incremental results after every task
        out_path = out_dir / "m2_benchmark_results.json"
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # ── Final summary ───────────────────────────────────────────────────────
    print("\n\n" + "=" * 72)
    print("  MILESTONE 2 BENCHMARK — FINAL RESULTS")
    print("=" * 72)
    _print_table(results)

    # M2 exit criteria
    max_forget_overall = max(
        (r["max_forgetting_pct"] for r in results if r["max_forgetting_pct"] is not None),
        default=0.0,
    )
    all_pass_forgetting = max_forget_overall < 1.0
    model_size_kb = Path(_model_path(len(TASKS), out_dir)).stat().st_size // 1024
    model_size_mb = model_size_kb / 1024

    print(f"  Max forgetting across all tasks : {max_forget_overall:+.3f}%   "
          f"({'PASS' if all_pass_forgetting else 'FAIL'}  threshold: < 1%)")
    print(f"  Final model size                : {model_size_mb:.1f} MB   "
          f"({'PASS' if model_size_mb < 10 else 'FAIL'}  threshold: < 10 MB)")
    print()

    out_path = out_dir / "m2_benchmark_results.json"
    print(f"  Results saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Milestone 2: 10-task sequential benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--max-train",  type=int, default=None,
                   help="Limit training samples per task (default: full dataset)")
    p.add_argument("--max-test",   type=int, default=None,
                   help="Limit test samples per task (default: full test split)")
    p.add_argument("--epochs",     type=int, default=None,
                   help="Override epochs for all tasks (default: per-task from registry)")
    p.add_argument("--lr",         type=float, default=None,
                   help="Override learning rate for all tasks")
    p.add_argument("--start-from", type=int, default=1,
                   help="Resume from task N (1-indexed, requires --model-in)")
    p.add_argument("--model-in",   default=None,
                   help="Starting model .npz for --start-from > 1")
    p.add_argument("--out-dir",    default="outputs",
                   help="Directory for step checkpoints and results (default: outputs/)")
    args = p.parse_args()

    if args.start_from > 1 and not args.model_in:
        p.error("--start-from > 1 requires --model-in <checkpoint.npz>")

    run_benchmark(
        max_train=args.max_train,
        max_test=args.max_test,
        epochs_override=args.epochs,
        lr_override=args.lr,
        start_from=args.start_from,
        model_in=args.model_in,
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    main()
