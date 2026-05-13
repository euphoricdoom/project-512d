"""Train learned gates by distilling explicit Helix channel trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.constants import Config512D
from experiments.run_classic_benchmarks import CLASSIC_TASKS, generate_classic_task_data
from system.distillation_trainer import DistillationTrainer
from system.distillation_trainer_v2 import DistillationTrainerV2
from system.learned_gates import LearnedGateNetwork, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill learned gates")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--k-shot", type=int, default=10)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--input-width", type=int, default=64)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="outputs/distilled_gates.pth")
    parser.add_argument("--gate-decay-min", type=float, default=0.0)
    parser.add_argument("--gate-decay-max", type=float, default=0.99)
    parser.add_argument("--sleep-cycles", type=int, default=0)
    parser.add_argument("--sleep-lr", type=float, default=None)
    parser.add_argument(
        "--trainer",
        choices=["v1", "v2"],
        default="v2",
        help="Distillation trainer implementation. v2 uses cached stable features.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config512D(seed=args.seed, train_samples=args.samples, train_epochs=args.epochs)
    tasks = {
        task: generate_classic_task_data(task, args.samples, args.seed + i, cfg)
        for i, task in enumerate(CLASSIC_TASKS)
    }
    gate_net = LearnedGateNetwork(
        input_dim=args.input_width,
        task_embedding_dim=128,
        num_dc_channels=args.feature_dim,
        feature_dim=args.feature_dim,
        decay_min=args.gate_decay_min,
        decay_max=args.gate_decay_max,
    )
    print("=" * 60)
    print("GATE DISTILLATION")
    print("=" * 60)
    print(f"Parameters: {count_parameters(gate_net):,}")
    print(f"Tasks: {list(tasks)}")
    trainer_cls = DistillationTrainerV2 if args.trainer == "v2" else DistillationTrainer
    print(f"Trainer: {args.trainer}")
    trainer = trainer_cls(
        gate_net,
        input_width=args.input_width,
        feature_dim=args.feature_dim,
        lr=args.lr,
        device=args.device,
        k_support=args.k_shot,
        seed=args.seed,
    )
    trainer.train(
        tasks,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        save_path=args.output,
        sleep_cycles=args.sleep_cycles,
        sleep_lr=args.sleep_lr,
    )
    gate_net.save(args.output)
    history_path = Path(args.output).parent / "distillation_history.json"
    trainer.save_history(history_path)
    print(f"Saved distilled gates to {args.output}")
    print(f"Saved history to {history_path}")


if __name__ == "__main__":
    main()
