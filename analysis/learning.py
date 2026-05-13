from __future__ import annotations

from typing import Dict

import numpy as np

from core.constants import Config512D
from core.kernel import mse
from system.modular_system import ModularFieldSystem
from system.trainer import ModularTrainer, make_dataset


class LearningAnalysis:
    """Compare routing strategies by training on a fixed regression task.

    Strategies:
        learned   — router is trained via reward signal (normal training)
        random    — router weights stay at initialization, no updates
        uniform   — router outputs equal weights for every module
        nearest   — only adjacent modules receive non-zero routing weight
    """

    def __init__(self, base_cfg: Config512D) -> None:
        self.base_cfg = base_cfg

    def compare_routing_strategies(
        self,
        task: str = "tanh",
        n_epochs: int = 30,
        n_samples: int = 30,
    ) -> Dict[str, float]:
        """Run each routing strategy and compare final test loss."""
        from dataclasses import replace

        cfg = replace(self.base_cfg, train_epochs=n_epochs, train_samples=n_samples)
        results: Dict[str, float] = {}

        strategies = {
            "learned":  self._run_learned,
            "random":   self._run_random,
            "uniform":  self._run_uniform,
            "nearest":  self._run_nearest,
        }

        for name, fn in strategies.items():
            print(f"  Strategy: {name}...")
            loss = fn(cfg, task)
            results[name] = loss
            print(f"    final test loss: {loss:.6f}")

        print("\n  Routing strategy ranking:")
        for name, loss in sorted(results.items(), key=lambda x: x[1]):
            print(f"    {name:<10}: {loss:.6f}")
        best = min(results, key=lambda k: results[k])
        print(f"\n  Best strategy: {best}")
        return results

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _run_learned(self, cfg: Config512D, task: str) -> float:
        system = ModularFieldSystem(cfg)
        trainer = ModularTrainer(system)
        result = trainer.train(task, verbose=False)
        return float(result["test_loss"])

    def _run_random(self, cfg: Config512D, task: str) -> float:
        """Evaluate with router frozen at init (no training updates)."""
        system = ModularFieldSystem(cfg)
        rng = np.random.default_rng(cfg.seed + 200)
        X, Y = make_dataset(cfg, task, rng, 40)
        losses = []
        for x, y in zip(X, Y):
            system.reset_state()
            pred, _ = system.process(x, cfg.process_steps)
            losses.append(mse(pred, y))
        return float(np.mean(losses))

    def _run_uniform(self, cfg: Config512D, task: str) -> float:
        """Evaluate with uniform routing (equal weight to all modules)."""
        system = ModularFieldSystem(cfg)
        system.router = np.ones_like(system.router) * (1.0 / cfg.num_modules)
        rng = np.random.default_rng(cfg.seed + 200)
        X, Y = make_dataset(cfg, task, rng, 40)
        losses = []
        for x, y in zip(X, Y):
            system.reset_state()
            pred, _ = system.process(x, cfg.process_steps)
            losses.append(mse(pred, y))
        return float(np.mean(losses))

    def _run_nearest(self, cfg: Config512D, task: str) -> float:
        """Evaluate with nearest-neighbor routing (each module routes only to adjacent)."""
        system = ModularFieldSystem(cfg)
        system.router = np.zeros_like(system.router)
        for i in range(cfg.num_modules):
            lo = max(0, i - 1)
            hi = min(cfg.num_modules, i + 2)
            system.router[lo:hi, :] = 0.05
        rng = np.random.default_rng(cfg.seed + 200)
        X, Y = make_dataset(cfg, task, rng, 40)
        losses = []
        for x, y in zip(X, Y):
            system.reset_state()
            pred, _ = system.process(x, cfg.process_steps)
            losses.append(mse(pred, y))
        return float(np.mean(losses))
