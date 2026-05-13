from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from core.constants import Config512D
from core.kernel import mse
from core.readout import module_features
from system.modular_system import ModularFieldSystem


# ============================================================
# DATASETS
# ============================================================

def make_dataset(
    cfg: Config512D,
    task: str,
    rng: np.random.Generator,
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate (inputs, targets) for a named regression task.

    Tasks:
        tanh    — element-wise tanh(2x)
        smooth  — 5-sample moving average
        cumsum  — cumulative sum with tanh clamp
        edge    — finite-difference edge detector with tanh clamp
    """
    X = rng.normal(0.0, 0.55, size=(n, cfg.input_dim))

    if task == "tanh":
        Y = np.tanh(2.0 * X[:, :cfg.output_dim])
    elif task == "smooth":
        Y = np.array([
            np.convolve(x[:cfg.output_dim], np.ones(5) / 5, mode="same") for x in X
        ])
    elif task == "cumsum":
        Y = np.cumsum(X[:, :cfg.output_dim] * 0.12, axis=1)
        Y = np.tanh(Y)
    elif task == "edge":
        y = np.zeros((n, cfg.output_dim))
        y[:, 1:] = np.abs(X[:, 1:cfg.output_dim] - X[:, :cfg.output_dim - 1])
        Y = np.tanh(y)
    else:
        raise ValueError(f"Unknown task '{task}'. Choose from: tanh, smooth, cumsum, edge")

    return X, Y


# ============================================================
# TRAINER
# ============================================================

class ModularTrainer:
    """Trains the readout layer, router, and module specialization weights.

    Update rules:
    - Readout: delta rule (true gradient of MSE for linear readout)
    - Router:  reward-weighted perturbation (not BPTT — intentionally simple)
    - Modules: local reward-scaled Hebbian update
    """

    def __init__(self, system: ModularFieldSystem) -> None:
        self.system = system
        self._sample_count = 0
        self.current_task: str | None = None

    def set_task(self, task: str) -> None:
        self.current_task = task
        if hasattr(self.system.readout, "set_task"):
            self.system.readout.set_task(task)

    def train_sample(
        self, input_vector: np.ndarray, target: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Process one (input, target) pair and apply all weight updates."""
        cfg = self.system.cfg
        self.system.reset_state()
        pred, route_weights = self.system.process(input_vector, cfg.process_steps)

        error = target - pred
        loss = mse(pred, target)

        # Readout delta rule — update against 92D feature vector, not raw state
        features = module_features(self.system.state, cfg)
        if hasattr(self.system.readout, "update"):
            self.system.readout.update(
                features, error, cfg.readout_lr, task_id=self.current_task
            )
        else:
            self.system.readout += cfg.readout_lr * np.outer(error, features)
            self.system.readout = np.clip(self.system.readout, -3.0, 3.0)

        # Router reward signal
        reward = -loss
        uniform = 1.0 / cfg.num_modules
        self.system.router += cfg.router_lr * reward * np.outer(
            route_weights - uniform, input_vector
        )
        self.system.router = np.clip(self.system.router, -2.0, 2.0)

        # Local module specialization
        input8 = input_vector[:cfg.module_size]
        for module in self.system.modules:
            module.update_specialization(input8, reward, cfg.specialization_lr)

        # Learnable K_inter — Hebbian update on U, V factors
        if cfg.learnable_inter_modules:
            self.system.kernel.update_inter(self.system.state, cfg.inter_lr)
            self._sample_count += 1
            if self._sample_count % cfg.inter_renorm_every == 0:
                self.system.kernel.renormalize_inter()

        if cfg.learnable_constraint_feedback:
            self.system.governance.update(
                module_norms=self.system.current_module_norms(),
                performance_error=loss,
                lr=cfg.constraint_feedback_lr,
            )

        return loss, route_weights

    def train(
        self, task: str, verbose: bool = True
    ) -> Dict[str, object]:
        """Full training run over cfg.train_epochs on cfg.train_samples samples.

        Returns a results dict with train losses, test loss, and per-epoch
        mean routing vectors.
        """
        cfg = self.system.cfg
        self.set_task(task)
        rng = np.random.default_rng(cfg.seed + 100)
        X, Y = make_dataset(cfg, task, rng, cfg.train_samples)

        losses: List[float] = []
        route_means: List[np.ndarray] = []

        for epoch in range(cfg.train_epochs):
            order = rng.permutation(cfg.train_samples)
            epoch_losses = []
            epoch_routes = []
            for idx in order:
                loss, routes = self.train_sample(X[idx], Y[idx])
                epoch_losses.append(loss)
                epoch_routes.append(routes)
            losses.append(float(np.mean(epoch_losses)))
            route_means.append(np.mean(epoch_routes, axis=0))

            if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
                print(f"  Epoch {epoch+1:02d}/{cfg.train_epochs} | loss={losses[-1]:.6f}")

        # Hold-out test
        Xt, Yt = make_dataset(cfg, task, rng, 40)
        test_losses = []
        for x, y in zip(Xt, Yt):
            self.system.reset_state()
            pred, _ = self.system.process(x, cfg.process_steps)
            test_losses.append(mse(pred, y))

        return {
            "losses": losses,
            "test_loss": float(np.mean(test_losses)),
            "route_means": np.array(route_means),
        }
