"""Adaptive Helix temporal adapter with learned gates.

``AdaptiveHelixTemporalAdapter`` keeps the Phase 1 Helix feature contract while
replacing explicit task-specific channel writes with learned gate responses.
The parent ``HelixTemporalAdapter`` remains available as the explicit baseline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

try:
    from system.helix_temporal import HelixTemporalAdapter
    from system.learned_gates import (
        LearnedGateConfig,
        LearnedGateController,
        LearnedGateNetwork,
    )
except ImportError:  # Allows ``python system/adaptive_helix.py``.
    from helix_temporal import HelixTemporalAdapter
    from learned_gates import (
        LearnedGateConfig,
        LearnedGateController,
        LearnedGateNetwork,
    )


class AdaptiveHelixTemporalAdapter(HelixTemporalAdapter):
    """Helix adapter whose DC writes are controlled by learned gates.

    Args mirror ``HelixTemporalAdapter``. ``gate_network`` may be supplied for
    task-example generated PyTorch gates; otherwise a lightweight supervised
    NumPy ``LearnedGateController`` drives the gate by default.
    """

    def __init__(
        self,
        input_dim: int,
        input_width: int = 64,
        projection_dim: int = 256,
        ac_decay: float = 0.25,
        dc_decay: float = 0.98,
        gate_scale: float = 1.0,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
        diagnostics: bool = False,
        projection_scale: float | None = None,
        gate_network: LearnedGateNetwork | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            input_width=input_width,
            projection_dim=projection_dim,
            ac_decay=ac_decay,
            dc_decay=dc_decay,
            gate_scale=gate_scale,
            seed=seed,
            rng=rng,
            diagnostics=diagnostics,
            projection_scale=projection_scale,
        )
        self.device = torch.device(device)
        self.gate_network = gate_network.to(self.device) if gate_network is not None else None
        self.current_gate_config: dict[str, torch.Tensor] | None = None
        self.gate_config_np: dict[str, np.ndarray] | None = None
        self.configured_task_id: str | None = None
        self.gate_controller = LearnedGateController(
            LearnedGateConfig(
                input_width=self.input_width,
                state_dim=self.projection_dim,
                learning_rate=0.2,
                seed=seed,
            )
        )
        self.gate_loss: float | None = None

    def configure_for_task(
        self, task_examples: np.ndarray | torch.Tensor, task_id: str | None = None
    ) -> dict[str, torch.Tensor]:
        """Generate a learned gate configuration from K task examples."""
        if self.gate_network is None:
            self.gate_network = LearnedGateNetwork(
                input_dim=self.input_width,
                num_dc_channels=self.projection_dim,
                feature_dim=self.projection_dim,
            ).to(self.device)
        if not isinstance(task_examples, torch.Tensor):
            task_examples = torch.as_tensor(task_examples, dtype=torch.float32)
        task_examples = task_examples.to(self.device, dtype=torch.float32)
        with torch.no_grad():
            self.current_gate_config = self.gate_network.configure_for_task(task_examples)
        self.gate_config_np = {
            key: value.detach().cpu().numpy().copy()
            for key, value in self.current_gate_config.items()
        }
        self.configured_task_id = task_id
        return self.current_gate_config

    @staticmethod
    def _safe_sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-x))

    def _learned_dc_update(
        self,
        input_arr: np.ndarray,
        projected: np.ndarray,
        task_id: str | None,
    ) -> np.ndarray:
        """Apply generated gates when configured, otherwise controller gates."""
        if self.gate_config_np is not None:
            write_weights = self.gate_config_np["write_gates"]
            update_weights = self.gate_config_np["update_weights"]
            decay_rates = self.gate_config_np["decay_rates"]
            write_gate = self._safe_sigmoid(projected @ write_weights.T)
            candidate = np.tanh(projected @ update_weights.T)
            self.dc_state = decay_rates[None, :] * self.dc_state + (
                1.0 - decay_rates[None, :]
            ) * write_gate * candidate
            return write_gate

        write_gate = self.gate_controller.forward(input_arr, self.ac_state, task_id)
        candidate = np.tanh(projected + self.ac_state)
        self.dc_state = (
            self.dc_decay * self.dc_state
            + (1.0 - self.dc_decay) * write_gate * candidate
        )
        return write_gate

    def step(
        self,
        input_t: np.ndarray,
        reservoir_features_t: np.ndarray | None = None,
        t: int | None = None,
        total_steps: int | None = None,
        task_id: str | None = None,
    ) -> np.ndarray:
        """Consume one timestep using learned gates.

        ``task_id`` is accepted only as optional conditioning for the lightweight
        controller. No explicit benchmark-specific channel logic is used.
        """
        if reservoir_features_t is None:
            feature_arr = np.asarray(input_t, dtype=float)
            if feature_arr.ndim == 1:
                feature_arr = feature_arr[None, :]
            elif feature_arr.shape[0] != self.batch_size:
                self.reset(feature_arr.shape[0])
            input_arr = np.zeros((feature_arr.shape[0], self.input_width))
        else:
            input_arr = self._as_batch(input_t, self.input_width, "input_t")
            feature_arr = self._as_batch(
                reservoir_features_t, self.input_dim, "reservoir_features_t"
            )

        projected = np.tanh(feature_arr @ self.W_project + self.b_project)
        event = projected - self.last_projected
        self.ac_state = self.ac_decay * self.ac_state + (1.0 - self.ac_decay) * event
        write_gate = self._learned_dc_update(input_arr, projected, task_id)

        self.step_count += 1
        self.last_projected = projected
        self.mean_projected += (projected - self.mean_projected) / self.step_count
        self.max_projected = np.maximum(self.max_projected, projected)
        self.last_gate = write_gate
        self.last_write_gate = write_gate
        self.phase_final = self._phase_code(
            0 if t is None else int(t),
            self.step_count if total_steps is None else int(total_steps),
        )

        if self.diagnostics_enabled:
            self._history["ac"].append(self.ac_state.copy())
            self._history["dc"].append(self.dc_state.copy())
            self._history["gate"].append(write_gate.copy())
        return projected[0] if projected.shape[0] == 1 else projected

    def update_gates(
        self,
        input_t: np.ndarray,
        ac_state: np.ndarray,
        target_gate: np.ndarray,
        task_id: str | None = None,
    ) -> dict[str, float]:
        """Supervise the lightweight gate controller on a target gate."""
        metrics = self.gate_controller.update(input_t, ac_state, target_gate, task_id)
        self.gate_loss = metrics["loss"]
        return metrics

    def get_diagnostics(self) -> dict[str, Any]:
        """Return Helix diagnostics plus learned-gate summaries."""
        payload = super().get_diagnostics()
        payload.update(
            {
                "adapter_type": "adaptive_helix",
                "learned_gate_enabled": True,
                "gate_loss": self.gate_loss,
            }
        )
        if self.gate_config_np is not None:
            payload["learned_gates"] = {
                "decay_rates": {
                    "mean": float(np.mean(self.gate_config_np["decay_rates"])),
                    "std": float(np.std(self.gate_config_np["decay_rates"])),
                    "min": float(np.min(self.gate_config_np["decay_rates"])),
                    "max": float(np.max(self.gate_config_np["decay_rates"])),
                },
                "write_gates": {
                    "mean": float(np.mean(self.gate_config_np["write_gates"])),
                    "std": float(np.std(self.gate_config_np["write_gates"])),
                },
                "update_weights": {
                    "mean": float(np.mean(self.gate_config_np["update_weights"])),
                    "std": float(np.std(self.gate_config_np["update_weights"])),
                },
            }
        return payload


class AdaptiveHelix(AdaptiveHelixTemporalAdapter):
    """Compatibility alias with the shorter class name from the Phase 2 plan."""


def compare_explicit_vs_learned(task_examples: np.ndarray, task_id: str) -> dict[str, float]:
    """Compare explicit and adaptive Helix feature outputs on dummy features."""
    input_width = task_examples.shape[2]
    explicit = HelixTemporalAdapter(input_dim=256, input_width=input_width, projection_dim=64, seed=1)
    learned = AdaptiveHelixTemporalAdapter(
        input_dim=256, input_width=input_width, projection_dim=64, seed=1
    )
    learned.configure_for_task(task_examples[: min(10, len(task_examples))], task_id)
    explicit.reset(1)
    learned.reset(1)
    rng = np.random.default_rng(1)
    for t in range(task_examples.shape[1]):
        input_t = task_examples[:1, t, :]
        features_t = rng.normal(size=(1, 256))
        explicit.step(input_t, features_t, t, task_examples.shape[1], task_id)
        learned.step(input_t, features_t, t, task_examples.shape[1], None)
    a = explicit.final_features().reshape(-1)
    b = learned.final_features().reshape(-1)
    return {
        "feature_difference": float(np.mean(np.abs(a - b))),
        "explicit_norm": float(np.linalg.norm(a)),
        "learned_norm": float(np.linalg.norm(b)),
        "correlation": float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else 0.0,
    }


def _self_test() -> None:
    adapter = AdaptiveHelixTemporalAdapter(
        input_dim=5, input_width=3, projection_dim=4, seed=7, diagnostics=True
    )
    adapter.reset(2)
    for t in range(3):
        adapter.step(np.ones((2, 3)), np.ones((2, 5)) * t, t=t, total_steps=3)
    features = adapter.final_features()
    assert features.shape == (2, adapter.feature_dim)
    assert adapter.feature_dim == 27
    assert np.isfinite(features).all()
    print("AdaptiveHelix tests passed")


if __name__ == "__main__":
    _self_test()


__all__ = [
    "AdaptiveHelixTemporalAdapter",
    "AdaptiveHelix",
    "compare_explicit_vs_learned",
]
