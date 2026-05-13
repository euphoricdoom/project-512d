from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from core.constants import Config512D
from core.governance import CompactGovernance
from core.kernel import CoupledModularKernel, rational_bound, softmax, stable_tanh_bound
from core.projector import BoundaryDetector, ConstraintProjector, TraceLedger
from core.readout import module_features, module_features_batch


class SpecializedModule:
    """Minimal per-module input encoder. Holds field-space indices and
    a learnable specialization weight vector for routing-weighted injection."""

    def __init__(self, module_id: int, module_size: int, rng: np.random.Generator) -> None:
        i0 = module_id * module_size
        self.dims = np.arange(i0, i0 + module_size)
        self.specialization_weights = rng.normal(0.0, 0.5, size=module_size)

    def encode_input(self, input_slice: np.ndarray) -> np.ndarray:
        return stable_tanh_bound(input_slice * self.specialization_weights)

    def update_specialization(
        self, input_slice: np.ndarray, reward: float, lr: float
    ) -> None:
        self.specialization_weights += lr * reward * input_slice
        np.clip(self.specialization_weights, -2.0, 2.0, out=self.specialization_weights)


def build_modules(
    cfg: "Config512D", rng: np.random.Generator
) -> "list[SpecializedModule]":
    return [SpecializedModule(i, cfg.module_size, rng) for i in range(cfg.num_modules)]


class ModularFieldSystem:
    """Assembled 512D modular field system.

    Combines:
    - CoupledModularKernel (dynamics substrate)
    - 60 SpecializedModules (cognitive structure)
    - Softmax router (input dispatch)
    - Linear readout layer (trainable output)
    - BoundaryDetector + ConstraintProjector (governance)
    - TraceLedger (observability)
    """

    def __init__(self, cfg: Config512D) -> None:
        cfg.validate()
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # Core components
        self.kernel = CoupledModularKernel(cfg, self.rng)
        self.frozen = False
        self.projector = ConstraintProjector(cfg)
        self.governance = CompactGovernance(cfg)
        self.boundary = BoundaryDetector(cfg)
        self.trace = TraceLedger(cfg)

        # State
        self.state = self.rng.normal(0.0, 0.08, size=cfg.dim)
        self.state_prev = np.zeros(cfg.dim)

        # Cognitive modules
        self.modules: List[SpecializedModule] = build_modules(cfg, self.rng)

        # Router: (num_modules, input_dim) — maps input vector to module weights
        self.router = self.rng.normal(0.0, 0.025, size=(cfg.num_modules, cfg.input_dim))

        # Readout: (output_dim, feature_dim) — 92D module-aggregate interface
        # feature_dim = num_modules + dim_c = 60 + 32 = 92
        feature_dim = cfg.num_modules + cfg.dim_c
        self.readout = self.rng.normal(0.0, 0.02, size=(cfg.output_dim, feature_dim))

        # Fixed input projection into the full field
        self.input_projection = self.rng.normal(
            0.0, 0.035, size=(cfg.dim, cfg.input_dim)
        )

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_state(self, scale: float = 0.08) -> None:
        """Re-randomize state to break correlations between training samples."""
        self.state = self.rng.normal(0.0, scale, size=self.cfg.dim)
        self.state_prev = np.zeros(self.cfg.dim)
        self.boundary.prev_norm = float(np.linalg.norm(self.state))

    def freeze(self) -> None:
        """Freeze learnable shared dynamics while preserving state evolution."""
        self.frozen = True
        self.kernel.freeze()

    def unfreeze(self) -> None:
        """Resume learnable shared dynamics."""
        self.frozen = False
        self.kernel.unfreeze()

    def is_frozen(self) -> bool:
        """Return whether shared dynamics are frozen."""
        return self.frozen or self.kernel.is_frozen()

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route_input(self, input_vector: np.ndarray) -> np.ndarray:
        """Compute routing weights. k_winners > 0: sparse top-k; else full softmax."""
        cfg = self.cfg
        scores = self.router @ input_vector
        k = cfg.k_winners
        if k <= 0 or k >= cfg.num_modules:
            return softmax(scores, cfg.route_temperature)
        weights = np.zeros(cfg.num_modules)
        top_k = np.argpartition(scores, -k)[-k:]
        weights[top_k] = softmax(scores[top_k], cfg.route_temperature)
        return weights

    def route_input_batch(self, input_vectors: np.ndarray) -> np.ndarray:
        """Batch routing weights for input_vectors shaped (B, input_dim)."""
        cfg = self.cfg
        input_vectors = np.asarray(input_vectors, dtype=float)
        assert input_vectors.ndim == 2 and input_vectors.shape[1] == cfg.input_dim
        scores = input_vectors @ self.router.T
        k = cfg.k_winners
        if k <= 0 or k >= cfg.num_modules:
            z = scores / max(cfg.route_temperature, 1e-8)
            z = z - np.max(z, axis=1, keepdims=True)
            exp_z = np.exp(z)
            return exp_z / (np.sum(exp_z, axis=1, keepdims=True) + 1e-12)

        weights = np.zeros_like(scores)
        top_k = np.argpartition(scores, -k, axis=1)[:, -k:]
        row_idx = np.arange(scores.shape[0])[:, None]
        selected = scores[row_idx, top_k]
        z = selected / max(cfg.route_temperature, 1e-8)
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        weights[row_idx, top_k] = exp_z / (
            np.sum(exp_z, axis=1, keepdims=True) + 1e-12
        )
        return weights

    # ------------------------------------------------------------------
    # Input injection
    # ------------------------------------------------------------------

    def inject(self, input_vector: np.ndarray) -> np.ndarray:
        """Inject input into the field via global projection and routed module encoding."""
        cfg = self.cfg
        input_vector = np.asarray(input_vector, dtype=float)
        assert input_vector.shape == (cfg.input_dim,)

        route_weights = self.route_input(input_vector)

        # Global perturbation across all dims
        self.state += cfg.input_strength * (self.input_projection @ input_vector)

        # Module-local routed perturbations
        input8 = input_vector[:cfg.module_size]
        for module, weight in zip(self.modules, route_weights):
            if weight >= cfg.active_route_floor:
                self.state[module.dims] += (
                    cfg.input_strength * weight * module.encode_input(input8)
                )

        self.state = rational_bound(self.state)
        return route_weights

    def reset_state_batch(self, batch_size: int, scale: float = 0.08) -> Tuple[np.ndarray, np.ndarray]:
        """Create independent initial states for a batch without mutating single state."""
        states = self.rng.normal(0.0, scale, size=(batch_size, self.cfg.dim))
        prev_states = np.zeros((batch_size, self.cfg.dim))
        return states, prev_states

    def inject_batch(
        self,
        input_vectors: np.ndarray,
        states: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Inject a batch of inputs into a batch of states."""
        cfg = self.cfg
        input_vectors = np.asarray(input_vectors, dtype=float)
        assert input_vectors.ndim == 2 and input_vectors.shape[1] == cfg.input_dim
        assert states.shape == (input_vectors.shape[0], cfg.dim)

        route_weights = self.route_input_batch(input_vectors)
        states += cfg.input_strength * (input_vectors @ self.input_projection.T)

        input8 = input_vectors[:, :cfg.module_size]
        specialization = np.stack([m.specialization_weights for m in self.modules])
        encoded = stable_tanh_bound(input8[:, None, :] * specialization[None, :, :])
        active = route_weights >= cfg.active_route_floor
        encoded *= (cfg.input_strength * route_weights * active)[:, :, None]
        states[:, :cfg.dim_f] += encoded.reshape(input_vectors.shape[0], cfg.dim_f)
        states = rational_bound(states)
        return states, route_weights

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def step(self, t: int = 0, pred_loss: Optional[float] = None) -> np.ndarray:
        """One evolution step: kernel mix → containment → boundary check."""
        cfg = self.cfg
        module_norms = self.current_module_norms()
        self.state[cfg.dim_f:] = module_norms

        KX = self.kernel.compute(self.state)
        field_correction = np.zeros(cfg.dim_f)
        if cfg.enable_projection:
            damping = -self.governance.apply(module_norms)
            field_correction = -self.state[:cfg.dim_f] * damping
            KX[:cfg.dim_f] += field_correction
        X_next = (
            (1.0 - cfg.epsilon) * self.state
            + cfg.epsilon * KX
            + cfg.alpha * self.state_prev
        )
        if cfg.enable_projection:
            X_next[:cfg.dim_f] += cfg.projection_damping * field_correction
        X_next = rational_bound(X_next)

        self.state_prev = self.state.copy()
        self.state = X_next

        boundary = self.boundary.check(self.state)
        projected = cfg.enable_projection and bool(boundary["trigger"])

        self.state[cfg.dim_f:] = self.current_module_norms()

        self.trace.record(t, self.state, boundary, projected, pred_loss)
        return self.state

    def current_module_norms_batch(self, states: np.ndarray) -> np.ndarray:
        """Batch L2 norm for each module."""
        cfg = self.cfg
        return np.linalg.norm(
            states[:, :cfg.dim_f].reshape(-1, cfg.num_modules, cfg.module_size),
            axis=2,
        )

    def _governance_batch(self, module_norms: np.ndarray) -> np.ndarray:
        """Vectorized CompactGovernance.apply for module_norms shaped (B, M)."""
        cfg = self.cfg
        damping = module_norms * self.governance.self_inhibition[None, :]
        for group_start, group_end in ((0, 20), (20, 40), (40, 50), (50, 60)):
            group = module_norms[:, group_start:group_end]
            group_max = np.max(group, axis=1, keepdims=True)
            damping[:, group_start:group_end] += (
                self.governance.lateral_strength * np.maximum(0.0, group_max - group)
            )
        return -np.repeat(damping, cfg.module_size, axis=1)

    def step_batch(
        self,
        states: np.ndarray,
        prev_states: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """One batched evolution step equivalent to step() without trace side effects."""
        cfg = self.cfg
        module_norms = self.current_module_norms_batch(states)
        states = states.copy()
        states[:, cfg.dim_f:] = module_norms

        KX = self.kernel.compute_batch(states)
        field_correction = np.zeros((states.shape[0], cfg.dim_f))
        if cfg.enable_projection:
            damping = -self._governance_batch(module_norms)
            field_correction = -states[:, :cfg.dim_f] * damping
            KX[:, :cfg.dim_f] += field_correction

        next_states = (
            (1.0 - cfg.epsilon) * states
            + cfg.epsilon * KX
            + cfg.alpha * prev_states
        )
        if cfg.enable_projection:
            next_states[:, :cfg.dim_f] += cfg.projection_damping * field_correction
        next_states = rational_bound(next_states)
        next_states[:, cfg.dim_f:] = self.current_module_norms_batch(next_states)
        return next_states, states

    def evolve(self, steps: int) -> None:
        """Free evolution without input injection."""
        for t in range(steps):
            self.step(t=t)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def predict(self) -> np.ndarray:
        """Linear readout via module aggregate and constraint scoreboard features."""
        self.state[self.cfg.dim_f:] = self.current_module_norms()
        features = module_features(self.state, self.cfg)
        if hasattr(self.readout, "predict"):
            return self.readout.predict(features)
        return self.readout @ features

    def predict_batch_states(self, states: np.ndarray) -> np.ndarray:
        states = states.copy()
        states[:, self.cfg.dim_f:] = self.current_module_norms_batch(states)
        features = module_features_batch(states, self.cfg)
        if hasattr(self.readout, "predict_batch"):
            return self.readout.predict_batch(features)
        return features @ self.readout.T

    def features_batch_states(self, states: np.ndarray) -> np.ndarray:
        states = states.copy()
        states[:, self.cfg.dim_f:] = self.current_module_norms_batch(states)
        return module_features_batch(states, self.cfg)

    def process_batch(
        self, input_vectors: np.ndarray, steps: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Inject/evolve a batch independently and return predictions, routes, states."""
        if steps is None:
            steps = self.cfg.process_steps
        states, prev_states = self.reset_state_batch(input_vectors.shape[0])
        states, route_weights = self.inject_batch(input_vectors, states)
        for _ in range(steps):
            states, prev_states = self.step_batch(states, prev_states)
        return self.predict_batch_states(states), route_weights, states

    def process(
        self, input_vector: np.ndarray, steps: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Inject input, evolve for steps, return (prediction, route_weights)."""
        if steps is None:
            steps = self.cfg.process_steps
        route_weights = self.inject(input_vector)
        for t in range(steps):
            self.step(t=t)
        return self.predict(), route_weights

    # ------------------------------------------------------------------
    # Module observability
    # ------------------------------------------------------------------

    def module_activations(self) -> np.ndarray:
        """Current activation level for each module."""
        return np.array([m.activation(self.state) for m in self.modules])

    def current_module_norms(self) -> np.ndarray:
        """Current L2 norm for each module, reused as constraint state."""
        cfg = self.cfg
        return np.linalg.norm(
            self.state[:cfg.dim_f].reshape(cfg.num_modules, cfg.module_size),
            axis=1,
        )

    def module_output_table(self) -> List[Dict[str, object]]:
        """Sorted list of module activations with id and expertise."""
        acts = self.module_activations()
        rows = [
            {"id": i, "expertise": m.expertise, "activation": float(acts[i])}
            for i, m in enumerate(self.modules)
        ]
        rows.sort(key=lambda r: float(r["activation"]), reverse=True)
        return rows
