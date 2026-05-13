"""Compact selective governance using module activations as constraints.

Nintendo-efficient design:
- zero adjacency lookup tables (compute structure on the fly)
- 60 scalar self-inhibition weights
- lateral inhibition within functional groups, derived from module ids
- interpretable winner-take-all competition
"""
from __future__ import annotations

import numpy as np

from core.constants import Config512D


GROUP_BOUNDARIES = ((0, 20), (20, 40), (40, 50), (50, 60))
GROUP_NAMES = ("perception", "reasoning", "memory", "action")


def module_group(module_id: int) -> int:
    for group_idx, (start, end) in enumerate(GROUP_BOUNDARIES):
        if start <= module_id < end:
            return group_idx
    raise ValueError(f"module_id out of range: {module_id}")


def group_means(module_norms: np.ndarray) -> np.ndarray:
    return np.array([
        float(np.mean(module_norms[start:end]))
        for start, end in GROUP_BOUNDARIES
    ])


class CompactGovernance:
    """Selective governance with minimal storage.

    Each module is governed by self-inhibition and lateral inhibition inside its
    functional group. The only learned state is one self-inhibition scalar per
    module.
    """

    def __init__(self, cfg: Config512D) -> None:
        self.num_modules = cfg.num_modules
        self.module_size = cfg.module_size
        self.self_inhibition = np.ones(self.num_modules) * 0.1
        self.lateral_strength = 0.05

    def apply(self, module_norms: np.ndarray) -> np.ndarray:
        """Return a field-space damping signal derived from module activations."""
        module_norms = np.asarray(module_norms, dtype=float)
        assert module_norms.shape == (self.num_modules,)

        damping = self.self_inhibition * module_norms
        for group_start, group_end in GROUP_BOUNDARIES:
            group = module_norms[group_start:group_end]
            group_max = float(np.max(group))
            damping[group_start:group_end] += self.lateral_strength * np.maximum(
                0.0, group_max - group
            )

        return -np.repeat(damping, self.module_size)

    def update(
        self,
        module_norms: np.ndarray,
        performance_error: float,
        lr: float = 0.001,
    ) -> None:
        """Strengthen inhibition on overactive modules when performance is poor."""
        if performance_error > 0.1:
            top5 = np.argsort(module_norms)[-5:]
            self.self_inhibition[top5] += lr * float(performance_error)
        np.clip(self.self_inhibition, 0.01, 0.5, out=self.self_inhibition)

    def stored_params(self) -> int:
        return int(self.self_inhibition.size)

    def get_parameters(self) -> list[np.ndarray]:
        return [self.self_inhibition]

    def governance_matrix(self) -> np.ndarray:
        matrix = np.zeros((self.num_modules, self.num_modules))
        for i in range(self.num_modules):
            matrix[i, i] = self.self_inhibition[i]
        for start, end in GROUP_BOUNDARIES:
            for target in range(start, end):
                for source in range(start, end):
                    if source != target:
                        matrix[target, source] = self.lateral_strength
        return matrix


class BinaryGovernance:
    """Binary winner-take-all governance.

    Only top-k modules per functional group remain unsuppressed. This is useful
    as an extreme efficiency and sparsity baseline.
    """

    def __init__(self, cfg: Config512D, k_winners: int = 2) -> None:
        self.num_modules = cfg.num_modules
        self.module_size = cfg.module_size
        self.k_winners = k_winners

    def apply(self, module_norms: np.ndarray) -> np.ndarray:
        """Return a binary suppression signal for all modules outside group top-k."""
        module_norms = np.asarray(module_norms, dtype=float)
        assert module_norms.shape == (self.num_modules,)

        damping = np.zeros(self.num_modules)
        for group_start, group_end in GROUP_BOUNDARIES:
            group = module_norms[group_start:group_end]
            k = min(self.k_winners, group_end - group_start)
            threshold = np.partition(group, -k)[-k]
            damping[group_start:group_end] = group < threshold

        return -np.repeat(damping, self.module_size)

    def update(
        self,
        module_norms: np.ndarray,
        performance_error: float,
        lr: float = 0.001,
    ) -> None:
        return None

    def stored_params(self) -> int:
        return 0

    def get_parameters(self) -> list[np.ndarray]:
        return []

    def governance_matrix(self) -> np.ndarray:
        matrix = np.zeros((self.num_modules, self.num_modules))
        for start, end in GROUP_BOUNDARIES:
            matrix[start:end, start:end] = 1.0
            np.fill_diagonal(matrix[start:end, start:end], 0.0)
        return matrix
