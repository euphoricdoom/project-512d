"""Hierarchical kernel network.

This keeps the same public training surface as KernelNetwork, but replaces the
flat all-kernel router/topology with a 4-ary hierarchy:

- kernels are partitioned into groups of four
- routing chooses a functional group, then kernels inside that group
- communication is dense inside groups and sparse between group representatives
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from core.constants import Config512D
from core.kernel import softmax
from experiments.run_4_kernel_network import KernelNetwork


TASK_DOMAINS = {
    "tanh": 0,
    "relu": 0,
    "sigmoid": 0,
    "abs": 0,
    "square": 0,
    "sqrt": 0,
    "cube": 0,
    "sin_abs": 0,
    "tanh_square": 0,
    "sigmoid_scale": 0,
    "sqrt_clip": 0,
    "smooth": 1,
    "cumsum": 1,
    "shift": 1,
    "scale": 1,
    "flip": 1,
    "noise": 1,
    "quantize": 1,
    "relu_shift": 1,
    "edge": 2,
    "threshold": 2,
    "clip": 2,
    "saturate": 2,
    "normalize": 2,
    "neg": 2,
    "sin": 3,
    "cos": 3,
    "exp": 3,
    "log": 3,
    "exp_clip": 3,
    "log_abs": 3,
    "cos_square": 3,
}


class HierarchicalKernelNetwork(KernelNetwork):
    """Sparse, group-routed kernel network.

    The hierarchy is deliberately piecewise: every four kernels form a local
    group. For larger networks, group representatives are connected through a
    sparse 4-ary tree plus a light ring so information can still propagate
    between domains without O(K^2) communication.
    """

    def __init__(
        self,
        num_kernels: int,
        cfg: Config512D,
        group_size: int = 4,
        use_stable_readout: bool = True,
        stable_base_lr: float = 0.01,
        use_projection: bool = False,
        projection_dim: int = 256,
    ) -> None:
        self.group_size = group_size
        super().__init__(
            num_kernels,
            cfg,
            use_stable_readout=use_stable_readout,
            stable_base_lr=stable_base_lr,
            use_projection=use_projection,
            projection_dim=projection_dim,
        )
        self.groups = self._build_groups(num_kernels)
        self.kernel_to_group = {
            kernel_idx: group_idx
            for group_idx, group in enumerate(self.groups)
            for kernel_idx in group
        }
        self.group_router = self.rng.normal(
            0.0, 0.01, size=(len(self.groups), cfg.input_dim)
        )
        self.topology = self._build_hierarchical_topology()

    def _build_groups(self, n: int) -> List[List[int]]:
        return [
            list(range(start, min(start + self.group_size, n)))
            for start in range(0, n, self.group_size)
        ]

    def _build_hierarchical_topology(self) -> np.ndarray:
        n = self.num_kernels
        W = np.zeros((n, n))

        # Local communication inside each group: ring, not complete graph.
        for group in self.groups:
            if len(group) == 1:
                continue
            for idx, i in enumerate(group):
                left = group[(idx - 1) % len(group)]
                right = group[(idx + 1) % len(group)]
                if i != left:
                    W[i, left] = 0.10
                if i != right:
                    W[i, right] = 0.10

        reps = [group[0] for group in self.groups]
        if len(reps) <= 1:
            return W

        # Light ring between group representatives for neighboring domains.
        for idx, rep in enumerate(reps):
            nxt = reps[(idx + 1) % len(reps)]
            if rep != nxt:
                W[rep, nxt] = W[nxt, rep] = max(W[rep, nxt], 0.04)

        # 4-ary tree among representatives: O(groups) sparse hierarchy.
        for child_idx in range(1, len(reps)):
            parent_idx = (child_idx - 1) // self.group_size
            child = reps[child_idx]
            parent = reps[parent_idx]
            W[child, parent] = max(W[child, parent], 0.06)
            W[parent, child] = max(W[parent, child], 0.03)

        np.fill_diagonal(W, 0.0)
        return W

    def task_group(self, task_id: str | None) -> int | None:
        if task_id is None or not self.groups:
            return None
        domain = TASK_DOMAINS.get(task_id)
        if domain is None:
            return None
        return domain % len(self.groups)

    def route(self, input_vec: np.ndarray, task_id: str | None = None) -> np.ndarray:
        return self.route_batch(input_vec[None, :], task_id)[0]

    def route_batch(self, input_vecs: np.ndarray, task_id: str | None = None) -> np.ndarray:
        input_vecs = np.asarray(input_vecs, dtype=float)
        batch_size = input_vecs.shape[0]
        weights = np.zeros((batch_size, self.num_kernels))

        group_scores = input_vecs @ self.group_router.T
        preferred_group = self.task_group(task_id)
        if preferred_group is not None:
            group_scores[:, preferred_group] += 1.5
        group_weights = self._softmax_rows(group_scores)

        kernel_scores = input_vecs @ self.kernel_router.T
        for group_idx, group in enumerate(self.groups):
            local_scores = kernel_scores[:, group]
            local_weights = self._softmax_rows(local_scores)
            weights[:, group] = group_weights[:, group_idx, None] * local_weights
        return weights

    def group_assignments(self) -> Dict[int, List[int]]:
        return {idx: list(group) for idx, group in enumerate(self.groups)}

    def communication_edges(self) -> int:
        return int(np.count_nonzero(self.topology))

    def hierarchy_summary(self) -> Dict[str, object]:
        return {
            "architecture": "hierarchical",
            "group_size": self.group_size,
            "num_groups": len(self.groups),
            "groups": self.group_assignments(),
            "edges": self.communication_edges(),
        }

    def resource_usage(self) -> Dict[str, int]:
        usage = super().resource_usage()
        usage["group_router_params"] = int(self.group_router.size)
        usage["total_params"] += int(self.group_router.size)
        usage["communication_edges"] = self.communication_edges()
        return usage

    @staticmethod
    def _softmax_rows(scores: np.ndarray) -> np.ndarray:
        z = scores - np.max(scores, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / (np.sum(exp_z, axis=1, keepdims=True) + 1e-12)
