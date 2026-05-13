from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import networkx as nx

from system.modular_system import ModularFieldSystem


# ============================================================
# MODULE-LEVEL INFORMATION FLOW
# ============================================================

class InformationFlowAnalysis:
    """Measure how perturbations propagate through the module graph."""

    def __init__(self, system: ModularFieldSystem) -> None:
        self.system = system
        self.cfg = system.cfg

    # ------------------------------------------------------------------
    # Propagation speed
    # ------------------------------------------------------------------

    def measure_propagation_speed(self, n_trials: int = 10) -> List[float]:
        """Inject a unit impulse into a random module and measure spread rate."""
        cfg = self.cfg
        speeds = []

        for _ in range(n_trials):
            module_id = np.random.randint(0, cfg.num_modules)
            self.system.state = np.zeros(cfg.dim)
            self.system.state[module_id * cfg.module_size] = 1.0

            activation_history = []
            for _ in range(50):
                self.system.step()
                acts = np.zeros(cfg.num_modules)
                for i in range(cfg.num_modules):
                    s = i * cfg.module_size
                    acts[i] = np.linalg.norm(self.system.state[s:s + cfg.module_size])
                activation_history.append(acts)

            activation_history = np.array(activation_history)
            active_counts = np.sum(activation_history > 0.01, axis=1)

            if len(active_counts) > 10:
                speed = (active_counts[25] - active_counts[5]) / 20.0
                speeds.append(float(speed))

        avg = float(np.mean(speeds))
        std = float(np.std(speeds))
        print(f"  Average propagation speed: {avg:.3f} modules/step (std {std:.3f})")
        if avg > 0:
            print(f"  Estimated saturation:      ~{cfg.num_modules / avg:.1f} steps")
        return speeds

    # ------------------------------------------------------------------
    # Propagation wavefront (first-crossing times)
    # ------------------------------------------------------------------

    def propagation_wavefront(
        self, source_module: int = 0, steps: int = 80
    ) -> Dict[str, object]:
        """Record first-activation timestep for every module from a source impulse."""
        cfg = self.cfg
        self.system.reset_state(scale=0.0)
        src = self.system.modules[source_module]
        self.system.state[src.dims] = np.ones(cfg.module_size) * 0.8

        activation_history: List[np.ndarray] = []
        first_crossing = np.full(cfg.num_modules, -1, dtype=int)

        for t in range(steps):
            acts = self.system.module_activations()
            activation_history.append(acts)
            for m in range(cfg.num_modules):
                if first_crossing[m] == -1 and acts[m] > cfg.propagation_threshold:
                    first_crossing[m] = t
            self.system.step(t=t)

        history = np.array(activation_history)
        reached = np.where(first_crossing >= 0)[0]

        if len(reached) > 1:
            distances = np.abs(reached - source_module)
            times = first_crossing[reached]
            valid = times > 0
            speed = float(np.mean(distances[valid] / times[valid])) if np.any(valid) else 0.0
        else:
            speed = 0.0

        return {
            "history": history,
            "first_crossing": first_crossing,
            "reached_count": int(len(reached)),
            "avg_speed_modules_per_step": speed,
            "max_activation": float(np.max(history)),
        }

    # ------------------------------------------------------------------
    # Module influence matrix
    # ------------------------------------------------------------------

    def module_influence_matrix(self) -> np.ndarray:
        """Aggregate the inter-coupling kernel to a (num_modules, num_modules) matrix."""
        cfg = self.cfg
        K_inter = self.system.kernel._U @ self.system.kernel._V.T
        influence = np.zeros((cfg.num_modules, cfg.num_modules))
        for i in range(cfg.num_modules):
            for j in range(cfg.num_modules):
                i_sl = slice(i * cfg.module_size, (i + 1) * cfg.module_size)
                j_sl = slice(j * cfg.module_size, (j + 1) * cfg.module_size)
                influence[i, j] = float(np.sum(K_inter[j_sl, i_sl]))
        return influence

    def report_influence(self) -> np.ndarray:
        """Print top influential and receptive modules and return the matrix."""
        influence = self.module_influence_matrix()
        outgoing = np.sum(influence, axis=1)
        incoming = np.sum(influence, axis=0)

        print("  Top 5 influential (outgoing):")
        for idx in np.argsort(outgoing)[::-1][:5]:
            print(f"    Module {idx:02d} ({self.system.modules[idx].expertise}): {outgoing[idx]:.6f}")

        print("  Top 5 receptive (incoming):")
        for idx in np.argsort(incoming)[::-1][:5]:
            print(f"    Module {idx:02d} ({self.system.modules[idx].expertise}): {incoming[idx]:.6f}")

        self._pagerank_report(influence)
        return influence

    def _pagerank_report(self, influence: np.ndarray) -> None:
        threshold = np.percentile(influence[influence > 0], 90) if np.any(influence > 0) else 0
        G = nx.DiGraph()
        for i in range(self.cfg.num_modules):
            G.add_node(i)
        for i in range(self.cfg.num_modules):
            for j in range(self.cfg.num_modules):
                if influence[i, j] > threshold:
                    G.add_edge(i, j, weight=float(influence[i, j]))

        print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (top 10%)")
        try:
            pr = nx.pagerank(G)
            top5 = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:5]
            print("  Top 5 central modules (PageRank):")
            for mid, score in top5:
                print(f"    Module {mid:02d} ({self.system.modules[mid].expertise}): {score:.6f}")
        except Exception:
            print("  (PageRank unavailable for this graph)")


# ============================================================
# STANDALONE PROPAGATION FUNCTION
# ============================================================

def analyze_propagation(
    system: ModularFieldSystem, source_module: int = 0, steps: int = 80
) -> Dict[str, object]:
    """Convenience wrapper around InformationFlowAnalysis.propagation_wavefront."""
    return InformationFlowAnalysis(system).propagation_wavefront(source_module, steps)
