from __future__ import annotations

from typing import List, Tuple

import numpy as np

from core.constants import Config512D
from core.kernel import CoupledModularKernel, spectral_radius, normalize_kernel


class StabilityAnalysis:
    """Map the stability boundary as a function of inter-module coupling strength.

    Tests a range of coupling values, computing the spectral radius at each
    point to find the maximum coupling before the system becomes unstable
    (spectral radius >= 1).
    """

    def __init__(self, base_cfg: Config512D) -> None:
        self.base_cfg = base_cfg

    def sweep_coupling(
        self,
        low: float = 0.001,
        high: float = 0.020,
        n_points: int = 10,
    ) -> Tuple[np.ndarray, List[float]]:
        """Sweep inter_coupling from low to high, measure spectral radius at each."""
        from dataclasses import replace

        coupling_values = np.linspace(low, high, n_points)
        radii: List[float] = []

        print(f"  {'Coupling':>10}  {'Spectral radius':>16}  {'Stable?':>8}")
        print("  " + "-" * 40)

        for coupling in coupling_values:
            cfg = replace(self.base_cfg, inter_coupling=float(coupling))
            rng = np.random.default_rng(cfg.seed)
            kernel = CoupledModularKernel(cfg, rng)
            radius = spectral_radius(kernel.K)
            radii.append(radius)
            stable = "yes" if radius < 1.0 else "NO"
            print(f"  {coupling:>10.4f}  {radius:>16.6f}  {stable:>8}")

        stable_couplings = [c for c, r in zip(coupling_values, radii) if r < 1.0]
        if stable_couplings:
            print(f"\n  Maximum stable inter_coupling: {max(stable_couplings):.4f}")
        else:
            print("\n  No stable coupling found in range.")

        return coupling_values, radii

    def condition_number(self, coupling: float) -> float:
        """Kernel condition number kappa = |lambda_max| / |lambda_min|.

        Large kappa means small input perturbations cause large output swings.
        """
        from dataclasses import replace
        cfg = replace(self.base_cfg, inter_coupling=coupling)
        rng = np.random.default_rng(cfg.seed)
        kernel = CoupledModularKernel(cfg, rng)
        eigvals = np.abs(np.linalg.eigvals(kernel.K))
        kappa = float(np.max(eigvals) / (np.min(eigvals) + 1e-30))
        print(f"  Condition number at coupling={coupling:.4f}: kappa = {kappa:.2f}")
        return kappa
