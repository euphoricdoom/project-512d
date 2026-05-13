from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.linalg import eig

from system.modular_system import ModularFieldSystem


# ============================================================
# EIGENMODE ANALYZER
# ============================================================

class EigenmodeAnalyzer:
    """Complete spectral decomposition and interrogation of a kernel matrix.

    Provides:
    - Full eigendecomposition (sorted by magnitude)
    - Per-mode analysis (phase, field/constraint energy split, localization)
    - Lyapunov exponent spectrum
    - Energy landscape and convergence time estimates
    - Mode participation ratio (effective dimensionality)
    - Pure eigenmode evolution simulation
    """

    def __init__(self, K: np.ndarray) -> None:
        self.K = K
        self.eigenvalues: Optional[np.ndarray] = None
        self.eigenvectors: Optional[np.ndarray] = None
        self._dim = K.shape[0]
        self._dim_f = 480  # field space boundary

    # ------------------------------------------------------------------
    # Decomposition
    # ------------------------------------------------------------------

    def decompose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Full eigendecomposition sorted by descending |lambda|."""
        print("Decomposing kernel into eigenmodes...")
        self.eigenvalues, self.eigenvectors = eig(self.K)

        idx = np.argsort(np.abs(self.eigenvalues))[::-1]
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = self.eigenvectors[:, idx]

        total = len(self.eigenvalues)
        stable   = int(np.sum(np.abs(self.eigenvalues) < 1))
        critical = int(np.sum(np.abs(self.eigenvalues) == 1))
        unstable = int(np.sum(np.abs(self.eigenvalues) > 1))
        complex_count = int(np.sum(np.abs(self.eigenvalues.imag) > 1e-10))

        print(f"  Total modes:    {total}")
        print(f"  Stable   |λ|<1: {stable}")
        print(f"  Critical |λ|=1: {critical}")
        print(f"  Unstable |λ|>1: {unstable}")
        print(f"  Complex modes:  {complex_count}")
        print(f"  Spectral radius: {np.max(np.abs(self.eigenvalues)):.6f}")

        return self.eigenvalues, self.eigenvectors

    def _ensure_decomposed(self) -> None:
        if self.eigenvalues is None:
            self.decompose()

    # ------------------------------------------------------------------
    # Per-mode analysis
    # ------------------------------------------------------------------

    def analyze_mode(self, mode_idx: int) -> Tuple[complex, np.ndarray]:
        """Print detailed analysis of a single eigenmode."""
        self._ensure_decomposed()
        lam = self.eigenvalues[mode_idx]
        v = self.eigenvectors[:, mode_idx]

        print(f"\nMode {mode_idx}:")
        print(f"  Eigenvalue: {lam.real:.6f} + {lam.imag:.6f}j")
        print(f"  Magnitude:  {np.abs(lam):.6f}")
        print(f"  Phase:      {np.angle(lam):.6f} rad")

        field_energy      = float(np.linalg.norm(v[:self._dim_f]))
        constraint_energy = float(np.linalg.norm(v[self._dim_f:]))
        print(f"  Field energy:      {field_energy:.6f}")
        print(f"  Constraint energy: {constraint_energy:.6f}")
        ratio = field_energy / constraint_energy if constraint_energy > 1e-10 else float("inf")
        print(f"  F/C ratio:         {ratio:.2f}")

        magnitudes = np.abs(v)
        top5 = np.argsort(magnitudes)[::-1][:5]
        print("  Top 5 active dims:")
        for d in top5:
            space = "Field" if d < self._dim_f else "Constraint"
            print(f"    dim {d:4d} ({space}): {magnitudes[d]:.6f}")

        # Participation ratio
        v_abs = np.abs(v)
        participation = (np.sum(v_abs**2)**2) / (np.sum(v_abs**4) + 1e-30)
        print(f"  Participation ratio: {participation:.1f} dims")

        return lam, v

    # ------------------------------------------------------------------
    # Lyapunov spectrum
    # ------------------------------------------------------------------

    def lyapunov_exponents(self, n_modes: int = 20) -> np.ndarray:
        """Lyapunov exponents = log|lambda| for each mode."""
        self._ensure_decomposed()
        return np.log(np.abs(self.eigenvalues[:n_modes]) + 1e-30)

    # ------------------------------------------------------------------
    # Energy landscape
    # ------------------------------------------------------------------

    def energy_landscape(self) -> Dict[str, object]:
        """Characterize the global attractor structure."""
        self._ensure_decomposed()
        lam_max = float(np.max(np.abs(self.eigenvalues)))
        closest_critical_idx = int(np.argmin(np.abs(self.eigenvalues - 1.0)))
        lam_critical = self.eigenvalues[closest_critical_idx]

        time_to_half = (
            np.log(0.5) / np.log(lam_max) if lam_max < 1.0 else float("inf")
        )

        print(f"  Slowest mode |λ|:    {lam_max:.6f}")
        print(f"  Nearest critical λ:  {lam_critical:.6f} (dist {abs(lam_critical - 1.0):.2e})")
        print(f"  Time to half energy: ~{time_to_half:.1f} steps")

        return {
            "lam_max": lam_max,
            "lam_critical": lam_critical,
            "time_to_half": time_to_half,
        }

    # ------------------------------------------------------------------
    # Effective dimensionality
    # ------------------------------------------------------------------

    def participation_ratio(self) -> Dict[str, object]:
        """Fraction of eigenspectrum carrying meaningful dynamics."""
        self._ensure_decomposed()
        lam_mag = np.abs(self.eigenvalues)
        ratio = (np.sum(lam_mag**2)**2) / (np.sum(lam_mag**4) + 1e-30)

        cumsum = np.cumsum(lam_mag**2) / (np.sum(lam_mag**2) + 1e-30)
        n_90 = int(np.argmax(cumsum > 0.90)) + 1
        n_95 = int(np.argmax(cumsum > 0.95)) + 1
        n_99 = int(np.argmax(cumsum > 0.99)) + 1

        total = self._dim
        print(f"  Participation ratio:   {ratio:.1f} / {total} dims")
        print(f"  Modes for 90% dynamics: {n_90}")
        print(f"  Modes for 95% dynamics: {n_95}")
        print(f"  Modes for 99% dynamics: {n_99}")

        return {"ratio": ratio, "cumsum": cumsum, "n_90": n_90, "n_95": n_95, "n_99": n_99}

    # ------------------------------------------------------------------
    # Degeneracy analysis
    # ------------------------------------------------------------------

    def degeneracy_analysis(self) -> Dict[str, object]:
        """Find degenerate eigenspaces (repeated eigenvalues)."""
        self._ensure_decomposed()
        unique: list = []
        counts: list = []
        for lam in self.eigenvalues:
            matched = False
            for k, u in enumerate(unique):
                if abs(lam - u) < 1e-6:
                    counts[k] += 1
                    matched = True
                    break
            if not matched:
                unique.append(lam)
                counts.append(1)

        max_deg = max(counts)
        print(f"  Unique eigenvalues: {len(unique)}")
        print(f"  Largest degeneracy: {max_deg}")

        top5_idx = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)[:5]
        print("  Top 5 degenerate eigenvalues:")
        for i in top5_idx:
            print(f"    λ = {unique[i]:.6f}, degeneracy = {counts[i]}")

        return {"unique": unique, "counts": counts}

    # ------------------------------------------------------------------
    # Pure eigenmode simulation
    # ------------------------------------------------------------------

    def simulate_mode(self, mode_idx: int, steps: int = 100) -> Tuple[list, list]:
        """Simulate how a pure eigenmode decays vs theoretical prediction."""
        self._ensure_decomposed()
        lam = self.eigenvalues[mode_idx]
        v = self.eigenvectors[:, mode_idx].real

        X = v.copy()
        simulated = []
        for _ in range(steps):
            simulated.append(float(np.linalg.norm(X)))
            X = self.K @ X

        theoretical = [float(np.abs(lam)**t * np.linalg.norm(v)) for t in range(steps)]
        return simulated, theoretical


# ============================================================
# QUICK SUMMARY FUNCTION
# ============================================================

def eigenmode_report(system: ModularFieldSystem) -> Dict[str, object]:
    """Fast eigenvalue summary without full EigenmodeAnalyzer overhead."""
    eigvals = np.linalg.eigvals(system.kernel.K)
    complex_count = int(np.sum(np.abs(eigvals.imag) > 1e-10))
    return {
        "spectral_radius": float(np.max(np.abs(eigvals))),
        "real_modes":      int(len(eigvals) - complex_count),
        "complex_modes":   complex_count,
        "top_abs":         np.sort(np.abs(eigvals))[::-1][:10],
    }
