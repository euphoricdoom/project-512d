from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .constants import Config512D
from .kernel import rational_bound


# ============================================================
# BOUNDARY DETECTOR
# ============================================================

class BoundaryDetector:
    """Watches the system norm and triggers when energy or rate of change
    exceeds configured thresholds."""

    def __init__(self, cfg: Config512D):
        self.cfg = cfg
        self.prev_norm = 0.0

    def check(self, X: np.ndarray) -> Dict[str, object]:
        norm = float(np.linalg.norm(X))
        delta = abs(norm - self.prev_norm)
        high_energy = norm > self.cfg.boundary_theta
        rapid_change = delta > self.cfg.boundary_delta
        self.prev_norm = norm
        return {
            "trigger": high_energy or rapid_change,
            "norm": norm,
            "delta": delta,
            "reason": "energy" if high_energy else "change" if rapid_change else None,
        }


# ============================================================
# CONSTRAINT PROJECTOR
# ============================================================

class ConstraintProjector:
    """When the boundary detector fires, projects the state through the
    constraint subspace with damping to drain energy."""

    def __init__(self, cfg: Config512D):
        self.cfg = cfg

    def project(self, X: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        out = X.copy()
        out *= cfg.projection_damping
        return rational_bound(out)


# ============================================================
# TRACE LEDGER
# ============================================================

class TraceLedger:
    """Append-only record of per-step system observables."""

    def __init__(self, cfg: Config512D) -> None:
        self.cfg = cfg
        self.rows: List[Dict[str, object]] = []

    def record(
        self,
        t: int,
        X: np.ndarray,
        boundary: Dict[str, object],
        projected: bool,
        pred_loss: Optional[float] = None,
    ) -> None:
        self.rows.append({
            "t": t,
            "norm": float(np.linalg.norm(X)),
            "field_norm": float(np.linalg.norm(X[:self.cfg.dim_f])),
            "constraint_norm": float(np.linalg.norm(X[self.cfg.dim_f:])),
            "delta": float(boundary.get("delta", 0.0)),
            "triggered": bool(boundary.get("trigger", False)),
            "projected": projected,
            "reason": boundary.get("reason", None),
            "loss": float(pred_loss) if pred_loss is not None else None,
        })

    def summary(self) -> Dict[str, object]:
        if not self.rows:
            return {}
        norms = [float(r["norm"]) for r in self.rows]
        return {
            "steps": len(self.rows),
            "max_norm": max(norms),
            "avg_norm": float(np.mean(norms)),
            "final_norm": norms[-1],
            "triggers": sum(1 for r in self.rows if r["triggered"]),
            "projections": sum(1 for r in self.rows if r["projected"]),
        }
