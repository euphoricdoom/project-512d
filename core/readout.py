from __future__ import annotations

import numpy as np

from .constants import Config512D


def module_features(x: np.ndarray, cfg: Config512D) -> np.ndarray:
    """Extract module norms plus the module-activation constraint scoreboard."""
    module_norms = np.linalg.norm(
        x[:cfg.dim_f].reshape(cfg.num_modules, cfg.module_size), axis=1
    )
    return np.concatenate([module_norms, x[cfg.dim_f:]])


def module_features_batch(x: np.ndarray, cfg: Config512D) -> np.ndarray:
    """Batch version of module_features for states shaped (B, dim)."""
    x = np.asarray(x, dtype=float)
    assert x.ndim == 2 and x.shape[1] == cfg.dim
    module_norms = np.linalg.norm(
        x[:, :cfg.dim_f].reshape(-1, cfg.num_modules, cfg.module_size), axis=2
    )
    return np.concatenate([module_norms, x[:, cfg.dim_f:]], axis=1)
