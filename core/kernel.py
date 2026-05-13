from __future__ import annotations

import numpy as np
from scipy.linalg import circulant

from .constants import Config512D


# ============================================================
# MATH HELPERS
# ============================================================

def rational_bound(x: np.ndarray) -> np.ndarray:
    """Elementwise containment: x / (1 + |x|). Smooth, differentiable, maps R -> (-1, 1)."""
    return x / (1.0 + np.abs(x))


def stable_tanh_bound(x: np.ndarray) -> np.ndarray:
    """Stronger containment via tanh. Preferred during training for gradient stability."""
    return np.tanh(x)


def spectral_radius(K: np.ndarray) -> float:
    """Largest absolute eigenvalue of K."""
    return float(np.max(np.abs(np.linalg.eigvals(K))))


def normalize_kernel(K: np.ndarray, target_radius: float) -> np.ndarray:
    """Rescale K so spectral radius <= target_radius. Preserves eigenvector structure."""
    r = spectral_radius(K)
    if r > target_radius and r > 0:
        return K * (target_radius / r)
    return K


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with temperature."""
    z = x / max(temperature, 1e-8)
    z = z - np.max(z)
    exp_z = np.exp(z)
    return exp_z / (np.sum(exp_z) + 1e-12)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


# ============================================================
# KERNEL
# ============================================================

class CoupledModularKernel:
    """Structured 512D kernel stored in factored implicit form.

    K = K_local  (diagonal: two scalars)
      + K_intra  (shared circulant template + per-module gain vector)
      + K_inter  (rank-r low-rank: U @ V.T, field subspace only)

    K is never materialized as a dense matrix during normal operation.
    compute(X) applies each component cheaply:
      K_local  — O(dim) elementwise multiply
      K_intra  — O(dim * log(module_size)) via per-module FFT convolution
      K_inter  — O(dim_f * rank) via two successive matmuls

    The .K property assembles the full (dim, dim) matrix on demand;
    use it only for tests and one-off analysis — not in the training loop.
    """

    def __init__(self, cfg: Config512D, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.frozen = False

        # K_local: two scalars, one per subspace
        self._lambda_f = cfg.lambda_f
        self._lambda_c = cfg.lambda_c

        # K_intra: one shared circulant template + per-module gain
        # template: (module_size,)  gains: (num_modules,)
        self._circ_template = self._build_circ_template()
        self._circ_gains = np.ones(cfg.num_modules)
        self._circ_freq = np.fft.rfft(self._circ_template)  # precomputed FFT

        # K_inter: rank-r low-rank factors, field subspace only
        # U: (dim_f, rank)  V: (dim_f, rank)  →  K_inter_field ≈ U @ V.T
        # RNG consumed identically to the original dense build, then SVD.
        self._U, self._V = self._build_inter_factors(rng)

        # Spectral scale: computed once from the assembled approximation.
        # Stored as a scalar so compute() never needs a full matrix.
        K_approx = self._assemble_unscaled()
        r = spectral_radius(K_approx)
        self._scale = (cfg.target_radius / r) if r > cfg.target_radius else 1.0
        self.radius = r * self._scale

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_circ_template(self) -> np.ndarray:
        cfg = self.cfg
        t = np.zeros(cfg.module_size)
        t[1]  = cfg.intra_coupling
        t[-1] = cfg.intra_coupling
        t[2]  = cfg.intra_coupling * 0.5
        t[-2] = cfg.intra_coupling * 0.5
        return t

    def _build_inter_factors(self, rng: np.random.Generator):
        """Build full K_inter field matrix (consuming RNG identically to the original
        dense method), then factor to rank cfg.inter_rank via truncated SVD."""
        cfg = self.cfg
        Kf = np.zeros((cfg.dim_f, cfg.dim_f))

        for i in range(cfg.num_modules):
            i0 = i * cfg.module_size
            i1 = i0 + cfg.module_size
            for j in range(cfg.num_modules):
                if i == j:
                    continue
                j0 = j * cfg.module_size
                j1 = j0 + cfg.module_size
                w = cfg.inter_coupling * np.exp(-abs(i - j) / 9.0)
                block = rng.normal(0.0, cfg.inter_mix_noise, (cfg.module_size, cfg.module_size))
                block += np.eye(cfg.module_size) * 0.35
                Kf[i0:i1, j0:j1] += w * block

        rank = cfg.inter_rank
        U_full, s, Vt = np.linalg.svd(Kf, full_matrices=False)
        sq_s = np.sqrt(s[:rank])
        U = U_full[:, :rank] * sq_s
        V = Vt[:rank, :].T * sq_s
        return U, V

    # ------------------------------------------------------------------
    # Matrix assembly (analysis / test use only)
    # ------------------------------------------------------------------

    def _assemble_unscaled(self) -> np.ndarray:
        """Build the full (dim, dim) K without the spectral scale factor.
        Called once at init to compute _scale; thereafter only via the .K property."""
        cfg = self.cfg
        K = np.zeros((cfg.dim, cfg.dim))

        # K_local
        np.fill_diagonal(K[:cfg.dim_f, :cfg.dim_f], self._lambda_f)
        np.fill_diagonal(K[cfg.dim_f:, cfg.dim_f:], self._lambda_c)

        # K_intra: shared circulant block scaled per module
        circ_block = circulant(self._circ_template)
        for i in range(cfg.num_modules):
            i0 = i * cfg.module_size
            i1 = i0 + cfg.module_size
            K[i0:i1, i0:i1] += self._circ_gains[i] * circ_block

        # K_inter: rank-r approximation
        K[:cfg.dim_f, :cfg.dim_f] += self._U @ self._V.T

        return K

    @property
    def K(self) -> np.ndarray:
        """Full assembled kernel matrix (dim, dim). Computed on demand.
        Do not call on the hot path — use compute() instead."""
        return self._assemble_unscaled() * self._scale

    # ------------------------------------------------------------------
    # Implicit application — hot path
    # ------------------------------------------------------------------

    def _apply_local(self, x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x)
        out[:self.cfg.dim_f] = self._lambda_f * x[:self.cfg.dim_f]
        out[self.cfg.dim_f:] = self._lambda_c * x[self.cfg.dim_f:]
        return out

    def _apply_local_batch(self, x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x)
        out[:, :self.cfg.dim_f] = self._lambda_f * x[:, :self.cfg.dim_f]
        out[:, self.cfg.dim_f:] = self._lambda_c * x[:, self.cfg.dim_f:]
        return out

    def _apply_intra(self, x: np.ndarray) -> np.ndarray:
        """Block-circulant apply via batched FFT across all modules at once."""
        cfg = self.cfg
        out = np.zeros(cfg.dim)
        xf_mat = x[:cfg.dim_f].reshape(cfg.num_modules, cfg.module_size)
        result = np.fft.irfft(
            np.fft.rfft(xf_mat, axis=1) * self._circ_freq, n=cfg.module_size, axis=1
        )
        out[:cfg.dim_f] = (result * self._circ_gains[:, None]).ravel()
        return out

    def _apply_intra_batch(self, x: np.ndarray) -> np.ndarray:
        """Block-circulant apply for a batch of states."""
        cfg = self.cfg
        out = np.zeros_like(x)
        xf_mat = x[:, :cfg.dim_f].reshape(-1, cfg.num_modules, cfg.module_size)
        result = np.fft.irfft(
            np.fft.rfft(xf_mat, axis=2) * self._circ_freq,
            n=cfg.module_size,
            axis=2,
        )
        out[:, :cfg.dim_f] = (result * self._circ_gains[None, :, None]).reshape(
            -1, cfg.dim_f
        )
        return out

    def _apply_inter(self, x: np.ndarray) -> np.ndarray:
        """Low-rank apply: U @ (V.T @ x_field). O(dim_f * rank)."""
        out = np.zeros(self.cfg.dim)
        out[:self.cfg.dim_f] = self._U @ (self._V.T @ x[:self.cfg.dim_f])
        return out

    def _apply_inter_batch(self, x: np.ndarray) -> np.ndarray:
        """Low-rank apply for a batch of states."""
        out = np.zeros_like(x)
        hidden = x[:, :self.cfg.dim_f] @ self._V
        out[:, :self.cfg.dim_f] = hidden @ self._U.T
        return out

    def compute(self, X: np.ndarray) -> np.ndarray:
        """Apply kernel via implicit factored form. No dense matmul on the hot path."""
        KX = (
            self._apply_local(X)
            + self._apply_intra(X)
            + self._apply_inter(X)
        ) * self._scale
        return rational_bound(KX)

    def compute_batch(self, X: np.ndarray) -> np.ndarray:
        """Apply the implicit kernel to a batch of states shaped (B, dim)."""
        X = np.asarray(X, dtype=float)
        assert X.ndim == 2 and X.shape[1] == self.cfg.dim
        KX = (
            self._apply_local_batch(X)
            + self._apply_intra_batch(X)
            + self._apply_inter_batch(X)
        ) * self._scale
        return rational_bound(KX)

    # ------------------------------------------------------------------
    # Learnable K_inter
    # ------------------------------------------------------------------

    def update_inter(self, x: np.ndarray, lr: float) -> None:
        """Hebbian update for the rank-r K_inter factors U, V.

        Rule: strengthen U[i,j] and V[k,j] when the hidden unit j
        co-activates with output i and input k.  Factors are soft-clipped
        after each update; call renormalize_inter() every N samples to
        restore the exact spectral bound.
        """
        if self.frozen:
            return
        xf = x[:self.cfg.dim_f]
        h = self._V.T @ xf          # (rank,) hidden activations
        y = self._U @ h             # (dim_f,) inter-kernel output
        self._U += lr * np.outer(y, h)
        self._V += lr * np.outer(xf, h)
        # lightweight containment between full renormalizations
        np.clip(self._U, -2.0, 2.0, out=self._U)
        np.clip(self._V, -2.0, 2.0, out=self._V)

    def renormalize_inter(self) -> None:
        """Recompute the global spectral scale from the current U, V.
        Call every cfg.inter_renorm_every training samples."""
        if self.frozen:
            return
        K_approx = self._assemble_unscaled()
        r = spectral_radius(K_approx)
        self._scale = (self.cfg.target_radius / r) if r > self.cfg.target_radius else 1.0
        self.radius = r * self._scale

    def freeze(self) -> None:
        """Disable kernel-parameter updates while leaving dynamics enabled."""
        self.frozen = True

    def unfreeze(self) -> None:
        """Re-enable kernel-parameter updates."""
        self.frozen = False

    def is_frozen(self) -> bool:
        """Return whether kernel-parameter updates are disabled."""
        return self.frozen


def freeze_all_kernels(kernels) -> None:
    """Freeze every kernel-like object in ``kernels``."""
    for kernel in kernels:
        if hasattr(kernel, "freeze"):
            kernel.freeze()


def unfreeze_all_kernels(kernels) -> None:
    """Unfreeze every kernel-like object in ``kernels``."""
    for kernel in kernels:
        if hasattr(kernel, "unfreeze"):
            kernel.unfreeze()
