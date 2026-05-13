"""Tests for SystemMemory._phase — validates the corrected sinusoidal encoding.

The original implementation had a divide-by-zero-masked bug: it computed
  angle = 2π * (T-1) / (T-1)
which collapses to 2π (→ sin≈0, cos≈1) for every T > 1, making phase
indistinguishable across sequence lengths. This test suite enforces the
corrected behavior.
"""

import numpy as np
import pytest

from weightless_model import SystemMemory


@pytest.fixture
def mem():
    return SystemMemory(reservoir_dim=4)


class TestPhaseEncoding:
    def test_single_timestep_is_neutral(self, mem):
        """T=1 phase should use the no-temporal-structure sentinel [0, 0, 1]."""
        phase = mem._phase(T=1, batch=1)[0]
        assert phase[0] == pytest.approx(0.0)
        assert phase[1] == pytest.approx(0.0)
        assert phase[2] == pytest.approx(1.0)

    def test_different_lengths_produce_different_phases(self, mem):
        """Sequences of different lengths must produce distinct phase codes."""
        phases = [mem._phase(T, 1)[0] for T in range(1, 13)]
        # All must be mutually distinct (regression test for the '/ (T-1)' bug)
        for i, pi in enumerate(phases):
            for j, pj in enumerate(phases):
                if i != j:
                    assert not np.allclose(pi, pj), (
                        f"Phase for T={i+1} and T={j+1} are identical: {pi}"
                    )

    def test_batch_dimension_tiles_correctly(self, mem):
        """Phase must be the same across all samples in a batch."""
        phase = mem._phase(T=6, batch=8)
        assert phase.shape == (8, 3)
        for i in range(1, 8):
            np.testing.assert_array_equal(phase[i], phase[0])

    def test_third_component_increases_with_length(self, mem):
        """Normalised position (third component) must increase toward 1 as T grows."""
        pos = [mem._phase(T, 1)[0, 2] for T in range(2, 20)]
        assert all(a < b for a, b in zip(pos, pos[1:])), (
            "Normalised position must be strictly increasing with T"
        )

    def test_third_component_approaches_one(self, mem):
        """Normalised position should approach 1 for large T."""
        p_large = mem._phase(T=1000, batch=1)[0, 2]
        assert p_large == pytest.approx(1.0, abs=0.002)

    def test_phase_components_are_finite(self, mem):
        """Phase must always be finite (no NaN / Inf)."""
        for T in [1, 2, 5, 6, 10, 100, 1000]:
            phase = mem._phase(T, 3)
            assert np.all(np.isfinite(phase)), f"Non-finite phase for T={T}: {phase}"

    def test_sin_cos_on_unit_circle(self, mem):
        """sin²+cos² must equal 1 for any T (up to float precision)."""
        for T in [2, 6, 10, 50, 100]:
            phase = mem._phase(T, 1)[0]
            norm = phase[0] ** 2 + phase[1] ** 2
            assert norm == pytest.approx(1.0, abs=1e-10), (
                f"sin²+cos²={norm} ≠ 1 for T={T}"
            )

    def test_process_sequence_feature_dim(self, mem):
        """End-to-end: process_sequence must produce the right feature width."""
        X = np.random.default_rng(0).normal(size=(3, 6, 4))
        feats = mem.process_sequence(X)
        assert feats.shape == (3, 4 * 6 + 3)
