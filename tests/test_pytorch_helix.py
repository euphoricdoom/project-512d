"""Contract tests for the torch-native Helix adapter.

The implementation is allowed to skip these tests until ``system.pytorch_helix``
lands, but the public constructor must stay aligned with the NumPy adapters:
``PyTorchHelix(input_dim, input_width=64, projection_dim=...)``.
"""

from __future__ import annotations

import inspect
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")
pytorch_helix = pytest.importorskip("system.pytorch_helix")


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _make_model(seed: int = 123):
    torch.manual_seed(seed)
    return pytorch_helix.PyTorchHelix(
        input_dim=5,
        input_width=3,
        projection_dim=4,
    )


def _gate_config():
    return {
        "write_gates": torch.eye(4, dtype=torch.float32),
        "update_weights": torch.eye(4, dtype=torch.float32),
        "decay_rates": torch.full((4,), 0.9, dtype=torch.float32),
    }


def _final_features(model):
    if hasattr(model, "final_features"):
        return model.final_features()
    return model.get_final_features()


def test_pytorch_helix_constructor_uses_shared_dimension_names():
    signature = inspect.signature(pytorch_helix.PyTorchHelix)

    assert "input_dim" in signature.parameters
    assert "input_width" in signature.parameters
    assert "projection_dim" in signature.parameters

    for legacy_name in ("ac_dim", "dc_dim"):
        if legacy_name in signature.parameters:
            assert signature.parameters[legacy_name].default is not inspect._empty

    model = _make_model()
    assert model.input_dim == 5
    assert model.input_width == 3
    assert model.projection_dim == 4


def test_pytorch_helix_keeps_fast_batched_feature_contract():
    model = _make_model(seed=7)
    model.reset(batch_size=2)

    inputs = torch.tensor(
        [
            [[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]],
            [[0.5, 1.0, 0.0], [-0.5, 0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    features = torch.arange(20, dtype=torch.float32).reshape(2, 2, 5) / 10.0

    for t in range(inputs.shape[1]):
        projected = model.step(
            inputs[:, t, :],
            features[:, t, :],
            _gate_config(),
            t=t,
            total_steps=inputs.shape[1],
        )

    final_features = _final_features(model)

    assert _as_numpy(projected).shape == (2, 4)
    assert _as_numpy(final_features).shape == (2, 27)
    assert np.isfinite(_as_numpy(final_features)).all()
    gate_trajectory = _as_numpy(model.get_gate_trajectory())
    assert np.all((0.0 <= gate_trajectory) & (gate_trajectory <= 1.0))


def test_pytorch_helix_seeded_torch_runs_are_deterministic():
    first = _make_model(seed=19)
    second = _make_model(seed=19)
    input_t = torch.tensor([[0.25, -0.5, 0.75]], dtype=torch.float32)
    features_t = torch.tensor([[0.2, -0.1, 0.4, -0.3, 0.6]], dtype=torch.float32)

    first.step(input_t, features_t, _gate_config(), t=0, total_steps=1)
    second.step(input_t, features_t, _gate_config(), t=0, total_steps=1)

    assert np.allclose(_as_numpy(_final_features(first)), _as_numpy(_final_features(second)))
