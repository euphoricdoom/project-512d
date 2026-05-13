"""Tests for task-specific projection heads."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from system.task_specific_projection import TaskSpecificProjection


def _make_projection(
    *,
    num_kernels: int = 2,
    base_feature_dim: int = 6,
    projection_dim: int = 12,
    output_dim: int = 3,
    base_lr: float = 0.05,
    grad_clip: float = 10.0,
    seed: int = 123,
) -> TaskSpecificProjection:
    np.random.seed(seed)
    return TaskSpecificProjection(
        num_kernels=num_kernels,
        base_feature_dim=base_feature_dim,
        projection_dim=projection_dim,
        output_dim=output_dim,
        base_lr=base_lr,
        grad_clip=grad_clip,
    )


def _set_task(model: TaskSpecificProjection, task_id: str) -> None:
    if hasattr(model, "set_task"):
        model.set_task(task_id)
    elif hasattr(model, "add_task"):
        model.add_task(task_id)
    else:
        model.predict(np.zeros(_total_feature_dim(model)), task_id=task_id)


def _total_feature_dim(model: TaskSpecificProjection) -> int:
    if hasattr(model, "total_feature_dim"):
        return int(model.total_feature_dim)
    return int(model.num_kernels * model.base_feature_dim)


def _task_projection(model: TaskSpecificProjection, task_id: str) -> np.ndarray:
    for name in ("projections", "task_projections", "projection_matrices", "projection_weights"):
        if hasattr(model, name):
            value = getattr(model, name)
            if isinstance(value, dict):
                return np.asarray(value[task_id])
            return np.asarray(value)
    raise AssertionError("TaskSpecificProjection must expose task-specific projection matrices")


def _task_readout(model: TaskSpecificProjection, task_id: str) -> np.ndarray:
    for name in ("readouts", "weights", "output_weights", "heads"):
        if hasattr(model, name):
            value = getattr(model, name)
            if isinstance(value, dict):
                return np.asarray(value[task_id])
            return np.asarray(value)
    raise AssertionError("TaskSpecificProjection must expose task-specific output weights")


def _task_state(model: TaskSpecificProjection, task_id: str) -> tuple[np.ndarray, np.ndarray]:
    return _task_projection(model, task_id).copy(), _task_readout(model, task_id).copy()


def _update_error(
    model: TaskSpecificProjection,
    features: np.ndarray,
    error: np.ndarray,
    *,
    task_id: str,
    lr: float | None = None,
) -> None:
    kwargs = {"task_id": task_id}
    if lr is not None:
        kwargs["lr"] = lr
    try:
        model.update(features, error, as_error=True, **kwargs)
    except TypeError:
        model.update(features, error, **kwargs)


def _update_batch_error(
    model: TaskSpecificProjection,
    features: np.ndarray,
    errors: np.ndarray,
    *,
    task_id: str,
    lr: float | None = None,
) -> None:
    kwargs = {"task_id": task_id}
    if lr is not None:
        kwargs["lr"] = lr
    try:
        model.update_batch(features, errors, as_error=True, **kwargs)
    except TypeError:
        model.update_batch(features, errors, **kwargs)


def _mse(model: TaskSpecificProjection, features: np.ndarray, targets: np.ndarray, task_id: str) -> float:
    pred = model.predict_batch(features, task_id=task_id)
    return float(np.mean((targets - pred) ** 2))


def test_initialization_creates_expected_shapes_and_info():
    model = _make_projection(num_kernels=3, base_feature_dim=5, projection_dim=7, output_dim=4)
    _set_task(model, "shape")

    assert model.num_kernels == 3
    assert model.base_feature_dim == 5
    assert model.projection_dim == 7
    assert model.output_dim == 4
    assert _total_feature_dim(model) == 15
    assert _task_projection(model, "shape").shape == (7, 15)
    assert _task_readout(model, "shape").shape == (4, 7)

    pred = model.predict(np.linspace(-1.0, 1.0, 15), task_id="shape")
    batch_pred = model.predict_batch(np.zeros((2, 15)), task_id="shape")
    assert pred.shape == (4,)
    assert batch_pred.shape == (2, 4)
    assert np.all(np.isfinite(pred))

    if hasattr(model, "stored_params"):
        assert model.stored_params() >= 7 * 15 + 4 * 7


def test_task_specific_projections_are_isolated_and_distinct():
    rng = np.random.default_rng(10)
    model = _make_projection(seed=11)
    features = rng.normal(size=_total_feature_dim(model))

    _set_task(model, "edge")
    _set_task(model, "smooth")

    edge_projection = _task_projection(model, "edge")
    smooth_projection = _task_projection(model, "smooth")
    assert not np.allclose(edge_projection, smooth_projection)

    edge_pred = model.predict(features, task_id="edge")
    smooth_pred = model.predict(features, task_id="smooth")
    assert edge_pred.shape == smooth_pred.shape == (model.output_dim,)
    assert not np.allclose(edge_pred, smooth_pred)


def test_learning_reduces_loss_on_fixed_linear_task():
    rng = np.random.default_rng(20)
    model = _make_projection(
        num_kernels=2,
        base_feature_dim=5,
        projection_dim=10,
        output_dim=3,
        base_lr=0.08,
        grad_clip=5.0,
        seed=21,
    )
    features = rng.normal(size=(24, _total_feature_dim(model)))
    teacher = rng.normal(scale=0.25, size=(_total_feature_dim(model), model.output_dim))
    targets = np.tanh(features @ teacher)
    _set_task(model, "linear")

    initial_loss = _mse(model, features, targets, "linear")
    for _ in range(80):
        pred = model.predict_batch(features, task_id="linear")
        _update_batch_error(model, features, targets - pred, task_id="linear")
    final_loss = _mse(model, features, targets, "linear")

    assert np.isfinite(final_loss)
    assert final_loss < initial_loss * 0.75


def test_zero_interference_between_tasks():
    rng = np.random.default_rng(30)
    model = _make_projection(seed=31)
    features = rng.normal(size=(8, _total_feature_dim(model)))
    errors = rng.normal(size=(8, model.output_dim))
    _set_task(model, "old")
    _set_task(model, "new")
    old_before = _task_state(model, "old")
    new_before = _task_state(model, "new")

    _update_batch_error(model, features, errors, task_id="new")

    old_after = _task_state(model, "old")
    new_after = _task_state(model, "new")
    assert np.allclose(old_after[0], old_before[0])
    assert np.allclose(old_after[1], old_before[1])
    assert not np.allclose(new_after[0], new_before[0])
    assert not np.allclose(new_after[1], new_before[1])


def test_projection_gradient_update_changes_projection_matrix():
    rng = np.random.default_rng(40)
    model = _make_projection(base_lr=0.04, seed=41)
    features = rng.normal(size=_total_feature_dim(model))
    error = rng.normal(size=model.output_dim)
    _set_task(model, "grad")
    projection_before = _task_projection(model, "grad").copy()

    _update_error(model, features, error, task_id="grad")

    projection_after = _task_projection(model, "grad")
    delta_norm = np.linalg.norm(projection_after - projection_before)
    assert np.isfinite(delta_norm)
    assert delta_norm > 0.0
    assert np.all(np.isfinite(projection_after))


def test_batched_updates_match_batch_prediction_shapes_and_learn():
    rng = np.random.default_rng(50)
    model = _make_projection(
        num_kernels=3,
        base_feature_dim=4,
        projection_dim=9,
        output_dim=2,
        base_lr=0.05,
        seed=51,
    )
    features = rng.normal(size=(10, _total_feature_dim(model)))
    teacher = rng.normal(scale=0.2, size=(_total_feature_dim(model), model.output_dim))
    targets = features @ teacher
    _set_task(model, "batch")

    before = _mse(model, features, targets, "batch")
    pred = model.predict_batch(features, task_id="batch")
    _update_batch_error(model, features, targets - pred, task_id="batch")
    after = _mse(model, features, targets, "batch")

    assert pred.shape == targets.shape
    assert np.isfinite(after)
    assert after < before


@pytest.mark.parametrize("projection_dim", [4, 8, 16])
def test_different_projection_dims(projection_dim):
    rng = np.random.default_rng(60 + projection_dim)
    model = _make_projection(
        num_kernels=2,
        base_feature_dim=5,
        projection_dim=projection_dim,
        output_dim=3,
        seed=61 + projection_dim,
    )
    features = rng.normal(size=(6, _total_feature_dim(model)))
    errors = rng.normal(size=(6, model.output_dim))
    _set_task(model, f"dim-{projection_dim}")

    pred = model.predict_batch(features, task_id=f"dim-{projection_dim}")
    projection = _task_projection(model, f"dim-{projection_dim}")
    readout = _task_readout(model, f"dim-{projection_dim}")
    _update_batch_error(model, features, errors, task_id=f"dim-{projection_dim}")

    assert pred.shape == (6, 3)
    assert projection.shape == (projection_dim, 10)
    assert readout.shape == (3, projection_dim)
