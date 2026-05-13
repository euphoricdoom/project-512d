"""Fast gate distillation tests for learned Helix gates."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

adaptive_helix = pytest.importorskip("system.adaptive_helix")
helix_temporal = pytest.importorskip("system.helix_temporal")


def _make_teacher(seed: int = 5):
    return helix_temporal.HelixTemporalAdapter(
        input_dim=5,
        input_width=3,
        projection_dim=4,
        ac_decay=0.2,
        dc_decay=0.7,
        seed=seed,
    )


def _make_student(seed: int = 11):
    return adaptive_helix.AdaptiveHelixTemporalAdapter(
        input_dim=5,
        input_width=3,
        projection_dim=4,
        ac_decay=0.2,
        dc_decay=0.7,
        seed=seed,
        diagnostics=True,
    )


def _toy_sequence():
    inputs = np.array(
        [
            [[1.0, 0.0, 0.5], [0.0, 1.0, -0.5], [0.5, 0.0, 1.0]],
            [[0.0, 1.0, 0.25], [1.0, 0.0, 0.0], [-0.5, 1.0, 0.5]],
        ],
        dtype=float,
    )
    features = np.array(
        [
            [[0.2, -0.1, 0.4, 0.0, 0.5], [0.1, 0.3, -0.2, 0.4, -0.1], [0.5, 0.0, 0.2, -0.3, 0.1]],
            [[-0.4, 0.2, 0.1, 0.3, 0.0], [0.0, -0.2, 0.6, -0.1, 0.2], [0.3, 0.1, -0.5, 0.2, 0.4]],
        ],
        dtype=float,
    )
    return inputs, features


def _teacher_gate_targets(task_id: str = "adding_task"):
    inputs, features = _toy_sequence()
    teacher = _make_teacher()
    teacher.reset(batch_size=inputs.shape[0])
    targets = []
    for t in range(inputs.shape[1]):
        teacher.step(
            inputs[:, t, :],
            features[:, t, :],
            t=t,
            total_steps=inputs.shape[1],
            task_id=task_id,
        )
        targets.append(teacher.last_write_gate.copy())
    return inputs, features, targets


def _trajectory_loss(student, inputs, features, targets, task_id: str):
    losses = []
    student.reset(batch_size=inputs.shape[0])
    for t, target in enumerate(targets):
        student.step(
            inputs[:, t, :],
            features[:, t, :],
            t=t,
            total_steps=inputs.shape[1],
            task_id=None,
        )
        losses.append(student.gate_controller.loss(inputs[:, t, :], student.ac_state, target, task_id))
    return float(np.mean(losses))


def test_controller_distills_explicit_helix_gate_trajectory():
    inputs, features, targets = _teacher_gate_targets()
    student = _make_student()
    task_id = "distilled_adding"

    before = _trajectory_loss(student, inputs, features, targets, task_id)
    for _ in range(16):
        student.reset(batch_size=inputs.shape[0])
        for t, target in enumerate(targets):
            student.step(
                inputs[:, t, :],
                features[:, t, :],
                t=t,
                total_steps=inputs.shape[1],
                task_id=None,
            )
            metrics = student.update_gates(inputs[:, t, :], student.ac_state, target, task_id)
    after = _trajectory_loss(student, inputs, features, targets, task_id)

    assert set(metrics) >= {"loss", "gate_mean"}
    assert np.isfinite(before)
    assert np.isfinite(after)
    assert after < before * 0.95


def test_distilled_student_keeps_helix_feature_contract_without_task_channels():
    inputs, features, targets = _teacher_gate_targets(task_id="copy_task")
    student = _make_student(seed=23)
    task_id = "distilled_copy"

    for _ in range(8):
        student.reset(batch_size=inputs.shape[0])
        for t, target in enumerate(targets):
            student.step(
                inputs[:, t, :],
                features[:, t, :],
                t=t,
                total_steps=inputs.shape[1],
                task_id=None,
            )
            student.update_gates(inputs[:, t, :], student.ac_state, target, task_id)

    student.reset(batch_size=inputs.shape[0])
    for t in range(inputs.shape[1]):
        projected = student.step(
            inputs[:, t, :],
            features[:, t, :],
            t=t,
            total_steps=inputs.shape[1],
            task_id=None,
        )

    final_features = student.final_features()
    diagnostics = student.get_diagnostics()

    assert projected.shape == (2, 4)
    assert final_features.shape == (2, student.feature_dim)
    assert student.feature_dim == 27
    assert np.isfinite(final_features).all()
    assert diagnostics["adapter_type"] == "adaptive_helix"
    assert diagnostics["gate_trajectory"].shape == (3, 2, 4)


def test_v2_cached_features_are_stable_and_input_informative():
    learned_gates = pytest.importorskip("system.learned_gates")
    distill_v2 = pytest.importorskip("system.distillation_trainer_v2")

    gate = learned_gates.LearnedGateNetwork(
        input_dim=3,
        task_embedding_dim=16,
        num_dc_channels=6,
        feature_dim=6,
    )
    trainer = distill_v2.DistillationTrainerV2(
        gate,
        input_width=3,
        feature_dim=6,
        k_support=2,
        seed=99,
    )
    inputs, _ = _toy_sequence()
    first = trainer.make_feature_stream(inputs, task_id="adding_task")
    second = trainer.make_feature_stream(inputs, task_id="adding_task")

    assert np.allclose(first, second)
    assert np.allclose(first[:, :, :3], inputs)
    assert np.allclose(first[:, :, 3:6], np.cumsum(inputs, axis=1))
