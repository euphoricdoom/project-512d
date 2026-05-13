"""Phase 2 contract tests for learned temporal gates.

These tests intentionally target the planned ``system.learned_gates`` API. They
skip while the module is absent, then become executable API checks as soon as the
implementation lands.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

@pytest.fixture
def learned_gates():
    return pytest.importorskip("system.learned_gates")


def _make_controller(learned_gates, seed=123):
    cfg = learned_gates.LearnedGateConfig(
        input_width=3,
        state_dim=4,
        learning_rate=0.2,
        seed=seed,
    )
    return learned_gates.LearnedGateController(cfg)


def test_gate_forward_is_batched_bounded_and_deterministic(learned_gates):
    first = _make_controller(learned_gates, seed=7)
    second = _make_controller(learned_gates, seed=7)
    input_t = np.array([[0.0, 1.0, -1.0], [0.5, 0.0, 0.25]])
    ac_state = np.array(
        [[0.1, -0.2, 0.3, -0.4], [0.0, 0.25, -0.5, 0.75]]
    )

    first_gate = first.forward(input_t, ac_state, task_id="copy_task")
    second_gate = second.forward(input_t, ac_state, task_id="copy_task")

    assert first_gate.shape == (2, 4)
    assert np.all(np.isfinite(first_gate))
    assert np.all((0.0 <= first_gate) & (first_gate <= 1.0))
    assert np.allclose(first_gate, second_gate)


def test_supervised_update_reduces_gate_target_loss(learned_gates):
    controller = _make_controller(learned_gates, seed=11)
    input_t = np.array([[1.0, 0.0, 0.5], [-0.5, 1.0, 0.0]])
    ac_state = np.array([[0.2, -0.1, 0.4, 0.0], [0.0, 0.3, -0.2, 0.1]])
    target_gate = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])

    before = controller.loss(input_t, ac_state, target_gate, task_id="adding_task")
    for _ in range(8):
        metrics = controller.update(input_t, ac_state, target_gate, task_id="adding_task")
    after = controller.loss(input_t, ac_state, target_gate, task_id="adding_task")

    assert set(metrics) >= {"loss", "gate_mean"}
    assert np.isfinite(metrics["loss"])
    assert after < before


def test_state_dict_round_trip_preserves_outputs_without_aliasing(learned_gates):
    controller = _make_controller(learned_gates, seed=19)
    input_t = np.array([[0.25, -0.5, 0.75]])
    ac_state = np.array([[0.4, 0.3, -0.2, -0.1]])
    target_gate = np.ones((1, 4))
    controller.update(input_t, ac_state, target_gate, task_id="parity_task")

    restored = _make_controller(learned_gates, seed=999)
    restored.load_state_dict(controller.state_dict())

    assert np.allclose(
        controller.forward(input_t, ac_state, task_id="parity_task"),
        restored.forward(input_t, ac_state, task_id="parity_task"),
    )

    state = controller.state_dict()
    state["weights"][0, 0] += 10.0
    assert np.allclose(
        controller.forward(input_t, ac_state, task_id="parity_task"),
        restored.forward(input_t, ac_state, task_id="parity_task"),
    )
