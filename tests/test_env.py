"""Tests for the RLVR environment, verifiers, and task bank."""

import pytest

from rlvr_env import RLVREnv, TASKS, get_task, list_tasks
from rlvr_env.rewards import verify, verify_code, verify_math


def test_task_bank_has_20_tasks():
    assert len(TASKS) == 20
    assert len(list_tasks("code")) == 10
    assert len(list_tasks("math")) == 10


def test_task_ids_unique():
    ids = [t["id"] for t in TASKS]
    assert len(ids) == len(set(ids))


def test_all_reference_solutions_pass():
    """Every reference solution must pass its own verifier (sanity check)."""
    for task in TASKS:
        passed, detail = verify(task, task["reference_solution"])
        assert passed, f"{task['id']} reference failed: {detail}"


def test_math_verifier_accepts_and_rejects():
    task = get_task("math-01")  # 17 * 23 + 41 = 432
    assert verify_math(task, "432")[0] is True
    assert verify_math(task, "The answer is 432")[0] is True
    assert verify_math(task, "431")[0] is False
    assert verify_math(task, "no number here")[0] is False


def test_code_verifier_accepts_and_rejects():
    task = get_task("code-01")  # add(a, b)
    assert verify_code(task, "def add(a, b):\n    return a + b\n")[0] is True
    assert verify_code(task, "def add(a, b):\n    return a - b\n")[0] is False
    assert verify_code(task, "def wrong(a, b):\n    return a + b\n")[0] is False
    assert verify_code(task, "this is not python(")[0] is False


def test_env_episode_flow():
    env = RLVREnv()
    obs, info = env.reset(options={"task_id": "math-02"})  # $40 shirt, 25% off -> 30
    assert obs["task_id"] == "math-02"
    assert obs["category"] == "math"
    _, reward, terminated, truncated, info = env.step("30")
    assert reward == 1.0
    assert terminated is True
    assert truncated is False
    assert info["passed"] is True
    with pytest.raises(RuntimeError):
        env.step("30")  # episode already ended


def test_env_wrong_answer_scores_zero():
    env = RLVREnv()
    env.reset(options={"task_id": "code-03"})  # reverse_string
    _, reward, terminated, _, info = env.step("def reverse_string(s):\n    return s\n")
    assert reward == 0.0
    assert terminated is True
    assert info["passed"] is False
