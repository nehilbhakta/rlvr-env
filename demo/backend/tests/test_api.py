"""Tests for the RLVR demo API.

Known scores (verified against the real env verifiers):
  baseline 2/20 (math-01, math-10 only), tiny 7/20 (code 5/10, math 2/10),
  oracle 20/20.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import app  # noqa: E402

client = TestClient(app)

EXPECTED = {"baseline": 2, "tiny": 7, "oracle": 20}


def test_tasks_endpoint_returns_20_tasks():
    r = client.get("/api/tasks")
    assert r.status_code == 200
    tasks = r.json()["tasks"]
    assert len(tasks) == 20
    assert [t["category"] for t in tasks].count("code") == 10
    assert [t["category"] for t in tasks].count("math") == 10
    for t in tasks:
        assert {"task_id", "category", "difficulty", "prompt"} <= set(t)
        # The expected math answer must never leak through this endpoint.
        assert "answer" not in t
        assert "reference_solution" not in t
        if t["category"] == "code":
            assert "function_name" in t and "test_count" in t


def test_health():
    assert client.get("/api/health").json() == {"status": "ok", "tasks": 20}


def test_unknown_agent_rejected():
    assert client.post("/api/evaluate", json={"agent": "gpt-5"}).status_code == 400


@pytest.mark.parametrize("agent,expected", list(EXPECTED.items()))
def test_agent_scores(agent, expected):
    r = client.post("/api/evaluate", json={"agent": agent}, timeout=300)
    assert r.status_code == 200
    data = r.json()
    assert data["agent"] == agent
    assert data["total"] == 20
    assert len(data["results"]) == 20
    assert data["passed"] == expected, f"{agent}: got {data['passed']}, expected {expected}"
    assert data["passed"] == data["code_passed"] + data["math_passed"]
    for row in data["results"]:
        assert row["reward"] in (0.0, 1.0)
        assert row["reward"] == (1.0 if row["passed"] else 0.0)
        assert isinstance(row["answer"], str)
        assert isinstance(row["detail"], str) and row["detail"]
