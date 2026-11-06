"""Generate demo/frontend/public/recorded-results.json.

Baseline and oracle are computed by running the REAL env verifiers
(rlvr_env.rewards.verify) over the original 20 tasks, using the same
heuristic/oracle logic as evaluate.py and demo/backend/main.py.
The tiny model's rows come verbatim from
experiments/tiny-model/recorded_results_tiny.json (a real run of the
checkpoint through the env verifiers: 7/20, code 5/10, math 2/10).

Run once from the repo root with the demo venv:
    .venv/bin/python demo/backend/generate_recorded_results.py
"""

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rlvr_env.rewards import verify  # noqa: E402
from rlvr_env.tasks import TASKS  # noqa: E402

ARITH_RE = re.compile(r"(-?\d+(?:\.\d+)?\s*(?:[+\-*/^]\s*-?\d+(?:\.\d+)?\s*)+)")


def heuristic_agent(task):
    if task["category"] == "math":
        match = ARITH_RE.search(task["prompt"])
        if match:
            expr = match.group(1).replace("^", "**")
            try:
                return str(eval(expr, {"__builtins__": {}}))  # noqa: S307
            except Exception:
                return "0"
        return "0"
    return "def f():\n    return None\n"


def build(agent_name, answer_fn):
    results = []
    for task in TASKS:
        answer = answer_fn(task)
        passed, detail = verify(task, answer)
        results.append(
            {
                "task_id": task["id"],
                "category": task["category"],
                "prompt": task["prompt"],
                "answer": answer,
                "passed": bool(passed),
                "reward": 1.0 if passed else 0.0,
                "detail": detail,
            }
        )
    return {
        "agent": agent_name,
        "total": len(results),
        "passed": sum(1 for r in results if r["passed"]),
        "code_passed": sum(1 for r in results if r["passed"] and r["category"] == "code"),
        "math_passed": sum(1 for r in results if r["passed"] and r["category"] == "math"),
        "results": results,
    }


def main():
    tiny_src = json.loads(
        (REPO_ROOT / "experiments" / "tiny-model" / "recorded_results_tiny.json").read_text()
    )
    tiny_rows = []
    for row in tiny_src["tasks"]:
        passed = bool(row["passed"])
        tiny_rows.append(
            {
                "task_id": row["task_id"],
                "category": row["category"],
                "prompt": row["prompt"],
                "answer": row["model_answer"],
                "passed": passed,
                "reward": 1.0 if passed else 0.0,
                "detail": row["detail"],
            }
        )
    tiny = {
        "agent": "tiny",
        "total": len(tiny_rows),
        "passed": sum(1 for r in tiny_rows if r["passed"]),
        "code_passed": sum(1 for r in tiny_rows if r["passed"] and r["category"] == "code"),
        "math_passed": sum(1 for r in tiny_rows if r["passed"] and r["category"] == "math"),
        "results": tiny_rows,
    }
    assert tiny["passed"] == tiny_src["total"] == 7, tiny["passed"]
    assert tiny["code_passed"] == 5 and tiny["math_passed"] == 2

    baseline = build("baseline", heuristic_agent)
    oracle = build("oracle", lambda t: t["reference_solution"])
    print(f"baseline: {baseline['passed']}/20 (passes: {[r['task_id'] for r in baseline['results'] if r['passed']]})")
    print(f"tiny:     {tiny['passed']}/20")
    print(f"oracle:   {oracle['passed']}/20")
    assert baseline["passed"] == 2, baseline["passed"]
    assert oracle["passed"] == 20, oracle["passed"]

    tasks_public = []
    for t in TASKS:
        out = {"task_id": t["id"], "category": t["category"], "difficulty": t["difficulty"], "prompt": t["prompt"]}
        if t["category"] == "code":
            out["function_name"] = t["function_name"]
            out["test_count"] = len(t["tests"])
        tasks_public.append(out)

    payload = {
        "note": "Recorded results from a real run of the RLVR verifiers. Not a live run — use the FastAPI backend for live evaluation.",
        "agents": {"baseline": baseline, "tiny": tiny, "oracle": oracle},
        "tasks": tasks_public,
    }
    out_path = REPO_ROOT / "demo" / "frontend" / "public" / "recorded-results.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
