"""Evaluate agents on the RLVR task bank.

Two agents, both with no LLM and no access to the verifiers' internals:

* heuristic — a rule-based baseline. For math tasks it extracts the
  arithmetic expression from the prompt and evaluates it with Python
  (so it solves plain arithmetic, not algebra or word problems). For
  code tasks it submits a stub that returns None, so it scores 0 —
  writing code needs a real model.
* oracle — submits each task's reference solution. It should score
  100%; anything less means a verifier or a reference solution is
  broken, not that the task is hard.

Usage: python evaluate.py
"""

import re

from rlvr_env import RLVREnv, TASKS

ARITH_RE = re.compile(r"(-?\d+(?:\.\d+)?\s*(?:[+\-*/^]\s*-?\d+(?:\.\d+)?\s*)+)")


def heuristic_agent(obs):
    """Rule-based baseline: evaluates a bare arithmetic expression if the
    prompt contains one, otherwise returns a stub."""
    if obs["category"] == "math":
        match = ARITH_RE.search(obs["prompt"])
        if match:
            expr = match.group(1).replace("^", "**")
            try:
                return str(eval(expr, {"__builtins__": {}}))  # noqa: S307 - numbers only, from our own prompts
            except Exception:
                return "0"
        return "0"
    # Code tasks: a stub that defines the function but returns None.
    fn = {"code": None}
    return "def f():\n    return None\n"


def oracle_agent(obs):
    for task in TASKS:
        if task["id"] == obs["task_id"]:
            return task["reference_solution"]
    raise KeyError(obs["task_id"])


def run_agent(agent, name):
    env = RLVREnv()
    results = {"code": [], "math": []}
    for task in TASKS:
        obs, _ = env.reset(options={"task_id": task["id"]})
        _, reward, _, _, info = env.step(agent(obs))
        results[task["category"]].append((task["id"], bool(reward), info["detail"]))
    return name, results


def report(name, results):
    total_pass = sum(p for cat in results.values() for _, p, _ in cat)
    total = sum(len(v) for v in results.values())
    print(f"\n{name}: {total_pass}/{total} passed ({100 * total_pass / total:.0f}%)")
    for category, rows in results.items():
        passed = sum(p for _, p, _ in rows)
        print(f"  {category}: {passed}/{len(rows)} passed")
        for task_id, p, detail in rows:
            mark = "PASS" if p else "FAIL"
            print(f"    [{mark}] {task_id}: {detail}")


def main():
    for agent, name in [(heuristic_agent, "heuristic baseline"), (oracle_agent, "oracle (reference solutions)")]:
        name_, results = run_agent(agent, name)
        report(name_, results)


if __name__ == "__main__":
    main()
