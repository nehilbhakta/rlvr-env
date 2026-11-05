"""Verifiable reward functions.

The reward is 1.0 if and only if the submitted answer passes the automated
verifier, else 0.0. There is no model judgement, no partial credit, no
heuristic scoring — this is the "verifiable" part of RLVR.
"""

import re

NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_number(text):
    """Return the last number appearing in the text, or None."""
    matches = NUMBER_RE.findall(str(text))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def verify_math(task, action, tol=1e-6):
    """Exact-answer match for math tasks.

    Returns (passed: bool, detail: str).
    """
    submitted = _extract_number(action)
    if submitted is None:
        return False, "no number found in answer"
    expected = float(task["answer"])
    if abs(submitted - expected) <= tol:
        return True, f"answer {submitted} matches {expected}"
    return False, f"answer {submitted} != {expected}"


def verify_code(task, action):
    """Run hidden unit tests against the submitted code, in-process.

    The agent's code is exec'd in an isolated namespace; the required
    function is then called with each test's arguments. All tests must
    pass. Returns (passed: bool, detail: str).
    """
    namespace = {}
    try:
        exec(str(action), namespace)
    except Exception as exc:  # noqa: BLE001 - any submission error is a fail
        return False, f"code did not run: {type(exc).__name__}: {exc}"

    fn = namespace.get(task["function_name"])
    if fn is None or not callable(fn):
        return False, f"function {task['function_name']} not defined"

    failures = []
    for i, (args, expected) in enumerate(task["tests"]):
        try:
            got = fn(*args)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"test {i}: raised {type(exc).__name__}: {exc}")
            continue
        if got != expected:
            failures.append(f"test {i}: got {got!r}, expected {expected!r}")

    if failures:
        return False, f"{len(failures)}/{len(task['tests'])} tests failed: " + "; ".join(failures)
    return True, f"all {len(task['tests'])} tests passed"


def verify(task, action):
    """Dispatch to the category verifier. Returns (passed, detail)."""
    if task["category"] == "math":
        return verify_math(task, action)
    if task["category"] == "code":
        return verify_code(task, action)
    raise ValueError(f"unknown category: {task['category']}")
