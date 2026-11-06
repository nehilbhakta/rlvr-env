"""Generate training / held-out variants of the 20 RLVR task templates.

Code tasks: prompts are fixed (the env's prompts); the target is the
reference solution from rlvr_env.tasks. Held-out evaluation for code uses
the same 10 prompts scored by the env's own hidden unit tests.

Math tasks: prompts are generated from the same templates with different
numbers. Training and held-out sets use DISJOINT parameter ranges
(split="train" vs split="test"), so no held-out math prompt can appear in
training — even for templates whose total prompt space is small.

Every generated example is a dict:
    {"task_id", "category", "prompt", "target", "answer"}  (answer: math only)
"""

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "rlvr-env"))
from rlvr_env.tasks import CODE_TASKS, MATH_TASKS  # noqa: E402

# ---------------------------------------------------------------------------
# Math variant generators: each takes (rng, split) and returns (prompt, answer)
# split="train" and split="test" draw from disjoint parameter ranges.
# ---------------------------------------------------------------------------


def _v_math_01(rng, split):
    lo, hi = (11, 40) if split == "train" else (41, 99)
    a, b = rng.randint(lo, hi), rng.randint(lo, hi)
    c = rng.randint(10, 99)
    return f"What is {a} * {b} + {c}? Give only the number.", float(a * b + c)


def _v_math_02(rng, split):
    if split == "train":
        price = rng.choice(list(range(20, 401, 20)))
        disc = rng.choice([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 75])
    else:
        price = rng.choice(list(range(420, 901, 20)))
        disc = rng.choice([5, 15, 35, 45, 55, 65, 70, 80, 85, 90, 95])
    ans = price * (100 - disc) / 100
    return (
        f"A shirt costs ${price} and is discounted by {disc}%. "
        f"What is the sale price in dollars? Give only the number.",
        float(ans),
    )


def _v_math_03(rng, split):
    a = rng.randint(2, 9)
    x = rng.randint(2, 12) if split == "train" else rng.randint(13, 25)
    b = rng.randint(1, 20)
    c = a * x + b
    return f"Solve for x: {a}x + {b} = {c}. Give only the number.", float(x)


def _v_math_04(rng, split):
    n = rng.randint(5, 60) if split == "train" else rng.randint(61, 200)
    return (
        f"What is the sum of the first {n} positive integers "
        f"(1 + 2 + ... + {n})? Give only the number.",
        float(n * (n + 1) // 2),
    )


def _v_math_05(rng, split):
    if split == "train":
        t = rng.choice([2, 2.5, 4, 5])
        speed = rng.randint(10, 80)
    else:
        t = rng.choice([1.5, 3, 3.5, 6, 8, 10])
        speed = rng.randint(81, 120)
    dist = speed * t
    dist_s = str(int(dist)) if float(dist).is_integer() else str(dist)
    return (
        f"A train travels {dist_s} km in {t} hours. "
        f"What is its average speed in km/h? Give only the number.",
        float(speed),
    )


def _v_math_06(rng, split):
    if split == "train":
        pct = rng.choice(list(range(5, 100, 5)))
        n = rng.choice(list(range(20, 601, 20)))
    else:
        pct = rng.choice(list(range(5, 100, 5)))
        n = rng.choice(list(range(620, 1001, 20)))
    return f"What is {pct}% of {n}? Give only the number.", float(pct * n / 100)


def _v_math_07(rng, split):
    # a(x - b) = c*x - d  ->  x = (a*b - d) / (a - c); pick a, c, b, x, derive d.
    # Keep d positive so the prompt reads cleanly.
    for _ in range(100):
        a = rng.randint(2, 6)
        c = rng.randint(1, 9)
        while c == a:
            c = rng.randint(1, 9)
        b = rng.randint(2, 9)
        x = rng.randint(2, 9) if split == "train" else rng.randint(10, 15)
        d = a * b - (a - c) * x
        if d > 0:
            return f"Solve for x: {a}(x - {b}) = {c}x - {d}. Give only the number.", float(x)
    raise RuntimeError("could not sample a clean math-07 variant")


def _v_math_08(rng, split):
    lo, hi = (12, 200) if split == "train" else (201, 400)
    a = rng.randint(lo, hi)
    b = rng.randint(lo, hi)
    return (
        f"What is the greatest common divisor (GCD) of {a} and {b}? "
        f"Give only the number.",
        float(math.gcd(a, b)),
    )


def _v_math_09(rng, split):
    # Pythagorean triple via Euclid's formula so the answer is an integer.
    m_hi = 10 if split == "train" else 15
    m_lo = 2 if split == "train" else 11
    m = rng.randint(m_lo, m_hi)
    n = rng.randint(1, m - 1)
    a = m * m - n * n
    b = 2 * m * n
    c = m * m + n * n
    if rng.random() < 0.5:
        a, b = b, a
    return (
        f"A rectangle has length {a} and width {b}. "
        f"What is the length of its diagonal? Give only the number.",
        float(c),
    )


def _v_math_10(rng, split):
    a = rng.randint(3, 8) if split == "train" else rng.randint(9, 14)
    b = rng.randint(1, a - 1)
    return f"What is 2^{a} - 2^{b}? Give only the number.", float(2**a - 2**b)


MATH_GENERATORS = {
    "math-01": _v_math_01,
    "math-02": _v_math_02,
    "math-03": _v_math_03,
    "math-04": _v_math_04,
    "math-05": _v_math_05,
    "math-06": _v_math_06,
    "math-07": _v_math_07,
    "math-08": _v_math_08,
    "math-09": _v_math_09,
    "math-10": _v_math_10,
}

# Original fixed tasks (the env's own prompts/answers), for the baseline check.
ORIGINAL = {t["id"]: t for t in CODE_TASKS + MATH_TASKS}


def fmt_answer(ans: float) -> str:
    """Format a numeric answer the way the reference solutions do."""
    return str(int(ans)) if float(ans).is_integer() else str(ans)


def build_math_variants(task_id, n, seed, split):
    rng = random.Random(seed)
    gen = MATH_GENERATORS[task_id]
    out, seen, attempts = [], set(), 0
    while len(out) < n and attempts < n * 200 + 1000:
        attempts += 1
        prompt, ans = gen(rng, split)
        if prompt in seen:
            continue
        seen.add(prompt)
        out.append(
            {
                "task_id": task_id,
                "category": "math",
                "prompt": prompt,
                "target": fmt_answer(ans),
                "answer": ans,
            }
        )
    return out


def build_code_examples(n_copies):
    out = []
    for task in CODE_TASKS:
        for _ in range(n_copies):
            out.append(
                {
                    "task_id": task["id"],
                    "category": "code",
                    "prompt": task["prompt"],
                    "target": task["reference_solution"],
                    "answer": None,
                }
            )
    return out


def build_train_set(math_per_task=400, code_copies=150, seed=1234):
    """Training set: math variants from the train ranges + repeated code prompts."""
    examples = build_code_examples(code_copies)
    for task_id in MATH_GENERATORS:
        examples.extend(build_math_variants(task_id, math_per_task, seed=seed, split="train"))
    random.Random(seed).shuffle(examples)
    return examples


def build_heldout_set(math_per_task=20, seed=987654):
    """Held-out evaluation set: math variants from the TEST ranges (disjoint
    from training) + the 10 fixed code tasks, scored by the hidden unit tests.
    """
    examples = []
    for task in CODE_TASKS:
        examples.append(
            {
                "task_id": task["id"],
                "category": "code",
                "prompt": task["prompt"],
                "target": task["reference_solution"],
                "answer": None,
                "tests": task["tests"],
                "function_name": task["function_name"],
            }
        )
    for task_id in MATH_GENERATORS:
        examples.extend(build_math_variants(task_id, math_per_task, seed=seed, split="test"))
    return examples


if __name__ == "__main__":
    train = build_train_set()
    held = build_heldout_set()
    print(f"train: {len(train)} examples")
    print(f"held-out: {len(held)} examples")
    train_prompts = {e["prompt"] for e in train if e["category"] == "math"}
    overlap = [
        e["prompt"] for e in held if e["category"] == "math" and e["prompt"] in train_prompts
    ]
    print(f"held-out math prompts overlapping train: {len(overlap)}")
    from collections import Counter

    print("train per task:", dict(Counter(e["task_id"] for e in train)))
    print("held-out per task:", dict(Counter(e["task_id"] for e in held)))
    for e in train[:1] + held[:1]:
        print("---", e["task_id"], "|", e["prompt"][:80], "->", repr(e["target"][:60]))
