# RLVR Environment — Verifiable Rewards for ML/Coding/Math Tasks

A small, real reinforcement-learning environment where the reward comes
**only** from an automated verifier — unit tests for code tasks, exact-answer
matching for math tasks. No LLM calls, no model judgement, no fake scores.

This is the core idea behind RLVR (reinforcement learning with verifiable
rewards): when a task has a checkable answer, you don't need a learned reward
model — you run the tests. That is what frontier labs use to teach models to
do ML engineering, and what this environment demonstrates at small scale.

## Tasks

20 tasks in `rlvr_env/tasks.py`:

| Category | Count | Verification |
| --- | --- | --- |
| Code | 10 | Hidden pytest-style unit tests run against the submitted function |
| Math | 10 | Exact numeric answer match (tolerance 1e-6) |

Code tasks: `add`, `is_prime`, `reverse_string`, `fizzbuzz_value`,
`factorial`, `fibonacci`, `count_vowels`, `merge_sorted`, `binary_search`,
`flatten`. Math tasks: arithmetic, percentages, linear equations, sums,
speed, GCD, Pythagoras, powers.

## How it works

```python
from rlvr_env import RLVREnv

env = RLVREnv()
obs, _ = env.reset(options={"task_id": "code-02"})
# obs = {"task_id": "code-02", "prompt": "Write a function `is_prime(n)` ...",
#        "category": "code", "difficulty": "easy"}

obs, reward, terminated, truncated, info = env.step(
    "def is_prime(n):\n    if n < 2:\n        return False\n    ..."
)
# reward == 1.0 only if ALL hidden tests pass, else 0.0
# info = {"task_id": ..., "category": ..., "passed": ..., "detail": ...}
```

One episode = one task. The agent submits one answer; the episode ends.
`env.py` follows the Gymnasium API (`reset` / `step`, `observation_space` /
`action_space`), so any Gymnasium-compatible training loop can use it.

## Run it

```bash
pip install -r requirements.txt
python -m pytest        # 7 tests
python evaluate.py      # baseline + oracle pass rates
```

Example output (`python evaluate.py`):

```
heuristic baseline: 2/20 passed (10%)
  code: 0/10 passed
  math: 2/10 passed

oracle (reference solutions): 20/20 passed (100%)
  code: 10/10 passed
  math: 10/10 passed
```

The heuristic baseline only evaluates a bare arithmetic expression from the
prompt, so it solves plain arithmetic and nothing else — and it can't write
code at all. The oracle submits each task's reference solution and must score
100%; if it ever doesn't, a verifier or a reference solution is broken.

## Files

- `rlvr_env/env.py` — Gymnasium-style env (`reset`/`step`, reward 1.0/0.0)
- `rlvr_env/tasks.py` — the 20-task bank with prompts, tests, answers, references
- `rlvr_env/rewards.py` — the verifiers (unit-test runner, numeric matcher)
- `evaluate.py` — heuristic baseline + oracle evaluation, pass rates per category
- `tests/test_env.py` — pytest suite

## Experiment: tiny model trained on this env

A 299,136-parameter decoder-only transformer trained from scratch on
generated variants of these 20 tasks lives in
[`experiments/tiny-model/`](experiments/tiny-model/) — 4,197 training
examples, 45 epochs, CPU-only, ~35 minutes, training loss 3.98 → 0.21.
No RL pass was run.

| Agent | Code | Math | Total |
| --- | --- | --- | --- |
| Heuristic baseline | 0/10 | 2/10 | **2/20** |
| Tiny model | 5/10 | 2/10 | **7/20** |
| Oracle | 10/10 | 10/10 | **20/20** |

On 210 held-out prompts with numbers never seen in training, the honest
picture is mixed: baseline 40/210, tiny model 17/210. The baseline's 40
come entirely from the two bare-arithmetic templates (math-01, math-10),
where its regex calculator is perfect (20/20 each) and the model scores
0/20 — it never learned general arithmetic. Where the model actually wins
on held-out data is code (5/10 vs 0/10). Its GCD score (11/20) is not GCD
learned: it answers "1" to nearly every GCD prompt and passes only when
the GCD actually is 1. Full results, per-example outputs, and the "what it
cannot do" list are in
[`experiments/tiny-model/README.md`](experiments/tiny-model/README.md).

## Honest limitations

- **A model was trained on it — once, small, supervised.**
  `experiments/tiny-model/` is a from-scratch SFT run, not an RL run. A
  next step would be wiring the env to a real RL loop (e.g. GRPO
  fine-tuning of a small model with TRL) using the verifiers as the
  reward — the env's Gymnasium API is designed for that.
- **Binary rewards.** A submission is 1.0 or 0.0; there is no partial credit
  for code that passes most tests. That is deliberate — verifiable rewards
  are sharp — but it makes the reward signal sparse.
- **Code is executed in-process** with `exec`. The tasks are tiny and the
  tests are fast, but this is not a sandbox: do not run untrusted agent code
  this way in production. A production version would run submissions in a
  container with a timeout.
- **Single-turn episodes.** The agent gets one shot per task; there is no
  multi-turn interaction, tool use, or iterative debugging.
- **Small task bank.** 20 hand-written tasks demonstrate the pattern; a
  production RLVR setup needs hundreds of tasks with held-out test sets to
  prevent overfitting to the verifier.
