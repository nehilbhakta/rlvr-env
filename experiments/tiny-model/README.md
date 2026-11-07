# Experiment: tiny model trained on this env

A tiny decoder-only transformer (299,136 parameters) trained from scratch
on generated variants of this environment's 20 tasks — no pretraining, no
fine-tuned LLM. It exists to answer one question: can a small model actually
learn on an RLVR environment whose only reward signal is the verifier?

## Results

### The env's own 20 tasks

| Agent | Code | Math | Total |
| --- | --- | --- | --- |
| Heuristic baseline (`evaluate.py`) | 0/10 | 2/10 | **2/20** |
| Tiny model (this experiment) | 5/10 | 2/10 | **7/20** |
| Oracle (reference solutions) | 10/10 | 10/10 | **20/20** |

The model writes 5 of the 10 functions correctly from the prompt alone
(`add`, `reverse_string`, `factorial`, `fibonacci`, `count_vowels`) and
solves 2 of the 10 original math tasks.

### Held-out variants — 210 fresh prompts, numbers never seen in training

| Agent | Code | Math | Total |
| --- | --- | --- | --- |
| Heuristic baseline | 0/10 | 40/200 | **40/210** |
| Tiny model | 5/10 | 12/200 | **17/210** |

Read this table carefully — it is the honest one. The baseline's 40 math
passes come entirely from the two bare-arithmetic templates (math-01 and
math-10), where a regex that evaluates the expression in the prompt is
perfect (20/20 each). The tiny model scores 0/20 on both: it never learned
general arithmetic. Where the model actually wins on held-out data is code
(5/10 vs 0/10).

On GCD (math-08) it scores 11/20, but that is **not** GCD learned: the model
answers "1" to nearly every GCD prompt, and passes only when the GCD
actually is 1. Say it that way — do not call it "partially learned".

Full per-example outputs are in `results.json`.

## GRPO reinforcement-learning pass (2026-11-06) — did not improve the model

A real GRPO pass was run on top of the SFT checkpoint (`train_grpo.py`):
group size 8, temperature 1.0, clipped surrogate (eps 0.2), KL penalty
(beta 0.04, k3 estimator) against the frozen SFT reference, binary
verifier reward — 1.0 iff the sampled answer passes
`rlvr_env.rewards.verify`, else 0.0. Prompts were fresh train-split math
variants (never the held-out ranges) plus the 10 fixed code prompts, 16
prompts per step (12 math + 4 code). Two runs: the first at lr 1e-5 was
stopped after 35 steps — sampled reward flat (0.113 → 0.109) and KL only
0.002, so the policy was barely moving. The second ran 80 steps at lr
5e-5 (cosine decay), 47.4 minutes on a 4-core CPU, KL settling around
0.012–0.02 nats. No divergence, no NaN.

Results, greedy decoding, same protocols as the SFT evals:

| Eval | SFT | RL (GRPO) | Baseline |
| --- | --- | --- | --- |
| Env's own 20 tasks | **7/20** (code 5/10, math 2/10) | **6/20** (code 5/10, math 1/10) | 2/20 |
| Held-out 210 variants | **17/210** (code 5/10, math 12/200) | **17/210** (code 5/10, math 12/200) | 40/210 |

The held-out per-task counts are identical in every cell. The fixed-20
regression is one task: math-01, which the SFT model answered correctly
(432) and the RL model now answers wrong (412). Sampled reward during
training stayed flat (~0.11–0.13 mean) and the dev pass rate stayed 4/10
at every 10-step check.

The policy did move — 86 of the 210 held-out greedy answers changed —
but toward modal wrong answers ("100", "10", "1") rather than correct
ones. RL sharpened the model's prior modes instead of finding correct
answers.

Why it didn't help, honestly:

1. **Sparse binary rewards.** Only ~4–5 of 16 prompt groups per step had
   any reward variance. On tasks the model never solves (math-01,
   math-10), every sample scores 0, so the advantage is 0 and there is no
   gradient at all. RL cannot teach arithmetic the model never samples
   correctly.
2. **The model is at its capability ceiling on these templates.** SFT
   already extracted what this 299K model can do; RL can only
   redistribute probability among answers the model already produces.
3. **The KL penalty capped drift near the SFT reference**, so the policy
   moved but stayed close.

So the SFT checkpoint stays the model in the demo
(`checkpoints/model.pt` / `model.pt.b64`). The RL checkpoint
(`checkpoints/model-rl.pt.b64`) is kept only for reproducibility, and
nothing here claims an RL improvement. The environment itself — verifiable
rewards, oracle/baseline, honest evals — is the durable artifact; the
tiny-model experiment's value is showing exactly where a small model
stops learning.

## What it cannot do

- **It was trained on variants of these same 20 task templates.** It shows
  a small model learning this task family from verifiable rewards, not
  general reasoning. Give it a task outside these 20 templates and it will
  fail.
- **It did not learn general arithmetic.** Two-digit multiplication,
  percentages, linear equations and Pythagorean triples are beyond what
  this 0.3M-parameter model learned in 45 epochs. A regex calculator still
  beats it on bare arithmetic expressions.
- **An RL pass was run and did not help.** See the GRPO section above:
  80 steps at lr 5e-5 moved the policy (KL ~0.02) but held-out stayed
  17/210 and the fixed-20 score went 7/20 → 6/20. The SFT checkpoint
  remains the demo model.

## How it was trained

- **Data** (`data.py`): for each math template, variants generated with
  random parameters and answers computed programmatically. Training and
  held-out sets use disjoint parameter ranges, so no held-out math prompt
  appears in training (verified: 0/200 overlap). Code targets are the
  reference solutions from `rlvr_env.tasks`. 4,197 training examples.
- **Model** (`model.py`): decoder-only transformer, character-level,
  d_model=96, 2 layers, 4 heads, 299,136 parameters.
- **Training** (`train.py`): supervised fine-tuning — the prompt is context,
  loss is only on the solution tokens. AdamW, cosine schedule, CPU-only,
  4 cores. 45 epochs total: 25 at lr 5e-4, then 20 more at lr 5e-5.
  Training loss 3.98 → 0.21. Total compute ~35 minutes.
- **Evaluation** (`evaluate_trained.py`): greedy decoding, scored by
  `rlvr_env.rewards.verify` — the exact same verifiers the environment
  uses. The heuristic baseline is scored on the same held-out variants.

## Reproduce

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install gymnasium
cd checkpoints && python decode_weights.py && cd ..   # weights are stored as base64 text
python evaluate_trained.py   # scores the checkpoint on held-out variants
python train.py              # retrains from scratch on CPU (~35 min)
python train_grpo.py         # GRPO pass on the SFT checkpoint (~50 min, CPU)
python eval_fixed20.py       # scores a checkpoint on the env's 20 original tasks
```

`train.py` defaults (d_model=128, 3 layers, lr 3e-4, 20 epochs) are not the
exact config of the published checkpoint (d_model=96, 2 layers, 45 epochs
in two phases); the checkpoint's own config is stored in `model.pt` and is
what `evaluate_trained.py` uses.

## Files

- `train.py` — training loop (with `--resume` and `--save-every`)
- `train_grpo.py` — GRPO reinforcement-learning pass on the SFT checkpoint
  (writes `checkpoints/model-rl.pt`; the SFT checkpoint is never overwritten)
- `model.py` — tiny transformer
- `data.py` — variant generators (train/test splits)
- `evaluate_trained.py` — held-out evaluation with the env verifiers
- `eval_fixed20.py` — greedy evaluation on the env's 20 original tasks
- `results.json` — full SFT evaluation output (per-example)
- `results-rl.json` — full RL evaluation output, held-out 210 (per-example)
- `results-rl-fixed20.json` — RL evaluation on the env's 20 original tasks
- `results-sft-fixed20.json` — SFT evaluation on the env's 20 original tasks
  (re-run with `eval_fixed20.py` for a clean before/after)
- `training_curve.json` / `training_curve.csv` — loss (and dev pass rate) per epoch
- `grpo_curve.json` — per-step reward, KL and dev pass rate for the 80-step
  GRPO run at lr 5e-5; `grpo_curve_lr1e-5.json` is the aborted 35-step run at
  lr 1e-5 (policy barely moved)
- `checkpoints/model.pt.b64` — trained weights, base64-encoded (1.2 MB decoded);
  run `checkpoints/decode_weights.py` to get `model.pt`
- `checkpoints/model-rl.pt.b64` — GRPO checkpoint (80 steps, lr 5e-5),
  base64-encoded; kept for reproducibility only — it scores worse than the
  SFT checkpoint (6/20 vs 7/20) and is NOT the demo model
- `checkpoints/decode_weights.py` — decodes the weights
