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

## What it cannot do

- **It was trained on variants of these same 20 task templates.** It shows
  a small model learning this task family from verifiable rewards, not
  general reasoning. Give it a task outside these 20 templates and it will
  fail.
- **It did not learn general arithmetic.** Two-digit multiplication,
  percentages, linear equations and Pythagorean triples are beyond what
  this 0.3M-parameter model learned in 45 epochs. A regex calculator still
  beats it on bare arithmetic expressions.
- **No RL pass was run.** The supervised model had not yet learned the
  skills an RL reward could reinforce, so RL on top would have been
  premature. That is the honest next step, not a hidden one.

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
```

`train.py` defaults (d_model=128, 3 layers, lr 3e-4, 20 epochs) are not the
exact config of the published checkpoint (d_model=96, 2 layers, 45 epochs
in two phases); the checkpoint's own config is stored in `model.pt` and is
what `evaluate_trained.py` uses.

## Files

- `train.py` — training loop (with `--resume` and `--save-every`)
- `model.py` — tiny transformer
- `data.py` — variant generators (train/test splits)
- `evaluate_trained.py` — held-out evaluation with the env verifiers
- `results.json` — full evaluation output (per-example)
- `training_curve.json` / `training_curve.csv` — loss (and dev pass rate) per epoch
- `checkpoints/model.pt.b64` — trained weights, base64-encoded (1.2 MB decoded);
  run `checkpoints/decode_weights.py` to get `model.pt`
- `checkpoints/decode_weights.py` — decodes the weights
