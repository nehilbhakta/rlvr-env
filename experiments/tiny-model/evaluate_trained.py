"""Evaluate the trained checkpoint on held-out RLVR variants.

Scoring uses the env's own verifiers (rlvr_env.rewards.verify) — no custom
scorer. The held-out set uses a different random seed from training, so no
held-out math prompt was seen in training. Code tasks use the env's fixed
prompts, scored by the hidden unit tests.

Also scores the heuristic baseline from evaluate.py on the SAME held-out
variants, for an apples-to-apples comparison.

Usage:
    python evaluate_trained.py [--ckpt checkpoints/model.pt] [--out results.json]
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "rlvr-env"))

from data import ORIGINAL, build_heldout_set, build_train_set  # noqa: E402
from model import EOS, TinyGPT  # noqa: E402
from rlvr_env.rewards import verify  # noqa: E402

HERE = Path(__file__).resolve().parent
ARITH_RE = re.compile(r"(-?\d+(?:\.\d+)?\s*(?:[+\-*/^]\s*-?\d+(?:\.\d+)?\s*)+)")


def heuristic_agent(obs):
    """The baseline from rlvr-env/evaluate.py, copied verbatim."""
    if obs["category"] == "math":
        match = ARITH_RE.search(obs["prompt"])
        if match:
            expr = match.group(1).replace("^", "**")
            try:
                return str(eval(expr, {"__builtins__": {}}))  # noqa: S307
            except Exception:
                return "0"
        return "0"
    return "def f():\n    return None\n"


@torch.no_grad()
def model_answer(model, prompt, stoi, itos, device, max_new):
    ids = [stoi["<bos>"]] + [stoi[c] for c in prompt] + [stoi["<sep>"]]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new, eos_id=stoi["<eos>"])
    gen = out[0, len(ids) :].tolist()
    return "".join(itos[i] for i in gen if itos[i] not in ("<eos>", "<pad>"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(HERE / "checkpoints" / "model.pt"))
    ap.add_argument("--math-per-task", type=int, default=20)
    ap.add_argument("--seed", type=int, default=987654)
    ap.add_argument("--out", type=str, default=str(HERE / "results.json"))
    args = ap.parse_args()

    torch.set_num_threads(4)
    device = torch.device("cpu")
    t0 = time.time()

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = TinyGPT(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    stoi, itos = ckpt["stoi"], ckpt["itos"]

    heldout = build_heldout_set(math_per_task=args.math_per_task, seed=args.seed)
    # Verify the held-out math prompts really are unseen in training.
    train_prompts = {e["prompt"] for e in build_train_set() if e["category"] == "math"}
    overlap = sum(
        1 for e in heldout if e["category"] == "math" and e["prompt"] in train_prompts
    )
    per_task, rows = {}, []
    for item in heldout:
        task = dict(ORIGINAL[item["task_id"]])
        if item["category"] == "math":
            task["answer"] = item["answer"]
        max_new = 16 if item["category"] == "math" else 420
        ans = model_answer(model, item["prompt"], stoi, itos, device, max_new)
        ok, detail = verify(task, ans)
        base_ans = heuristic_agent({"category": item["category"], "prompt": item["prompt"]})
        base_ok, _ = verify(task, base_ans)
        rows.append(
            {
                "task_id": item["task_id"],
                "category": item["category"],
                "prompt": item["prompt"],
                "model_answer": ans,
                "passed": bool(ok),
                "detail": detail,
                "baseline_passed": bool(base_ok),
            }
        )
        per_task.setdefault(item["task_id"], {"passed": 0, "total": 0, "baseline_passed": 0})
        per_task[item["task_id"]]["total"] += 1
        per_task[item["task_id"]]["passed"] += int(ok)
        per_task[item["task_id"]]["baseline_passed"] += int(base_ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {item['task_id']}: {ans[:70]!r} :: {detail}", flush=True)

    def summ(cat):
        rs = [r for r in rows if r["category"] == cat]
        return {
            "passed": sum(int(r["passed"]) for r in rs),
            "total": len(rs),
            "baseline_passed": sum(int(r["baseline_passed"]) for r in rs),
        }

    results = {
        "model": "tiny decoder-only transformer trained from scratch (see README)",
        "checkpoint": str(args.ckpt),
        "eval": {
            "math_per_task": args.math_per_task,
            "heldout_seed": args.seed,
            "n_tasks": len(rows),
            "split": (
                "Held-out math variants generated from parameter ranges disjoint "
                "from training (see data.py splits); no held-out math prompt appears "
                "in training (verified overlap: "
                f"{overlap} of {sum(1 for e in heldout if e['category'] == 'math')} math prompts). "
                "Code tasks use the env's 10 fixed prompts, scored by the hidden unit tests."
            ),
        },
        "overall": {
            "trained_passed": sum(int(r["passed"]) for r in rows),
            "baseline_passed": sum(int(r["baseline_passed"]) for r in rows),
            "total": len(rows),
        },
        "by_category": {"code": summ("code"), "math": summ("math")},
        "per_task": per_task,
        "rows": rows,
        "eval_seconds": round(time.time() - t0, 1),
    }
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    o = results["overall"]
    print(f"\nTRAINED: {o['trained_passed']}/{o['total']}  |  BASELINE: {o['baseline_passed']}/{o['total']}")
    print(f"code: {results['by_category']['code']}  math: {results['by_category']['math']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
