"""Evaluate a checkpoint greedily on the env's 20 ORIGINAL tasks.

Same protocol as rlvr-env/evaluate.py (the env's own prompts, scored by
rlvr_env.rewards.verify), plus the heuristic baseline on the same tasks
for an apples-to-apples before/after. This is the fixed-20 number quoted
in experiments/tiny-model/README.md (SFT model: 7/20).

Usage:
    python eval_fixed20.py [--ckpt checkpoints/model-rl.pt] [--out results-rl-fixed20.json]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rlvr-env"))

from evaluate_trained import heuristic_agent, model_answer  # noqa: E402
from rlvr_env.rewards import verify  # noqa: E402
from rlvr_env.tasks import TASKS  # noqa: E402
from model import TinyGPT  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(HERE / "checkpoints" / "model-rl.pt"))
    ap.add_argument("--out", type=str, default=str(HERE / "results-rl-fixed20.json"))
    args = ap.parse_args()

    torch.set_num_threads(4)
    device = torch.device("cpu")
    t0 = time.time()

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = TinyGPT(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    stoi, itos = ckpt["stoi"], ckpt["itos"]

    rows = []
    for task in TASKS:
        max_new = 16 if task["category"] == "math" else 420
        ans = model_answer(model, task["prompt"], stoi, itos, device, max_new)
        ok, detail = verify(task, ans)
        base_ans = heuristic_agent({"category": task["category"], "prompt": task["prompt"]})
        base_ok, _ = verify(task, base_ans)
        rows.append({
            "task_id": task["id"], "category": task["category"],
            "prompt": task["prompt"], "model_answer": ans,
            "passed": bool(ok), "detail": detail,
            "baseline_passed": bool(base_ok),
        })
        print(f"[{'PASS' if ok else 'FAIL'}] {task['id']}: {ans[:70]!r} :: {detail}",
              flush=True)

    def summ(cat):
        rs = [r for r in rows if r["category"] == cat]
        return {"passed": sum(int(r["passed"]) for r in rs), "total": len(rs),
                "baseline_passed": sum(int(r["baseline_passed"]) for r in rs)}

    results = {
        "checkpoint": str(args.ckpt),
        "eval": "greedy decoding on the env's 20 original tasks, scored by "
                "rlvr_env.rewards.verify (same protocol as rlvr-env/evaluate.py)",
        "overall": {"trained_passed": sum(int(r["passed"]) for r in rows),
                    "baseline_passed": sum(int(r["baseline_passed"]) for r in rows),
                    "total": len(rows)},
        "by_category": {"code": summ("code"), "math": summ("math")},
        "rows": rows,
        "eval_seconds": round(time.time() - t0, 1),
    }
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    o = results["overall"]
    print(f"\nFIXED-20: {o['trained_passed']}/{o['total']}  |  BASELINE: {o['baseline_passed']}/{o['total']}")
    print(f"code: {results['by_category']['code']}  math: {results['by_category']['math']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
