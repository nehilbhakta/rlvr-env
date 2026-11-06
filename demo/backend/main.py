"""RLVR demo backend.

FastAPI app that runs the REAL env verifiers (rlvr_env.rewards.verify)
over the 20 tasks in rlvr_env/tasks.py for three agents:

* baseline — the heuristic from evaluate.py (regex arithmetic for math,
  a `def f(): return None` stub for code).
* tiny — greedy decoding from the trained 299,136-param checkpoint in
  experiments/tiny-model/checkpoints/ (decoded from model.pt.b64 on
  startup if model.pt is missing).
* oracle — each task's reference_solution.

CORS allows only http://localhost:3000. Runs on port 8000.
"""

import base64
import re
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Repo root is two levels up from this file (demo/backend -> rlvr-env).
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TINY_DIR = REPO_ROOT / "experiments" / "tiny-model"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(TINY_DIR))

from rlvr_env.rewards import verify  # noqa: E402
from rlvr_env.tasks import TASKS  # noqa: E402

ARITH_RE = re.compile(r"(-?\d+(?:\.\d+)?\s*(?:[+\-*/^]\s*-?\d+(?:\.\d+)?\s*)+)")

app = FastAPI(title="RLVR Demo API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

def heuristic_agent(task):
    """The baseline from rlvr-env/evaluate.py, copied verbatim in logic."""
    if task["category"] == "math":
        match = ARITH_RE.search(task["prompt"])
        if match:
            expr = match.group(1).replace("^", "**")
            try:
                return str(eval(expr, {"__builtins__": {}}))  # noqa: S307 - numbers only, from our own prompts
            except Exception:
                return "0"
        return "0"
    # Code tasks: a stub that defines a function but returns None.
    return "def f():\n    return None\n"


def oracle_agent(task):
    return task["reference_solution"]


# --- Tiny model (lazy-loaded, cached; ~12s for all 20 tasks on CPU) --------
_tiny_lock = threading.Lock()
_tiny = None  # (model, stoi, itos, device)


def _ensure_tiny_model():
    global _tiny
    if _tiny is not None:
        return _tiny
    with _tiny_lock:
        if _tiny is not None:
            return _tiny
        import torch

        from model import TinyGPT  # noqa: E402  (experiments/tiny-model/model.py)

        ckpt_dir = TINY_DIR / "checkpoints"
        pt_path = ckpt_dir / "model.pt"
        if not pt_path.exists():
            # Weights are stored as base64 text because the repo push path
            # is UTF-8 only; decode exactly like decode_weights.py does.
            data = base64.b64decode((ckpt_dir / "model.pt.b64").read_text())
            pt_path.write_bytes(data)
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
        model = TinyGPT(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        torch.set_num_threads(4)
        _tiny = (model, ckpt["stoi"], ckpt["itos"], torch.device("cpu"))
        return _tiny


def tiny_agent(task):
    import torch

    model, stoi, itos, device = _ensure_tiny_model()
    max_new = 16 if task["category"] == "math" else 420
    with torch.no_grad():
        ids = [stoi["<bos>"]] + [stoi[c] for c in task["prompt"]] + [stoi["<sep>"]]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(idx, max_new, eos_id=stoi["<eos>"])
        gen = out[0, len(ids):].tolist()
    return "".join(itos[i] for i in gen if itos[i] not in ("<eos>", "<pad>"))


AGENTS = {"baseline": heuristic_agent, "tiny": tiny_agent, "oracle": oracle_agent}

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class EvaluateRequest(BaseModel):
    agent: str


def public_task(task):
    """Task as exposed to the frontend. The expected math answer is NEVER
    included — the verifier checks it server-side."""
    out = {
        "task_id": task["id"],
        "category": task["category"],
        "difficulty": task["difficulty"],
        "prompt": task["prompt"],
    }
    if task["category"] == "code":
        out["function_name"] = task["function_name"]
        out["test_count"] = len(task["tests"])
    return out


@app.get("/api/tasks")
def list_tasks():
    return {"tasks": [public_task(t) for t in TASKS]}


@app.get("/api/health")
def health():
    return {"status": "ok", "tasks": len(TASKS)}


@app.post("/api/evaluate")
def evaluate(req: EvaluateRequest):
    agent_name = req.agent
    if agent_name not in AGENTS:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail=f"unknown agent: {agent_name}. Use one of: baseline, tiny, oracle")
    agent = AGENTS[agent_name]
    t0 = time.time()
    results = []
    for task in TASKS:
        answer = agent(task)
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
    passed_total = sum(1 for r in results if r["passed"])
    return {
        "agent": agent_name,
        "total": len(results),
        "passed": passed_total,
        "code_passed": sum(1 for r in results if r["passed"] and r["category"] == "code"),
        "math_passed": sum(1 for r in results if r["passed"] and r["category"] == "math"),
        "eval_seconds": round(time.time() - t0, 1),
        "results": results,
    }
