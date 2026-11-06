# RLVR Demo — tiny model vs baseline vs oracle

A working demo of the RLVR environment: 20 tasks (10 code checked by hidden
unit tests, 10 math checked by exact answer), three agents, and the env's own
verifiers (`rlvr_env.rewards.verify`) scoring every answer. Reward is 1.0 for
a pass, 0.0 for a fail — no model judgement, no partial credit.

| Agent | Code | Math | Total |
| --- | --- | --- | --- |
| Baseline (regex arithmetic + return-None stub) | 0/10 | 2/10 | **2/20** |
| Tiny model (299,136 params, trained from scratch) | 5/10 | 2/10 | **7/20** |
| Oracle (reference solutions) | 10/10 | 10/10 | **20/20** |

## Layout

```
demo/
  backend/    FastAPI app (port 8000) — runs the real verifiers
    main.py
    requirements.txt
    tests/    pytest suite (6 tests, incl. the three known scores)
    generate_recorded_results.py   (one-off generator for the recorded file)
  frontend/   Next.js 15 / React 19 / TypeScript (port 3000)
    public/recorded-results.json   precomputed REAL results, for demo mode
```

The frontend is dark navy + indigo/violet with Inter — its own look, distinct
from TaskBoard (light) and Heart Risk (teal).

## Run it (Windows, two terminals)

Open two terminals (PowerShell or Command Prompt). Both start in the repo
folder you cloned — adjust the path to where you put `rlvr-env`.

### Terminal 1 — backend (port 8000)

```powershell
cd <path-to>\rlvr-env\demo\backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000
```

What you should see:

```
INFO:     Started server process [...]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Notes:

- First run decodes the tiny-model weights from
  `experiments\tiny-model\checkpoints\model.pt.b64` into `model.pt`
  (1.2 MB) automatically — no separate step needed. The model itself loads
  lazily, on the first Tiny-model run.
- `pip install -r requirements.txt` installs CPU-only torch from the
  PyTorch CPU index (the `--index-url` line in the file does that). It is
  the big download (~200 MB); the rest is small.
- If `python` is not on your PATH, use the Python 3.12 installer from
  python.org and tick "Add python.exe to PATH".

### Terminal 2 — frontend (port 3000)

```powershell
cd <path-to>\rlvr-env\demo\frontend
npm install
npm run dev
```

What you should see:

```
▲ Next.js 15.5.29
- Local:        http://localhost:3000
✓ Ready in ...
```

Then open **http://localhost:3000** in your browser (use `localhost`, not
`127.0.0.1` — the backend's CORS allow-list is `http://localhost:3000` only).

## Use it

1. Browse the 20 tasks on the left; click one to see its prompt and what the
   verifier checks. For code tasks you also see the hidden-test count. The
   expected math answer is never shown — the verifier checks it server-side.
2. Press **Run Baseline**, **Run Tiny model**, or **Run Oracle**. Results fill
   into the table row by row with a score summary (e.g. 7/20) and each
   agent's actual answer per task (code in a block, math as text).
3. The tiny model runs on CPU and takes about **12–20 seconds** for all
   20 tasks (the first run also loads the weights, so it is the slowest).
   Baseline and oracle return in under a second.

## Demo mode (no backend)

If the backend is not running, the frontend loads
`public/recorded-results.json` instead and shows an amber banner:
**"Recorded results from a real run"**. Those numbers are precomputed by
running the real verifiers — baseline 2/20, tiny 7/20, oracle 20/20 — not
made up, and the UI labels the replay as a replay. Start the backend and
press a run button again for a live evaluation.

## API (backend, port 8000)

- `GET /api/tasks` — all 20 tasks (task_id, category, difficulty, prompt;
  code tasks also give `function_name` and `test_count`; math answers are
  never returned).
- `POST /api/evaluate` with `{"agent": "baseline" | "tiny" | "oracle"}` —
  per-task `{task_id, category, prompt, answer, passed, reward, detail}`
  plus totals.
- `GET /api/health` — `{"status": "ok", "tasks": 20}`.

## Tests

```powershell
cd <path-to>\rlvr-env\demo\backend
.\.venv\Scripts\activate
python -m pytest tests/ -v
```

6 tests: the tasks endpoint returns 20 tasks with no leaked math answers,
and each agent returns 20 results whose totals match the known scores
(baseline 2/20, tiny 7/20, oracle 20/20) with rewards of exactly 0.0/1.0.
The tiny test takes ~15–20 seconds (model load + greedy decoding on CPU).
