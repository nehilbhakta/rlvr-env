# Deploying the RLVR demo

Two services, no accounts created yet — each deploy below is a one-click
(or one-form) step once you are signed in to Render / Vercel.

Repo: **nehilbhakta/rlvr-env**, branch `main`. The demo lives in `demo/`:
`demo/backend/` is the FastAPI app, `demo/frontend/` is the Next.js 15 app.

## 1. Backend — Render (FastAPI)

**Option A — Blueprint (recommended):** Render Dashboard → New →
Blueprint → pick repo `nehilbhakta/rlvr-env`. Render reads
`demo/render.yaml` (note: Render looks for `render.yaml` at the repo root
or in a subfolder you point it at — if the Blueprint flow does not pick up
`demo/render.yaml` automatically, use Option B).

**Option B — manual web service:**

| Setting | Value |
|---|---|
| Service type | Web Service |
| Runtime / Environment | Python |
| Repository | `nehilbhakta/rlvr-env`, branch `main` |
| Root Directory | `demo/backend` |
| Build Command | `pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| Instance | Free (or Starter if the free tier sleeps too much — the tiny model takes ~12 s per run once awake) |
| Env var `FRONTEND_URL` | `http://localhost:3000` for now; change to the Vercel URL once the frontend is live (comma-separated if several origins) |
| Env var `PYTHON_VERSION` | `3.12.3` |

Notes:

- The start module is `main:app` in `demo/backend/main.py` (verified — the
  app object is `app = FastAPI(...)` there). Run it from `demo/backend`,
  not the repo root.
- **CPU torch is the big install (~200 MB).** It is installed first from
  the PyTorch CPU index, then the rest from PyPI — do not put an
  `--index-url` line in `requirements.txt` (it would replace PyPI entirely
  and break the other installs). The first build takes several minutes.
- The backend imports `rlvr_env` from the repo root and the tiny-model
  weights from `experiments/tiny-model/checkpoints/model.pt.b64` — both are
  in the same repo, so a whole-repo clone (what Render does) just works.
  On first tiny-model run it decodes `model.pt` (1.2 MB) automatically.
- The Render filesystem is ephemeral: decoded weights and any `__pycache__`
  disappear on redeploy/restart, but they are regenerated on demand.
- Health check: `GET /api/health` → `{"status": "ok", "tasks": 20}`.
- Known scores (live, from the real verifiers): baseline **2/20**, tiny
  model **7/20**, oracle **20/20**.

## 2. Frontend — Vercel (Next.js)

Vercel Dashboard → Add New → Project → Import `nehilbhakta/rlvr-env`.

| Setting | Value |
|---|---|
| Framework Preset | Next.js (auto-detected) |
| Root Directory | `demo/frontend` |
| Build Command | `npm run build` (default) |
| Output Directory | `.next` (default — do not change) |
| Install Command | `npm install` (default) |
| Env var `NEXT_PUBLIC_API_URL` | The Render backend URL, e.g. `https://rlvr-demo-api.onrender.com` — **no trailing slash** |

Important:

- `NEXT_PUBLIC_API_URL` is inlined into the JavaScript **at build time**.
  Set the env var **before** the first deploy (or change it and redeploy)
  — changing it alone does not update a live build.
- If the env var is unset, the frontend falls back to
  `http://localhost:8000` (local dev) and, when the backend is unreachable,
  to the recorded-results file with an amber banner: “Recorded results
  from a real run” (baseline 2/20, tiny 7/20, oracle 20/20). The page
  works with no backend at all.
- After the frontend is live, go back to Render and set the backend's
  `FRONTEND_URL` env var to the Vercel URL (e.g.
  `https://<project>.vercel.app`) so the browser's cross-origin calls are
  allowed, then redeploy/restart the Render service.

## 3. Verify after both are live

1. Open the Vercel URL — the banner should read **“Live mode — connected
   to the FastAPI backend”** (not “Recorded results”).
2. Run Baseline → expect **2/20**; Run Tiny model → expect **7/20**
   (~12 s on CPU); Run Oracle → expect **20/20**.
3. If the banner says “Recorded results”, the frontend cannot reach the
   backend — check `NEXT_PUBLIC_API_URL` (set before build?) and the
   backend's `FRONTEND_URL` (Vercel origin allowed?).

## Local dev (unchanged)

Backend: `cd demo/backend`, venv, `pip install torch --index-url
https://download.pytorch.org/whl/cpu`, `pip install -r requirements.txt`,
`uvicorn main:app --host 127.0.0.1 --port 8000`. Frontend: `cd
demo/frontend`, `npm install`, `npm run dev` → http://localhost:3000.
