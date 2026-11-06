"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Task = {
  task_id: string;
  category: "code" | "math";
  difficulty: string;
  prompt: string;
  function_name?: string;
  test_count?: number;
};

type ResultRow = {
  task_id: string;
  category: string;
  prompt: string;
  answer: string;
  passed: boolean;
  reward: number;
  detail: string;
};

type EvalResponse = {
  agent: string;
  total: number;
  passed: number;
  code_passed: number;
  math_passed: number;
  eval_seconds?: number;
  results: ResultRow[];
};

type AgentName = "baseline" | "tiny" | "oracle";

const AGENTS: { id: AgentName; label: string; blurb: string }[] = [
  { id: "baseline", label: "Baseline", blurb: "Regex arithmetic for math, a return-None stub for code. No model at all." },
  { id: "tiny", label: "Tiny model", blurb: "299K-param transformer, trained from scratch. Greedy decoding, ~12s on CPU." },
  { id: "oracle", label: "Oracle", blurb: "Each task's reference solution. 20/20 means the verifiers work — not that the task is hard." },
];

export default function Home() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [tasksError, setTasksError] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [running, setRunning] = useState<AgentName | null>(null);
  const [activeAgent, setActiveAgent] = useState<AgentName | null>(null);
  const [mode, setMode] = useState<"live" | "recorded" | "unknown">("unknown");
  const [results, setResults] = useState<ResultRow[]>([]);
  const [summary, setSummary] = useState<Omit<EvalResponse, "results"> | null>(null);
  const [runError, setRunError] = useState("");
  const revealTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  // Load the task list from the API; fall back to the recorded file's tasks.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch(`${API}/api/tasks`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const data = await r.json();
        if (cancelled) return;
        setTasks(data.tasks);
        setSelectedId(data.tasks[0]?.task_id ?? null);
        setMode("live");
      } catch {
        try {
          const r = await fetch("/recorded-results.json");
          const data = await r.json();
          if (cancelled) return;
          setTasks(data.tasks);
          setSelectedId(data.tasks[0]?.task_id ?? null);
          setMode("recorded");
          setTasksError("Backend unreachable — task list loaded from the recorded file.");
        } catch {
          if (!cancelled) setTasksError("Could not reach the backend and the recorded file failed to load.");
        }
      }
    })();
    return () => {
      cancelled = true;
      if (revealTimer.current) clearInterval(revealTimer.current);
    };
  }, []);

  // Reveal results one row at a time so the table fills in as the run lands.
  const reveal = useCallback((data: EvalResponse) => {
    if (revealTimer.current) clearInterval(revealTimer.current);
    setResults([]);
    setSummary({
      agent: data.agent,
      total: data.total,
      passed: data.passed,
      code_passed: data.code_passed,
      math_passed: data.math_passed,
      eval_seconds: data.eval_seconds,
    });
    let i = 0;
    revealTimer.current = setInterval(() => {
      i += 1;
      setResults(data.results.slice(0, i));
      if (i >= data.results.length && revealTimer.current) {
        clearInterval(revealTimer.current);
        revealTimer.current = null;
      }
    }, 120);
  }, []);

  const runAgent = useCallback(
    async (agent: AgentName) => {
      setRunning(agent);
      setActiveAgent(agent);
      setRunError("");
      setResults([]);
      setSummary(null);
      try {
        const r = await fetch(`${API}/api/evaluate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ agent }),
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const data: EvalResponse = await r.json();
        setMode("live");
        reveal(data);
      } catch {
        // Demo mode: precomputed REAL results from a recorded run — never faked.
        try {
          const r = await fetch("/recorded-results.json");
          const data = await r.json();
          const recorded: EvalResponse = data.agents[agent];
          setMode("recorded");
          setRunError(
            "Backend unreachable — showing recorded results from a real run (see the banner). Start the backend and run again for a live evaluation.",
          );
          reveal(recorded);
        } catch {
          setRunError("Backend unreachable and the recorded results file could not be loaded.");
        }
      } finally {
        setRunning(null);
      }
    },
    [reveal],
  );

  const selected = tasks.find((t) => t.task_id === selectedId) ?? null;
  const selectedResult = results.find((r) => r.task_id === selectedId) ?? null;
  const resultById = new Map(results.map((r) => [r.task_id, r]));

  return (
    <div className="page">
      <h1>Can a tiny model learn from verifiable rewards?</h1>
      <p className="lede">
        Twenty tasks — <strong>10 code</strong> (checked by hidden unit tests) and{" "}
        <strong>10 math</strong> (checked by exact answer). Three agents attempt them and
        the environment&apos;s own verifiers score every answer: reward{" "}
        <strong>1.0</strong> for a pass, <strong>0.0</strong> for a fail. No model
        judgement, no partial credit. Pick a task to see its prompt and what the
        verifier checks, then run an agent.
      </p>

      {mode === "live" && (
        <div className="mode-banner live">
          <span className="dot" /> <strong>Live mode</strong> — connected to the FastAPI
          backend. Every run executes the real verifiers.
        </div>
      )}
      {mode === "recorded" && (
        <div className="mode-banner recorded">
          <span className="dot" /> <strong>Recorded results from a real run</strong> — the
          backend is not reachable, so these scores come from precomputed results
          recorded by running the real verifiers (baseline 2/20, tiny model 7/20,
          oracle 20/20). This is a replay, not a live run: start the backend and
          press a run button again for live evaluation.
        </div>
      )}
      {tasksError && <p className="error">{tasksError}</p>}

      <div className="run-row">
        {AGENTS.map((a) => (
          <button
            key={a.id}
            className={`run-btn${activeAgent === a.id ? " active" : ""}`}
            disabled={running !== null}
            onClick={() => runAgent(a.id)}
          >
            <b>{running === a.id ? `Running ${a.label}…` : `Run ${a.label}`}</b>
            <span>{a.blurb}</span>
          </button>
        ))}
      </div>
      <p className="run-note">
        The tiny model runs on CPU and takes about 12 seconds for all 20 tasks. The
        baseline and oracle return in under a second.
      </p>
      {runError && <p className="error">{runError}</p>}

      {summary && (
        <div className="score-strip" aria-live="polite">
          <div className="score-card">
            <div className="big">
              {summary.passed}/{summary.total}
            </div>
            <div className="lbl">
              {AGENTS.find((a) => a.id === summary.agent)?.label} — total score
              {summary.eval_seconds !== undefined ? ` · ${summary.eval_seconds}s` : ""}
            </div>
            <div className="sub">
              Code {summary.code_passed}/10 · Math {summary.math_passed}/10
            </div>
          </div>
          <div className="score-card">
            <div className="big">{summary.code_passed}/10</div>
            <div className="lbl">Code — hidden unit tests must all pass</div>
          </div>
          <div className="score-card">
            <div className="big">{summary.math_passed}/10</div>
            <div className="lbl">Math — exact answer match</div>
          </div>
        </div>
      )}

      <div className="columns">
        <section className="panel" aria-label="Task browser">
          <h2>Tasks — 10 code + 10 math</h2>
          {tasks.length === 0 ? (
            <p className="empty">Loading tasks…</p>
          ) : (
            <ul className="task-list">
              {tasks.map((t) => {
                const res = resultById.get(t.task_id);
                return (
                  <li key={t.task_id}>
                    <button
                      className={selectedId === t.task_id ? "selected" : ""}
                      onClick={() => setSelectedId(t.task_id)}
                    >
                      <span className={`tcat ${t.category}`}>{t.category}</span>
                      <span className="tid">{t.task_id}</span> · {t.difficulty}
                      <span className="tprompt">{t.prompt}</span>
                      {res && (
                        <span className={`tverdict ${res.passed ? "pass" : "fail"}`}>
                          {res.passed ? "✓ PASS" : "✗ FAIL"} · reward {res.reward.toFixed(1)}
                        </span>
                      )}
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </section>

        <section className="panel" aria-label="Task detail">
          <h2>{selected ? `${selected.task_id} — detail` : "Task detail"}</h2>
          {!selected ? (
            <p className="empty">Select a task on the left.</p>
          ) : (
            <div className="detail">
              <h3>{selected.prompt}</h3>
              <div className="meta">
                {selected.category} · {selected.difficulty}
                {selected.function_name ? ` · function \`${selected.function_name}\`` : ""}
              </div>
              <div className="check">
                <b>What the verifier checks: </b>
                {selected.category === "code"
                  ? `The submitted code is executed and \`${selected.function_name}\` is called against ${selected.test_count} hidden unit tests. All ${selected.test_count} must pass — one failure is a reward of 0.0.`
                  : "The last number in the submitted answer must match the expected answer exactly (tolerance 1e-6). The expected answer is kept server-side and is never shown here."}
              </div>
              {selectedResult ? (
                <>
                  <div className={`verdict-line ${selectedResult.passed ? "pass" : "fail"}`}>
                    {selectedResult.passed ? "✓ PASS" : "✗ FAIL"} — reward{" "}
                    {selectedResult.reward.toFixed(1)}
                  </div>
                  <p className="detail-text">{selectedResult.detail}</p>
                  <p>
                    <b>{AGENTS.find((a) => a.id === activeAgent)?.label ?? "Agent"} answered:</b>
                  </p>
                  {selected.category === "code" ? (
                    <pre className="answer">{selectedResult.answer || "(no answer)"}</pre>
                  ) : (
                    <pre className="answer">{selectedResult.answer || "(no answer)"}</pre>
                  )}
                </>
              ) : (
                <p className="detail-text" style={{ marginTop: 14 }}>
                  No result for this task yet — run an agent above.
                </p>
              )}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
