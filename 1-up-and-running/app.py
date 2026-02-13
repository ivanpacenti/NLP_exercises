"""
DTU Sentiment Demo Frontend (FastAPI)

This app is a small Python-based frontend web app meant to demonstrate and test an
independently-built REST web service for sentiment analysis of short course evaluations.

It targets a student-built web service with:
- POST /v1/sentiment
- Input JSON: {"text": "<course evaluation>"}
- Output JSON: {"score": <number between -5 and 5>}

The UI:
- Lets users enter text and get a sentiment score.
- Includes a predefined dataset of labeled examples for quick testing.
- Can run batch evaluation to show per-item results and aggregated accuracy.
- Shows operational metrics (counts + latency).

Pedagogical requirements:
- Few dependencies, easy installation.
- No external JS libraries.
- Simple inline styling (DTU colors).
- Clear, pedagogic error messages.
- Reasonable timeout for the external service.

The app is intentionally simple so students (in an NLP course) can read/understand it.

Run the application:
- uvicorn app:app --reload --port 8001

---------------------------------------------------------------------------
PROMPT for dataset (embedded verbatim as requested)
---------------------------------------------------------------------------
I would like a dataset for sentiment analysis where the text is
fictional course evaluations from the Technical University of Denmark.
The dataset should be in JSON-like in a list of list. Could be (either
single quote or double quote)
[['this was a good course', 'positive'],
 ['this was a bad course', 'negative']]

The text should have different features that should be selected with a
certain probability before the text is generated. These features are:

- Language: Danish (50%), English (50%)
- Sentiment: positive (70%), neutral (20%), negative (10%)
- Length: Short sentence (33%), one or two sentence (33%), multiple sentences (33%)
- Language style: bad, e.g., with spelling and capitalization errors (33%), middle of the road writing slightly informal (33%), well-written with excellent vocabulary (33%)
- Topic mentioned (non-exclusive):
  - teacher (40%)
  - book and course (30%)
  - feedback (20%)
  - other (40%)
- Teacher name mentioned: mentioned by name (40%), not mentioned by name (60%)
- Teachers that may be mentioned:
  - Finn, teaches NLP and receives generally negative evaluation
  - Nicki, teaches MLOps, lively and with excellent evaluation
  - Tue, teaches reinforcement learning, very difficult hard course, but generally good.
  - Bjørn, teaches machine learning and is generally good.
  - Ivana, teaches human cognitiion with excellent evaluations.

Generate 40 data items
---------------------------------------------------------------------------
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Literal, Optional, Tuple


import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field # , HttpUrl


SentimentLabel = Literal["positive", "neutral", "negative"]


class ScoreRequest(BaseModel):
    """Request payload for scoring a single text.

    Parameters
    ----------
    service_url : str
        Base URL for the external sentiment service, e.g. "http://localhost:8000".
        The app will call "{service_url}/v1/sentiment".
    text : str
        Input text to score.

    Notes
    -----
    We route external calls via the backend to avoid CORS issues and keep JS simple.
    """

    service_url: str = Field(..., description="Base URL of external service, e.g. http://localhost:8000")
    text: str = Field(..., min_length=1, description="Text to score")


class BatchRequest(BaseModel):
    """Request payload for batch evaluation on a dataset.

    Parameters
    ----------
    service_url : str
        Base URL for the external sentiment service.
    dataset : list
        List of [text, gold_label] pairs.

    Examples
    --------
    >>> req = BatchRequest(service_url="http://localhost:8000", dataset=[["Good course", "positive"]])
    >>> req.dataset[0][1]
    'positive'
    """

    service_url: str
    dataset: List[List[str]] = Field(..., description='List like [["text", "positive"], ...]')


@dataclass
class Metrics:
    """Simple in-memory operational metrics."""
    total_requests: int = 0
    success_requests: int = 0
    failed_requests: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    last_latency_ms: Optional[float] = None

    def record(self, ok: bool, latency_ms: float) -> None:
        self.total_requests += 1
        if ok:
            self.success_requests += 1
        else:
            self.failed_requests += 1
        self.last_latency_ms = latency_ms
        self.latencies_ms.append(latency_ms)

    def snapshot(self) -> Dict[str, Any]:
        avg = statistics.mean(self.latencies_ms) if self.latencies_ms else None
        p95 = None
        if len(self.latencies_ms) >= 20:
            xs = sorted(self.latencies_ms)
            p95 = xs[int(0.95 * (len(xs) - 1))]
        return {
            "total_requests": self.total_requests,
            "success_requests": self.success_requests,
            "failed_requests": self.failed_requests,
            "last_latency_ms": self.last_latency_ms,
            "avg_latency_ms": avg,
            "p95_latency_ms": p95,
        }


app = FastAPI(title="DTU Sentiment Demo Frontend", version="1.0.0")
metrics = Metrics()

# ---- Configuration you can tweak ----
DEFAULT_SERVICE_URL = "http://localhost:8000"
REQUEST_TIMEOUT_SECONDS = 4.0  # pedagogical: fail fast with a helpful message

# ---- Paste your 40 items here (or keep the small starter list) ----
# Format: [["text", "positive"], ["text", "neutral"], ...]
DATASET: List[List[str]] = [
  ['Great course, learned a lot.', 'positive'],
  ['Really solid DTU course with clear structure and useful exercises.', 'positive'],
  ['Nicki was energetic and made MLOps feel practical and fun.', 'positive'],
  ['The lectures were okay, but the pace felt uneven.', 'neutral'],
  ['This course was hard, but worth it.', 'positive'],
  ['Tue’s reinforcement learning course is brutal, yet the learning outcome is amazing.', 'positive'],
  ['Bjørn explained the core ML ideas clearly and the project was motivating.', 'positive'],
  ['I liked the course book and how it matched the weekly plan.', 'positive'],
  ['The feedback on assignments came a bit late.', 'neutral'],
  ['Finn’s NLP lectures were confusing and the slides had too many gaps.', 'negative'],
  ['Overall fine, nothing special.', 'neutral'],
  ['Excellent vocabulary and examples; I left each week with new tools.', 'positive'],
  ['good course but the typos in the material was annoying lol', 'neutral'],
  ['Nicki’s demos were sharp, and the TA feedback was super actionable.', 'positive'],
  ['The course is well organized, but I wish there were more office hours.', 'neutral'],
  ['Finn taught NLP, but honestly it felt messy and underprepared.', 'negative'],
  ['Ivana’s cognitive science lectures were inspiring and beautifully presented.', 'positive'],
  ['The teacher was nice and helpful.', 'positive'],
  ['Too many mandatory readings, but the exams were fair.', 'neutral'],
  ['Loved the project work and the way we got iterative feedback.', 'positive'],

  ['Mega godt kursus!', 'positive'],
  ['Rigtig god struktur og gode øvelser på DTU, jeg følte mig tryg gennem hele forløbet.', 'positive'],
  ['Nicki gjorde MLOps levende med hands-on demoer, og feedbacken var hurtig og konkret.', 'positive'],
  ['Kurset var okay, men tempoet svingede lidt fra uge til uge.', 'neutral'],
  ['Svært kursus, men jeg lærte virkelig meget.', 'positive'],
  ['Tue’s reinforcement learning var vildt svært, men undervisningen var stærk og gav mening til sidst.', 'positive'],
  ['Bjørn var god til at forklare maskinlæring, og projektet bandt det hele sammen.', 'positive'],
  ['Bogen passede fint til kurset, og kapitlerne blev brugt på en fornuftig måde.', 'positive'],
  ['Jeg savnede lidt mere feedback på de tidlige afleveringer.', 'neutral'],
  ['Finns NLP-kursus var rodet, og jeg forstod ofte ikke pointen med øvelserne.', 'negative'],
  ['Helt fint, ikke noget wow.', 'neutral'],
  ['Sproget i materialet var præcist, og eksemplerne var elegante og velvalgte.', 'positive'],
  ['det var ok kursus men opgaverne var lidt mærkelige og der var mange fejl', 'neutral'],
  ['Nicki var mega engageret, og man fik god, hurtig feedback på pipeline-opgaverne.', 'positive'],
  ['Kurset fungerede, men der kunne godt være lidt bedre koordinering mellem forelæsning og øvelsestime.', 'neutral'],
  ['Finn underviser i NLP, men det var frustrerende: uklare krav og for få forklaringer.', 'negative'],
  ['Ivana var fantastisk—tydelig formidling, stærke diskussioner, og jeg gik derfra med nye perspektiver.', 'positive'],
  ['Underviseren var hjælpsom, og jeg følte mig set i timerne.', 'positive'],
  ['For meget læsning nogle uger, men eksamen virkede rimelig.', 'neutral'],
  ['Jeg elskede projektet, og feedback-loopet gjorde, at vi faktisk blev bedre undervejs.', 'positive']
]


def score_to_label(score: float) -> SentimentLabel:
    """Map numeric score [-5, 5] to a coarse label.

    We keep this mapping simple and explicit for teaching.

    Parameters
    ----------
    score : float
        External service score, expected in [-5, 5].

    Returns
    -------
    SentimentLabel
        "negative" if score <= -1, "neutral" if -1 < score < 1, else "positive".

    Examples
    --------
    >>> score_to_label(3)
    'positive'
    >>> score_to_label(0)
    'neutral'
    >>> score_to_label(-3)
    'negative'
    """
    if score <= -1:
        return "negative"
    if score >= 1:
        return "positive"
    return "neutral"


async def call_external_service(service_url: str, text: str) -> Tuple[Optional[float], Dict[str, Any]]:
    """Call the student-built sentiment service and return (score, debug_info).

    Returns
    -------
    score : float | None
        Score if successful, otherwise None.
    debug_info : dict
        Includes latency_ms and any error information.
    """
    endpoint = service_url.rstrip("/") + "/v1/sentiment"
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            r = await client.post(endpoint, json={"text": text})
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if r.status_code != 200:
            metrics.record(ok=False, latency_ms=latency_ms)
            return None, {
                "latency_ms": latency_ms,
                "error": (
                    f"External service responded with HTTP {r.status_code}. "
                    f"Expected 200. Response body (truncated): {r.text[:200]!r}"
                ),
                "endpoint": endpoint,
            }

        data = r.json()
        if "score" not in data:
            metrics.record(ok=False, latency_ms=latency_ms)
            return None, {
                "latency_ms": latency_ms,
                "error": (
                    "External service returned JSON without the required field 'score'. "
                    f"Got keys: {list(data.keys())!r}"
                ),
                "endpoint": endpoint,
            }

        score = float(data["score"])
        if not (-5 <= score <= 5):
            # We still accept but warn; pedagogical.
            metrics.record(ok=True, latency_ms=latency_ms)
            return score, {
                "latency_ms": latency_ms,
                "warning": f"Score {score} is outside expected range [-5, 5].",
                "endpoint": endpoint,
            }

        metrics.record(ok=True, latency_ms=latency_ms)
        return score, {"latency_ms": latency_ms, "endpoint": endpoint}

    except httpx.TimeoutException:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        metrics.record(ok=False, latency_ms=latency_ms)
        return None, {
            "latency_ms": latency_ms,
            "error": (
                f"Timeout after {REQUEST_TIMEOUT_SECONDS:.1f}s while calling the external service. "
                "This usually means the container is not running, the URL/port is wrong, or the model is too slow. "
                "Try: (1) open the service docs at /docs, (2) verify /v1/sentiment exists, (3) reduce model size."
            ),
            "endpoint": endpoint,
        }
    except httpx.RequestError as e:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        metrics.record(ok=False, latency_ms=latency_ms)
        return None, {
            "latency_ms": latency_ms,
            "error": (
                "Could not reach the external service (network error). "
                "Check the base URL and whether the container is running. "
                f"Details: {type(e).__name__}: {e}"
            ),
            "endpoint": endpoint,
        }
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        metrics.record(ok=False, latency_ms=latency_ms)
        return None, {
            "latency_ms": latency_ms,
            "error": (
                "Unexpected error while calling/parsing the external service response. "
                f"Details: {type(e).__name__}: {e}"
            ),
            "endpoint": endpoint,
        }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    dataset_js = json.dumps(DATASET, ensure_ascii=False)

    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>DTU Sentiment Demo</title>
  <style>
    :root {
      --dtu-red: rgb(153,0,0);
      --bg: #ffffff;
      --fg: #111111;
      --muted: #666666;
      --card: #f6f6f6;
      --border: #dddddd;
    }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      color: var(--fg);
      background: var(--bg);
    }
    header {
      background: var(--dtu-red);
      color: white;
      padding: 14px 18px;
    }
    header h1 {
      font-size: 18px;
      margin: 0;
      font-weight: 700;
      letter-spacing: 0.2px;
    }
    header p {
      margin: 6px 0 0 0;
      font-size: 13px;
      opacity: 0.9;
    }
    main {
      max-width: 1100px;
      margin: 18px auto;
      padding: 0 14px 30px 14px;
    }
    .row {
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 14px;
    }
    @media (max-width: 900px) {
      .row { grid-template-columns: 1fr; }
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
    }
    label {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
    }
    input[type="text"], textarea, select {
      width: 100%;
      box-sizing: border-box;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px;
      font-size: 14px;
      background: white;
      color: var(--fg);
      outline: none;
    }
    textarea { min-height: 130px; resize: vertical; }
    .buttons {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 10px;
    }
    button {
      border: 0;
      border-radius: 10px;
      padding: 10px 12px;
      font-weight: 700;
      cursor: pointer;
      background: var(--dtu-red);
      color: white;
    }
    button.secondary {
      background: #2b2b2b;
    }
    button.ghost {
      background: transparent;
      color: var(--dtu-red);
      border: 1px solid var(--dtu-red);
    }
    .small {
      font-size: 12px;
      color: var(--muted);
      margin-top: 8px;
      line-height: 1.35;
    }
    .result {
      margin-top: 12px;
      background: white;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px;
    }
    .pill {
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      border: 1px solid var(--border);
      background: #fff;
      margin-right: 8px;
    }
    .grid2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    @media (max-width: 600px) {
      .grid2 { grid-template-columns: 1fr; }
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      background: white;
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
    }
    th, td {
      padding: 8px 10px;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }
    th {
      text-align: left;
      background: #fbfbfb;
      font-size: 12px;
      color: var(--muted);
    }
    tr:last-child td { border-bottom: 0; }
    .mono {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size: 12px;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .error {
      border: 1px solid rgba(153,0,0,0.35);
      background: rgba(153,0,0,0.06);
      color: #4a0000;
      padding: 10px;
      border-radius: 12px;
      margin-top: 10px;
    }
  </style>
</head>
<body>
  <header>
    <h1>DTU Sentiment Demo Frontend</h1>
    <p>Calls a student-built <span class="mono">POST /v1/sentiment</span> service and visualizes scores, labels, and batch accuracy.</p>
  </header>

  <main>
    <div class="row">
      <div class="card">
        <div class="grid2">
          <div>
            <label for="serviceUrl">External service base URL</label>
            <input id="serviceUrl" type="text" value="__DEFAULT_SERVICE_URL__" />
            <div class="small">
              Example: <span class="mono">http://localhost:8000</span> (service docs often at <span class="mono">/docs</span>)
            </div>
          </div>
          <div>
            <label for="sampleSelect">Predefined sample (optional)</label>
            <select id="sampleSelect"></select>
            <div class="small">Pick a sample to copy into the text box.</div>
          </div>
        </div>

        <div style="margin-top: 12px;">
          <label for="textInput">Course evaluation text</label>
          <textarea id="textInput" placeholder="Type or paste a short evaluation..."></textarea>
        </div>

        <div class="buttons">
          <button id="scoreBtn">Score text</button>
          <button id="fillBtn" class="ghost">Use selected sample</button>
          <button id="batchBtn" class="secondary">Run dataset & accuracy</button>
        </div>

        <div id="singleResult" class="result" style="display:none;"></div>
        <div id="singleError" class="error" style="display:none;"></div>
      </div>

      <div class="card">
        <h3 style="margin:0 0 10px 0; font-size: 16px;">Operational metrics</h3>
        <div id="metricsBox" class="result">
          <div class="small">No requests yet.</div>
        </div>

        <div style="margin-top: 14px;">
          <h3 style="margin:0 0 10px 0; font-size: 16px;">Notes for students</h3>
          <div class="small">
            <ul style="margin: 6px 0 0 18px; padding: 0;">
              <li>This frontend calls <span class="mono">/api/score</span> on the backend, which then calls your service.</li>
              <li>If you get timeouts, open your service docs at <span class="mono">/docs</span> and try <span class="mono">/v1/sentiment</span> manually.</li>
              <li>Batch accuracy uses a simple mapping: score ≤ -1 → negative, -1..1 → neutral, score ≥ 1 → positive.</li>
            </ul>
          </div>
        </div>
      </div>
    </div>

    <div class="card" style="margin-top: 14px;">
      <h3 style="margin:0 0 10px 0; font-size: 16px;">Batch results</h3>
      <div id="batchSummary" class="result" style="display:none;"></div>
      <div id="batchError" class="error" style="display:none;"></div>
      <div id="batchTableWrap" style="margin-top: 10px; display:none;"></div>
    </div>
  </main>

  <script>
    // Dataset injected from backend (edit DATASET in app.py).
    const DATASET = __DATASET_JSON__;

    function escapeHtml(s) {
      s = String(s);
      return s.replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;");
    }

    function setVisible(id, visible) {
      document.getElementById(id).style.display = visible ? "" : "none";
    }

    function renderMetrics(m) {
      const box = document.getElementById("metricsBox");
      if (!m) {
        box.innerHTML = '<div class="small">No metrics.</div>';
        return;
      }
      const avg = (m.avg_latency_ms == null) ? "—" : m.avg_latency_ms.toFixed(1);
      const last = (m.last_latency_ms == null) ? "—" : m.last_latency_ms.toFixed(1);
      const p95 = (m.p95_latency_ms == null) ? "—" : m.p95_latency_ms.toFixed(1);
      box.innerHTML = `
        <div style="display:flex; flex-wrap:wrap; gap:8px;">
          <span class="pill">Total: ${m.total_requests}</span>
          <span class="pill">Success: ${m.success_requests}</span>
          <span class="pill">Failed: ${m.failed_requests}</span>
        </div>
        <div class="small" style="margin-top:10px;">
          Latency (ms): last <span class="mono">${last}</span>, avg <span class="mono">${avg}</span>, p95 <span class="mono">${p95}</span>
        </div>
      `;
    }

    async function refreshMetrics() {
      const r = await fetch("/api/metrics");
      const m = await r.json();
      renderMetrics(m);
    }

    function initSamples() {
      const sel = document.getElementById("sampleSelect");
      sel.innerHTML = "";
      const opt0 = document.createElement("option");
      opt0.value = "";
      opt0.textContent = "— select —";
      sel.appendChild(opt0);

      DATASET.forEach((row, i) => {
        const [text, gold] = row;
        const opt = document.createElement("option");
        opt.value = String(i);
        opt.textContent = `#${i+1} (${gold}) ${text.slice(0, 70)}${text.length > 70 ? "…" : ""}`;
        sel.appendChild(opt);
      });
    }

    function showSingleError(msg) {
      const box = document.getElementById("singleError");
      box.innerHTML = `<strong>Could not score the text.</strong><br/><div class="small" style="margin-top:6px;">${escapeHtml(msg)}</div>`;
      setVisible("singleError", true);
    }

    function clearSingle() {
      setVisible("singleError", false);
      setVisible("singleResult", false);
    }

    document.getElementById("fillBtn").addEventListener("click", () => {
      const sel = document.getElementById("sampleSelect").value;
      if (!sel) return;
      const [text, gold] = DATASET[Number(sel)];
      document.getElementById("textInput").value = text;
    });

    document.getElementById("scoreBtn").addEventListener("click", async () => {
      clearSingle();
      const serviceUrl = document.getElementById("serviceUrl").value.trim();
      const text = document.getElementById("textInput").value;

      if (!serviceUrl) {
        showSingleError("Please provide the base URL of the external service (e.g., http://localhost:8000).");
        return;
      }
      if (!text.trim()) {
        showSingleError("Please enter some text to score.");
        return;
      }
      const r = await fetch("/api/score", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ service_url: serviceUrl, text: text })
      });

      const data = await r.json();
      await refreshMetrics();

      if (!r.ok) {
        showSingleError(data.detail || "Unknown error.");
        return;
      }

      const box = document.getElementById("singleResult");
      box.innerHTML = `
        <div style="display:flex; flex-wrap:wrap; gap:8px;">
          <span class="pill">Score: <span class="mono">${data.score}</span></span>
          <span class="pill">Label: <span class="mono">${data.label}</span></span>
          <span class="pill">Latency: <span class="mono">${data.latency_ms.toFixed(1)} ms</span></span>
        </div>
        ${data.warning ? `<div class="small" style="margin-top:10px;">⚠ ${escapeHtml(data.warning)}</div>` : ""}
      `;
      setVisible("singleResult", true);
    });

    function showBatchError(msg) {
      const box = document.getElementById("batchError");
      box.innerHTML = `<strong>Batch run failed.</strong><br/><div class="small" style="margin-top:6px;">${escapeHtml(msg)}</div>`;
      setVisible("batchError", true);
    }

    function clearBatch() {
      setVisible("batchError", false);
      setVisible("batchSummary", false);
      setVisible("batchTableWrap", false);
      document.getElementById("batchTableWrap").innerHTML = "";
    }

    document.getElementById("batchBtn").addEventListener("click", async () => {
      clearBatch();
      const serviceUrl = document.getElementById("serviceUrl").value.trim();
      if (!serviceUrl) {
        showBatchError("Please provide the base URL of the external service (e.g., http://localhost:8000).");
        return;
      }
      if (!DATASET.length) {
        showBatchError("Dataset is empty. Paste your 40 examples into DATASET in app.py.");
        return;
      }

      const r = await fetch("/api/batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ service_url: serviceUrl, dataset: DATASET })
      });

      const data = await r.json();
      await refreshMetrics();

      if (!r.ok) {
        showBatchError(data.detail || "Unknown error.");
        return;
      }

      // Summary
      const sum = document.getElementById("batchSummary");
      sum.innerHTML = `
        <div style="display:flex; flex-wrap:wrap; gap:8px;">
          <span class="pill">Items: <span class="mono">${data.n}</span></span>
          <span class="pill">Accuracy: <span class="mono">${(100*data.accuracy).toFixed(1)}%</span></span>
          <span class="pill">Avg latency/item: <span class="mono">${data.avg_latency_ms.toFixed(1)} ms</span></span>
        </div>
        <div class="small" style="margin-top:10px;">
          Label mapping: score ≤ -1 → negative, -1..1 → neutral, score ≥ 1 → positive.
        </div>
      `;
      setVisible("batchSummary", true);

      // Table
      let html = `
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Text</th>
              <th>Gold</th>
              <th>Score</th>
              <th>Pred</th>
              <th>OK</th>
              <th>Latency (ms)</th>
            </tr>
          </thead>
          <tbody>
      `;
      data.rows.forEach((row, i) => {
        html += `
          <tr>
            <td class="mono">${i+1}</td>
            <td>${escapeHtml(row.text)}</td>
            <td class="mono">${row.gold}</td>
            <td class="mono">${row.score == null ? "—" : row.score}</td>
            <td class="mono">${row.pred == null ? "—" : row.pred}</td>
            <td class="mono">${row.ok ? "✓" : "×"}</td>
            <td class="mono">${row.latency_ms.toFixed(1)}</td>
          </tr>
        `;
      });
      html += "</tbody></table>";

      const wrap = document.getElementById("batchTableWrap");
      wrap.innerHTML = html;
      setVisible("batchTableWrap", true);
    });

    // Boot
    initSamples();
    refreshMetrics();
  </script>
</body>
</html>
""".replace("__DATASET_JSON__", dataset_js).replace("__DEFAULT_SERVICE_URL__", DEFAULT_SERVICE_URL)


@app.post("/api/score")
async def api_score(req: ScoreRequest) -> JSONResponse:
    """Score a single text via the external service."""
    score, info = await call_external_service(str(req.service_url), req.text)
    if score is None:
        # Pedagogic error messages come from call_external_service()
        return JSONResponse(status_code=502, content={"detail": info.get("error", "Unknown error.")})

    label = score_to_label(score)
    payload: Dict[str, Any] = {
        "score": score,
        "label": label,
        "latency_ms": info.get("latency_ms", None),
    }
    if "warning" in info:
        payload["warning"] = info["warning"]
    return JSONResponse(content=payload)


@app.post("/api/batch")
async def api_batch(req: BatchRequest) -> JSONResponse:
    """Run a batch evaluation on a [text, gold_label] dataset.

    Notes
    -----
    This runs requests sequentially for clarity (teaching). The student-built
    service should still be capable of handling concurrent clients; you can
    trivially parallelize this later if desired.
    """
    rows_out: List[Dict[str, Any]] = []
    latencies: List[float] = []
    correct = 0
    n = 0

    for item in req.dataset:
        if not (isinstance(item, list) and len(item) == 2):
            return JSONResponse(
                status_code=400,
                content={"detail": "Dataset must be a list of [text, gold_label] pairs."},
            )

        text, gold = item[0], item[1]
        if gold not in ("positive", "neutral", "negative"):
            return JSONResponse(
                status_code=400,
                content={"detail": f"Gold label must be positive/neutral/negative. Got: {gold!r}"},
            )

        score, info = await call_external_service(str(req.service_url), text)
        latency_ms = float(info.get("latency_ms", 0.0))
        latencies.append(latency_ms)

        if score is None:
            # Count as incorrect but keep going; useful for demos.
            rows_out.append(
                {
                    "text": text,
                    "gold": gold,
                    "score": None,
                    "pred": None,
                    "ok": False,
                    "latency_ms": latency_ms,
                    "error": info.get("error"),
                }
            )
            n += 1
            continue

        pred = score_to_label(score)
        ok = (pred == gold)
        correct += int(ok)
        n += 1
        rows_out.append(
            {
                "text": text,
                "gold": gold,
                "score": score,
                "pred": pred,
                "ok": ok,
                "latency_ms": latency_ms,
            }
        )

    accuracy = (correct / n) if n else 0.0
    avg_latency = statistics.mean(latencies) if latencies else 0.0

    return JSONResponse(
        content={
            "n": n,
            "correct": correct,
            "accuracy": accuracy,
            "avg_latency_ms": avg_latency,
            "rows": rows_out,
        }
    )


@app.get("/api/metrics")
def api_metrics() -> JSONResponse:
    """Return a snapshot of in-memory metrics."""
    return JSONResponse(content=metrics.snapshot())
