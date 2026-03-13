"""DTU RAG frontend web app.

This module implements a small FastAPI-based frontend web application for
interacting with and evaluating an independently built Retrieval-Augmented
Generation (RAG) web service.

The design goal is pedagogical simplicity:

- few dependencies
- a single Python file for the application
- minimal JavaScript without external libraries
- inline CSS in the HTML
- explicit and readable error messages
- asynchronous HTTP calls to support multiple concurrent requests during
  evaluation

The original assignment prompt from the user is intentionally included below.
This makes the generated code self-contained and preserves the rationale and
requirements behind the implementation.

Original prompt
---------------
You should make a Python-based frontend Web app perhaps with
Javascript that can use another Web service (that is built
independently by students). The Web app should make it easy to
demonstrate and test the Web service. The Web service is
retrieval-augmented generation. This exercise is describe in detail
below.

I want a frontend Web app with few dependencies so that installation
would be painless.

If there is any Javascript is should be relatively simple. Do not
include jQuery or other external library unless absolutely necessary.

The Web app may include some styling, but I would like to have it
simple and the style within HTML code rather than as a separate style
file. It is running at DTU where the primary colors are corporate red
(153,0,0), white and black. Some more colors at
https://designguide.dtu.dk/colours if needed.

The interface language of the Web app should be English.

The Web app could be implemented in FastAPI, Streamlit or other
framework, depending on what you would think is the most pedagogical
and has the least dependencies. I as a teacher and the students should
be able to understand the code even though the course is not about
frontend development. If docstrings are included make it in numpydoc
format and do not be afraid to add doctests if that is relevant.

There is a small test dataset (excerpt) for one of the endpoints: The
'ask' endpoint where there is a query and a ground truth answer,
corresponding to 'query' and 'answer' in the endpoint.

[
    {
        "query": "Which courses do Bjørn Sand Jensen teach?",
        "ground_truth_answer": "Bjørn Sand Jensen teaches 02451 Introduction to Machine Learning and 02452 Machine Learning.",
    },
    {
        "query": "Hvilke kurser underviser Bjørn Sand Jensen i?",
        "ground_truth_answer": "Bjørn Sand Jensen underviser i 02451 Introduktion til Machine Learning og 02452 Machine Learning.",
    },
    {
        "query": "Which other course besides 02451 does Bjorn Jensen teaches?",
        "ground_thruth_answer", "Bjørn Sand Jensen also teaches 02452 Machine Learning.",
    },
    ...


I can copy and paste the full dataset into the generated code.

If possible this small dataset can be automatically be tested and
result displayed in the Web app. I believe the test needs to be with
LLM-as-a-judge. There is access to a LLM API where the API key can be
read from a .env file that can be called to act as the LLM-as-a-judge.

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
CAMPUSAI_API_KEY = os.getenv("CAMPUSAI_API_KEY")

client = OpenAI(
    api_key=CAMPUSAI_API_KEY,
    base_url="https://chat.campusai.compute.dtu.dk/api/v1"
)

The LLM model is "Gemma3".

There are 20 test examples, so I would go for a case where one can
execute the full dataset or just part of it, one testcase.

I believe the LLM-as-a-judge show show: correctness, completeness,
hallucination, and a text explanation. Summarize the full dataset with
a table showing these aspects. The score could be 0 and 1.
There should be a summary for the full dataset which could be
summarized in a table.

For the other endpoint there could be a simple interface to enter query.

Please also make any error message pedagogic, and include appropriate
time out for the response from the Web service. Include operational
metrics, e.g., response latency from the Web service and/or number of
successful request. The Web service ought to handle multiple
asynchronous Web requests, so this could be

If you use Python or Javascript templates/formatting remember the level
of interpolation and escaping.

Include this prompt (the above text) as part of the generated code,
e.g., in a docstring.

Now I am showing the web service exercise text (do not implement this
- I and students will do this independently). This is not necessary to
include in the generated Web app.

Notes
-----
The frontend expects the student service to expose at least these two
endpoints:

- GET /v1/search?query=<text>&top_k=<int>&mode=<text>
- GET /v1/ask?query=<text>&top_k=<int>&mode=<text>

Examples
--------
Run locally::

    uvicorn rag-ui:app --port 8001 --reload

Open http://127.0.0.1:8001 in a browser.

The service URL for the student backend can then be set in the web UI,
for example::

    http://127.0.0.1:8000

"""

from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

load_dotenv()

APP_TITLE = "DTU RAG Frontend"
DEFAULT_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://127.0.0.1:8000")
DEFAULT_TIMEOUT_SECONDS = float(os.getenv("RAG_TIMEOUT_SECONDS", "30"))
DEFAULT_CONCURRENCY = int(os.getenv("RAG_EVAL_CONCURRENCY", "5"))
CAMPUSAI_API_KEY = os.getenv("CAMPUSAI_API_KEY")
CAMPUSAI_BASE_URL = os.getenv(
    "CAMPUSAI_BASE_URL", "https://chat.campusai.compute.dtu.dk/api/v1"
)
CAMPUSAI_MODEL = os.getenv("CAMPUSAI_MODEL", "Gemma 3 (Chat)")

# Paste the full dataset here if desired. The textarea in the web UI can also
# override this list during runtime.
DEFAULT_TEST_DATASET = [ 
    {
	"query": "Which courses do Bjørn Sand Jensen teach?",
	"ground_truth_answer": "Bjørn Sand Jensen teaches 02451 Introduction to Machine Learning and 02452 Machine Learning.",
    },
    {
	"query": "Hvilke kurser underviser Bjørn Sand Jensen i?",
	"ground_truth_answer": "Bjørn Sand Jensen underviser i 02451 Introduktion til Machine Learning og 02452 Machine Learning.",
    },
    {
	"query": "Which other course besides 02451 does Bjorn Jensen teaches?",
	"ground_truth_answer": "Bjørn Sand Jensen also teaches 02452 Machine Learning.",
    },
    {
	"query": "When does the MLops course run?",
	"ground_truth_answer": "The Machine Learning Operations course (02476) runs in January.",
    },
    {
	"query": "How many ECTS is the 02476?",
	"ground_truth_answer": "02476 Machine Learning Operations is 5 ECTS.",
    },
    {
	"query": "etcs 02476?",
	"ground_truth_answer": "The course 02476 Machine Learning Operations is 5 ECTS.",
    },
    {
	"query": "Does Ivana Kovalenka teach a course together with another teacher?",
	"ground_truth_answer": "Yes, Ivana Konvalinka 02464 Artificial Intelligence and Human Cognition with Tobias Andersen."
    },
    {
	"query": "Does the teacher in experiment in cognitive science also teach other courses?",
	"ground_truth_answer": "Yes, Ivana Konvalinka is the teacher and she also teaches 02464 Artificial Intelligence and Human Cognition.",
    },
    {
	"query": "Which courses in chemical engineering do Bjørn Sand Jensen teach?",
	"ground_truth_answer": "Bjørn Sand Jensen does not teach any course in chemical engineering. He teaches machine learning which might be useful in chemical engineer.",
    },
    {
	"query": "Which course teaches PyTorch?",
	"ground_truth_answer": "PyTorch in a Python library for deep learning and it is taught in 02456 Deep learning, 02461 Introduction to Intelligent Systems, 02981 Generative Modeling Summer School (GeMMS) and possible other courses.",
    },
    {
	"query": "Which machine learning courses run in January?",
	"ground_truth_answer": "02476 Machine Learning Operations runs in January and 10316 Materials design with machine learning and artificial intelligence.",
    },
    {
	"query": "Hvilke machinelearning kurser kører i januar",
	"ground_truth_answer": "02476 Machine Learning Operations kører i januar og 10316 Materialedesign med machine learning og kunstig intelligens.",
    },
    {
	"query": "I would like the course that Ivana is teaching, but not the one that Tobias is also teaching",
	"ground_truth_answer": "02455 Experiment in Cognitive Science is taught by Ivana Konvalinka only.",
    },
    {
	"query": "What is 2451?",
	"ground_truth_answer": "02451 Introduction to Machine Learning is a 5 ECTS Bachelor course running Tuesday afternoons in spring.",
    },
    {
	"query": "What is the course most similar to 02451?",
	"ground_truth_answer": "02452 Machine Learning is most similar to 02451 Introduction to Machine Learning.",
    },
    {
	"query": "What is the difference between 02451 and 2452?",
	"ground_truth_answer": "02451 Introduction to Machine Learning is a Bachelor course in the Spring F4A module while 02452 Machine Learning is a Master course in the Autumn E4A module."
    },
    {
	"query": "How many ECTS points is Tue Herlau's course?",
	"ground_truth_answer": "Tue Herlau is responsible for 02465 Introduction to reinforcement learning and control, which gives 5 ECTS points.",
    },
    {
	"query": "Are there any courses about MRI?",
	"ground_truth_answer": "There are several courses on MRI: 22476 Electromagnetism for Health Technology, 22481 Introduction to medical imaging, 22506 Medical magnetic resonance imaging and 22507 Advanced magnetic resonance imaging and some others."
    },
    {
	"query": "Which course is Hiba Nassar involved in?",
	"ground_truth_answer": "Hiba Nassar teaches 02462 Signal and Data."
    },
    {
	"query": "Underviser Hibar Nassa i signler ogdta?",
	"ground_truth_answer": "Ja, Hiba Nassar underviser i 02462 Signal og Data."
    },
]



app = FastAPI(title=APP_TITLE)


class SearchRequest(BaseModel):
    """Request model for proxying the student's search endpoint."""

    service_url: str = Field(default=DEFAULT_SERVICE_URL)
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    mode: str = Field(default="sparse")
    timeout_seconds: float = Field(default=DEFAULT_TIMEOUT_SECONDS, ge=1.0, le=120.0)


class AskRequest(BaseModel):
    """Request model for proxying the student's ask endpoint."""

    service_url: str = Field(default=DEFAULT_SERVICE_URL)
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    mode: str = Field(default="sparse")
    timeout_seconds: float = Field(default=DEFAULT_TIMEOUT_SECONDS, ge=1.0, le=120.0)


class EvalRequest(BaseModel):
    """Request model for running dataset-based evaluation."""

    service_url: str = Field(default=DEFAULT_SERVICE_URL)
    dataset_text: str = ""
    top_k: int = Field(default=5, ge=1, le=50)
    mode: str = Field(default="sparse")
    timeout_seconds: float = Field(default=DEFAULT_TIMEOUT_SECONDS, ge=1.0, le=120.0)
    start_index: int = Field(default=0, ge=0)
    max_cases: int | None = Field(default=None, ge=1, le=1000)
    concurrency: int = Field(default=DEFAULT_CONCURRENCY, ge=1, le=20)


@dataclass
class JudgeResult:
    """Evaluation for one test case.

    Attributes
    ----------
    correctness : int
        Binary score where 1 means correct.
    completeness : int
        Binary score where 1 means sufficiently complete.
    hallucination : int
        Binary score where 1 means the answer contains hallucinated content.
        Lower is better for this metric.
    explanation : str
        Short explanation from the judge model.

    Examples
    --------
    >>> result = JudgeResult(1, 1, 0, "Looks correct.")
    >>> asdict(result)["correctness"]
    1
    """

    correctness: int
    completeness: int
    hallucination: int
    explanation: str


async def fetch_json(
    url: str,
    params: dict[str, Any],
    timeout_seconds: float,
) -> tuple[dict[str, Any], float, int]:
    """Fetch JSON from an HTTP endpoint.

    Parameters
    ----------
    url : str
        Absolute URL.
    params : dict[str, Any]
        Query parameters sent with the request.
    timeout_seconds : float
        Request timeout in seconds.

    Returns
    -------
    tuple[dict[str, Any], float, int]
        Parsed JSON payload, latency in milliseconds, and HTTP status code.

    Raises
    ------
    HTTPException
        Raised with a pedagogical error message when the upstream service
        fails or cannot be reached.
    """
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.get(url, params=params)
        latency_ms = (time.perf_counter() - start) * 1000.0
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail=(
                f"The student web service did not respond within {timeout_seconds:.1f} "
                "seconds. Please check whether the service is running, whether the "
                "URL is correct, and whether the endpoint itself is stuck."
            ),
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "The frontend could not reach the student web service. Please check "
                "the base URL, network connectivity, port number, and whether Docker "
                "or uvicorn is actually running."
            ),
        ) from exc

    if response.status_code >= 400:
        response_text = response.text.strip()
        raise HTTPException(
            status_code=502,
            detail=(
                f"The student web service returned HTTP {response.status_code}. "
                f"Response body: {response_text[:500]}"
            ),
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "The student web service responded, but the body was not valid JSON. "
                "Please ensure the endpoint returns JSON and not HTML or plain text."
            ),
        ) from exc

    return payload, latency_ms, response.status_code


async def judge_answer(
    query: str,
    ground_truth_answer: str,
    predicted_answer: str,
) -> JudgeResult:
    """Judge one answer with an LLM.

    Parameters
    ----------
    query : str
        Original question.
    ground_truth_answer : str
        Expected answer.
    predicted_answer : str
        Answer returned by the student service.

    Returns
    -------
    JudgeResult
        Structured evaluation.

    Notes
    -----
    This function requires the environment variable ``CAMPUSAI_API_KEY``.
    If the key is missing, the function falls back to a transparent placeholder
    result rather than crashing.
    """
    if not CAMPUSAI_API_KEY:
        return JudgeResult(
            correctness=0,
            completeness=0,
            hallucination=0,
            explanation=(
                "No CAMPUSAI_API_KEY was found in the environment, so LLM-as-a-judge "
                "could not run. Add the key to .env to enable automatic judging."
            ),
        )

    system_prompt = (
        "You are a strict but fair evaluator of answers from a retrieval-augmented "
        "generation system for DTU course questions. Compare the predicted answer "
        "with the ground truth answer. Return JSON only with the keys: correctness, "
        "completeness, hallucination, explanation. Use binary scores 0 or 1. "
        "For hallucination, return 1 if the predicted answer contains unsupported or "
        "invented information, otherwise 0."
    )
    user_prompt = (
        f"Query:\n{query}\n\n"
        f"Ground truth answer:\n{ground_truth_answer}\n\n"
        f"Predicted answer:\n{predicted_answer}\n\n"
        "Return JSON only. Example: "
        '{"correctness": 1, "completeness": 1, "hallucination": 0, '
        '"explanation": "The answer matches the ground truth."}'
    )

    client = AsyncOpenAI(api_key=CAMPUSAI_API_KEY, base_url=CAMPUSAI_BASE_URL)
    try:
        completion = await client.chat.completions.create(
            model=CAMPUSAI_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = completion.choices[0].message.content or "{}"
        parsed = json.loads(content)
        return JudgeResult(
            correctness=int(parsed.get("correctness", 0)),
            completeness=int(parsed.get("completeness", 0)),
            hallucination=int(parsed.get("hallucination", 0)),
            explanation=str(parsed.get("explanation", "No explanation returned.")),
        )
    except Exception as exc:  # noqa: BLE001
        return JudgeResult(
            correctness=0,
            completeness=0,
            hallucination=0,
            explanation=(
                "The judge model could not complete the evaluation. This does not "
                f"necessarily mean the student answer was wrong. Technical error: {exc}"
            ),
        )


def parse_dataset(dataset_text: str) -> list[dict[str, str]]:
    """Parse a dataset entered in the user interface.

    Parameters
    ----------
    dataset_text : str
        JSON text representing a list of test cases.

    Returns
    -------
    list[dict[str, str]]
        Cleaned list with ``query`` and ``ground_truth_answer`` keys.

    Raises
    ------
    HTTPException
        If the JSON cannot be parsed or does not follow the expected format.
    """
    if not dataset_text.strip():
        return DEFAULT_TEST_DATASET

    try:
        parsed = json.loads(dataset_text)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "The dataset text is not valid JSON. Please paste a JSON list where "
                "each item contains at least 'query' and 'ground_truth_answer'."
            ),
        ) from exc

    if not isinstance(parsed, list):
        raise HTTPException(
            status_code=400,
            detail="The dataset must be a JSON list of test cases.",
        )

    cleaned: list[dict[str, str]] = []
    for index, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise HTTPException(
                status_code=400,
                detail=f"Dataset item {index} is not a JSON object.",
            )
        query = item.get("query")
        ground_truth = item.get("ground_truth_answer") or item.get("ground_thruth_answer")
        if not isinstance(query, str) or not isinstance(ground_truth, str):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Dataset item {index} must contain string fields 'query' and "
                    "'ground_truth_answer'."
                ),
            )
        cleaned.append(
            {"query": query.strip(), "ground_truth_answer": ground_truth.strip()}
        )
    return cleaned


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Render the single-page frontend."""
    dataset_pretty = json.dumps(DEFAULT_TEST_DATASET, ensure_ascii=False, indent=2)
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{APP_TITLE}</title>
  <style>
    :root {{
      --dtu-red: rgb(153, 0, 0);
      --dtu-red-dark: rgb(110, 0, 0);
      --white: #ffffff;
      --black: #111111;
      --gray: #f4f4f4;
      --border: #d8d8d8;
      --ok: #1f6f3f;
      --warn: #8a6700;
      --error: #8f1d1d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--black);
      background: var(--gray);
      line-height: 1.45;
    }}
    header {{
      background: var(--dtu-red);
      color: var(--white);
      padding: 1rem 1.25rem;
    }}
    main {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 1rem;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 1rem;
      align-items: start;
    }}
    .card {{
      background: var(--white);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1rem;
      box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }}
    h1, h2, h3 {{ margin-top: 0; }}
    label {{ display: block; font-weight: bold; margin-top: 0.6rem; }}
    input, select, textarea, button {{
      width: 100%;
      padding: 0.65rem;
      margin-top: 0.25rem;
      border: 1px solid #bcbcbc;
      border-radius: 4px;
      font: inherit;
    }}
    textarea {{ min-height: 10rem; resize: vertical; }}
    button {{
      background: var(--dtu-red);
      color: var(--white);
      border: none;
      cursor: pointer;
      margin-top: 0.8rem;
    }}
    button:hover {{ background: var(--dtu-red-dark); }}
    button.secondary {{ background: #333; }}
    .row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; }}
    .metrics {{ display: flex; flex-wrap: wrap; gap: 0.75rem; }}
    .metric {{
      background: #fafafa;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 0.6rem 0.8rem;
      min-width: 180px;
    }}
    .status {{
      padding: 0.75rem;
      border-radius: 6px;
      margin-top: 0.75rem;
      white-space: pre-wrap;
    }}
    .status.info {{ background: #eef4ff; border: 1px solid #c5d6ff; }}
    .status.ok {{ background: #edf8f0; border: 1px solid #bddcc6; }}
    .status.error {{ background: #fff0f0; border: 1px solid #f0c7c7; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 0.75rem; }}
    th, td {{ border: 1px solid var(--border); padding: 0.5rem; vertical-align: top; }}
    th {{ background: #fafafa; text-align: left; }}
    code, pre {{ white-space: pre-wrap; word-break: break-word; }}
    .small {{ font-size: 0.92rem; color: #444; }}
  </style>
</head>
<body>
  <header>
    <h1>{APP_TITLE}</h1>
    <div>Simple frontend for demonstrating and evaluating a student RAG web service.</div>
  </header>
  <main>
    <div class="card">
      <h2>Connection settings</h2>
      <div class="row">
        <div>
          <label for="serviceUrl">Student service base URL</label>
          <input id="serviceUrl" value="{DEFAULT_SERVICE_URL}" />
        </div>
        <div>
          <label for="timeoutSeconds">Timeout in seconds</label>
          <input id="timeoutSeconds" type="number" value="{DEFAULT_TIMEOUT_SECONDS}" min="1" max="120" step="1" />
        </div>
      </div>
      <div class="row">
        <div>
          <label for="mode">Mode</label>
          <select id="mode">
            <option value="sparse">sparse</option>
            <option value="dense">dense</option>
            <option value="hybrid">hybrid</option>
          </select>
        </div>
        <div>
          <label for="topK">Top k</label>
          <input id="topK" type="number" value="5" min="1" max="50" step="1" />
        </div>
      </div>
      <div class="metrics" id="globalMetrics"></div>
    </div>

    <div class="grid">
      <section class="card">
        <h2>Search endpoint</h2>
        <label for="searchQuery">Query</label>
        <input id="searchQuery" placeholder="e.g. MRI" />
        <button onclick="runSearch()">Run search</button>
        <div id="searchStatus"></div>
        <pre id="searchResult"></pre>
      </section>

      <section class="card">
        <h2>Ask endpoint</h2>
        <label for="askQuery">Question</label>
        <input id="askQuery" placeholder="e.g. Which course is Hiba Nassar involved in?" />
        <button onclick="runAsk()">Run ask</button>
        <div id="askStatus"></div>
        <pre id="askResult"></pre>
      </section>
    </div>

    <section class="card" style="margin-top: 1rem;">
      <h2>Dataset evaluation with LLM-as-a-judge</h2>
      <p class="small">
        Paste a JSON list of test cases. Each item should contain <code>query</code>
        and <code>ground_truth_answer</code>. The frontend will call the student
        <code>/v1/ask</code> endpoint and then use the configured CampusAI model as judge.
      </p>
      <label for="datasetText">Dataset JSON</label>
      <textarea id="datasetText">{dataset_pretty}</textarea>
      <div class="row">
        <div>
          <label for="startIndex">Start index</label>
          <input id="startIndex" type="number" value="0" min="0" step="1" />
        </div>
        <div>
          <label for="maxCases">Maximum number of cases</label>
          <input id="maxCases" type="number" value="" min="1" step="1" placeholder="leave empty to run all" />
        </div>
      </div>
      <div class="row">
        <div>
          <label for="concurrency">Concurrency</label>
          <input id="concurrency" type="number" value="{DEFAULT_CONCURRENCY}" min="1" max="20" step="1" />
        </div>
        <div>
          <label>&nbsp;</label>
          <button class="secondary" onclick="runEvaluation()">Run evaluation</button>
        </div>
      </div>
      <div id="evalStatus"></div>
      <div id="evalSummary"></div>
      <div id="evalTable"></div>
    </section>
  </main>

<script>
let appMetrics = {{
  totalRequests: 0,
  successfulRequests: 0,
  failedRequests: 0,
  lastLatencyMs: null,
  averageLatencyMs: null,
  latencies: []
}};

function escapeHtml(value) {{
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}}

function setStatus(id, type, text) {{
  const el = document.getElementById(id);
  el.className = 'status ' + type;
  el.textContent = text;
}}

function clearStatus(id) {{
  const el = document.getElementById(id);
  el.className = '';
  el.textContent = '';
}}

function getCommonPayload() {{
  return {{
    service_url: document.getElementById('serviceUrl').value.trim(),
    timeout_seconds: Number(document.getElementById('timeoutSeconds').value),
    top_k: Number(document.getElementById('topK').value),
    mode: document.getElementById('mode').value
  }};
}}

function updateMetrics(latencyMs, ok) {{
  appMetrics.totalRequests += 1;
  if (ok) appMetrics.successfulRequests += 1;
  else appMetrics.failedRequests += 1;
  if (typeof latencyMs === 'number' && !Number.isNaN(latencyMs)) {{
    appMetrics.lastLatencyMs = latencyMs;
    appMetrics.latencies.push(latencyMs);
    const sum = appMetrics.latencies.reduce((a, b) => a + b, 0);
    appMetrics.averageLatencyMs = sum / appMetrics.latencies.length;
  }}

  const metrics = [
    ['Total frontend requests', appMetrics.totalRequests],
    ['Successful requests', appMetrics.successfulRequests],
    ['Failed requests', appMetrics.failedRequests],
    ['Last latency', appMetrics.lastLatencyMs ? appMetrics.lastLatencyMs.toFixed(1) + ' ms' : '—'],
    ['Average latency', appMetrics.averageLatencyMs ? appMetrics.averageLatencyMs.toFixed(1) + ' ms' : '—']
  ];
  document.getElementById('globalMetrics').innerHTML = metrics
    .map(([label, value]) => `<div class="metric"><div><strong>${{escapeHtml(label)}}</strong></div><div>${{escapeHtml(value)}}</div></div>`)
    .join('');
}}

async function postJson(url, payload) {{
  const response = await fetch(url, {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify(payload)
  }});
  const data = await response.json();
  if (!response.ok) {{
    throw new Error(data.detail || 'Unknown error');
  }}
  return data;
}}

async function runSearch() {{
  clearStatus('searchStatus');
  document.getElementById('searchResult').textContent = '';
  const payload = getCommonPayload();
  payload.query = document.getElementById('searchQuery').value;
  try {{
    setStatus('searchStatus', 'info', 'Running /v1/search ...');
    const data = await postJson('/api/search', payload);
    updateMetrics(data.metrics.latency_ms, true);
    setStatus('searchStatus', 'ok', `Search completed in ${{data.metrics.latency_ms.toFixed(1)}} ms.`);
    document.getElementById('searchResult').textContent = JSON.stringify(data, null, 2);
  }} catch (error) {{
    updateMetrics(null, false);
    setStatus('searchStatus', 'error', error.message);
  }}
}}

async function runAsk() {{
  clearStatus('askStatus');
  document.getElementById('askResult').textContent = '';
  const payload = getCommonPayload();
  payload.query = document.getElementById('askQuery').value;
  try {{
    setStatus('askStatus', 'info', 'Running /v1/ask ...');
    const data = await postJson('/api/ask', payload);
    updateMetrics(data.metrics.latency_ms, true);
    setStatus('askStatus', 'ok', `Ask completed in ${{data.metrics.latency_ms.toFixed(1)}} ms.`);
    document.getElementById('askResult').textContent = JSON.stringify(data, null, 2);
  }} catch (error) {{
    updateMetrics(null, false);
    setStatus('askStatus', 'error', error.message);
  }}
}}

function renderSummary(summary) {{
  return `
    <table>
      <thead>
        <tr>
          <th>Number of cases</th>
          <th>Mean correctness</th>
          <th>Mean completeness</th>
          <th>Mean hallucination</th>
          <th>Mean service latency (ms)</th>
          <th>Successful service calls</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>${{summary.number_of_cases}}</td>
          <td>${{summary.mean_correctness.toFixed(3)}}</td>
          <td>${{summary.mean_completeness.toFixed(3)}}</td>
          <td>${{summary.mean_hallucination.toFixed(3)}}</td>
          <td>${{summary.mean_latency_ms.toFixed(1)}}</td>
          <td>${{summary.successful_service_calls}}</td>
        </tr>
      </tbody>
    </table>
  `;
}}

function renderTable(rows) {{
  const header = `
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Query</th>
          <th>Ground truth answer</th>
          <th>Predicted answer</th>
          <th>Correctness</th>
          <th>Completeness</th>
          <th>Hallucination</th>
          <th>Latency (ms)</th>
          <th>Explanation</th>
        </tr>
      </thead>
      <tbody>
  `;
  const body = rows.map((row) => `
      <tr>
        <td>${{row.index}}</td>
        <td>${{escapeHtml(row.query)}}</td>
        <td>${{escapeHtml(row.ground_truth_answer)}}</td>
        <td>${{escapeHtml(row.predicted_answer)}}</td>
        <td>${{row.judge.correctness}}</td>
        <td>${{row.judge.completeness}}</td>
        <td>${{row.judge.hallucination}}</td>
        <td>${{row.latency_ms.toFixed(1)}}</td>
        <td>${{escapeHtml(row.judge.explanation)}}</td>
      </tr>
  `).join('');
  return header + body + '</tbody></table>';
}}

async function runEvaluation() {{
  clearStatus('evalStatus');
  document.getElementById('evalSummary').innerHTML = '';
  document.getElementById('evalTable').innerHTML = '';
  const payload = getCommonPayload();
  payload.dataset_text = document.getElementById('datasetText').value;
  payload.start_index = Number(document.getElementById('startIndex').value);
  const maxCasesValue = document.getElementById('maxCases').value.trim();
  payload.max_cases = maxCasesValue === '' ? null : Number(maxCasesValue);
  payload.concurrency = Number(document.getElementById('concurrency').value);
  try {{
    setStatus('evalStatus', 'info', 'Running evaluation. This may take a little while because each case calls both the student service and the judge model.');
    const data = await postJson('/api/evaluate', payload);
    updateMetrics(data.summary.mean_latency_ms, true);
    setStatus('evalStatus', 'ok', `Evaluation completed for ${{data.summary.number_of_cases}} case(s).`);
    document.getElementById('evalSummary').innerHTML = renderSummary(data.summary);
    document.getElementById('evalTable').innerHTML = renderTable(data.rows);
  }} catch (error) {{
    updateMetrics(null, false);
    setStatus('evalStatus', 'error', error.message);
  }}
}}

updateMetrics(null, true);
</script>
</body>
</html>
"""


@app.post("/api/search")
async def api_search(request: SearchRequest) -> JSONResponse:
    """Proxy the student's ``/v1/search`` endpoint."""
    payload, latency_ms, status_code = await fetch_json(
        f"{request.service_url.rstrip('/')}/v1/search",
        params={
            "query": request.query,
            "top_k": request.top_k,
            "mode": request.mode,
        },
        timeout_seconds=request.timeout_seconds,
    )
    return JSONResponse(
        {
            "upstream": payload,
            "metrics": {
                "latency_ms": latency_ms,
                "status_code": status_code,
                "endpoint": "/v1/search",
            },
        }
    )


@app.post("/api/ask")
async def api_ask(request: AskRequest) -> JSONResponse:
    """Proxy the student's ``/v1/ask`` endpoint."""
    payload, latency_ms, status_code = await fetch_json(
        f"{request.service_url.rstrip('/')}/v1/ask",
        params={
            "query": request.query,
            "top_k": request.top_k,
            "mode": request.mode,
        },
        timeout_seconds=request.timeout_seconds,
    )
    return JSONResponse(
        {
            "upstream": payload,
            "metrics": {
                "latency_ms": latency_ms,
                "status_code": status_code,
                "endpoint": "/v1/ask",
            },
        }
    )


@app.post("/api/evaluate")
async def api_evaluate(request: EvalRequest) -> JSONResponse:
    """Evaluate a subset of a dataset against the student's ask endpoint."""
    dataset = parse_dataset(request.dataset_text)
    subset = dataset[request.start_index :]
    if request.max_cases is not None:
        subset = subset[: request.max_cases]

    if not subset:
        raise HTTPException(
            status_code=400,
            detail="The chosen dataset slice is empty. Adjust start_index or max_cases.",
        )

    import asyncio

    semaphore = asyncio.Semaphore(request.concurrency)

    async def evaluate_one(index: int, item: dict[str, str]) -> dict[str, Any]:
        async def _run() -> dict[str, Any]:
            ask_payload, latency_ms, _ = await fetch_json(
                f"{request.service_url.rstrip('/')}/v1/ask",
                params={
                    "query": item["query"],
                    "top_k": request.top_k,
                    "mode": request.mode,
                },
                timeout_seconds=request.timeout_seconds,
            )
            predicted_answer = str(ask_payload.get("answer", "")).strip()
            judge = await judge_answer(
                query=item["query"],
                ground_truth_answer=item["ground_truth_answer"],
                predicted_answer=predicted_answer,
            )
            return {
                "index": request.start_index + index,
                "query": item["query"],
                "ground_truth_answer": item["ground_truth_answer"],
                "predicted_answer": predicted_answer,
                "latency_ms": latency_ms,
                "judge": asdict(judge),
                "upstream": ask_payload,
            }

        async with semaphore:
            return await _run()

    rows = await asyncio.gather(
        *(evaluate_one(i, item) for i, item in enumerate(subset)),
        return_exceptions=True,
    )

    cleaned_rows: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        if isinstance(row, Exception):
            cleaned_rows.append(
                {
                    "index": request.start_index + i,
                    "query": subset[i]["query"],
                    "ground_truth_answer": subset[i]["ground_truth_answer"],
                    "predicted_answer": "",
                    "latency_ms": 0.0,
                    "judge": asdict(
                        JudgeResult(
                            correctness=0,
                            completeness=0,
                            hallucination=0,
                            explanation=(
                                "Evaluation failed before judging. Error: "
                                f"{row}"
                            ),
                        )
                    ),
                    "upstream": None,
                }
            )
        else:
            cleaned_rows.append(row)

    latencies = [row["latency_ms"] for row in cleaned_rows]
    correctness = [row["judge"]["correctness"] for row in cleaned_rows]
    completeness = [row["judge"]["completeness"] for row in cleaned_rows]
    hallucination = [row["judge"]["hallucination"] for row in cleaned_rows]
    successful_service_calls = sum(1 for row in cleaned_rows if row["upstream"] is not None)

    summary = {
        "number_of_cases": len(cleaned_rows),
        "mean_correctness": statistics.fmean(correctness) if correctness else 0.0,
        "mean_completeness": statistics.fmean(completeness) if completeness else 0.0,
        "mean_hallucination": statistics.fmean(hallucination) if hallucination else 0.0,
        "mean_latency_ms": statistics.fmean(latencies) if latencies else 0.0,
        "successful_service_calls": successful_service_calls,
    }
    return JSONResponse({"summary": summary, "rows": cleaned_rows})
