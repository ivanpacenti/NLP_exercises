"""
Frontend Web App for demonstrating/testing an independently built Web service.

INCLUDE THIS PROMPT (the above text) as part of the generated code, e.g., in a docstring.

You should make a Python-based frontend Web app perhaps with
Javascript that can use another Web service (that is built
independently). The Web app should make it easy to demonstrate and
test the Web service. The Web service extracts sentences from a PDF
file. This exercise is describe in detail below.

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

The Web app implements entity recognition on a text. There is a small
test dataset (excerpt):

[
  {
    "text": "Ms Mette Frederiksen is in New York today.",
    "persons": ["Mette Frederiksen"]
  },
  {
    "text": "Einstein and von Neumann meet each other.",
    "persons": ["Einstein", "von Neumann"]
  },
  {
    "text": "Dr. Jane Goodall spoke with Prof. Brian Cox after the event.",
    "persons": ["Jane Goodall", "Brian Cox"]
  },
  {
    "text": "I ran the Einstein sum on the dataset, then exported the report.",
    "persons": []
  },

If possible this small dataset can be automatically be tested and
result displayed in the Web app.

Please also make any error message pedagogic, and include appropriate
time out for the response from the Web service. Include operational
metrics, e.g., response latency from the Web service and/or number of
successful request. The Web service ought to handle multiple
asynchronous Web requests, so this could be

Include this prompt (the above text) as part of the generated code,
e.g., in a docstring.

Now I am showing the web service exercise text (do not implement this
- I and students will do this independently). This is not necessary to
include in the generated Web app.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field


# ----------------------------
# Configuration
# ----------------------------

SERVICE_BASE_URL = os.environ.get("PERSON_SERVICE_URL", "http://localhost:8000").rstrip("/")
SERVICE_ENDPOINT = os.environ.get("PERSON_SERVICE_ENDPOINT", "/v1/extract-persons")
SERVICE_TIMEOUT_SECONDS = float(os.environ.get("PERSON_SERVICE_TIMEOUT_SECONDS", "10.0"))


# ----------------------------
# Small built-in test dataset
# ----------------------------

TEST_DATASET: List[Dict[str, Any]] = [
  {
    "text": "Ms Mette Frederiksen is in New York today.",
    "persons": ["Mette Frederiksen"]
  },
  {
    "text": "Einstein and von Neumann meet each other.",
    "persons": ["Einstein", "von Neumann"]
  },
  {
    "text": "Dr. Jane Goodall spoke with Prof. Brian Cox after the event.",
    "persons": ["Jane Goodall", "Brian Cox"]
  },
  {
    "text": "I ran the Einstein sum on the dataset, then exported the report.",
    "persons": []
  },
  {
    "text": "Mrs. Oprah Winfrey thanked Mr. John Legend for the introduction.",
    "persons": ["Oprah Winfrey", "John Legend"]
  },
  {
    "text": "The package was signed by A. Smith, but the label says 'Smithson Tools'.",
    "persons": ["A. Smith"]
  },
  {
    "text": "We discussed Ada Lovelace Jr. in class, but the slides were titled 'Lovelace Framework'.",
    "persons": ["Ada Lovelace Jr."]
  },
  {
    "text": "Captain Jean-Luc Picard reviewed the Picard protocol.",
    "persons": ["Jean-Luc Picard"]
  },
  {
    "text": "I emailed Mr. Tim Cook and Sundar Pichai about the schedule.",
    "persons": ["Tim Cook", "Sundar Pichai"]
  },
  {
    "text": "The mural by Banksy was mentioned, but no one knows the real name.",
    "persons": ["Banksy"]
  },
  {
    "text": "Yesterday, Serena Williams joined Rafael Nadal on stage.",
    "persons": ["Serena Williams", "Rafael Nadal"]
  },
  {
    "text": "We met with Ms. Mary-Jane Watson; later we saw the Watson & Co. sign.",
    "persons": ["Mary-Jane Watson"]
  },
  {
    "text": "The station is called Newton Road, not 'Isaac Newton'.",
    "persons": []
  },
  {
    "text": "Mr. Keanu Reeves was mistaken for 'Reeves Boulevard' on the map.",
    "persons": ["Keanu Reeves"]
  },
  {
    "text": "I read a quote by Søren Kierkegaard and one by Hannah Arendt.",
    "persons": ["Søren Kierkegaard", "Hannah Arendt"]
  },
  {
    "text": "Please forward this to Dr. Martin Luther King Jr., if you have his contact.",
    "persons": ["Martin Luther King Jr."]
  },
  {
    "text": "The seminar featured von der Leyen and Emmanuel Macron.",
    "persons": ["von der Leyen", "Emmanuel Macron"]
  },
  {
    "text": "We used the Turing test example, but Alan Turing was not the topic today.",
    "persons": ["Alan Turing"]
  },
  {
    "text": "I bought a 'Tesla' mug; Nikola Tesla wasn't mentioned anywhere on it.",
    "persons": ["Nikola Tesla"]
  },
  {
    "text": "Ms. Angelina Jolie visited the clinic; the sign outside reads 'Jolie Care'.",
    "persons": ["Angelina Jolie"]
  },
  {
    "text": "I saw 'Einstein Café' and thought of Albert Einstein.",
    "persons": ["Albert Einstein"]
  },
  {
    "text": "Barack Obama and Michelle Obama recorded a joint message.",
    "persons": ["Barack Obama", "Michelle Obama"]
  },
  {
    "text": "Mr. José Mourinho Jr. arrived late, but Mourinho Street was closed anyway.",
    "persons": ["José Mourinho Jr."]
  },
  {
    "text": "The library named after Virginia Woolf hosted a talk by Neil Gaiman.",
    "persons": ["Virginia Woolf", "Neil Gaiman"]
  },
  {
    "text": "We compared 'Darwin OS' with Linux; Charles Darwin was only a footnote.",
    "persons": ["Charles Darwin"]
  },
  {
    "text": "The letter was addressed to Prof. Noam Chomsky, care of Chomsky Hall.",
    "persons": ["Noam Chomsky"]
  },
  {
    "text": "I messaged Mr. Elon Musk, but the reply came from 'Musk Logistics'.",
    "persons": ["Elon Musk"]
  },
  {
    "text": "Mrs. Malala Yousafzai met with Greta Thunberg in Oslo.",
    "persons": ["Malala Yousafzai", "Greta Thunberg"]
  },

  {
    "text": "Hr. Lars Løkke Rasmussen deltog i mødet i dag.",
    "persons": ["Lars Løkke Rasmussen"]
  },
  {
    "text": "Fru Helle Thorning-Schmidt besøgte campus, men skiltet sagde 'Thorning Plads'.",
    "persons": ["Helle Thorning-Schmidt"]
  },
  {
    "text": "Betzy Meyers Høj i Ballerup er spærret på grund af vejarbejde.",
    "persons": []
  },
  {
    "text": "Jeg så Mads Mikkelsen på plakaten, men det var bare 'Mikkelsen & Søn'.",
    "persons": ["Mads Mikkelsen"]
  },
  {
    "text": "Ms. Pia Kjærsgaard holdt tale, og bagefter gik vi forbi Kjærsgaard Allé.",
    "persons": ["Pia Kjærsgaard"]
  },
  {
    "text": "Professor Niels Bohr nævnes ofte, men her handler det om Bohr-variablen i koden.",
    "persons": ["Niels Bohr"]
  },
  {
    "text": "Vi drak kaffe paa 'Andersen Cafe' - ikke noget med H.C. Andersen i teksten.",
    "persons": ["H.C. Andersen"]
  },
  {
    "text": "Jeg mødte Sofie Linde og Jakob Ellemann-Jensen ved indgangen.",
    "persons": ["Sofie Linde", "Jakob Ellemann-Jensen"]
  },
  {
    "text": "Dr. Anders Fogh Rasmussen kom for sent, fordi Foghvej var lukket.",
    "persons": ["Anders Fogh Rasmussen"]
  },
  {
    "text": "Kald lige på frk. Karen Blixen, hvis hun er i bygningen.",
    "persons": ["Karen Blixen"]
  },
  {
    "text": "Vi snakkede om 'einstein sum' i regnearket, ikke om Albert Einstein.",
    "persons": []
  },
  {
    "text": "Statsminister Mette Frederiksen og Kong Frederik mødtes til receptionen.",
    "persons": ["Mette Frederiksen", "Kong Frederik"]
  }
]


# ----------------------------
# Metrics (simple in-memory)
# ----------------------------

@dataclass
class Metrics:
    """In-memory operational metrics.

    Notes
    -----
    This is intentionally simple and process-local (no database).
    If you run multiple worker processes, each process will have its own metrics.
    """
    total_requests: int = 0
    success_requests: int = 0
    failed_requests: int = 0
    last_latency_ms: Optional[float] = None
    latencies_ms: List[float] = field(default_factory=list)

    def record(self, ok: bool, latency_ms: float) -> None:
        """Record one request outcome.

        Parameters
        ----------
        ok:
            Whether the request was successful.
        latency_ms:
            Latency in milliseconds.
        """
        self.total_requests += 1
        self.last_latency_ms = latency_ms
        self.latencies_ms.append(latency_ms)
        if ok:
            self.success_requests += 1
        else:
            self.failed_requests += 1

    def summary(self) -> Dict[str, Any]:
        """Return metrics summary as a JSON-serializable dict."""
        avg = (sum(self.latencies_ms) / len(self.latencies_ms)) if self.latencies_ms else None
        return {
            "total_requests": self.total_requests,
            "success_requests": self.success_requests,
            "failed_requests": self.failed_requests,
            "last_latency_ms": self.last_latency_ms,
            "avg_latency_ms": avg,
            "service_base_url": SERVICE_BASE_URL,
            "service_endpoint": SERVICE_ENDPOINT,
            "timeout_seconds": SERVICE_TIMEOUT_SECONDS,
        }


METRICS = Metrics()


# ----------------------------
# API models
# ----------------------------

class ExtractPersonsRequest(BaseModel):
    """Request schema for extracting person names."""
    text: str = Field(..., min_length=1, description="Input text to run person extraction on.")


class ExtractPersonsResponse(BaseModel):
    """Response schema expected from the external Web service."""
    persons: List[str] = Field(default_factory=list)


# ----------------------------
# Service client
# ----------------------------

async def call_person_service(text: str) -> Tuple[ExtractPersonsResponse, float]:
    """Call the external person-extraction service.

    Parameters
    ----------
    text:
        The input text.

    Returns
    -------
    response, latency_ms:
        Parsed response and request latency in milliseconds.

    Raises
    ------
    HTTPException
        If the service is unreachable, times out, or returns an invalid response.

    Examples
    --------
    The function expects the external service to accept::

        POST {SERVICE_BASE_URL}{SERVICE_ENDPOINT}
        {"text": "Einstein and von Neumann meet each other."}

    and respond with::

        {"persons": ["Einstein", "von Neumann"]}
    """
    url = f"{SERVICE_BASE_URL}{SERVICE_ENDPOINT}"
    start = time.perf_counter()

    timeout = httpx.Timeout(SERVICE_TIMEOUT_SECONDS)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(url, json={"text": text})
        except httpx.ConnectError as e:
            latency_ms = (time.perf_counter() - start) * 1000.0
            METRICS.record(ok=False, latency_ms=latency_ms)
            raise HTTPException(
                status_code=502,
                detail=(
                    "Could not connect to the Web service.\n\n"
                    f"Checked: {url}\n"
                    "Typical fixes:\n"
                    "- Start the container / service\n"
                    "- Verify port mapping (e.g., -p 8000:8000)\n"
                    "- If running remotely, check firewall / host\n\n"
                    f"Technical detail: {type(e).__name__}"
                ),
            )
        except httpx.ReadTimeout:
            latency_ms = (time.perf_counter() - start) * 1000.0
            METRICS.record(ok=False, latency_ms=latency_ms)
            raise HTTPException(
                status_code=504,
                detail=(
                    "The Web service did not respond before the timeout.\n\n"
                    f"Timeout: {SERVICE_TIMEOUT_SECONDS:.1f}s\n"
                    f"Endpoint: {url}\n\n"
                    "Typical fixes:\n"
                    "- Increase PERSON_SERVICE_TIMEOUT_SECONDS\n"
                    "- Make the service faster (e.g., smaller model / caching)\n"
                    "- Check service logs for slow LLM calls"
                ),
            )

    latency_ms = (time.perf_counter() - start) * 1000.0

    # Pedagogic handling of non-2xx statuses
    if resp.status_code // 100 != 2:
        METRICS.record(ok=False, latency_ms=latency_ms)
        raise HTTPException(
            status_code=502,
            detail=(
                "The Web service returned an error.\n\n"
                f"HTTP status: {resp.status_code}\n"
                f"Endpoint: {url}\n"
                "Response body (first 500 chars):\n"
                f"{resp.text[:500]}"
            ),
        )

    # Validate JSON shape
    try:
        data = resp.json()
    except ValueError:
        METRICS.record(ok=False, latency_ms=latency_ms)
        raise HTTPException(
            status_code=502,
            detail=(
                "The Web service responded, but the response was not valid JSON.\n\n"
                f"Endpoint: {url}\n"
                "Tip: ensure the service returns application/json with a body like:\n"
                '{"persons": ["Name 1", "Name 2"]}'
            ),
        )

    if not isinstance(data, dict) or "persons" not in data or not isinstance(data["persons"], list):
        METRICS.record(ok=False, latency_ms=latency_ms)
        raise HTTPException(
            status_code=502,
            detail=(
                "The Web service returned JSON, but not in the expected format.\n\n"
                "Expected a JSON object with a key 'persons' holding a list of strings.\n"
                "Example:\n"
                '{"persons": ["Einstein", "von Neumann"]}\n\n'
                f"Received (first 500 chars): {str(data)[:500]}"
            ),
        )

    # Coerce to strings (pedagogic robustness)
    persons = [str(x) for x in data.get("persons", [])]
    METRICS.record(ok=True, latency_ms=latency_ms)
    return ExtractPersonsResponse(persons=persons), latency_ms


def normalize_person_list(xs: List[str]) -> List[str]:
    """Normalize a list of person strings for comparison.

    Strategy
    --------
    - Strip whitespace
    - Drop empty strings
    - Keep order-insensitive comparisons outside this function
    """
    out: List[str] = []
    for x in xs:
        y = x.strip()
        if y:
            out.append(y)
    return out


# ----------------------------
# FastAPI app
# ----------------------------

app = FastAPI(
    title="Text -> Persons Demo Frontend",
    description="A minimal frontend web app to demonstrate/test an external person-extraction service.",
    version="1.0.0",
)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Serve the demo UI as a single HTML page (inline CSS + small JS)."""
    # NOTE: Inline styling only (as requested). DTU-ish colors: red (153,0,0), white, black.
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Text -> Persons (Demo Frontend)</title>
  <style>
    :root {{
      --dtu-red: rgb(153,0,0);
      --bg: #ffffff;
      --ink: #111111;
      --muted: #666666;
      --panel: #fafafa;
      --border: #e5e5e5;
      --ok: #0f7b0f;
      --warn: #9a5b00;
      --err: #b00020;
    }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Noto Sans", "Liberation Sans", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    header {{
      background: var(--dtu-red);
      color: white;
      padding: 18px 16px;
    }}
    header h1 {{
      margin: 0;
      font-size: 18px;
      font-weight: 700;
      letter-spacing: 0.2px;
    }}
    header .sub {{
      margin-top: 6px;
      font-size: 13px;
      opacity: 0.95;
    }}
    main {{
      max-width: 980px;
      margin: 0 auto;
      padding: 16px;
      display: grid;
      grid-template-columns: 1fr;
      gap: 14px;
    }}
    .row {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
    }}
    @media (min-width: 960px) {{
      .row {{
        grid-template-columns: 1.1fr 0.9fr;
        align-items: start;
      }}
    }}
    .card {{
      border: 1px solid var(--border);
      background: var(--panel);
      border-radius: 12px;
      padding: 14px;
      box-shadow: 0 1px 0 rgba(0,0,0,0.03);
    }}
    .card h2 {{
      margin: 0 0 10px 0;
      font-size: 15px;
    }}
    .hint {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 6px;
      line-height: 1.35;
    }}
    textarea {{
      width: 100%;
      min-height: 130px;
      resize: vertical;
      font: inherit;
      border-radius: 10px;
      border: 1px solid var(--border);
      padding: 10px;
      background: white;
      box-sizing: border-box;
    }}
    button {{
      appearance: none;
      border: 0;
      background: var(--dtu-red);
      color: white;
      padding: 10px 12px;
      border-radius: 10px;
      cursor: pointer;
      font-weight: 700;
      letter-spacing: 0.2px;
    }}
    button.secondary {{
      background: white;
      color: var(--dtu-red);
      border: 1px solid var(--dtu-red);
    }}
    button:disabled {{
      opacity: 0.65;
      cursor: not-allowed;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      margin: 0;
      font-size: 13px;
      line-height: 1.35;
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px;
    }}
    .pill {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 12px;
      border: 1px solid var(--border);
      background: white;
      margin-right: 6px;
      margin-bottom: 6px;
    }}
    .ok {{ color: var(--ok); font-weight: 700; }}
    .warn {{ color: var(--warn); font-weight: 700; }}
    .err {{ color: var(--err); font-weight: 700; }}
    .grid2 {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
    }}
    @media (min-width: 720px) {{
      .grid2 {{ grid-template-columns: 1fr 1fr; }}
    }}
    footer {{
      color: var(--muted);
      font-size: 12px;
      padding: 10px 16px 18px 16px;
      max-width: 980px;
      margin: 0 auto;
    }}
    a {{ color: var(--dtu-red); }}
    code {{ background: rgba(0,0,0,0.04); padding: 1px 4px; border-radius: 6px; }}
  </style>
</head>
<body>
  <header>
    <h1>Text -> Persons (Demo Frontend)</h1>
    <div class="sub">
      Calls <code>{SERVICE_BASE_URL}{SERVICE_ENDPOINT}</code> - Timeout: <code>{SERVICE_TIMEOUT_SECONDS:.1f}s</code>
    </div>
  </header>

  <main>
    <div class="row">
      <section class="card">
        <h2>Try it</h2>
        <textarea id="textInput">Einstein and von Neumann meet each other.</textarea>
        <div style="display:flex; gap:10px; flex-wrap: wrap; margin-top: 10px;">
          <button id="btnExtract">Extract persons</button>
          <button id="btnClear" class="secondary">Clear</button>
        </div>
        <div class="hint">
          Tip: The backend service is built independently (e.g., students' container).
          If this page cannot connect, check that the service is running and reachable.
        </div>
      </section>

      <section class="card">
        <h2>Result</h2>
        <div id="statusLine" class="hint">No request yet.</div>
        <div style="margin-top: 10px;">
          <div id="personsPills"></div>
          <pre id="rawJson">{{}}</pre>
        </div>
      </section>
    </div>

    <div class="grid2">
      <section class="card">
        <h2>Small test suite</h2>
        <div class="hint">
          Runs the built-in dataset (4 examples) by calling the external service once per example.
          Comparison is order-insensitive.
        </div>
        <div style="display:flex; gap:10px; flex-wrap: wrap; margin-top: 10px;">
          <button id="btnRunTests">Run tests</button>
          <button id="btnLoadExample" class="secondary">Load random test text</button>
        </div>
        <div style="margin-top: 10px;">
          <pre id="testOutput">Press "Run tests".</pre>
        </div>
      </section>

      <section class="card">
        <h2>Operational metrics</h2>
        <div class="hint">Process-local counters + latency stats (ms).</div>
        <div style="margin-top: 10px;">
          <pre id="metricsOutput">Loading...</pre>
        </div>
        <div class="hint" style="margin-top: 10px;">
          Note: If you run multiple workers (e.g., <code>--workers 4</code>),
          each worker keeps its own metrics.
        </div>
      </section>
    </div>

    <section class="card">
      <h2>Service configuration</h2>
      <div class="hint">
        Change these via environment variables when starting this frontend:
        <ul>
          <li><code>PERSON_SERVICE_URL</code> (default: <code>http://localhost:8000</code>)</li>
          <li><code>PERSON_SERVICE_ENDPOINT</code> (default: <code>/v1/extract-persons</code>)</li>
          <li><code>PERSON_SERVICE_TIMEOUT_SECONDS</code> (default: <code>10</code>)</li>
        </ul>
      </div>
    </section>
  </main>

  <footer>
    Minimal JS + inline CSS by design (pedagogical). If you need CORS, prefer calling the service through this frontend's
    <code>/api/*</code> endpoints (already same-origin).
  </footer>

<script>
  async function fetchJson(url, opts) {{
    const res = await fetch(url, opts);
    const contentType = res.headers.get("content-type") || "";
    let payload = null;
    if (contentType.includes("application/json")) {{
      payload = await res.json();
    }} else {{
      payload = await res.text();
    }}
    if (!res.ok) {{
      // Standardize errors from FastAPI (detail field)
      if (payload && typeof payload === "object" && payload.detail) {{
        throw new Error(payload.detail);
      }}
      throw new Error(typeof payload === "string" ? payload : "Request failed.");
    }}
    return payload;
  }}

  function setStatus(text, cls) {{
    const el = document.getElementById("statusLine");
    el.textContent = text;
    el.className = cls ? cls : "hint";
  }}

  function renderPersons(persons) {{
    const wrap = document.getElementById("personsPills");
    wrap.innerHTML = "";
    if (!persons || persons.length === 0) {{
      const p = document.createElement("div");
      p.className = "hint";
      p.textContent = "No persons returned.";
      wrap.appendChild(p);
      return;
    }}
    persons.forEach(x => {{
      const pill = document.createElement("span");
      pill.className = "pill";
      pill.textContent = x;
      wrap.appendChild(pill);
    }});
  }}

  async function refreshMetrics() {{
    try {{
      const m = await fetchJson("/api/metrics", {{ method: "GET" }});
      document.getElementById("metricsOutput").textContent = JSON.stringify(m, null, 2);
    }} catch (e) {{
      document.getElementById("metricsOutput").textContent = String(e);
    }}
  }}

  async function doExtract() {{
    const text = document.getElementById("textInput").value || "";
    if (!text.trim()) {{
      setStatus("Please enter some text first.", "warn");
      return;
    }}
    const btn = document.getElementById("btnExtract");
    btn.disabled = true;
    setStatus("Calling service...", "hint");
    try {{
      const payload = await fetchJson("/api/extract-persons", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{ text }})
      }});
      renderPersons(payload.persons || []);
      document.getElementById("rawJson").textContent = JSON.stringify(payload, null, 2);
      const ms = (payload && payload._meta && payload._meta.latency_ms) ? payload._meta.latency_ms.toFixed(1) : "?";
      setStatus(`OK - latency ${{ms}} ms`, "ok");
    }} catch (e) {{
      renderPersons([]);
      document.getElementById("rawJson").textContent = "{{}}";
      setStatus(String(e), "err");
    }} finally {{
      btn.disabled = false;
      refreshMetrics();
    }}
  }}

  async function runTests() {{
    const btn = document.getElementById("btnRunTests");
    btn.disabled = true;
    document.getElementById("testOutput").textContent = "Running tests...";
    try {{
      const payload = await fetchJson("/api/run-tests", {{ method: "POST" }});
      document.getElementById("testOutput").textContent = JSON.stringify(payload, null, 2);
    }} catch (e) {{
      document.getElementById("testOutput").textContent = String(e);
    }} finally {{
      btn.disabled = false;
      refreshMetrics();
    }}
  }}

  function loadRandomTestText() {{
    const examples = {repr([x["text"] for x in TEST_DATASET])};
    const text = examples[Math.floor(Math.random() * examples.length)];
    document.getElementById("textInput").value = text;
    setStatus("Loaded a random test example into the text box.", "hint");
  }}

  document.getElementById("btnExtract").addEventListener("click", doExtract);
  document.getElementById("btnClear").addEventListener("click", () => {{
    document.getElementById("textInput").value = "";
    renderPersons([]);
    document.getElementById("rawJson").textContent = "{{}}";
    setStatus("Cleared.", "hint");
  }});
  document.getElementById("btnRunTests").addEventListener("click", runTests);
  document.getElementById("btnLoadExample").addEventListener("click", loadRandomTestText);

  refreshMetrics();
  setInterval(refreshMetrics, 2500);
</script>
</body>
</html>
"""


@app.post("/api/extract-persons")
async def api_extract_persons(req: ExtractPersonsRequest) -> JSONResponse:
    """Proxy endpoint (same-origin) that calls the external service.

    Returns
    -------
    JSONResponse
        Includes a small `_meta` field with latency.
    """
    resp, latency_ms = await call_person_service(req.text)
    payload = {'persons': resp.persons}
    payload["_meta"] = {"latency_ms": latency_ms}
    return JSONResponse(payload)


@app.get("/api/metrics")
async def api_metrics() -> JSONResponse:
    """Return in-memory operational metrics."""
    return JSONResponse(METRICS.summary())


@app.post("/api/run-tests")
async def api_run_tests() -> JSONResponse:
    """Run the built-in test dataset against the external service.

    Notes
    -----
    - Calls the external service once per example.
    - Comparison is order-insensitive.
    - This is designed for pedagogy, not for large-scale benchmarking.
    """
    results: List[Dict[str, Any]] = []
    passed = 0

    for ex in TEST_DATASET:
        text = ex["text"]
        expected = normalize_person_list(list(ex["persons"]))

        try:
            got_resp, latency_ms = await call_person_service(text)
            got = normalize_person_list(got_resp.persons)
            ok = sorted(got) == sorted(expected)
            if ok:
                passed += 1
            results.append(
                {
                    "text": text,
                    "expected": expected,
                    "got": got,
                    "pass": ok,
                    "latency_ms": latency_ms,
                }
            )
        except HTTPException as e:
            results.append(
                {
                    "text": text,
                    "expected": expected,
                    "got": None,
                    "pass": False,
                    "error": e.detail,
                }
            )

    summary = {
        "total": len(TEST_DATASET),
        "passed": passed,
        "failed": len(TEST_DATASET) - passed,
        "pass_rate": passed / len(TEST_DATASET) if TEST_DATASET else None,
    }
    return JSONResponse({"summary": summary, "results": results})
