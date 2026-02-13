from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import httpx, os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse


app = FastAPI(title="DTU PDF Words Demo Frontend", version="1.0.0")

DEFAULT_SERVICE_URL = "http://pdf_service:8000"
REQUEST_TIMEOUT_SECONDS = 10.0

#call to the pdf_service endpoint
async def call_external_pdf_words(service_url: str, file: UploadFile) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    endpoint = service_url.rstrip("/") + "/v1/pdf-to-words"
    t0 = time.perf_counter()
    try:
        content = await file.read()
        files = {"file": (file.filename or "upload.pdf", content, file.content_type or "application/pdf")}

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            r = await client.post(endpoint, files=files)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        if r.status_code != 200:
            return None, {
                "latency_ms": latency_ms,
                "error": f"HTTP {r.status_code}. Body (truncated): {r.text[:200]!r}",
                "endpoint": endpoint,
            }

        return r.json(), {"latency_ms": latency_ms, "endpoint": endpoint}

    except httpx.TimeoutException:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return None, {
            "latency_ms": latency_ms,
            "error": f"Timeout after {REQUEST_TIMEOUT_SECONDS:.1f}s calling external service.",
            "endpoint": endpoint,
        }
    except httpx.RequestError as e:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return None, {
            "latency_ms": latency_ms,
            "error": f"Network error: {type(e).__name__}: {e}",
            "endpoint": endpoint,
        }
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return None, {
            "latency_ms": latency_ms,
            "error": f"Unexpected error: {type(e).__name__}: {e}",
            "endpoint": endpoint,
        }

def validate_pdf_filename(filename: str) -> bool:
    """
    Accept ONLY filenames ending with a .pdf extension.
    Reject:
      - double extensions (file.pdf.exe)
      - missing extension
      - non-pdf extensions
    """
    if not filename:
        return False

    base = os.path.basename(filename)
    lower = base.lower()

    # Must end with .pdf
    if not lower.endswith(".pdf"):
        return False

    # Reject multiple dots like file.pdf.exe or file.v1.pdf.exe
    name_without_pdf = lower[:-4]  # remove ".pdf"
    if "." in name_without_pdf:
        return False

    return True

@app.post("/api/pdf-words")
async def api_pdf_words(service_url: str = Form(...), file: UploadFile = File(...)) -> JSONResponse:
    if not service_url.strip():
        return JSONResponse(status_code=400, content={"detail": "Missing service_url."})
    if not validate_pdf_filename(file.filename or ""):
        return JSONResponse(
            status_code=400,
            content={"detail": "Only files with a single .pdf extension are allowed."}
        )

    if (file.content_type or "").lower() not in ("application/pdf", "application/x-pdf", "application/octet-stream"):
        return JSONResponse(status_code=400, content={"detail": f"File must be a PDF. Got: {file.content_type!r}"})

    data, info = await call_external_pdf_words(service_url, file)
    if data is None:
        return JSONResponse(status_code=502, content={"detail": info.get("error", "Unknown error.")})

    return JSONResponse(content={"latency_ms": info["latency_ms"], "endpoint": info["endpoint"], "data": data})


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>DTU PDF Words Demo</title>
  <style>
    :root {{
      --dtu-red: rgb(153,0,0);
      --bg: #ffffff;
      --fg: #111111;
      --muted: #666666;
      --card: #f6f6f6;
      --border: #dddddd;
    }}
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      color: var(--fg);
      background: var(--bg);
    }}
    header {{
      background: var(--dtu-red);
      color: white;
      padding: 14px 18px;
    }}
    main {{
      max-width: 900px;
      margin: 18px auto;
      padding: 0 14px 30px 14px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
    }}
    label {{
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
      margin-top: 10px;
    }}
    input[type="text"], input[type="file"] {{
      width: 100%;
      box-sizing: border-box;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px;
      font-size: 14px;
      background: white;
      color: var(--fg);
      outline: none;
    }}
    button {{
      margin-top: 12px;
      border: 0;
      border-radius: 10px;
      padding: 10px 12px;
      font-weight: 700;
      cursor: pointer;
      background: var(--dtu-red);
      color: white;
    }}
    .result {{
      margin-top: 12px;
      background: white;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px;
    }}
    .mono {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size: 12px;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .error {{
      border: 1px solid rgba(153,0,0,0.35);
      background: rgba(153,0,0,0.06);
      color: #4a0000;
      padding: 10px;
      border-radius: 12px;
      margin-top: 10px;
    }}
  </style>
</head>
<body>
  <header>
    <h1 style="margin:0; font-size:18px;">DTU PDF Words Demo</h1>
    <p style="margin:6px 0 0 0; font-size:13px; opacity:0.9;">
      Upload a PDF and show the external service response (expects <span class="mono">/v1/pdf-to-words</span>).
    </p>
  </header>

  <main>
    <div class="card">
      <label for="serviceUrl">External service base URL</label>
      <input id="serviceUrl" type="text" value="{DEFAULT_SERVICE_URL}" />

      <label for="pdfFile">PDF file</label>
      <input id="pdfFile" type="file" accept="application/pdf" />

      <button id="pdfBtn">Upload & extract words</button>

      <div id="pdfResult" class="result" style="display:none;"></div>
      <div id="pdfError" class="error" style="display:none;"></div>
    </div>
  </main>

  <script>
    function escapeHtml(s) {{
      s = String(s);
      return s.replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;");
    }}
    function setVisible(id, visible) {{
      document.getElementById(id).style.display = visible ? "" : "none";
    }}
    function showPdfError(msg) {{
      const box = document.getElementById("pdfError");
      box.innerHTML = `<strong>PDF request failed.</strong><br/><div style="margin-top:6px;">${{escapeHtml(msg)}}</div>`;
      setVisible("pdfError", true);
    }}
    function clearPdf() {{
      setVisible("pdfError", false);
      setVisible("pdfResult", false);
      document.getElementById("pdfResult").innerHTML = "";
    }}

    document.getElementById("pdfBtn").addEventListener("click", async () => {{
      clearPdf();

      const serviceUrl = document.getElementById("serviceUrl").value.trim();
      const fileInput = document.getElementById("pdfFile");
      const file = fileInput.files && fileInput.files[0];

      if (!serviceUrl) {{
        showPdfError("Please provide the base URL of the external service (e.g., http://localhost:8000).");
        return;
      }}
      if (!file) {{
        showPdfError("Please choose a PDF file first.");
        return;
      }}

      const form = new FormData();
      form.append("service_url", serviceUrl);
      form.append("file", file);

      const r = await fetch("/api/pdf-words", {{ method: "POST", body: form }});
      const data = await r.json();

      if (!r.ok) {{
        showPdfError(data.detail || "Unknown error.");
        return;
      }}

      const payload = data.data || {{}};
      const words = payload.words || payload.tokens || null;

      const box = document.getElementById("pdfResult");
      box.innerHTML = `
        <div><strong>Latency:</strong> <span class="mono">${{data.latency_ms.toFixed(1)}} ms</span></div>
        <div><strong>Endpoint:</strong> <span class="mono">${{escapeHtml(data.endpoint)}}</span></div>
        <hr/>
        ${{words ? `
          <div><strong>Words (${{words.length}}):</strong></div>
          <div class="mono" style="margin-top:6px;">${{escapeHtml(words.join(" "))}}</div>
        ` : `
          <div><strong>No words/tokens field found.</strong> Showing raw JSON:</div>
        `}}
        <div style="margin-top:10px;"><strong>Raw JSON:</strong></div>
        <div class="mono" style="margin-top:6px;">${{escapeHtml(JSON.stringify(payload, null, 2))}}</div>
      `;
      setVisible("pdfResult", true);
    }});
  </script>
</body>
</html>"""