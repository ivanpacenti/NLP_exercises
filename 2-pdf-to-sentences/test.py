#!/usr/bin/env python3
"""
Red Team Tests for DTU PDF-to-Words service.

Usage examples:

# 1) Test the FRONTEND
python test.py --target frontend --base-url http://localhost:8001 --service-url http://pdf_service:8000

# 2) Test the PDF SERVICE directly
python test.py --target service --base-url http://localhost:8000

"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import string
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx


# ----------------------------
# Helpers
# ----------------------------

@dataclass
class TestResult:
    name: str
    ok: bool
    status_code: Optional[int]
    elapsed_ms: float
    detail: str

def now_ms() -> float:
    return time.perf_counter() * 1000.0

def rand_bytes(n: int) -> bytes:
    return os.urandom(n)

def fake_pdf_bytes() -> bytes:
    """
    A file that *pretends* to be PDF-like but is not a valid PDF.
    Some servers check for %PDF header; this starts with that but is still invalid.
    """
    return b"%PDF-1.4\n%Fake\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\nNOT_A_REAL_PDF"

def tiny_valid_pdf_bytes_minimal() -> bytes:
    # This is a very small PDF with one empty page (common minimal sample).
    return (
        b"%PDF-1.4\n"
        b"1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n"
        b"2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj\n"
        b"3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] "
        b"/Contents 4 0 R /Resources <<>> >>endobj\n"
        b"4 0 obj<< /Length 0 >>stream\nendstream\nendobj\n"
        b"xref\n0 5\n0000000000 65535 f \n"
        b"0000000010 00000 n \n"
        b"0000000062 00000 n \n"
        b"0000000117 00000 n \n"
        b"0000000246 00000 n \n"
        b"trailer<< /Size 5 /Root 1 0 R >>\nstartxref\n320\n%%EOF\n"
    )

def random_filename(ext: str = ".pdf") -> str:
    s = "".join(random.choice(string.ascii_lowercase) for _ in range(8))
    return f"{s}{ext}"

def snippet(text: str, n: int = 200) -> str:
    t = text.replace("\n", "\\n")
    return t[:n] + ("…" if len(t) > n else "")


# ----------------------------
# HTTP calls
# ----------------------------

async def post_to_service(
    client: httpx.AsyncClient,
    base_url: str,
    pdf_bytes: bytes,
    filename: str = "upload.pdf",
    content_type: str = "application/pdf",
) -> httpx.Response:
    """
    Direct call to PDF service:
      POST {base_url}/v1/pdf-to-words  (multipart field "file")
    """
    url = base_url.rstrip("/") + "/v1/pdf-to-words"
    files = {"file": (filename, pdf_bytes, content_type)}
    return await client.post(url, files=files)

async def post_to_frontend(
    client: httpx.AsyncClient,
    base_url: str,
    service_url: str,
    pdf_bytes: bytes,
    filename: str = "upload.pdf",
    content_type: str = "application/pdf",
) -> httpx.Response:
    """
    Call frontend:
      POST {base_url}/api/pdf-words  (multipart fields "service_url" + "file")
    """
    url = base_url.rstrip("/") + "/api/pdf-words"
    files = {"file": (filename, pdf_bytes, content_type)}
    data = {"service_url": service_url}
    return await client.post(url, data=data, files=files)

async def post_wrong_content_type_json(
    client: httpx.AsyncClient,
    url: str,
) -> httpx.Response:
    """
    Send JSON instead of multipart to test wrong content type handling.
    """
    return await client.post(url, json={"hello": "world"})


# ----------------------------
# Tests
# ----------------------------

async def run_one_test(name: str, coro) -> TestResult:
    t0 = now_ms()
    try:
        r: httpx.Response = await coro
        elapsed = now_ms() - t0
        return TestResult(name=name, ok=True, status_code=r.status_code, elapsed_ms=elapsed, detail=r.text)
    except Exception as e:
        elapsed = now_ms() - t0
        return TestResult(name=name, ok=False, status_code=None, elapsed_ms=elapsed, detail=f"{type(e).__name__}: {e}")

def expect_status_in(res: TestResult, allowed: Tuple[int, ...], note: str) -> TestResult:
    if res.status_code in allowed:
        return TestResult(res.name, True, res.status_code, res.elapsed_ms, note)
    return TestResult(res.name, False, res.status_code, res.elapsed_ms, f"{note}. Got {res.status_code}. Body: {snippet(res.detail)}")

def expect_body_contains(res: TestResult, needle: str, note: str) -> TestResult:
    if needle in (res.detail or ""):
        return TestResult(res.name, True, res.status_code, res.elapsed_ms, note)
    return TestResult(res.name, False, res.status_code, res.elapsed_ms, f"{note}. Missing '{needle}'. Body: {snippet(res.detail)}")

async def red_team_suite(
    target: str,
    base_url: str,
    service_url: str,
    timeout_s: float,
    pdf_paths: List[str],
    concurrency: int,
) -> List[TestResult]:
    results: List[TestResult] = []

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        # Decide which caller to use
        async def send(pdf_bytes: bytes, filename="upload.pdf", ctype="application/pdf"):
            if target == "service":
                return await post_to_service(client, base_url, pdf_bytes, filename=filename, content_type=ctype)
            return await post_to_frontend(client, base_url, service_url, pdf_bytes, filename=filename, content_type=ctype)

        # --- 1) Validation ---
        r = await run_one_test("T1 fake pdf renamed txt", send(b"hello i am not a pdf", filename="note.pdf"))
        # acceptable: 400-ish (service) OR 502 (frontend proxy) depending on architecture
        results.append(expect_status_in(r, (400, 415, 422, 500, 502), "Should reject non-PDF / fail gracefully"))

        r = await run_one_test("T2 double extension file.pdf.exe", send(fake_pdf_bytes(), filename="file.pdf.exe"))
        results.append(expect_status_in(r, (400, 415, 422, 500, 502), "Should reject suspicious / invalid file"))

        r = await run_one_test("T3 corrupted random bytes", send(rand_bytes(4096), filename="corrupt.pdf"))
        results.append(expect_status_in(r, (400, 415, 422, 500, 502), "Should reject corrupted PDF gracefully"))

        # Wrong content-type (octet-stream) but pdf-ish bytes
        r = await run_one_test("T4 octet-stream content-type", send(fake_pdf_bytes(), filename="upload.pdf", ctype="application/octet-stream"))
        results.append(expect_status_in(r, (200, 400, 500, 502), "Should handle octet-stream (browser often uses it)"))

        # Missing service_url only applies to frontend
        if target == "frontend":
            url = base_url.rstrip("/") + "/api/pdf-words"
            r0 = await run_one_test("T5 missing service_url field", client.post(url, files={"file": ("x.pdf", fake_pdf_bytes(), "application/pdf")}))
            results.append(expect_status_in(r0, (400, 422), "Frontend should reject missing service_url"))

            r1 = await run_one_test("T6 wrong content-type (json instead of multipart)", post_wrong_content_type_json(client, url))
            results.append(expect_status_in(r1, (400, 415, 422), "Frontend should reject non-multipart request"))

        # Resource / DoS-ish (lightweight)
        # Large payload (tune size as needed). Keep modest to avoid killing your own machine.
        big = fake_pdf_bytes() + rand_bytes(2_000_000)  # ~2MB extra
        r = await run_one_test("T7 large upload (~2MB extra)", send(big, filename="big.pdf"))
        results.append(expect_status_in(r, (200, 400, 413, 500, 502), "Should handle or reject large upload gracefully"))

        # Concurrency test
        async def one_small(i: int):
            return await send(fake_pdf_bytes(), filename=f"c{i}.pdf")

        t0 = now_ms()
        rs = await asyncio.gather(*[run_one_test(f"T8 concurrency upload #{i+1}", one_small(i)) for i in range(concurrency)])
        elapsed = now_ms() - t0

        # here accept mixed outcomes; key is "no crashes/hangs". Mark PASS if most returned a response.
        responded = sum(1 for x in rs if x.status_code is not None)
        ok = responded >= max(1, int(0.8 * concurrency))
        results.append(TestResult(
            name=f"T8 concurrency summary ({concurrency} requests)",
            ok=ok,
            status_code=None,
            elapsed_ms=elapsed,
            detail=f"Responses received: {responded}/{concurrency}",
        ))

        # --- 4) Optional: real PDFs provided (quality tests) ---
        # check hyphenation / soft hyphen in PDF files.
        for p in pdf_paths:
            name = os.path.basename(p)
            try:
                with open(p, "rb") as f:
                    b = f.read()
            except Exception as e:
                results.append(TestResult(f"T9 open real pdf {name}", False, None, 0.0, f"Could not read file: {e}"))
                continue

            r = await run_one_test(f"T9 real pdf upload {name}", send(b, filename=name))
            # If it succeeds,check presence of "words" in response (depends on target response shape).
            if r.status_code == 200:
                results.append(expect_body_contains(r, "words", "Response should include a 'words' field (or at least show it in JSON)"))
            else:
                results.append(expect_status_in(r, (200,), "Real PDF should be accepted (if it's valid)"))

    return results


# ----------------------------
# CLI / main
# ----------------------------

def print_report(results: List[TestResult]) -> None:
    print("\n=== RED TEAM TEST REPORT ===\n")
    passed = 0
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        code = "-" if r.status_code is None else str(r.status_code)
        print(f"[{status}] {r.name} | HTTP={code} | {r.elapsed_ms:.1f} ms")
        if not r.ok:
            print(f"       {snippet(r.detail, 300)}")
        else:
            print(f"       {r.detail}")
        print()
        passed += int(r.ok)
    print(f"Summary: {passed}/{len(results)} passed\n")

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["frontend", "service"], default="frontend",
                    help="Test frontend (/api/pdf-words) or service directly (/v1/pdf-to-words).")
    ap.add_argument("--base-url", required=True,
                    help="Base URL of target. Frontend example: http://localhost:8001 ; Service example: http://localhost:8000")
    ap.add_argument("--service-url", default="http://pdf_service:8000",
                    help="Only used when target=frontend. The URL that frontend will call (e.g., http://pdf_service:8000).")
    ap.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout seconds.")
    ap.add_argument("--pdf", action="append", default=[], help="Optional path to a real PDF to include in tests (repeatable).")
    ap.add_argument("--concurrency", type=int, default=10, help="Number of concurrent upload requests for the stress test.")
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    results = asyncio.run(red_team_suite(
        target=args.target,
        base_url=args.base_url,
        service_url=args.service_url,
        timeout_s=args.timeout,
        pdf_paths=args.pdf,
        concurrency=args.concurrency,
    ))
    print_report(results)

if __name__ == "__main__":
    main()