"""
IR service for DTU courses with two-stage retrieval:
1) candidate retrieval with sparse BM25
2) mandatory CampusAI reranking (for search/similar/objectives)

The `/v1/ask` endpoint also uses CampusAI to generate an answer grounded
on the top retrieved course context.
"""

import json
import math
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

app = FastAPI(title="DTU IR - CampusAI Chat-Only")

DATA_PATH = Path(__file__).with_name("dtu_courses.jsonl")
ROOT_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
TOKEN_RE = re.compile(r"[a-z0-9]+")
JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")
BM25_K1 = 1.5
BM25_B = 0.75


def _load_env_file(path: Path) -> None:
    """Load KEY=VALUE pairs from .env into process env if not already set."""
    if not path.exists():
        return

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# Load root-level .env automatically (../.env) before reading env vars.
_load_env_file(ROOT_ENV_PATH)

CHAT_COMPLETIONS_URL = os.getenv(
    "CAMPUSAI_CHAT_URL",
    "https://chat.campusai.compute.dtu.dk/api/chat/completions",
)
CAMPUSAI_API_KEY = os.getenv("CAMPUS_AI_API_KEY") or os.getenv("CAMPUSAI_API_KEY", "")
CAMPUSAI_CHAT_MODEL = os.getenv("CAMPUSAI_CHAT_MODEL", "Gemma3")
HTTP_TIMEOUT = float(os.getenv("CAMPUSAI_TIMEOUT", "30"))
RERANK_TOP_N_DEFAULT = int(os.getenv("CAMPUSAI_RERANK_TOP_N", "20"))


def _normalize_chat_url(raw_url: str) -> str:
    """Ensure CampusAI URL has protocol; default to chat/completions endpoint."""
    url = (raw_url or "").strip()
    if not url:
        return "https://chat.campusai.compute.dtu.dk/api/chat/completions"
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return f"https://{url}"


class CourseResult(BaseModel):
    course_id: str
    title: str
    score: float


class ObjectiveResult(BaseModel):
    course_id: str
    title: str
    objective: str
    score: float


class SimilarResponse(BaseModel):
    query_course_id: str
    results: List[CourseResult]
    mode: Literal["sparse"]
    top_k: int


class SearchResponse(BaseModel):
    query: str
    results: List[CourseResult]
    mode: Literal["sparse"]


class ObjectivesSearchResponse(BaseModel):
    query: str
    results: List[ObjectiveResult]
    mode: Literal["sparse"]


class AskRequest(BaseModel):
    query: str
    top_k: int = 5


class AskResponse(BaseModel):
    query: str
    answer: str
    context_courses: List[CourseResult]
    model: str


class HealthResponse(BaseModel):
    status: str
    provider: str
    index_sizes: Dict[str, int]


COURSES: Dict[str, Dict[str, object]] = {}
COURSE_IDS: List[str] = []
COURSE_TERM_FREQS: Dict[str, Dict[str, int]] = {}
COURSE_DOC_LEN: Dict[str, int] = {}
COURSE_BM25_IDF: Dict[str, float] = {}
COURSE_AVG_DL: float = 1.0

OBJECTIVE_DOCS: List[Dict[str, str]] = []
OBJECTIVE_TERM_FREQS: List[Dict[str, int]] = []
OBJECTIVE_DOC_LEN: List[int] = []
OBJECTIVE_BM25_IDF: Dict[str, float] = {}
OBJECTIVE_AVG_DL: float = 1.0


class CampusAIClient:
    """Minimal client for CampusAI chat completions."""

    def __init__(self) -> None:
        self.url = _normalize_chat_url(CHAT_COMPLETIONS_URL)
        self.api_key = CAMPUSAI_API_KEY
        self.chat_model = CAMPUSAI_CHAT_MODEL

    def configured(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        if not self.configured():
            raise RuntimeError("CampusAI not configured: set CAMPUS_AI_API_KEY (or CAMPUSAI_API_KEY)")

        payload = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }

        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            res = client.post(self.url, headers=self._headers(), json=payload)

        if res.status_code >= 400:
            raise RuntimeError(f"CampusAI error {res.status_code}: {res.text[:400]}")

        body = res.json()
        choices = body.get("choices", [])
        if not choices:
            raise RuntimeError("CampusAI response has no choices")

        return choices[0]["message"]["content"].strip()


client = CampusAIClient()


def _normalize_text(text: str) -> str:
    """Lowercase + strip accents to make matching more robust."""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(_normalize_text(text))


def _tokenize_sparse(text: str) -> List[str]:
    """Tokenize into unigrams + bigrams for BM25 scoring."""
    tokens = _tokenize(text)
    bigrams = [f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]
    return tokens + bigrams


def _collect_field_text(value: object) -> str:
    """Flatten nested JSON-like fields into searchable plain text."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(_collect_field_text(v) for v in value)
    if isinstance(value, dict):
        return " ".join(_collect_field_text(v) for v in value.values())
    return ""


def _build_bm25_stats(token_lists: List[List[str]]) -> Tuple[List[Dict[str, int]], List[int], Dict[str, float], float]:
    """Precompute tf/doclen/idf/avgdl used by BM25 ranking."""
    n_docs = len(token_lists)
    term_freqs: List[Dict[str, int]] = []
    doc_lens: List[int] = []
    df: Dict[str, int] = {}

    for tokens in token_lists:
        tf = dict(Counter(tokens))
        term_freqs.append(tf)
        doc_lens.append(len(tokens))
        for term in tf:
            df[term] = df.get(term, 0) + 1

    avg_dl = sum(doc_lens) / n_docs if n_docs else 1.0
    idf = {
        term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
        for term, freq in df.items()
    }
    return term_freqs, doc_lens, idf, avg_dl


def _bm25_score(
    query_tokens: List[str],
    doc_tf: Dict[str, int],
    doc_len: int,
    idf: Dict[str, float],
    avg_dl: float,
) -> float:
    """Compute BM25 score for one query against one document."""
    qtf = Counter(query_tokens)
    score = 0.0
    for term, q_weight in qtf.items():
        tf = doc_tf.get(term, 0)
        if tf == 0:
            continue
        term_idf = idf.get(term, 0.0)
        denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * (doc_len / max(avg_dl, 1e-9)))
        score += term_idf * ((tf * (BM25_K1 + 1.0)) / max(denom, 1e-9)) * q_weight
    return score


def _build_indexes() -> None:
    """Load dataset and build sparse indexes for courses and objectives."""
    global COURSES, COURSE_IDS
    global COURSE_TERM_FREQS, COURSE_DOC_LEN, COURSE_BM25_IDF, COURSE_AVG_DL
    global OBJECTIVE_DOCS, OBJECTIVE_TERM_FREQS, OBJECTIVE_DOC_LEN, OBJECTIVE_BM25_IDF, OBJECTIVE_AVG_DL

    if not DATA_PATH.exists():
        raise RuntimeError(f"Dataset not found: {DATA_PATH}")

    courses: Dict[str, Dict[str, object]] = {}
    course_ids: List[str] = []
    course_tokens_all: List[List[str]] = []

    objective_docs: List[Dict[str, str]] = []
    objective_tokens_all: List[List[str]] = []

    for line in DATA_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)

        course_id = str(row.get("course_code", "")).strip()
        title = str(row.get("title", "")).strip()
        objectives = [str(obj).strip() for obj in (row.get("learning_objectives", []) or []) if str(obj).strip()]
        fields_text = _collect_field_text(row.get("fields", {}))

        if not course_id:
            continue

        full_text = f"{title} {' '.join(objectives)} {fields_text}".strip()
        courses[course_id] = {"title": title, "text": full_text, "learning_objectives": objectives}

        course_ids.append(course_id)
        course_tokens_all.append(_tokenize_sparse(full_text))

        for obj in objectives:
            objective_docs.append({"course_id": course_id, "title": title, "objective": obj})
            objective_tokens_all.append(_tokenize_sparse(obj))

    course_tf_list, course_dl_list, course_idf, course_avg_dl = _build_bm25_stats(course_tokens_all)
    objective_tf_list, objective_dl_list, objective_idf, objective_avg_dl = _build_bm25_stats(objective_tokens_all)

    COURSES = courses
    COURSE_IDS = course_ids
    COURSE_TERM_FREQS = {cid: tf for cid, tf in zip(course_ids, course_tf_list)}
    COURSE_DOC_LEN = {cid: dl for cid, dl in zip(course_ids, course_dl_list)}
    COURSE_BM25_IDF = course_idf
    COURSE_AVG_DL = course_avg_dl

    OBJECTIVE_DOCS = objective_docs
    OBJECTIVE_TERM_FREQS = objective_tf_list
    OBJECTIVE_DOC_LEN = objective_dl_list
    OBJECTIVE_BM25_IDF = objective_idf
    OBJECTIVE_AVG_DL = objective_avg_dl


def rank_courses_sparse(query: str, top_k: int, exclude_course_id: Optional[str] = None) -> List[Tuple[str, float]]:
    """First-stage retrieval: rank courses with BM25."""
    q_tokens = _tokenize_sparse(query)
    scored: List[Tuple[str, float]] = []
    for cid in COURSE_IDS:
        if exclude_course_id and cid == exclude_course_id:
            continue
        score = _bm25_score(
            query_tokens=q_tokens,
            doc_tf=COURSE_TERM_FREQS[cid],
            doc_len=COURSE_DOC_LEN[cid],
            idf=COURSE_BM25_IDF,
            avg_dl=COURSE_AVG_DL,
        )
        scored.append((cid, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def rank_objectives_sparse(query: str, top_k: int) -> List[Tuple[int, float]]:
    """First-stage retrieval: rank learning objectives with BM25."""
    q_tokens = _tokenize_sparse(query)
    scored: List[Tuple[int, float]] = []
    for idx in range(len(OBJECTIVE_DOCS)):
        score = _bm25_score(
            query_tokens=q_tokens,
            doc_tf=OBJECTIVE_TERM_FREQS[idx],
            doc_len=OBJECTIVE_DOC_LEN[idx],
            idf=OBJECTIVE_BM25_IDF,
            avg_dl=OBJECTIVE_AVG_DL,
        )
        scored.append((idx, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def _llm_rerank(query: str, candidates: List[Tuple[str, str]], temperature: float = 0.0) -> List[str]:
    """Second-stage reranking with CampusAI.

    Input candidates are pairs (id, text). The model must return an ordered
    JSON array of IDs. Parsing is tolerant to minor output format drift.
    """
    if not candidates:
        return [cid for cid, _ in candidates]
    if not client.configured():
        raise RuntimeError("CampusAI not configured: set CAMPUS_AI_API_KEY (or CAMPUSAI_API_KEY)")

    candidate_lines = "\n".join([f"- {cid}: {text}" for cid, text in candidates])
    messages = [
        {
            "role": "system",
            "content": (
                "You are a ranking assistant. Return ONLY a JSON array of candidate IDs "
                "ordered from most relevant to least relevant for the query. "
                "Do not add explanations. No markdown fences. No extra keys."
            ),
        },
        {
            "role": "user",
            "content": f"Query: {query}\n\nCandidates:\n{candidate_lines}",
        },
    ]

    raw = client.chat(messages=messages, temperature=temperature)
    ordered: object
    try:
        ordered = json.loads(raw)
    except json.JSONDecodeError:
        # Handle common LLM outputs: fenced JSON or surrounding prose.
        match = JSON_ARRAY_RE.search(raw)
        if match:
            ordered = json.loads(match.group(0))
        else:
            # Last-resort extraction: keep candidate IDs in textual mention order.
            ids_in_order: List[str] = []
            for cid, _ in candidates:
                pos = raw.find(cid)
                if pos >= 0:
                    ids_in_order.append((pos, cid))
            if ids_in_order:
                ordered = [cid for _, cid in sorted(ids_in_order, key=lambda x: x[0])]
            else:
                raise RuntimeError("CampusAI rerank response is not parseable as ID list")

    if not isinstance(ordered, list):
        raise RuntimeError("CampusAI rerank response is not a JSON list")
    allowed = {cid for cid, _ in candidates}
    filtered = [cid for cid in ordered if isinstance(cid, str) and cid in allowed]
    missing = [cid for cid, _ in candidates if cid not in filtered]
    return filtered + missing


def _apply_course_rerank(query: str, ranked: List[Tuple[str, float]], rerank_top_n: int) -> List[Tuple[str, float]]:
    """Rerank only the head of BM25 results, keep tail order unchanged."""
    head = ranked[:rerank_top_n]
    tail = ranked[rerank_top_n:]
    score_map = {cid: score for cid, score in head}
    cands = [(cid, str(COURSES[cid]["title"])) for cid, _ in head]
    new_order = _llm_rerank(query, cands)
    reranked_head = [(cid, score_map[cid]) for cid in new_order]
    return reranked_head + tail


def _apply_objective_rerank(query: str, ranked: List[Tuple[int, float]], rerank_top_n: int) -> List[Tuple[int, float]]:
    """Rerank objective hits using synthetic IDs mapped back to list indices."""
    head = ranked[:rerank_top_n]
    tail = ranked[rerank_top_n:]
    score_map = {idx: score for idx, score in head}

    cands: List[Tuple[str, str]] = []
    id_to_idx: Dict[str, int] = {}
    for idx, _ in head:
        cid = OBJECTIVE_DOCS[idx]["course_id"]
        oid = f"O{idx}"
        text = f"{cid} | {OBJECTIVE_DOCS[idx]['objective']}"
        cands.append((oid, text))
        id_to_idx[oid] = idx

    new_order = _llm_rerank(query, cands)
    reranked_head = [(id_to_idx[oid], score_map[id_to_idx[oid]]) for oid in new_order if oid in id_to_idx]
    return reranked_head + tail


def _map_upstream_error(exc: Exception) -> HTTPException:
    """Map CampusAI/network errors to stable HTTP responses."""
    if isinstance(exc, httpx.TimeoutException):
        return HTTPException(status_code=504, detail="CampusAI request timed out")
    if isinstance(exc, httpx.HTTPError):
        return HTTPException(status_code=502, detail=f"CampusAI transport error: {exc}")
    return HTTPException(status_code=502, detail=str(exc))


@app.on_event("startup")
def startup() -> None:
    """Build indexes once at service startup."""
    _build_indexes()


@app.get("/v1/courses/{course_id}/similar", response_model=SimilarResponse)
def similar_courses(
    course_id: str,
    top_k: int = Query(10, ge=1, le=100),
    mode: Literal["sparse"] = Query("sparse"),
    rerank_top_n: int = Query(RERANK_TOP_N_DEFAULT, ge=1, le=100),
) -> SimilarResponse:
    """Return courses similar to a given course, always with CampusAI reranking."""
    if not client.configured():
        raise HTTPException(status_code=503, detail="CampusAI chat not configured")
    if course_id not in COURSES:
        raise HTTPException(status_code=404, detail=f"Unknown course_id: {course_id}")

    query_text = str(COURSES[course_id]["text"])
    ranked = rank_courses_sparse(query=query_text, top_k=max(top_k, rerank_top_n), exclude_course_id=course_id)
    try:
        ranked = _apply_course_rerank(query_text, ranked, rerank_top_n=rerank_top_n)
    except Exception as exc:
        raise _map_upstream_error(exc)

    results = [
        CourseResult(course_id=cid, title=str(COURSES[cid]["title"]), score=round(score, 3))
        for cid, score in ranked[:top_k]
    ]

    return SimilarResponse(query_course_id=course_id, results=results, mode=mode, top_k=top_k)


@app.get("/v1/search", response_model=SearchResponse)
def search_courses(
    query: str = Query(..., min_length=1),
    top_k: int = Query(10, ge=1, le=100),
    mode: Literal["sparse"] = Query("sparse"),
    rerank_top_n: int = Query(RERANK_TOP_N_DEFAULT, ge=1, le=100),
) -> SearchResponse:
    """Search courses by text query with mandatory CampusAI reranking."""
    if not client.configured():
        raise HTTPException(status_code=503, detail="CampusAI chat not configured")
    ranked = rank_courses_sparse(query=query, top_k=max(top_k, rerank_top_n))
    try:
        ranked = _apply_course_rerank(query, ranked, rerank_top_n=rerank_top_n)
    except Exception as exc:
        raise _map_upstream_error(exc)

    results = [
        CourseResult(course_id=cid, title=str(COURSES[cid]["title"]), score=round(score, 3))
        for cid, score in ranked[:top_k]
    ]
    return SearchResponse(query=query, results=results, mode=mode)


@app.get("/v1/objectives/search", response_model=ObjectivesSearchResponse)
def search_objectives(
    query: str = Query(..., min_length=1),
    top_k: int = Query(10, ge=1, le=100),
    mode: Literal["sparse"] = Query("sparse"),
    rerank_top_n: int = Query(RERANK_TOP_N_DEFAULT, ge=1, le=100),
) -> ObjectivesSearchResponse:
    """Search learning objectives with mandatory CampusAI reranking."""
    if not client.configured():
        raise HTTPException(status_code=503, detail="CampusAI chat not configured")
    ranked = rank_objectives_sparse(query=query, top_k=max(top_k, rerank_top_n))
    try:
        ranked = _apply_objective_rerank(query, ranked, rerank_top_n=rerank_top_n)
    except Exception as exc:
        raise _map_upstream_error(exc)

    results = [
        ObjectiveResult(
            course_id=OBJECTIVE_DOCS[idx]["course_id"],
            title=OBJECTIVE_DOCS[idx]["title"],
            objective=OBJECTIVE_DOCS[idx]["objective"],
            score=round(score, 3),
        )
        for idx, score in ranked[:top_k]
    ]
    return ObjectivesSearchResponse(query=query, results=results, mode=mode)


@app.post("/v1/ask", response_model=AskResponse)
def ask_courses(req: AskRequest) -> AskResponse:
    """Answer a user question using retrieved course context + CampusAI chat."""
    if not client.configured():
        raise HTTPException(status_code=503, detail="CampusAI chat not configured")

    ranked = rank_courses_sparse(req.query, top_k=req.top_k)
    context = [
        CourseResult(course_id=cid, title=str(COURSES[cid]["title"]), score=round(score, 3))
        for cid, score in ranked
    ]

    context_block = "\n\n".join(
        [
            f"[{c.course_id}] {c.title}\n"
            f"Objectives: {', '.join(COURSES[c.course_id]['learning_objectives'][:3])}"
            for c in context
        ]
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a DTU course assistant. Answer using only the provided context. "
                "If context is insufficient, state the limitation clearly."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {req.query}\n\nCourse context:\n{context_block}",
        },
    ]

    try:
        answer = client.chat(messages=messages, temperature=0.2)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="CampusAI request timed out")
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"CampusAI transport error: {exc}")
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return AskResponse(query=req.query, answer=answer, context_courses=context, model=client.chat_model)


@app.get("/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness + index-size check."""
    return HealthResponse(
        status="ok",
        provider=("campusai_chat" if client.configured() else "not_configured"),
        index_sizes={
            "courses": len(COURSES),
            "objectives": len(OBJECTIVE_DOCS),
        },
    )


@app.post("/v1/reindex")
def reindex() -> Dict[str, object]:
    """Rebuild in-memory indexes without restarting the service."""
    _build_indexes()
    return {
        "ok": True,
        "courses": len(COURSES),
        "objectives": len(OBJECTIVE_DOCS),
    }
