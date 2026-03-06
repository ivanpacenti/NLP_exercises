import hashlib
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Literal, Tuple

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

app = FastAPI()

TOKEN_RE = re.compile(r"[a-z0-9]+")
DENSE_DIM = 256
DATA_PATH = Path(__file__).with_name("dtu_courses.jsonl")


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
    mode: Literal["dense", "sparse", "hybrid"]
    top_k: int


class SearchResponse(BaseModel):
    query: str
    results: List[CourseResult]
    mode: Literal["dense", "sparse", "hybrid"]


class ObjectivesSearchResponse(BaseModel):
    query: str
    results: List[ObjectiveResult]
    mode: Literal["dense", "sparse", "hybrid"]


class HealthResponse(BaseModel):
    status: str
    index_sizes: Dict[str, int]


COURSES: Dict[str, Dict[str, object]] = {}
COURSE_SPARSE_VECTORS: Dict[str, Dict[str, float]] = {}
COURSE_DENSE_VECTORS: Dict[str, List[float]] = {}
COURSE_IDF: Dict[str, float] = {}

OBJECTIVE_DOCS: List[Dict[str, str]] = []
OBJECTIVE_SPARSE_VECTORS: List[Dict[str, float]] = []
OBJECTIVE_DENSE_VECTORS: List[List[float]] = []
OBJECTIVE_IDF: Dict[str, float] = {}


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def _dense_hash(token: str, dim: int = DENSE_DIM) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


def _l2_norm(values: List[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


def _dot_dense(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _dot_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def _build_idf(token_lists: List[List[str]]) -> Dict[str, float]:
    doc_freq: Dict[str, int] = {}
    for tokens in token_lists:
        for term in set(tokens):
            doc_freq[term] = doc_freq.get(term, 0) + 1

    num_docs = max(1, len(token_lists))
    return {
        term: math.log((1 + num_docs) / (1 + df)) + 1.0
        for term, df in doc_freq.items()
    }


def _build_sparse_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    tf: Dict[str, int] = {}
    for tok in tokens:
        tf[tok] = tf.get(tok, 0) + 1

    vec: Dict[str, float] = {}
    for tok, cnt in tf.items():
        vec[tok] = float(cnt) * idf.get(tok, 1.0)

    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm > 0:
        for tok in list(vec.keys()):
            vec[tok] /= norm

    return vec


def _build_dense_vector(tokens: List[str]) -> List[float]:
    vec = [0.0] * DENSE_DIM
    for tok in tokens:
        vec[_dense_hash(tok)] += 1.0

    norm = _l2_norm(vec)
    if norm > 0:
        vec = [v / norm for v in vec]

    return vec


def _query_sparse_vector(query: str, idf: Dict[str, float]) -> Dict[str, float]:
    return _build_sparse_vector(_tokenize(query), idf)


def _query_dense_vector(query: str) -> List[float]:
    return _build_dense_vector(_tokenize(query))


def _hybrid_score(sparse_score: float, dense_score: float, mode: str, alpha: float) -> float:
    if mode == "sparse":
        return sparse_score
    if mode == "dense":
        return dense_score
    return alpha * dense_score + (1.0 - alpha) * sparse_score


def _build_indexes() -> None:
    global COURSES
    global COURSE_SPARSE_VECTORS, COURSE_DENSE_VECTORS, COURSE_IDF
    global OBJECTIVE_DOCS, OBJECTIVE_SPARSE_VECTORS, OBJECTIVE_DENSE_VECTORS, OBJECTIVE_IDF

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
        objectives = row.get("learning_objectives", []) or []

        if not course_id:
            continue

        objectives = [str(obj).strip() for obj in objectives if str(obj).strip()]
        full_text = f"{title} {' '.join(objectives)}".strip()

        courses[course_id] = {
            "title": title,
            "text": full_text,
            "learning_objectives": objectives,
        }

        course_ids.append(course_id)
        course_tokens_all.append(_tokenize(full_text))

        for objective in objectives:
            objective_docs.append(
                {
                    "course_id": course_id,
                    "title": title,
                    "objective": objective,
                    "text": objective,
                }
            )
            objective_tokens_all.append(_tokenize(objective))

    course_idf = _build_idf(course_tokens_all)
    course_sparse_vectors = {
        cid: _build_sparse_vector(tokens, course_idf)
        for cid, tokens in zip(course_ids, course_tokens_all)
    }
    course_dense_vectors = {
        cid: _build_dense_vector(tokens)
        for cid, tokens in zip(course_ids, course_tokens_all)
    }

    objective_idf = _build_idf(objective_tokens_all)
    objective_sparse_vectors = [
        _build_sparse_vector(tokens, objective_idf)
        for tokens in objective_tokens_all
    ]
    objective_dense_vectors = [
        _build_dense_vector(tokens)
        for tokens in objective_tokens_all
    ]

    COURSES = courses
    COURSE_IDF = course_idf
    COURSE_SPARSE_VECTORS = course_sparse_vectors
    COURSE_DENSE_VECTORS = course_dense_vectors

    OBJECTIVE_DOCS = objective_docs
    OBJECTIVE_IDF = objective_idf
    OBJECTIVE_SPARSE_VECTORS = objective_sparse_vectors
    OBJECTIVE_DENSE_VECTORS = objective_dense_vectors


@app.on_event("startup")
def on_startup() -> None:
    _build_indexes()


@app.get("/v1/courses/{course_id}/similar", response_model=SimilarResponse)
def similar_courses(
    course_id: str,
    top_k: int = Query(10, ge=1, le=100),
    mode: Literal["dense", "sparse", "hybrid"] = Query("dense"),
    alpha: float = Query(0.5, ge=0.0, le=1.0),
) -> SimilarResponse:
    if course_id not in COURSES:
        raise HTTPException(status_code=404, detail=f"Unknown course_id: {course_id}")

    query_sparse = COURSE_SPARSE_VECTORS[course_id]
    query_dense = COURSE_DENSE_VECTORS[course_id]

    scored: List[Tuple[str, float]] = []
    for cid in COURSES:
        if cid == course_id:
            continue

        sparse_score = _dot_sparse(query_sparse, COURSE_SPARSE_VECTORS[cid])
        dense_score = _dot_dense(query_dense, COURSE_DENSE_VECTORS[cid])
        score = _hybrid_score(sparse_score, dense_score, mode, alpha)
        scored.append((cid, score))

    scored.sort(key=lambda item: item[1], reverse=True)

    results = [
        CourseResult(
            course_id=cid,
            title=str(COURSES[cid]["title"]),
            score=round(score, 3),
        )
        for cid, score in scored[:top_k]
    ]

    return SimilarResponse(
        query_course_id=course_id,
        results=results,
        mode=mode,
        top_k=top_k,
    )


@app.get("/v1/search", response_model=SearchResponse)
def search_courses(
    query: str = Query(..., min_length=1),
    top_k: int = Query(10, ge=1, le=100),
    mode: Literal["dense", "sparse", "hybrid"] = Query("dense"),
    alpha: float = Query(0.5, ge=0.0, le=1.0),
) -> SearchResponse:
    query_sparse = _query_sparse_vector(query, COURSE_IDF)
    query_dense = _query_dense_vector(query)

    scored: List[Tuple[str, float]] = []
    for cid in COURSES:
        sparse_score = _dot_sparse(query_sparse, COURSE_SPARSE_VECTORS[cid])
        dense_score = _dot_dense(query_dense, COURSE_DENSE_VECTORS[cid])
        score = _hybrid_score(sparse_score, dense_score, mode, alpha)
        scored.append((cid, score))

    scored.sort(key=lambda item: item[1], reverse=True)

    results = [
        CourseResult(
            course_id=cid,
            title=str(COURSES[cid]["title"]),
            score=round(score, 3),
        )
        for cid, score in scored[:top_k]
    ]

    return SearchResponse(query=query, results=results, mode=mode)


@app.get("/v1/objectives/search", response_model=ObjectivesSearchResponse)
def search_objectives(
    query: str = Query(..., min_length=1),
    top_k: int = Query(10, ge=1, le=100),
    mode: Literal["dense", "sparse", "hybrid"] = Query("dense"),
    alpha: float = Query(0.5, ge=0.0, le=1.0),
) -> ObjectivesSearchResponse:
    query_sparse = _query_sparse_vector(query, OBJECTIVE_IDF)
    query_dense = _query_dense_vector(query)

    scored: List[Tuple[int, float]] = []
    for idx in range(len(OBJECTIVE_DOCS)):
        sparse_score = _dot_sparse(query_sparse, OBJECTIVE_SPARSE_VECTORS[idx])
        dense_score = _dot_dense(query_dense, OBJECTIVE_DENSE_VECTORS[idx])
        score = _hybrid_score(sparse_score, dense_score, mode, alpha)
        scored.append((idx, score))

    scored.sort(key=lambda item: item[1], reverse=True)

    results = [
        ObjectiveResult(
            course_id=OBJECTIVE_DOCS[idx]["course_id"],
            title=OBJECTIVE_DOCS[idx]["title"],
            objective=OBJECTIVE_DOCS[idx]["objective"],
            score=round(score, 3),
        )
        for idx, score in scored[:top_k]
    ]

    return ObjectivesSearchResponse(query=query, results=results, mode=mode)


@app.get("/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        index_sizes={
            "courses": len(COURSES),
            "objectives": len(OBJECTIVE_DOCS),
        },
    )
