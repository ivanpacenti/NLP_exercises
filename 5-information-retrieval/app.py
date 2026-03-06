import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

app = FastAPI()

TOKEN_RE = re.compile(r"[a-z0-9]+")
DENSE_DIM = 256
DATA_PATH = Path(__file__).with_name("dtu_courses.jsonl")
# BM25 hyperparameters (standard defaults used in IR literature)
BM25_K1 = 1.5
BM25_B = 0.75


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
COURSE_DENSE_VECTORS: Dict[str, List[float]] = {}
COURSE_TERM_FREQS: Dict[str, Dict[str, int]] = {}
COURSE_DOC_LEN: Dict[str, int] = {}
COURSE_BM25_IDF: Dict[str, float] = {}
COURSE_AVG_DL: float = 1.0

OBJECTIVE_DOCS: List[Dict[str, str]] = []
OBJECTIVE_DENSE_VECTORS: List[List[float]] = []
OBJECTIVE_TERM_FREQS: List[Dict[str, int]] = []
OBJECTIVE_DOC_LEN: List[int] = []
OBJECTIVE_BM25_IDF: Dict[str, float] = {}
OBJECTIVE_AVG_DL: float = 1.0


def _normalize_text(text: str) -> str:
    # Normalize accents/diacritics so queries like "bjorn" match "Bjørn".
    normalized = unicodedata.normalize("NFKD", text)
    without_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return without_accents.lower()


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(_normalize_text(text))


def _tokenize_sparse(text: str) -> List[str]:
    # Sparse retrieval uses unigrams + bigrams to capture short phrases/names.
    tokens = _tokenize(text)
    bigrams = [f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]
    return tokens + bigrams


def _dense_hash(token: str, dim: int = DENSE_DIM) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


def _l2_norm(values: List[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


def _dot_dense(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _build_dense_vector(tokens: List[str]) -> List[float]:
    # Lightweight dense embedding via hashing trick (no external model download).
    vec = [0.0] * DENSE_DIM
    for tok in tokens:
        vec[_dense_hash(tok)] += 1.0

    norm = _l2_norm(vec)
    if norm > 0:
        vec = [v / norm for v in vec]

    return vec


def _normalize_scores(values: List[float]) -> List[float]:
    # Min-max normalization used before hybrid combination.
    if not values:
        return []

    vmin = min(values)
    vmax = max(values)
    if math.isclose(vmin, vmax):
        return [0.0 for _ in values]

    return [(v - vmin) / (vmax - vmin) for v in values]


def _build_bm25_stats(token_lists: List[List[str]]) -> Tuple[List[Dict[str, int]], List[int], Dict[str, float], float]:
    # Precompute BM25 statistics: tf per doc, doc length, idf, average doc length.
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
    # Standard BM25 scoring for one query against one document.
    if not query_tokens:
        return 0.0

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


def _collect_field_text(value: object) -> str:
    # Flatten nested JSON values (dict/list/str) into a searchable text string.
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(_collect_field_text(v) for v in value)
    if isinstance(value, dict):
        return " ".join(_collect_field_text(v) for v in value.values())
    return ""


def _hybrid_scores(sparse_scores: List[float], dense_scores: List[float], alpha: float) -> List[float]:
    # Combine sparse and dense scores after normalization:
    # final = alpha * dense + (1 - alpha) * sparse
    sparse_norm = _normalize_scores(sparse_scores)
    dense_norm = _normalize_scores(dense_scores)
    return [alpha * d + (1.0 - alpha) * s for s, d in zip(sparse_norm, dense_norm)]


def _build_indexes() -> None:
    global COURSES
    global COURSE_DENSE_VECTORS, COURSE_TERM_FREQS, COURSE_DOC_LEN, COURSE_BM25_IDF, COURSE_AVG_DL
    global OBJECTIVE_DOCS, OBJECTIVE_DENSE_VECTORS, OBJECTIVE_TERM_FREQS, OBJECTIVE_DOC_LEN
    global OBJECTIVE_BM25_IDF, OBJECTIVE_AVG_DL

    if not DATA_PATH.exists():
        raise RuntimeError(f"Dataset not found: {DATA_PATH}")

    courses: Dict[str, Dict[str, object]] = {}
    course_ids: List[str] = []
    course_sparse_tokens_all: List[List[str]] = []
    course_dense_tokens_all: List[List[str]] = []

    objective_docs: List[Dict[str, str]] = []
    objective_sparse_tokens_all: List[List[str]] = []
    objective_dense_tokens_all: List[List[str]] = []

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

        # Course-level searchable text includes title + objectives + metadata fields.
        full_text = f"{title} {' '.join(objectives)} {fields_text}".strip()

        courses[course_id] = {
            "title": title,
            "text": full_text,
            "learning_objectives": objectives,
        }

        course_ids.append(course_id)
        course_sparse_tokens_all.append(_tokenize_sparse(full_text))
        course_dense_tokens_all.append(_tokenize(full_text))

        for objective in objectives:
            objective_docs.append(
                {
                    "course_id": course_id,
                    "title": title,
                    "objective": objective,
                }
            )
            objective_sparse_tokens_all.append(_tokenize_sparse(objective))
            objective_dense_tokens_all.append(_tokenize(objective))

    # Build course-level sparse (BM25) and dense indices.
    course_term_freqs_list, course_doc_len_list, course_idf, course_avg_dl = _build_bm25_stats(course_sparse_tokens_all)
    course_term_freqs = {cid: tf for cid, tf in zip(course_ids, course_term_freqs_list)}
    course_doc_len = {cid: dl for cid, dl in zip(course_ids, course_doc_len_list)}
    course_dense_vectors = {
        cid: _build_dense_vector(tokens)
        for cid, tokens in zip(course_ids, course_dense_tokens_all)
    }

    # Build objective-level sparse (BM25) and dense indices.
    objective_term_freqs, objective_doc_len, objective_idf, objective_avg_dl = _build_bm25_stats(objective_sparse_tokens_all)
    objective_dense_vectors = [_build_dense_vector(tokens) for tokens in objective_dense_tokens_all]

    COURSES = courses
    COURSE_TERM_FREQS = course_term_freqs
    COURSE_DOC_LEN = course_doc_len
    COURSE_BM25_IDF = course_idf
    COURSE_AVG_DL = course_avg_dl
    COURSE_DENSE_VECTORS = course_dense_vectors

    OBJECTIVE_DOCS = objective_docs
    OBJECTIVE_TERM_FREQS = objective_term_freqs
    OBJECTIVE_DOC_LEN = objective_doc_len
    OBJECTIVE_BM25_IDF = objective_idf
    OBJECTIVE_AVG_DL = objective_avg_dl
    OBJECTIVE_DENSE_VECTORS = objective_dense_vectors


def rank_courses_for_query(
    query: str,
    top_k: int = 10,
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid",
    alpha: float = 0.25,
    exclude_course_id: Optional[str] = None,
) -> List[Tuple[str, float]]:
    # Internal ranker shared by /v1/search and /similar.
    query_sparse = _tokenize_sparse(query)
    query_dense = _build_dense_vector(_tokenize(query))

    ids: List[str] = []
    sparse_scores: List[float] = []
    dense_scores: List[float] = []

    for cid in COURSES:
        if exclude_course_id and cid == exclude_course_id:
            continue

        ids.append(cid)
        sparse_scores.append(
            _bm25_score(
                query_tokens=query_sparse,
                doc_tf=COURSE_TERM_FREQS[cid],
                doc_len=COURSE_DOC_LEN[cid],
                idf=COURSE_BM25_IDF,
                avg_dl=COURSE_AVG_DL,
            )
        )
        dense_scores.append(_dot_dense(query_dense, COURSE_DENSE_VECTORS[cid]))

    if mode == "sparse":
        final_scores = sparse_scores
    elif mode == "dense":
        final_scores = dense_scores
    else:
        final_scores = _hybrid_scores(sparse_scores, dense_scores, alpha=alpha)

    ranked = list(zip(ids, final_scores))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:top_k]


def rank_objectives_for_query(
    query: str,
    top_k: int = 10,
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid",
    alpha: float = 0.25,
) -> List[Tuple[int, float]]:
    # Internal ranker for objective-level retrieval.
    query_sparse = _tokenize_sparse(query)
    query_dense = _build_dense_vector(_tokenize(query))

    sparse_scores: List[float] = []
    dense_scores: List[float] = []

    for idx in range(len(OBJECTIVE_DOCS)):
        sparse_scores.append(
            _bm25_score(
                query_tokens=query_sparse,
                doc_tf=OBJECTIVE_TERM_FREQS[idx],
                doc_len=OBJECTIVE_DOC_LEN[idx],
                idf=OBJECTIVE_BM25_IDF,
                avg_dl=OBJECTIVE_AVG_DL,
            )
        )
        dense_scores.append(_dot_dense(query_dense, OBJECTIVE_DENSE_VECTORS[idx]))

    if mode == "sparse":
        final_scores = sparse_scores
    elif mode == "dense":
        final_scores = dense_scores
    else:
        final_scores = _hybrid_scores(sparse_scores, dense_scores, alpha=alpha)

    ranked = list(enumerate(final_scores))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:top_k]


@app.on_event("startup")
def on_startup() -> None:
    _build_indexes()


@app.get("/v1/courses/{course_id}/similar", response_model=SimilarResponse)
def similar_courses(
    course_id: str,
    top_k: int = Query(10, ge=1, le=100),
    mode: Literal["dense", "sparse", "hybrid"] = Query("sparse"),
    alpha: float = Query(0.0, ge=0.0, le=1.0),
) -> SimilarResponse:
    # Retrieve nearest courses by using the selected course text as query.
    if course_id not in COURSES:
        raise HTTPException(status_code=404, detail=f"Unknown course_id: {course_id}")

    query_text = str(COURSES[course_id]["text"])
    ranked = rank_courses_for_query(
        query=query_text,
        top_k=top_k,
        mode=mode,
        alpha=alpha,
        exclude_course_id=course_id,
    )

    results = [
        CourseResult(course_id=cid, title=str(COURSES[cid]["title"]), score=round(score, 3))
        for cid, score in ranked
    ]

    return SimilarResponse(query_course_id=course_id, results=results, mode=mode, top_k=top_k)


@app.get("/v1/search", response_model=SearchResponse)
def search_courses(
    query: str = Query(..., min_length=1),
    top_k: int = Query(10, ge=1, le=100),
    mode: Literal["dense", "sparse", "hybrid"] = Query("sparse"),
    alpha: float = Query(0.0, ge=0.0, le=1.0),
) -> SearchResponse:
    # Free-text course search endpoint used by evaluation/UI.
    ranked = rank_courses_for_query(query=query, top_k=top_k, mode=mode, alpha=alpha)

    results = [
        CourseResult(course_id=cid, title=str(COURSES[cid]["title"]), score=round(score, 3))
        for cid, score in ranked
    ]

    return SearchResponse(query=query, results=results, mode=mode)


@app.get("/v1/objectives/search", response_model=ObjectivesSearchResponse)
def search_objectives(
    query: str = Query(..., min_length=1),
    top_k: int = Query(10, ge=1, le=100),
    mode: Literal["dense", "sparse", "hybrid"] = Query("sparse"),
    alpha: float = Query(0.0, ge=0.0, le=1.0),
) -> ObjectivesSearchResponse:
    # Free-text objective search returning (course, objective) matches.
    ranked = rank_objectives_for_query(query=query, top_k=top_k, mode=mode, alpha=alpha)

    results = [
        ObjectiveResult(
            course_id=OBJECTIVE_DOCS[idx]["course_id"],
            title=OBJECTIVE_DOCS[idx]["title"],
            objective=OBJECTIVE_DOCS[idx]["objective"],
            score=round(score, 3),
        )
        for idx, score in ranked
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
