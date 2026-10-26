"""Deterministic rank fusion and whole-passage answer context selection."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import zip_longest

from src import config, db
from src.index_store import IndexHandle
from src.query_rewriting import is_book_overview


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    text: str
    metadata: dict[str, str | int]
    backend: str
    rank: int
    score: float


@dataclass(frozen=True)
class MergedCandidate:
    chunk_id: str
    text: str
    metadata: dict[str, str | int]
    dense_rank: int | None
    bm25_rank: int | None
    dense_distance: float | None
    bm25_score: float | None


def merge_ranked_candidates(
    dense: Sequence[SearchHit],
    bm25: Sequence[SearchHit],
    limit: int = config.HYBRID_CANDIDATE_LIMIT,
) -> list[MergedCandidate]:
    if not 1 <= limit <= config.HYBRID_CANDIDATE_LIMIT:
        raise ValueError("Invalid candidate count")
    dense = sorted(dense, key=lambda h: (h.score, h.chunk_id))
    bm25 = sorted(bm25, key=lambda h: (h.score, h.chunk_id))
    documents = {}
    for h in [*dense, *bm25]:
        if not math.isfinite(h.score):
            raise ValueError("Invalid retrieval score")
        content = (h.text, h.metadata)
        if h.chunk_id in documents and documents[h.chunk_id] != content:
            raise ValueError("Conflicting content/metadata for shared chunk ID")
        documents[h.chunk_id] = content
    dr = {h.chunk_id: (i + 1, h.score) for i, h in enumerate(dense)}
    br = {h.chunk_id: (i + 1, h.score) for i, h in enumerate(bm25)}
    ordered = []
    seen = set()
    for pair in zip_longest(dense, bm25):
        for hit in pair:
            if hit is not None and hit.chunk_id not in seen:
                ordered.append(hit)
                seen.add(hit.chunk_id)
                if len(ordered) == limit:
                    break
        if len(ordered) == limit:
            break
    return [
        MergedCandidate(
            h.chunk_id,
            h.text,
            h.metadata,
            dr.get(h.chunk_id, (None, None))[0],
            br.get(h.chunk_id, (None, None))[0],
            dr.get(h.chunk_id, (None, None))[1],
            br.get(h.chunk_id, (None, None))[1],
        )
        for h in ordered
    ]


def retrieve_hybrid(
    index: IndexHandle,
    query: str,
    page_filter: tuple[int, int] | None,
    dense_limit: int,
    bm25_limit: int,
    candidate_limit: int = config.HYBRID_CANDIDATE_LIMIT,
    *,
    distance_threshold: float = config.DISTANCE_THRESHOLD,
) -> list[MergedCandidate]:
    vector = db.embed_query(query, prompt_policy=index.query_prompt_policy)
    if not index.legacy and (index.bm25_path is None or not index.bm25_path.is_file()):
        raise RuntimeError("BM25 index unavailable; rebuild required")
    if not math.isfinite(distance_threshold) or distance_threshold < 0:
        raise ValueError("Invalid dense distance threshold")
    dense_count = index.dense_collection.count()
    terms = re.findall(r"\w+", query, flags=re.UNICODE)
    expression = " OR ".join('"' + t + '"' for t in terms)
    dense_n = min(dense_limit, config.HYBRID_MAX_FETCH)
    sparse_n = min(bm25_limit, config.HYBRID_MAX_FETCH)
    if dense_n < 1 or sparse_n < 1:
        raise ValueError("Fetch limits must be positive")
    while True:
        dense = []
        sparse = []
        if dense_count:
            args = {
                "query_embeddings": [vector],
                "n_results": min(dense_n, dense_count),
                "include": ["documents", "metadatas", "distances"],
            }
            if page_filter:
                lo, hi = page_filter
                args["where"] = (
                    {"physical_page": {"$in": list(range(lo, hi + 1))}}
                    if index.legacy
                    else {
                        "$and": [
                            {"physical_page_start": {"$lte": hi}},
                            {"physical_page_end": {"$gte": lo}},
                        ]
                    }
                )
            result = index.dense_collection.query(**args)
            for i, (cid, text, meta, score) in enumerate(
                zip(
                    result["ids"][0],
                    result["documents"][0],
                    result["metadatas"][0],
                    result["distances"][0],
                    strict=True,
                )
            ):
                if score < distance_threshold:
                    dense.append(
                        SearchHit(cid, text, meta, "dense", i + 1, float(score))
                    )
        raw_sparse = []
        if index.bm25_path is not None and expression:
            with sqlite3.connect(
                f"file:{index.bm25_path}?mode=ro", uri=True
            ) as connection:
                sql = "SELECT c.chunk_id,c.document,c.metadata_json,bm25(chunks_fts) AS score FROM chunks_fts JOIN chunks c ON c.chunk_id=chunks_fts.chunk_id WHERE chunks_fts MATCH ?"
                parameters = [expression]
                if page_filter:
                    sql += " AND json_extract(c.metadata_json,'$.physical_page_start')<=? AND json_extract(c.metadata_json,'$.physical_page_end')>=?"
                    parameters.extend([page_filter[1], page_filter[0]])
                sql += " ORDER BY score,c.chunk_id LIMIT ?"
                parameters.append(sparse_n)
                raw_sparse = connection.execute(sql, parameters).fetchall()
            sparse = [
                SearchHit(cid, text, json.loads(meta), "bm25", i + 1, float(score))
                for i, (cid, text, meta, score) in enumerate(raw_sparse)
            ]
        merged = merge_ranked_candidates(dense, sparse, candidate_limit)
        dense_exhausted = (
            min(dense_n, dense_count) < dense_n
            or dense_n >= dense_count
            or dense_n >= config.HYBRID_MAX_FETCH
        )
        # Filtered dense results can exhaust earlier than the collection-wide count.
        if dense_count:
            dense_exhausted = dense_exhausted or len(result["ids"][0]) < min(
                dense_n, dense_count
            )
        sparse_exhausted = (
            index.bm25_path is None
            or not expression
            or len(raw_sparse) < sparse_n
            or sparse_n >= config.HYBRID_MAX_FETCH
        )
        if len(merged) >= candidate_limit or (dense_exhausted and sparse_exhausted):
            return merged
        if not dense_exhausted:
            dense_n = min(config.HYBRID_MAX_FETCH, dense_n + config.HYBRID_FETCH_BATCH)
        if not sparse_exhausted:
            sparse_n = min(
                config.HYBRID_MAX_FETCH, sparse_n + config.HYBRID_FETCH_BATCH
            )


def is_broad_question(question: str) -> bool:
    return is_book_overview(question) or bool(
        re.search(
            r"\b(?:overview|summarize|compare|contrast|relationship|across chapters|what is this book about)\b|\bhow do\b.*\brelate\b",
            question,
            re.IGNORECASE,
        )
    )


def select_context(
    question: str,
    candidates: Sequence[MergedCandidate],
    requested_k: int,
    answer_token_counter,
    prompt_budget: int,
) -> list[MergedCandidate]:
    broad = is_broad_question(question)
    desired = (
        max(
            config.BROAD_CONTEXT_MIN_CHUNKS,
            min(config.BROAD_CONTEXT_MAX_CHUNKS, requested_k),
        )
        if broad
        else min(config.HYBRID_CANDIDATE_LIMIT, requested_k)
    )
    count = min(len(candidates), desired)
    floor = min(config.BROAD_CONTEXT_MIN_CHUNKS, count) if broad else min(1, count)
    for size in range(count, floor - 1, -1):
        chosen = list(candidates[:size])
        if answer_token_counter(chosen) <= prompt_budget:
            return chosen
    raise ValueError(
        "Answer prompt exceeds token budget; shorten the question/history or lower context count"
    )
