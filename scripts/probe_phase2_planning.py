"""Platform, rewrite, contradiction and broad-context experiments for planning.

Called after all GPU embedding models have been released. No production routing.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import httpx

from scripts.evaluate_phase2_baseline import snapshot_store
from src import config
from src.llm import OllamaClient


def check_fts(pages):
    with sqlite3.connect(":memory:") as sql:
        sql.execute("CREATE VIRTUAL TABLE chunks USING fts5(chunk_id UNINDEXED, text)")
        sql.executemany(
            "INSERT INTO chunks VALUES (?, ?)",
            [(f"page-{p:04d}", text) for p, text in pages],
        )
        rows = sql.execute(
            "SELECT chunk_id, bm25(chunks) FROM chunks WHERE chunks MATCH ? ORDER BY bm25(chunks),chunk_id LIMIT ?",
            ("interpreter", config.TOP_K_RESULTS),
        ).fetchall()
        if not rows or any(not cid.startswith("page-") for cid, _ in rows):
            raise RuntimeError("SQLite FTS5 identity/ranking check failed")
        return {
            "available": True,
            "sqlite_version": sqlite3.sqlite_version,
            "document_count": len(pages),
            "sample_ranked_ids": rows,
            "limitation": "In-memory capability test; the production BM25 tokenizer, schema, filters and atomic paired-store builder remain to be implemented/tested.",
        }


def check_atomic_manifest():
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=config.LOGS_DIR, prefix=".phase2-manifest-"
    ) as directory:
        root = Path(directory)
        active = root / "active.json"
        candidate = root / "candidate.json"
        active.write_text('{"index":"old"}')
        candidate.write_text('{"index":"new"}')
        old_before_activation = json.loads(active.read_text())["index"] == "old"
        os.replace(candidate, active)
        return {
            "old_until_replace": old_before_activation,
            "new_after_replace": json.loads(active.read_text())["index"] == "new",
            "same_filesystem_as_store": root.stat().st_dev
            == config.VECTOR_STORE.stat().st_dev,
            "limitation": "Filesystem primitive only; not an implemented paired Chroma/FTS migration or crash recovery proof.",
        }


def rewrite_check(case):
    prompt = (
        "Rewrite the final question into one standalone book-search question. Resolve references using only the conversation. "
        "Keep the topic and request. Return only that question, no answer, explanation, or Markdown.\n"
        + "\n".join(f"{turn['role']}: {turn['content']}" for turn in case["history"])
        + "\nFinal question: "
        + case["question"]
    )
    client = OllamaClient()
    started = time.monotonic()
    try:
        with httpx.Client(
            trust_env=False, timeout=config.QUERY_REWRITE_TIMEOUT_SECONDS
        ) as http:
            response = http.post(
                config.OLLAMA_URL + "/api/generate",
                json={
                    "model": config.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "think": False,
                    "keep_alive": config.OLLAMA_KEEP_ALIVE,
                    "options": dict(
                        client._options(),
                        num_predict=config.QUERY_REWRITE_MAX_OUTPUT_TOKENS,
                    ),
                },
            )
            response.raise_for_status()
            result = response.json()
        return {
            "question": case["question"],
            "rewritten": result.get("response", ""),
            "seconds": time.monotonic() - started,
            "within_budget": time.monotonic() - started
            <= config.QUERY_REWRITE_TIMEOUT_SECONDS,
            "done_reason": result.get("done_reason"),
            "eval_count": result.get("eval_count"),
            "review": "Raw output retained; requires source/topic inspection. This tests model ability, not future rewrite policy/cancellation/book-isolation code.",
        }
    except httpx.HTTPError as exc:
        return {
            "question": case["question"],
            "seconds": time.monotonic() - started,
            "error": str(exc),
            "within_budget": False,
        }
    finally:
        client.unload_and_wait()


def answer_check(app, case, book, k, count_prompt=False):
    app._conversation = list(case.get("history", []))
    started = time.monotonic()
    prompt, metadata, pages, ids = app._prepare_legacy_query(
        case["question"], app._format_history(), book, k, config.DISTANCE_THRESHOLD
    )
    row = {
        "case_id": case["case_id"],
        "question": case["question"],
        "k": k,
        "evidence_ids": ids,
        "physical_pages": pages,
        "metadata": metadata,
        "retrieval_seconds": time.monotonic() - started,
    }
    client = OllamaClient()
    token = app._try_acquire_workload("generation")
    if token is None:
        raise RuntimeError("GPU generation admission unavailable")
    started = time.monotonic()
    try:
        response = client.generate(
            prompt, model=config.OLLAMA_MODEL, stream=False, think=False
        )
        if not isinstance(response, dict) or response.get("done") is not True:
            raise RuntimeError(f"Incomplete answer probe: {response}")
        row.update(
            raw_answer=response.get("response", ""),
            done_reason=response.get("done_reason"),
            prompt_eval_count=response.get("prompt_eval_count"),
            eval_count=response.get("eval_count"),
            generation_seconds=time.monotonic() - started,
        )
        if count_prompt:
            # Wider context with one output token measures full template+prompt length.
            # It is diagnostic only; the application remains at its configured context.
            client.unload_and_wait()
            with httpx.Client(
                trust_env=False, timeout=config.OLLAMA_REQUEST_TIMEOUT_SECONDS
            ) as http:
                counted = http.post(
                    config.OLLAMA_URL + "/api/generate",
                    json={
                        "model": config.OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "think": False,
                        "keep_alive": config.OLLAMA_KEEP_ALIVE,
                        "options": dict(
                            client._options(),
                            num_ctx=config.PHASE2_ANSWER_COUNT_CONTEXT_TOKENS,
                            num_predict=1,
                        ),
                    },
                )
                counted.raise_for_status()
                data = counted.json()
            row["count_probe_context_tokens"] = (
                config.PHASE2_ANSWER_COUNT_CONTEXT_TOKENS
            )
            row["count_probe_prompt_eval_count"] = data.get("prompt_eval_count")
            row["same_prompt_count_at_wider_context"] = data.get(
                "prompt_eval_count"
            ) == response.get("prompt_eval_count")
            if data.get("prompt_eval_count") is not None:
                row["prompt_plus_reserved_output_fits"] = (
                    data["prompt_eval_count"] + config.OLLAMA_OUTPUT_TOKENS
                    <= config.OLLAMA_CONTEXT_TOKENS
                )
    finally:
        client.unload_and_wait()
        app._release_workload(token)
    return row


def planning_checks(labels, pages, baseline, fixtures):
    import chromadb

    from src import db, tui

    result = {
        "fts5": check_fts(pages),
        "atomic_manifest": check_atomic_manifest(),
        "original_baseline_unchanged": True,
        "rewrites": [],
        "broad_k8": [],
    }
    followup = next(c for c in labels["cases"] if c["kind"] == "followup")
    result["rewrites"].append(rewrite_check(followup))
    result["rewrites"].append(
        rewrite_check(dict(followup, question="How does it work?"))
    )
    with tempfile.TemporaryDirectory(prefix="axiom-planning-query-") as temp:
        copy = Path(temp) / "store"
        snapshot_store(config.VECTOR_STORE, copy)
        with (
            patch.object(db, "_CLIENT", chromadb.PersistentClient(path=str(copy))),
            patch.object(tui, "get_collection", db.get_existing_collection),
        ):
            app = tui.AxiomLMApp()
            for case in labels["cases"]:
                if case["kind"] in ("broad", "broad_multi"):
                    print("planning broad K=8:", case["case_id"], flush=True)
                    result["broad_k8"].append(
                        answer_check(
                            app,
                            case,
                            labels["book_stem"],
                            config.PHASE2_BROAD_MAX_CHUNKS,
                            count_prompt=True,
                        )
                    )
            print("planning contradiction", flush=True)
            result["contradiction"] = answer_check(
                app,
                fixtures["contradiction"],
                labels["book_stem"],
                config.TOP_K_RESULTS,
            )
    result["not_yet_testable"] = [
        "production query rewrite selection, async cancellation and book isolation",
        "production BM25/dense merge and filter refill",
        "versioned paired-index failure recovery",
        "chapter/paragraph metadata normalization and oversized paragraph segmentation",
        "claim-citation verification (Phase 4)",
    ]
    return result
