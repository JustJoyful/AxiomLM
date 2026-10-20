"""Evaluate Phase 2 in an isolated store; preserve the frozen baseline and sources."""

from __future__ import annotations

import asyncio
import gc
import json
import os
import tempfile
import time
import traceback
import weakref
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_phase2_baseline import (
    digest,
    load_labels,
    retrieval_metrics,
    snapshot_store,
    tree_identity,
    verify_sources,
)
from src import config
from src.index_metadata import validate_book


def load_inputs(book_stem):
    validate_book(book_stem)
    label_path = config.PHASE2_LABELS_DIR / f"{book_stem}_context_labels.json"
    labels = load_labels(label_path)
    baseline = json.loads(config.PHASE2_EVALUATION_PATH.read_text())
    if (
        baseline.get("status") != "complete"
        or baseline.get("book_stem") != book_stem
        or labels["book_stem"] != book_stem
        or baseline.get("dataset_sha256") != digest(label_path)
        or baseline.get("runtime", {}).get("embedding") != config.EMBED_MODEL
        or baseline.get("runtime", {}).get("embedding_revision")
        != config.EMBED_REVISION
    ):
        raise ValueError("Frozen baseline/label/model identity mismatch")
    return labels, baseline


def save_report(report):
    output = config.PHASE2_AFTER_EVALUATION_PATH
    if output.resolve() == config.PHASE2_EVALUATION_PATH.resolve():
        raise ValueError("After report must be distinct from original baseline")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output)


def protected_inputs(book_stem):
    return {
        "active_store": tree_identity(config.VECTOR_STORE),
        "parsed_artifacts": tree_identity(config.PARSED_MD),
        "baseline": digest(config.PHASE2_EVALUATION_PATH),
        "labels": digest(config.PHASE2_LABELS_DIR / f"{book_stem}_context_labels.json"),
        "pooling_validation": digest(config.PHASE2_GEOMETRY_REPORT_PATH),
    }


def _run_staged(book, labels, baseline, root, report):
    import torch

    from scripts.capture_runtime_manifest import capture_manifest
    from scripts.validate_foundation import _current_index_identity
    from src import db, index_store, tui
    from src.answer_tokens import load_answer_counter
    from src.indexer import contextual_chunks
    from src.llm import OllamaClient
    from src.query_rewriting import ConversationExchange, rewrite_followup
    from src.ui.render import sanitize_answer
    from src.workloads import WorkloadCoordinator

    report["runtime"] = capture_manifest()
    runtime = report["runtime"]
    if (
        runtime.get("digest") != baseline["runtime"]["digest"]
        or runtime.get("model") != baseline["runtime"]["model"]
    ):
        raise ValueError("Answer model identity differs from frozen baseline")
    client = OllamaClient()
    with client._http_client() as http:
        response = http.get(config.OLLAMA_URL + "/api/ps")
        response.raise_for_status()
        if response.json().get("models"):
            raise RuntimeError("Finish active generation before staged GPU ingest")
    legacy_root = root / "legacy_copy"
    snapshot_store(config.VECTOR_STORE, legacy_root)
    with patch.object(db, "_CLIENT", index_store.client_for(legacy_root)):
        if _current_index_identity(book) != baseline["index_ids_sha256"]:
            raise ValueError("Frozen baseline index identity is stale")
    coordinator = WorkloadCoordinator()
    admission = coordinator.try_acquire("ingest")
    if admission is None:
        raise RuntimeError("Ingest workload unavailable")
    model_refs = []
    parameter_refs = []
    seen = set()
    get_model = db.get_ingest_model

    def track_model(**kwargs):
        model = get_model(**kwargs)
        if id(model) not in seen:
            seen.add(id(model))
            model_refs.append(weakref.ref(model))
            parameter_refs.extend(weakref.ref(p) for p in model.parameters())
        return model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA target-device acceptance unavailable")
    warm = torch.ones((64, 64), device="cuda")
    product = warm @ warm
    del warm, product
    gc.collect()
    torch.cuda.empty_cache()
    idle = torch.cuda.memory_allocated()
    stage = root / "phase2"
    started = time.monotonic()
    try:
        assert coordinator.try_acquire("generation") is None
        with patch.object(db, "get_ingest_model", track_model):
            chunks = contextual_chunks(book)
            first = index_store.build_index_version(book, chunks, root=stage)
            first_ids = [c.chunk_id for c in chunks]
            rebuilt = contextual_chunks(book)
            second = index_store.build_index_version(book, rebuilt, root=stage)
        gc.collect()
        torch.cuda.empty_cache()
        cleaned = (
            db._INGEST_MODEL is None
            and all(r() is None for r in model_refs)
            and all(r() is None for r in parameter_refs)
            and torch.cuda.memory_allocated() <= idle
        )
        if not cleaned:
            raise RuntimeError("Ingest model/parameter ownership did not clear")
        report["ingest"] = {
            "seconds": time.monotonic() - started,
            "chunk_count": len(chunks),
            "version_id": first.version_id,
            "embedding_config": index_store.embedding_config(),
            "model_released": all(r() is None for r in model_refs),
            "parameters_released": all(r() is None for r in parameter_refs),
            "allocated_after": torch.cuda.memory_allocated(),
            "initialized_baseline": idle,
            "repeated_version_identical": first.version_id == second.version_id,
            "repeated_ids_identical": first_ids == [c.chunk_id for c in rebuilt],
        }
        if (
            not report["ingest"]["repeated_version_identical"]
            or not report["ingest"]["repeated_ids_identical"]
        ):
            raise ValueError("Nonreproducible rebuild")
        active_before = index_store.manifest_path(stage).read_bytes()

        def fail_bm25(*args, **kwargs):
            raise OSError("staged failure injection")

        try:
            with patch.object(index_store, "write_bm25", fail_bm25):
                index_store.build_index_version(
                    book,
                    [replace(chunks[0], text=chunks[0].text + "\nSTAGED FAILURE ONLY")],
                    root=stage,
                )
        except OSError:
            pass
        else:
            raise AssertionError("Failure injection did not fail")
        report["failed_build_preserved_old_pair"] = (
            index_store.manifest_path(stage).read_bytes() == active_before
            and index_store.resolve_index(book, root=stage).version_id
            == first.version_id
        )
        if not report["failed_build_preserved_old_pair"]:
            raise RuntimeError("Failed build changed active pair")
    finally:
        db.release_ingest_model()
        coordinator.release(admission)
    report["workload_order"] = [
        "GPU contextual ingest/release twice",
        "CPU live query/retrieval",
        "serialized rewrite/answer generation, keep_alive=0",
    ]
    counter = load_answer_counter(config.OLLAMA_MODEL)
    app = tui.AxiomLMApp()
    resolve = index_store.resolve_index
    report["rows"] = []
    cases = list(labels["cases"])
    supplemental = json.loads(
        (config.PHASE2_LABELS_DIR / "planning_cases.json").read_text()
    )["contradiction"]
    report["supplemental_rows"] = []
    for case in [*cases, supplemental]:
        row = {
            "case_id": case["case_id"],
            "kind": case["kind"],
            "question": case["question"],
            "status": "running",
        }
        destination = report["rows"] if case in cases else report["supplemental_rows"]
        destination.append(row)
        admission = coordinator.try_acquire("generation")
        if admission is None:
            raise RuntimeError("Generation workload unavailable")
        try:
            history = case.get("history", [])
            exchanges = [
                ConversationExchange(
                    history[i]["content"], history[i + 1]["content"], book
                )
                for i in range(0, len(history), 2)
            ]
            started = time.monotonic()
            rewritten = asyncio.run(
                rewrite_followup(
                    case["question"],
                    exchanges,
                    book,
                    model=config.OLLAMA_MODEL,
                    client=client,
                )
            )
            row["rewrite_seconds"] = time.monotonic() - started
            row["retrieval_query"] = rewritten.retrieval_query
            row["rewrite_applied"] = rewritten.applied
            row["rewrite_fallback"] = rewritten.fallback_reason
            app._conversation = list(history)
            started = time.monotonic()
            with patch.object(
                index_store, "resolve_index", lambda name: resolve(name, root=stage)
            ):
                prompt, metadata, pages, ids = app._prepare_query(
                    case["question"],
                    app._format_history(),
                    book,
                    config.TOP_K_RESULTS,
                    config.DISTANCE_THRESHOLD,
                    rewritten.retrieval_query,
                    config.OLLAMA_MODEL,
                    counter,
                )
            row.update(
                retrieval_seconds=time.monotonic() - started,
                evidence_metadata=metadata,
                physical_pages=pages,
                evidence_ids=ids,
                prompt_tokens=counter(prompt),
                prompt=prompt,
                query_device=str(db.get_query_model().device),
            )
            if row["query_device"] != "cpu":
                raise RuntimeError("Live queries are not CPU")
            if (
                row["prompt_tokens"] + config.OLLAMA_OUTPUT_TOKENS
                > config.OLLAMA_CONTEXT_TOKENS
            ):
                raise ValueError("Prompt budget overflow")
            started = time.monotonic()
            response = asyncio.run(
                client.agenerate(
                    prompt,
                    model=config.OLLAMA_MODEL,
                    think=False,
                    num_predict=config.OLLAMA_OUTPUT_TOKENS,
                    timeout=config.OLLAMA_REQUEST_TIMEOUT_SECONDS,
                )
            )
            row.update(
                answer_seconds=time.monotonic() - started,
                raw_answer=response.get("response", ""),
                response_metrics={
                    k: v
                    for k, v in response.items()
                    if k not in {"response", "context", "thinking"}
                },
                application_citation=tui.build_page_citation(book, pages),
            )
            row["sanitized_answer"] = sanitize_answer(row["raw_answer"])
            row["answer_hit_output_limit"] = (
                response.get("done_reason") == "length"
                or response.get("eval_count", 0) >= config.OLLAMA_OUTPUT_TOKENS
            )
            row["served_prompt_tokens"] = response.get("prompt_eval_count")
            row["tokenizer_matches_served_count"] = (
                row["served_prompt_tokens"] == row["prompt_tokens"]
            )
            row["abstained"] = row["sanitized_answer"].startswith(
                "The indexed material does not provide enough evidence"
            )
            row["status"] = "complete"
            print(
                case["case_id"],
                len(ids),
                "chunks",
                row["prompt_tokens"],
                "tokens",
                "count_match",
                row["tokenizer_matches_served_count"],
                flush=True,
            )
        finally:
            client.unload_and_wait()
            coordinator.release(admission)
            save_report(report)
    report["retrieval_metrics"] = retrieval_metrics(cases, report["rows"])
    report["baseline_retrieval_metrics"] = baseline["retrieval_metrics"]
    cited = [p for row in report["rows"] for p in row["physical_pages"]]
    valid = sum(
        (config.CHECKPOINTS / book / f"page_{p-1:04d}.md").is_file() for p in cited
    )
    report["citation_page_validity"] = {
        "value": valid / len(cited) if cited else None,
        "valid_pages": valid,
        "cited_pages": len(cited),
        "limitation": "Physical page validity is not claim support.",
    }
    report["claim_review"] = {
        "reviewer": None,
        "reviewed_claims": 0,
        "supported_claims": 0,
        "unsupported_claims": 0,
        "citation_accuracy": None,
        "hallucination_rate": None,
        "answer_coverage": None,
        "status": "unreviewed; inspect raw answers and evidence before claim-quality acceptance",
    }
    report["unanswerable"] = [
        {
            "case_id": case["case_id"],
            "abstained": next(
                r["abstained"]
                for r in report["rows"]
                if r["case_id"] == case["case_id"]
            ),
        }
        for case in cases
        if not case["answerable"]
    ]
    report["acceptance"] = {
        "pooling_gate": True,
        "reproducible_staged_rebuild": True,
        "atomic_activation_failure_recovery": report["failed_build_preserved_old_pair"],
        "CPU_queries": all(r["query_device"] == "cpu" for r in report["rows"]),
        "prompt_fit_and_exact_tokenizer": all(
            r["tokenizer_matches_served_count"]
            and r["prompt_tokens"] + config.OLLAMA_OUTPUT_TOKENS
            <= config.OLLAMA_CONTEXT_TOKENS
            for r in report["rows"]
        ),
        "all_answers_completed": not any(
            r["answer_hit_output_limit"] for r in report["rows"]
        ),
        "claim_support_review_complete": False,
    }


def validate(book_stem: str, *, staging: bool = True) -> dict[str, object]:
    if not staging:
        raise ValueError(
            "Only staging evaluation is authorized; active indexes are preserved"
        )
    if (
        config.PHASE2_AFTER_EVALUATION_PATH.resolve()
        == config.PHASE2_EVALUATION_PATH.resolve()
    ):
        raise ValueError("After report must be distinct from baseline")
    if config.PHASE2_AFTER_EVALUATION_PATH.exists():
        raise FileExistsError("Preserve previous after report before rerunning")
    labels, baseline = load_inputs(book_stem)
    verify_sources(labels, config.PARSED_MD)
    if (
        digest(config.PARSED_MD / book_stem / "page_manifest.json")
        != baseline["source_manifest_sha256"]
    ):
        raise ValueError("Source manifest identity changed")
    gate = json.loads(config.PHASE2_GEOMETRY_REPORT_PATH.read_text())
    if (
        not gate.get("passed")
        or config.PHASE2_POOLING_METHOD not in gate["compatible_methods"]
        or gate["revision"] != config.EMBED_REVISION
        or gate["protocol"]["chunk_content_token_ceiling"]
        != config.RETRIEVAL_CHUNK_TOKENS
    ):
        raise ValueError("Pooling capability gate is stale or failed")
    before = protected_inputs(book_stem)
    report = {
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
        "book_stem": book_stem,
        "staging": True,
        "dataset_sha256": digest(
            config.PHASE2_LABELS_DIR / f"{book_stem}_context_labels.json"
        ),
        "baseline_path": str(config.PHASE2_EVALUATION_PATH),
        "baseline_sha256": before["baseline"],
        "protected_before": before,
        "code_sha256": {
            str(p.relative_to(config.ROOT)): digest(p)
            for p in [Path(__file__), *sorted((config.ROOT / "src").glob("*.py"))]
        },
    }
    save_report(report)
    try:
        with tempfile.TemporaryDirectory(prefix="axiom-phase2-staging-") as directory:
            _run_staged(book_stem, labels, baseline, Path(directory), report)
        report["status"] = "complete"
    except BaseException as exc:
        report.update(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        try:
            after = protected_inputs(book_stem)
            report["protected_after"] = after
            report["preserved"] = before == after
            if not report["preserved"]:
                report.update(
                    status="failed",
                    error="Protected baseline/index/parsed artifacts changed",
                )
        except (OSError, ValueError) as exc:
            report.update(
                status="failed",
                preserved=False,
                integrity_error=f"{type(exc).__name__}: {exc}",
            )
        report["finished_utc"] = datetime.now(UTC).isoformat()
        save_report(report)
    if report["status"] == "failed":
        raise RuntimeError("Phase 2 evaluation integrity check failed; inspect report")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("book_stem")
    args = parser.parse_args()
    result = validate(args.book_stem)
    print(
        json.dumps(
            {
                "status": result["status"],
                "acceptance": result.get("acceptance"),
                "report": str(config.PHASE2_AFTER_EVALUATION_PATH),
            },
            indent=2,
        )
    )
