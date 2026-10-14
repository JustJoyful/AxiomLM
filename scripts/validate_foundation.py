"""Collect Phase 0 evidence without touching a user's normal vector index."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import httpx

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# huggingface_hub reads this when its modules import; set it before importing
# SentenceTransformers through src.db so validation never downloads weights.
os.environ["HF_HUB_OFFLINE"] = "1"

import chromadb

from scripts.capture_runtime_manifest import capture_manifest
from scripts.validate_ollama_tools import run_diagnostic
from src import db
from src.config import (
    FOUNDATION_REPORT_PATH,
    FOUNDATION_VECTOR_STORE,
    DISTANCE_THRESHOLD,
    OLLAMA_CONTEXT_TOKENS,
    OLLAMA_MODEL,
    OLLAMA_URL,
    PARSED_MD,
    RUNTIME_MANIFEST_PATH,
    TOOLS_REPORT_PATH,
)
from src.llm import OllamaClient
from src.page_querying import build_citation, extract_page_intent, strip_model_footnotes
from src.tui import PROMPT_TEMPLATE, latex_to_unicode
from src.workloads import WorkloadCoordinator

BASELINE_CASES = [
    ("direct", "What is an interpreter?", None),
    ("followup", "Give another example of that.", None),
    ("page", "Summarize physical PDF page 1.", 1),
    ("broad", "What is this book about?", None),
    ("unanswerable", "What was the weather in Kolkata yesterday?", None),
]


def evaluate_report(report: dict[str, Any]) -> dict[str, Any]:
    """Return named Phase 0 gates; missing evidence is a failed gate."""
    model = report.get("model") or {}
    query = report.get("query") or {}
    ingest = report.get("ingest") or {}
    generation = report.get("generation") or {}
    baseline = report.get("baseline") or {}

    def integer(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def probability(value: Any) -> bool:
        return isinstance(value, (int, float)) and 0.0 <= value <= 1.0

    review = baseline.get("after_review") or {}
    metrics = baseline.get("retrieval_metrics") or {}
    before_metrics = metrics.get("before") or {}
    after_metrics = metrics.get("after") or {}
    workload_evidence = report.get("workload_evidence") or {}

    gates = {
        "local_service": report.get("local_service") is True,
        "model_identity": (
            model.get("name") == OLLAMA_MODEL
            and bool(re.fullmatch(r"[0-9a-f]{64}", str(model.get("digest", ""))))
            and bool(model.get("ollama_version"))
            and model.get("quantization") == "Q4_K_M"
        ),
        "tool_roundtrip": (report.get("tools") or {}).get("passed") is True,
        "ollama_unload": (report.get("tools") or {}).get("unload")
        == "all Ollama models unloaded and confirmed",
        "generation_unload": generation.get("unload")
        == "all Ollama models unloaded and confirmed",
        "query_cpu": query.get("device") == "cpu"
        and query.get("parameters_on_cpu") is True,
        "ingest_gpu": ingest.get("device") == "cuda" and ingest.get("encoded") is True,
        "ingest_release": ingest.get("released") is True
        and ingest.get("failure_cleanup") is True,
        "workload_serialization": (
            report.get("workload_serialization") is True
            and workload_evidence.get("tui_dispatch_verified") is True
            and workload_evidence.get("cancellation_cleanup_verified") is True
        ),
        "context_fit": (
            not generation.get("oom", True)
            and integer(generation.get("prompt_eval_count"))
            >= int(OLLAMA_CONTEXT_TOKENS * 0.9)
            and generation.get("vram_sampled") is True
            and integer(generation.get("ollama_vram_peak_bytes")) > 0
        ),
        "baseline_comparison": (
            baseline.get("before") is True
            and baseline.get("after") is True
            and integer(baseline.get("question_count")) >= len(BASELINE_CASES)
            and baseline.get("before_schema_valid") is True
            and baseline.get("book_identity_matches") is True
            and baseline.get("index_identity_matches") is True
            and bool(baseline.get("index_identity"))
            and baseline.get("labels_complete") is True
            and review.get("all_questions_reviewed") is True
            and review.get("unsupported_claims_checked") is True
            and probability(review.get("citation_accuracy"))
            and probability(review.get("hallucination_rate"))
            and all(
                probability(metric.get(key))
                for metric in (before_metrics, after_metrics)
                for key in ("precision", "recall")
            )
            and all(
                integer(metric.get("answerable_case_denominator")) > 0
                for metric in (before_metrics, after_metrics)
            )
        ),
        "local_only": report.get("local_only") is True
        and report.get("local_failure_behavior") is True,
    }
    return {"passed": all(gates.values()), "gates": gates}


def _gpu_snapshot() -> dict[str, Any] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        row = result.stdout.strip().splitlines()[0].split(",")
        return {
            "name": row[0].strip(),
            "total_mb": int(row[1].strip()),
            "used_mb": int(row[2].strip()),
            "free_mb": int(row[3].strip()),
        }
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return None


def _ollama_process_snapshot() -> list[dict[str, Any]] | None:
    try:
        with httpx.Client(timeout=1, trust_env=False) as client:
            response = client.get(f"{OLLAMA_URL}/api/ps")
            response.raise_for_status()
        return response.json().get("models", [])
    except (httpx.HTTPError, ValueError, TypeError):
        return None


def _measure_gpu(operation: Callable[[], Any]) -> tuple[Any, list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    stop = threading.Event()

    def sample() -> None:
        last_ollama_sample = 0.0
        while not stop.is_set():
            snapshot = _gpu_snapshot()
            if snapshot is not None:
                now = time.monotonic()
                if now - last_ollama_sample >= 1.0:
                    snapshot["ollama_models"] = _ollama_process_snapshot()
                    last_ollama_sample = now
                samples.append({"at_seconds": time.monotonic(), **snapshot})
            stop.wait(0.25)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    try:
        return operation(), samples
    finally:
        stop.set()
        monitor.join(timeout=2)
        snapshot = _gpu_snapshot()
        if snapshot is not None:
            snapshot["ollama_models"] = _ollama_process_snapshot()
            samples.append({"at_seconds": time.monotonic(), **snapshot})


def _read_sample(book_stem: str) -> tuple[str, int]:
    pages = sorted((PARSED_MD / "checkpoints" / book_stem).glob("page_*.md"))
    for path in pages:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            match = re.search(r"page_(\d+)", path.stem)
            return text[:4000], (int(match.group(1)) + 1) if match else 1
    full = PARSED_MD / book_stem / "full.md"
    if full.exists():
        text = full.read_text(encoding="utf-8").strip()
        if text:
            return text[:4000], 1
    raise FileNotFoundError(
        f"No existing parsed Markdown found for book {book_stem!r}."
    )


def _run_after_baseline(book_stem: str) -> list[dict[str, Any]]:
    collection = db.get_existing_collection(book_stem)
    db.ensure_collection_ready(collection)
    results: list[dict[str, Any]] = []
    conversation: list[dict[str, str]] = []
    for kind, question, _ in BASELINE_CASES:
        query_lower = question.lower()
        target_pages = extract_page_intent(question)
        query_words = set(query_lower.split())
        vague_terms = {
            "it",
            "this",
            "that",
            "they",
            "example",
            "question",
            "more",
            "explain",
            "help",
            "detail",
        }
        is_vague = bool(query_words & vague_terms) or len(query_words) < 5
        retrieval_query = question
        if is_vague and conversation:
            for turn in reversed(conversation):
                if turn["role"] == "user":
                    if any(
                        term in query_lower
                        for term in ("example", "question", "exercise", "problem")
                    ):
                        retrieval_query = f"{turn['content']} illustrative examples exercises questions problems"
                    else:
                        retrieval_query = (
                            f"{' '.join(turn['content'].split()[:20])} {question}"
                        )
                    break

        query_vector = db.embed_query(retrieval_query)
        query_args: dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": 8,
            "include": ["documents", "metadatas", "distances"],
        }
        if target_pages:
            query_args["where"] = (
                {"physical_page": target_pages[0]}
                if len(target_pages) == 1
                else {"physical_page": {"$in": target_pages}}
            )
        retrieved = collection.query(**query_args)
        documents = (retrieved.get("documents") or [[]])[0]
        metadatas = (retrieved.get("metadatas") or [[]])[0]
        distances = (retrieved.get("distances") or [[]])[0]
        evidence_ids = (retrieved.get("ids") or [[]])[0]
        if target_pages and not documents:
            fallback = collection.query(
                query_embeddings=[query_vector],
                n_results=8,
                include=["documents", "metadatas", "distances"],
            )
            documents = (fallback.get("documents") or [[]])[0]
            metadatas = (fallback.get("metadatas") or [[]])[0]
            distances = (fallback.get("distances") or [[]])[0]
            evidence_ids = (fallback.get("ids") or [[]])[0]

        blocks, metadata_rows, selected_ids, retrieved_pages = [], [], [], []
        limit = DISTANCE_THRESHOLD * 1.4 if is_vague else DISTANCE_THRESHOLD
        for index, (document, metadata) in enumerate(zip(documents, metadatas)):
            if (
                index < len(distances)
                and isinstance(distances[index], (int, float))
                and distances[index] >= limit
            ):
                continue
            physical = metadata.get("physical_page", metadata.get("page", "N/A"))
            try:
                physical = int(physical)
            except (TypeError, ValueError):
                pass
            blocks.append(
                f"[{metadata.get('Header 1', 'Unknown Chapter')} · "
                f"{metadata.get('Header 2', 'Unknown Section')} · p.{physical}]\n{document}"
            )
            metadata_rows.append(metadata)
            if index < len(evidence_ids):
                selected_ids.append(evidence_ids[index])
            if isinstance(physical, int) and physical >= 1:
                retrieved_pages.append(physical)
        evidence_ids = selected_ids

        history_lines = ["Conversation so far:"]
        for turn in conversation[-6:]:
            role = "User" if turn["role"] == "user" else "Axie"
            history_lines.append(f"{role}: {turn['content']}")
        history = "\n".join(history_lines) + "\n" if conversation else ""
        chapter = (
            ",".join(
                sorted({str(meta.get("Header 1", "Unknown")) for meta in metadata_rows})
            )
            if metadata_rows
            else "General Knowledge"
        )
        cited_pages = (
            ",".join(
                sorted(
                    {
                        str(meta.get("physical_page", meta.get("page", "?")))
                        for meta in metadata_rows
                    }
                )
            )
            if metadata_rows
            else "N/A"
        )
        prompt = PROMPT_TEMPLATE.format(
            book=book_stem,
            chapter=chapter,
            pages=cited_pages,
            history=history,
            context=(
                "\n\n".join(blocks)
                if blocks
                else "No direct context retrieved from indexed files."
            ),
            question=question,
        )
        started = time.monotonic()
        response = OllamaClient().generate(
            prompt, model=OLLAMA_MODEL, stream=False, think=False
        )
        if not isinstance(response, dict):
            raise RuntimeError("Expected a non-stream Ollama baseline response.")
        answer = strip_model_footnotes(latex_to_unicode(str(response.get("response", ""))))
        citation = build_citation(book_stem, retrieved_pages)
        if citation and not answer.rstrip().endswith(citation):
            answer = f"{answer.rstrip()}\n\n{citation}"
        results.append(
            {
                "kind": kind,
                "question": question,
                "retrieval_query": retrieval_query,
                "evidence_ids": evidence_ids,
                "evidence_metadata": metadata_rows,
                "answer": answer,
                "latency_seconds": time.monotonic() - started,
                "load_duration_ns": response.get("load_duration"),
                "total_duration_ns": response.get("total_duration"),
                "prompt_eval_count": response.get("prompt_eval_count"),
                "generation_count": response.get("eval_count"),
                "unsupported_claim_review": "manual review required",
            }
        )
        conversation.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )
    return results


def _valid_before_baseline(path: Path | None, book_stem: str) -> bool:
    if path is None or not path.is_file():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        rows = report.get("questions", [])
        expected = {kind: question for kind, question, _ in BASELINE_CASES}
        actual = {
            row.get("kind"): row
            for row in rows
            if isinstance(row, dict)
        }
        return (
            report.get("schema_version") == 1
            and report.get("book_stem") == book_stem
            and bool(report.get("index_identity"))
            and isinstance(report.get("model_identity"), dict)
            and bool(report["model_identity"].get("name"))
            and bool(re.fullmatch(
                r"[0-9a-f]{64}", str(report["model_identity"].get("digest", ""))
            ))
            and bool(report["model_identity"].get("ollama_version"))
            and len(rows) == len(expected)
            and set(actual) == set(expected)
            and all(
                actual[kind].get("question") == question
                and bool(actual[kind].get("answer"))
                and isinstance(actual[kind].get("evidence_ids"), list)
                and isinstance(actual[kind].get("relevant_evidence_ids"), list)
                and isinstance(actual[kind].get("expected_claims"), list)
                and isinstance(actual[kind].get("unsupported_claims"), list)
                and isinstance(actual[kind].get("citation_reviews"), list)
                and actual[kind].get("unsupported_claims_reviewed") is True
                and actual[kind].get("citation_review_complete") is True
                and isinstance(actual[kind].get("latency_seconds"), (int, float))
                for kind, question in expected.items()
            )
        )
    except (OSError, json.JSONDecodeError, AttributeError, TypeError):
        return False


def _retrieval_metrics(
    before_data: dict[str, Any], after_rows: list[dict[str, Any]]
) -> dict[str, Any] | None:
    before_by_kind = {
        row.get("kind"): row
        for row in before_data.get("questions", [])
        if isinstance(row, dict)
    }
    after_by_kind = {row.get("kind"): row for row in after_rows}
    per_phase: dict[str, dict[str, float | int]] = {}
    for phase, rows_by_kind in (("before", before_by_kind), ("after", after_by_kind)):
        precision_values: list[float] = []
        recall_values: list[float] = []
        for kind, baseline_row in before_by_kind.items():
            relevant = set(baseline_row.get("relevant_evidence_ids", []))
            if not relevant:
                continue
            measured_row = rows_by_kind.get(kind)
            if measured_row is None:
                return None
            retrieved = set(
                measured_row.get(
                    "evidence_ids", measured_row.get("retrieved_evidence_ids", [])
                )
            )
            hits = len(relevant & retrieved)
            precision_values.append(hits / len(retrieved) if retrieved else 0.0)
            recall_values.append(hits / len(relevant))
        if not precision_values:
            return None
        per_phase[phase] = {
            "precision": sum(precision_values) / len(precision_values),
            "recall": sum(recall_values) / len(recall_values),
            "answerable_case_denominator": len(precision_values),
        }
    return per_phase


def _current_index_identity(book_stem: str) -> str:
    collection = db.get_existing_collection(book_stem)
    identifiers = collection.get(include=[]).get("ids", [])
    signature = "\n".join(sorted(str(identifier) for identifier in identifiers))
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()


def _probe_local_failure_behavior() -> bool:
    """Confirm connection failures surface as errors without a remote fallback."""
    def unavailable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated local service unavailable", request=request)

    try:
        OllamaClient(transport=httpx.MockTransport(unavailable)).generate(
            "offline probe", stream=False
        )
    except httpx.ConnectError:
        return True
    return False


def _should_run_ingest(tools_report: dict[str, Any]) -> bool:
    return (
        tools_report.get("unload")
        == "all Ollama models unloaded and confirmed"
    )


def _probe_ingest_failure_cleanup(sample: str) -> bool:
    """Prove a failed encode releases module and stack-owned model references."""
    model = None
    original_encode = None
    failure_seen = False
    try:
        model = db.get_ingest_model()
        original_encode = model.encode

        def fail_encode(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("injected embedding failure")

        model.encode = fail_encode
        try:
            db.JinaV5DocumentEmbeddingFunction()([sample])
        except RuntimeError as exc:
            failure_seen = str(exc) == "injected embedding failure"
        finally:
            model.encode = original_encode
    finally:
        # A bound method owns its model. Drop it, then the local model, before
        # release_ingest_model runs GC and clears CUDA's free allocator blocks.
        if original_encode is not None:
            del original_encode
        if model is not None:
            del model
        db.release_ingest_model()
    return failure_seen and db._INGEST_MODEL is None


def _run_tui_workload_probe() -> dict[str, Any]:
    """Collect actual Textual dispatch/cancellation evidence in an isolated process."""
    script = Path(__file__).with_name("validate_tui_workload.py")
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=script.resolve().parents[1],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"probe exited {result.returncode}")
        return json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, RuntimeError) as exc:
        return {
            "tui_dispatch_verified": False,
            "cancellation_cleanup_verified": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def collect_evidence(
    book_stem: str, test_collection: str, baseline_before: Path | None
) -> dict[str, Any]:
    if not test_collection.startswith("foundation_"):
        raise ValueError(
            "Test collection must begin with 'foundation_' to protect normal indexes."
        )
    sample, physical_page = _read_sample(book_stem)
    workload_evidence = _run_tui_workload_probe()
    report: dict[str, Any] = {
        "book_stem": book_stem,
        "test_collection": test_collection,
        "test_vector_store": str(FOUNDATION_VECTOR_STORE),
        "config": {
            "context_tokens": OLLAMA_CONTEXT_TOKENS,
            "query_device": db.QUERY_EMBED_DEVICE,
            "ingest_device": db.INGEST_EMBED_DEVICE,
        },
        "local_only": OLLAMA_URL.startswith(
            ("http://localhost:", "http://127.0.0.1:", "http://[::1]:")
        ),
        "local_failure_behavior": _probe_local_failure_behavior(),
        "workload_serialization": (
            workload_evidence.get("tui_dispatch_verified") is True
            and workload_evidence.get("cancellation_cleanup_verified") is True
        ),
        "workload_evidence": workload_evidence,
    }

    manifest = capture_manifest()
    RUNTIME_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    report["local_service"] = "ollama_error" not in manifest
    report["model"] = {
        "name": manifest.get("model"),
        "digest": manifest.get("digest"),
        "ollama_version": manifest.get("ollama_version"),
        "quantization": manifest.get("model_details", {}).get("quantization_level"),
    }

    report["tools"] = run_diagnostic()
    TOOLS_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOOLS_REPORT_PATH.write_text(
        json.dumps(report["tools"], indent=2) + "\n", encoding="utf-8"
    )
    coordinator = WorkloadCoordinator()
    generation_token = coordinator.try_acquire("generation")
    report["workload_coordinator_smoke"] = bool(
        generation_token and coordinator.try_acquire("ingest") is None
    )
    if generation_token:
        coordinator.release(generation_token)
    ingest_token = coordinator.try_acquire("ingest")
    report["workload_coordinator_smoke"] = (
        report["workload_coordinator_smoke"]
        and bool(ingest_token)
        and coordinator.try_acquire("generation") is None
    )
    if ingest_token:
        coordinator.release(ingest_token)
    query_evidence: dict[str, Any] = {}
    ingest_evidence: dict[str, Any] = {
        "device": None,
        "encoded": False,
        "released": False,
        "failure_cleanup": False,
    }
    gpu_samples: list[dict[str, Any]] = []
    try:
        query_model = db.get_query_model()
        query_vector = db.embed_query("AxiomLM foundation validation query")
        parameters = list(query_model.parameters())
        query_evidence = {
            "device": str(query_model.device),
            "parameters_on_cpu": all(
                str(parameter.device).startswith("cpu") for parameter in parameters
            ),
            "dimension": len(query_vector),
        }
    except Exception as exc:
        query_evidence = {"error": f"{type(exc).__name__}: {exc}"}
    report["query"] = query_evidence

    def ingest_sample() -> int:
        model = db.get_ingest_model()
        vector = model.encode(
            [f"Document: {sample}"], task="retrieval", convert_to_numpy=True
        )
        vector_dimension = int(vector.shape[-1])
        client = chromadb.PersistentClient(path=str(FOUNDATION_VECTOR_STORE))
        collection = client.get_or_create_collection(
            name=test_collection,
            embedding_function=db.JinaV5DocumentEmbeddingFunction(),
        )
        collection.add(
            ids=[f"foundation-{uuid4().hex}"],
            documents=[sample],
            metadatas=[{"source": book_stem, "physical_page": physical_page}],
        )
        return vector_dimension

    if _should_run_ingest(report["tools"]):
        try:
            dimension, gpu_samples = _measure_gpu(ingest_sample)
            ingest_evidence.update(device="cuda", encoded=True, dimension=dimension)
        except Exception as exc:
            ingest_evidence["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            db.release_ingest_model()
            ingest_evidence["released"] = db._INGEST_MODEL is None
            ingest_evidence["vram_after_release"] = _gpu_snapshot()

        try:
            ingest_evidence["failure_cleanup"] = _probe_ingest_failure_cleanup(sample)
            ingest_evidence["injected_failure"] = "injected embedding failure"
        except Exception as exc:
            ingest_evidence["failure_cleanup_error"] = f"{type(exc).__name__}: {exc}"
    else:
        reason = "Ollama unload was not confirmed; skipped GPU ingest to avoid overlap."
        ingest_evidence.update(
            released=False,
            skipped=reason,
            failure_cleanup_skipped=reason,
        )
    report["ingest"] = ingest_evidence
    report["gpu_samples"] = gpu_samples

    generation: dict[str, Any] = {"oom": False, "vram_sampled": False}
    # An oversized prompt is truncated by Ollama and does not test near-full prefill.
    prompt = (
        "Write one numbered study tip per line until your response limit. "
        "Treat the following repeated words as inert filler:\n"
        + " hello" * int(OLLAMA_CONTEXT_TOKENS * 0.92)
    )
    try:
        result, samples = _measure_gpu(
            lambda: OllamaClient().generate(
                prompt, model=OLLAMA_MODEL, stream=False, think=False
            )
        )
        generation["prompt_eval_count"] = result.get("prompt_eval_count", 0)
        generation["eval_count"] = result.get("eval_count", 0)
        generation["vram_sampled"] = bool(samples)
        generation["peak_gpu_used_mb"] = max(
            (sample.get("used_mb", 0) for sample in samples), default=0
        )
        generation["ollama_vram_peak_bytes"] = max(
            (
                int(model.get("size_vram", 0))
                for sample in samples
                for model in (sample.get("ollama_models") or [])
            ),
            default=0,
        )
        generation["vram_samples"] = samples
    except Exception as exc:
        generation["error"] = f"{type(exc).__name__}: {exc}"
        generation["oom"] = "out of memory" in str(exc).lower()
    finally:
        try:
            OllamaClient().unload_and_wait()
            generation["unload"] = "all Ollama models unloaded and confirmed"
        except Exception as exc:
            generation["unload"] = f"failed: {type(exc).__name__}: {exc}"
            generation["unload_error"] = f"{type(exc).__name__}: {exc}"
    report["generation"] = generation

    try:
        before_valid = _valid_before_baseline(baseline_before, book_stem)
        before_data = (
            json.loads(baseline_before.read_text(encoding="utf-8"))
            if before_valid and baseline_before is not None
            else {}
        )
        current_index = _current_index_identity(book_stem)
        after_rows = _run_after_baseline(book_stem)
        index_matches = before_data.get("index_identity") == current_index
        report["baseline"] = {
            "before": before_valid,
            "before_path": str(baseline_before) if baseline_before else None,
            "before_capture_mode": before_data.get("capture_mode"),
            "before_source_commit": before_data.get("source_commit"),
            "before_model_identity": before_data.get("model_identity"),
            "after_results": after_rows,
            "after": True,
            "question_count": len(BASELINE_CASES),
            "before_schema_valid": before_valid,
            "book_identity_matches": before_data.get("book_stem") == book_stem,
            "index_identity": current_index,
            "index_identity_matches": index_matches,
            "labels_complete": before_valid,
            "after_review": {
                "all_questions_reviewed": False,
                "unsupported_claims_checked": False,
                "note": "Manually review every generated answer/citation before acceptance.",
            },
            "retrieval_metrics": (
                _retrieval_metrics(before_data, after_rows)
                if before_valid and index_matches
                else None
            ),
            "comparison_note": (
                "Review the before artifact's capture mode and provenance before interpreting answer quality."
            ),
        }
    except Exception as exc:
        report["baseline"] = {
            "before": _valid_before_baseline(baseline_before, book_stem),
            "before_path": str(baseline_before) if baseline_before else None,
            "after": False,
            "question_count": 0,
            "before_schema_valid": False,
            "book_identity_matches": False,
            "labels_complete": False,
            "after_review": {"all_questions_reviewed": False},
            "error": f"{type(exc).__name__}: {exc}",
        }

    report["verdict"] = evaluate_report(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "book_stem",
        nargs="?",
        help="Existing parsed book used for a small validation sample.",
    )
    parser.add_argument(
        "--test-collection",
        help="Explicit isolated collection name; must start with foundation_.",
    )
    parser.add_argument(
        "--evaluate-report",
        type=Path,
        help="Re-evaluate a manually reviewed evidence report without rerunning models.",
    )
    parser.add_argument(
        "--baseline-before",
        type=Path,
        help="Saved pre-change baseline JSON, if available.",
    )
    parser.add_argument("--report", type=Path, default=FOUNDATION_REPORT_PATH)
    args = parser.parse_args()
    if args.evaluate_report is not None:
        report = json.loads(args.evaluate_report.read_text(encoding="utf-8"))
        report["verdict"] = evaluate_report(report)
        args.evaluate_report.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report["verdict"], indent=2))
        return 0 if report["verdict"]["passed"] else 1
    if not args.book_stem or not args.test_collection:
        parser.error("book_stem and --test-collection are required unless --evaluate-report is used")
    report = collect_evidence(
        args.book_stem, args.test_collection, args.baseline_before
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report.get("verdict"), indent=2))
    print(f"Evidence report: {args.report}")
    return 0 if report["verdict"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
