"""Freeze current SICP retrieval/answers against reviewed labels, on an index copy.

Run with: uv run --no-sync python scripts/evaluate_phase2_baseline.py
The active store and parsed artifacts are never opened by Chroma or rewritten.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The baseline must never repair the model cache by downloading missing weights.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src import config

LABELS_PATH = config.ROOT / "tests/fixtures/phase2/sicp_context_labels.json"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_labels(path: Path) -> dict:
    labels = json.loads(path.read_text())
    if labels.get("schema_version") != 1 or not labels.get("book_stem"):
        raise ValueError("Unsupported labels or missing book identity")
    cases = labels.get("cases", [])
    seen = set()
    for row in cases:
        cid = row.get("case_id")
        if not cid or cid in seen or row.get("book_stem") != labels["book_stem"]:
            raise ValueError("Duplicate/missing case ID or mismatched book")
        seen.add(cid)
        if not row.get("kind") or not row.get("question"):
            raise ValueError(f"Missing kind/question for {cid}")
        if type(row.get("answerable")) is not bool:
            raise ValueError(f"Invalid answerable flag for {cid}")
        pages = row.get("relevant_pages")
        if not isinstance(pages, list) or any(
            type(p) is not int or p < 1 for p in pages
        ):
            raise ValueError(f"Invalid physical pages for {cid}")
        if bool(pages) != row["answerable"] or len(pages) != len(set(pages)):
            raise ValueError(f"Answerability/pages mismatch for {cid}")
        if any(str(p) not in labels.get("source_pages", {}) for p in pages):
            raise ValueError(f"Missing source hash for {cid}")
        claims = row.get("expected_claims")
        if not claims:
            raise ValueError(f"Missing expected claims for {cid}")
        for claim in claims:
            sources = claim.get("source_pages", [])
            if not claim.get("claim") or not set(sources).issubset(pages):
                raise ValueError(f"Claim evidence outside labeled pages for {cid}")
            if row["answerable"] and not sources:
                raise ValueError(f"Claim missing evidence for {cid}")
        history = row.get("history", [])
        if len(history) % 2 or any(
            turn.get("role") != ("user" if i % 2 == 0 else "assistant")
            or not turn.get("content")
            for i, turn in enumerate(history)
        ):
            raise ValueError(f"History must contain completed turns for {cid}")
    if not cases:
        raise ValueError("No evaluation cases")
    return labels


def verify_sources(labels: dict, parsed_root: Path) -> None:
    book = labels["book_stem"]
    manifest = json.loads((parsed_root / book / "page_manifest.json").read_text())
    if manifest["book_stem"] != book:
        raise ValueError("Manifest book changed")
    rows = {r["physical_page"]: r for r in manifest["pages"]}
    for page, expected_hash in labels["source_pages"].items():
        physical = int(page)
        row = rows[physical]
        expected_file = f"page_{physical - 1:04d}.md"
        if row["page_idx"] != physical - 1 or row["checkpoint_file"] != expected_file:
            raise ValueError(f"Physical-page mapping changed for page {page}")
        path = parsed_root / "checkpoints" / book / expected_file
        if digest(path) != expected_hash:
            raise ValueError(
                f"Source text changed for page {page}; review labels again"
            )


def tree_identity(root: Path) -> str:
    entries = [
        (str(p.relative_to(root)), digest(p))
        for p in sorted(root.rglob("*"))
        if p.is_file()
    ]
    if not entries:
        raise ValueError(f"Empty index store: {root}")
    return hashlib.sha256(json.dumps(entries).encode()).hexdigest()


def snapshot_store(source: Path, destination: Path) -> str:
    before = tree_identity(source)
    shutil.copytree(source, destination)
    if tree_identity(source) != before or tree_identity(destination) != before:
        raise RuntimeError("Index changed while copying; stop queries/ingest and retry")
    return before


def retrieval_metrics(cases: list[dict], rows: list[dict]) -> dict:
    by_id = {row["case_id"]: row for row in rows}
    if len(by_id) != len(cases) or set(by_id) != {c["case_id"] for c in cases}:
        raise ValueError("Metrics require exactly one result for every frozen case")
    precision, recall = [], []
    unanswerable_with_evidence = 0
    for case in cases:
        retrieved = set(by_id[case["case_id"]]["physical_pages"])
        if not case["answerable"]:
            unanswerable_with_evidence += bool(retrieved)
            continue
        relevant = set(case["relevant_pages"])
        hits = len(relevant & retrieved)
        precision.append(hits / len(retrieved) if retrieved else 0.0)
        recall.append(hits / len(relevant))
    return {
        "macro_page_precision": sum(precision) / len(precision) if precision else None,
        "macro_page_recall": sum(recall) / len(recall) if recall else None,
        "answerable_case_denominator": len(precision),
        "unanswerable_cases_returning_evidence": unanswerable_with_evidence,
        "definition": "Mean per-answerable-case precision=gold hits/unique retrieved pages; recall=gold hits/unique gold pages. Empty retrieval scores zero. Unanswerable excluded.",
        "limitation": "Gold pages are curated sufficient evidence, not exhaustive relevance labels; other valid pages can be counted as misses. Eight cases cannot establish general quality.",
    }


def save_report(report: dict) -> None:
    path = config.PHASE2_EVALUATION_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def run() -> dict:
    if config.PHASE2_EVALUATION_PATH.exists():
        raise FileExistsError(
            "Baseline report already exists; preserve it before a new run"
        )
    import chromadb
    from huggingface_hub import snapshot_download

    from scripts.capture_runtime_manifest import capture_manifest
    from scripts.validate_foundation import _current_index_identity
    from src import db, tui
    from src.llm import OllamaClient
    from src.ui.render import sanitize_answer

    labels = load_labels(LABELS_PATH)
    verify_sources(labels, config.PARSED_MD)
    snapshot = snapshot_download(
        repo_id=config.EMBED_MODEL,
        revision=config.EMBED_REVISION,
        local_files_only=True,
    )
    report = {
        "schema_version": 1,
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
        "book_stem": labels["book_stem"],
        "dataset_sha256": digest(LABELS_PATH),
        "dataset_path": str(LABELS_PATH.relative_to(config.ROOT)),
        "source_manifest_sha256": digest(
            config.PARSED_MD / labels["book_stem"] / "page_manifest.json"
        ),
        "config_sha256": digest(config.ROOT / "src/config.py"),
        "code_sha256": {
            p: digest(config.ROOT / p)
            for p in (
                "src/tui.py",
                "src/db.py",
                "src/llm.py",
                "scripts/evaluate_phase2_baseline.py",
            )
        },
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=config.ROOT, text=True
        ).strip(),
        "runtime": capture_manifest(),
        "settings": {
            "k": config.TOP_K_RESULTS,
            "distance_threshold": config.DISTANCE_THRESHOLD,
            "options": OllamaClient._options(),
            "think": False,
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
        },
        "prompt_semantics": "Unmodified Phase 1 db.embed_query; manual Query prefix plus cached default document prompt. Baseline only; normalization requires reindex.",
        "answer_review": {
            "status": "unreviewed",
            "reviewer": None,
            "citation_accuracy": None,
            "hallucination_rate": None,
        },
        "rows": [],
    }
    if "ollama_error" in report["runtime"] or "model_error" in report["runtime"]:
        raise RuntimeError(f"Local runtime unavailable: {report['runtime']}")
    client = OllamaClient()
    app = tui.AxiomLMApp()
    before = tree_identity(config.VECTOR_STORE)
    report["active_store_sha256_before"] = before
    try:
        with tempfile.TemporaryDirectory(prefix="axiom-phase2-baseline-") as temp:
            copy_path = Path(temp) / "vector_store"
            snapshot_store(config.VECTOR_STORE, copy_path)
            snapshot_client = chromadb.PersistentClient(path=str(copy_path))
            with (
                patch.object(db, "_CLIENT", snapshot_client),
                patch.object(db, "_MODEL_SNAPSHOT", snapshot),
                patch.object(tui, "get_collection", db.get_existing_collection),
            ):
                report["index_ids_sha256"] = _current_index_identity(
                    labels["book_stem"]
                )
                collection = db.get_existing_collection(labels["book_stem"])
                report["collection_count"] = collection.count()
                report["collection_metadata"] = collection.metadata
                for case in labels["cases"]:
                    app._conversation = list(case["history"])
                    retrieval_queries = []

                    def traced_embed(query, queries=retrieval_queries):
                        queries.append(query)
                        vector = db.embed_query(query)
                        model = db.get_query_model()
                        report["query_embedding"] = {
                            "device": str(model.device),
                            "dimension": len(vector),
                            "task": "retrieval",
                            "internal_prompts": model.prompts,
                            "default_prompt_name": model.default_prompt_name,
                        }
                        return vector

                    started = time.monotonic()
                    with patch.object(tui, "embed_query", traced_embed):
                        prompt, metadata, pages, ids = app._prepare_legacy_query(
                            case["question"],
                            app._format_history(),
                            labels["book_stem"],
                            config.TOP_K_RESULTS,
                            config.DISTANCE_THRESHOLD,
                        )
                    row = {
                        "case_id": case["case_id"],
                        "kind": case["kind"],
                        "question": case["question"],
                        "retrieval_query": retrieval_queries[-1],
                        "history": case["history"],
                        "evidence_ids": ids,
                        "evidence_metadata": metadata,
                        "physical_pages": pages,
                        "retrieval_seconds": time.monotonic() - started,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    }
                    report["rows"].append(row)
                    save_report(report)
                    print(
                        f"{case['case_id']}: {len(ids)} chunks, pages {pages}; generating",
                        flush=True,
                    )
                    token = app._try_acquire_workload("generation")
                    if token is None:
                        raise RuntimeError("GPU admission unavailable")
                    started = time.monotonic()
                    try:
                        response = client.generate(
                            prompt, model=config.OLLAMA_MODEL, stream=False, think=False
                        )
                        if (
                            not isinstance(response, dict)
                            or response.get("error")
                            or response.get("done") is not True
                            or not response.get("response")
                        ):
                            raise RuntimeError(f"Invalid Ollama response: {response}")
                        row["raw_answer"] = response.get("response", "")
                        row["copy_text"] = sanitize_answer(row["raw_answer"])
                        row["generation_metrics"] = {
                            k: response.get(k)
                            for k in (
                                "done",
                                "done_reason",
                                "prompt_eval_count",
                                "eval_count",
                                "total_duration",
                                "load_duration",
                                "eval_duration",
                            )
                        }
                    finally:
                        row["generation_seconds"] = time.monotonic() - started
                        client.unload_and_wait()
                        app._release_workload(token)
                    save_report(report)
            report["retrieval_metrics"] = retrieval_metrics(
                labels["cases"], report["rows"]
            )
        verify_sources(labels, config.PARSED_MD)
        report["status"] = "complete"
    except BaseException as exc:
        # Preserve interruption evidence before propagating Ctrl+C/SystemExit.
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["active_store_sha256_after"] = tree_identity(config.VECTOR_STORE)
        report["active_store_unchanged"] = report["active_store_sha256_after"] == before
        if not report["active_store_unchanged"]:
            report.update(
                status="failed",
                error="Active store changed during baseline; results require a new run",
            )
        report["finished_utc"] = datetime.now(UTC).isoformat()
        save_report(report)
    if report["status"] != "complete":
        raise RuntimeError(report["error"])
    return report


if __name__ == "__main__":
    run()
    print(config.PHASE2_EVALUATION_PATH)
