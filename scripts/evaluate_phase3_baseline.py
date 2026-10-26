"""Capture the toolless Phase 2 path on the frozen Phase 3 cases BEFORE agent routing.

Runs follow-up rewriting, hybrid retrieval/context selection and one toolless
generation per case against an isolated copy of the vector store. The active
store, labels and every prior report stay untouched; existing outputs are refused.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config

LABELS = config.ROOT / "tests/fixtures/phase3/sicp_agent_labels.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_identity(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(str(path.relative_to(root)).encode() + b"\0")
        h.update(digest(path).encode())
    return h.hexdigest()


def load_cases(labels_path: Path = LABELS) -> tuple[dict, list[dict]]:
    """Frozen Phase 3 cases plus the hash-pinned Phase 2 cases they reference."""
    labels = json.loads(labels_path.read_text())
    reference = labels["phase2_reference"]
    phase2_path = config.ROOT / reference["path"]
    if digest(phase2_path) != reference["sha256"]:
        raise ValueError("Phase 2 labels changed since the Phase 3 fixture was frozen")
    phase2 = {c["case_id"]: c for c in json.loads(phase2_path.read_text())["cases"]}
    cases = [
        {
            "case_id": cid,
            "kind": phase2[cid]["kind"],
            "question": phase2[cid]["question"],
            "history": phase2[cid]["history"],
            "relevant_pages": phase2[cid]["relevant_pages"],
            "answerable": phase2[cid]["answerable"],
            "expected_mode": reference["expected_mode"],
            "source": "phase2",
        }
        for cid in reference["case_ids"]
    ]
    cases += [
        {**case, "history": case.get("history", []), "source": "phase3"}
        for case in labels["cases"]
    ]
    return labels, cases


def page_metrics(cases: list[dict], rows: list[dict]) -> dict:
    """Micro page precision/recall over cases with page labels, with denominators."""
    relevant = retrieved = hit = labelled = 0
    by_id = {r["case_id"]: r for r in rows}
    for case in cases:
        if not case["relevant_pages"] or case["case_id"] not in by_id:
            continue
        labelled += 1
        want = set(case["relevant_pages"])
        got = set(by_id[case["case_id"]]["physical_pages"])
        relevant += len(want)
        retrieved += len(got)
        hit += len(want & got)
    return {
        "labelled_cases": labelled,
        "relevant_pages": relevant,
        "retrieved_pages": retrieved,
        "hits": hit,
        "precision": hit / retrieved if retrieved else None,
        "recall": hit / relevant if relevant else None,
    }


def identities(paths: tuple[str, ...]) -> dict:
    from scripts.probe_phase3_runtime import _identities

    return {
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=config.ROOT, text=True
        ).strip(),
        "runtime": _identities(),
        "config_sha256": digest(config.ROOT / "src/config.py"),
        "code_sha256": {p: digest(config.ROOT / p) for p in paths},
    }


def reserve(output: Path):
    """Exclusively create the report before any work; prior evidence is kept."""
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    stream = output.open("x")
    stream.write('{"status":"running"}\n')
    stream.flush()
    return stream


def finish(stream, report: dict) -> None:
    stream.seek(0)
    json.dump(report, stream, indent=2, ensure_ascii=False)
    stream.truncate()
    stream.write("\n")
    stream.close()


def exchanges_for(history: list[dict], book: str) -> tuple:
    from src.query_rewriting import ConversationExchange

    return tuple(
        ConversationExchange(q["content"], a["content"], book)
        for q, a in zip(history[::2], history[1::2], strict=False)
        if q["role"] == "user" and a["role"] == "assistant"
    )


async def _run_case(app, client, case: dict, book: str) -> dict:
    from src.query_rewriting import rewrite_followup
    from src.ui.render import sanitize_answer

    history = list(case["history"])
    app._conversation = history
    row = {
        "case_id": case["case_id"],
        "kind": case["kind"],
        "question": case["question"],
    }
    started = time.monotonic()
    rewritten = await rewrite_followup(
        case["question"],
        exchanges_for(history, book),
        book,
        model=config.OLLAMA_MODEL,
        client=client,
    )
    prompt, metadata, pages, ids = await asyncio.to_thread(
        app._prepare_query,
        case["question"],
        app._format_history(),
        book,
        config.TOP_K_RESULTS,
        config.DISTANCE_THRESHOLD,
        rewritten.retrieval_query,
        config.OLLAMA_MODEL,
    )
    row.update(
        retrieval_query=(
            metadata[0]["retrieval_query"] if metadata else rewritten.retrieval_query
        ),
        evidence_ids=ids,
        physical_pages=pages,
        retrieval_seconds=time.monotonic() - started,
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        tools_used=[],
    )
    started = time.monotonic()
    try:
        response = await client.agenerate(
            prompt,
            model=config.OLLAMA_MODEL,
            think=False,
            num_predict=config.OLLAMA_OUTPUT_TOKENS,
            timeout=config.OLLAMA_REQUEST_TIMEOUT_SECONDS,
        )
        row["raw_answer"] = response["response"]
        row["copy_text"] = sanitize_answer(response["response"])
        row["done_reason"] = response.get("done_reason")
        # An output-cap stop is a failed completion, never an answered case.
        row["completed"] = response.get("done_reason") == "stop"
        row["eval_count"] = response.get("eval_count")
    except (TimeoutError, ValueError, OSError) as exc:
        row.update(completed=False, error=f"{type(exc).__name__}: {exc}")
    finally:
        row["generation_seconds"] = time.monotonic() - started
        await asyncio.to_thread(client.unload_and_wait)
    return row


async def _evaluate(report: dict, cases: list[dict], book: str, store: Path) -> None:
    from src import index_store, tui
    from src.llm import OllamaClient

    resolve = index_store.resolve_index
    client = OllamaClient()
    app = tui.AxiomLMApp()
    with patch.object(
        index_store, "resolve_index", lambda stem: resolve(stem, root=store)
    ):
        for case in cases:
            row = await _run_case(app, client, case, book)
            report["rows"].append(row)
            print(f"{case['case_id']}: pages {row['physical_pages']}", flush=True)


def evaluate(book: str, *, output: Path) -> dict:
    stream = reserve(output)
    report = {
        "schema_version": 1,
        "kind": "phase3_before_toolless",
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
        "book_stem": book,
        "labels_sha256": digest(LABELS),
        "answer_review": {"status": "unreviewed", "reviewer": None},
        "rows": [],
    }
    before = tree_identity(config.VECTOR_STORE)
    report["active_store_sha256_before"] = before
    try:
        labels, cases = load_cases()
        if labels["book_stem"] != book:
            raise ValueError("Labels are frozen for a different book")
        report.update(
            identities(
                (
                    "src/tui.py",
                    "src/retrieval.py",
                    "src/hybrid_retrieval.py",
                    "src/query_rewriting.py",
                    "scripts/evaluate_phase3_baseline.py",
                )
            )
        )
        with tempfile.TemporaryDirectory(prefix="axiom-phase3-baseline-") as temp:
            store = Path(temp) / "vector_store"
            shutil.copytree(config.VECTOR_STORE, store)
            asyncio.run(_evaluate(report, cases, book, store))
        report["retrieval_metrics"] = page_metrics(cases, report["rows"])
        report["completion"] = {
            "cases": len(report["rows"]),
            "completed": sum(1 for r in report["rows"] if r.get("completed")),
        }
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        if not isinstance(exc, Exception):
            raise
    finally:
        after = tree_identity(config.VECTOR_STORE)
        report["active_store_sha256_after"] = after
        report["active_store_unchanged"] = after == before
        if after != before:
            report.update(status="failed", error="Active store changed during run")
        report["finished_utc"] = datetime.now(UTC).isoformat()
        finish(stream, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("book")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.book, output=args.output)
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in ("status", "error", "retrieval_metrics", "completion")
            }
        )
    )
    raise SystemExit(0 if result["status"] == "complete" else 1)
