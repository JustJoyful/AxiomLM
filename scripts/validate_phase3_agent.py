"""Evaluate the Phase 3 agent on the frozen cases against the toolless before-report.

Runs the production agent (real local model, real native sandbox) over an
isolated vector-store copy. By default web permission is OFF (the offline
path); --web grants it and needs the configured search endpoint (the online
path), checking web use against each case's frozen web expectation.
Answer quality, citation accuracy and hallucination review stay "unreviewed"
until a person attributes them; nothing here is a page-existence proxy.
"""

from __future__ import annotations

import argparse
import asyncio
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

import httpx

from scripts.evaluate_phase3_baseline import (
    LABELS,
    digest,
    exchanges_for,
    finish,
    identities,
    load_cases,
    page_metrics,
    reserve,
    tree_identity,
)
from src import config

CODE = (
    "src/agent.py",
    "src/chat_tokens.py",
    "src/llm.py",
    "src/retrieval.py",
    "src/tools/protocol.py",
    "src/tools/registry.py",
    "src/tools/local.py",
    "src/tools/web.py",
    "src/tools/math_checks.py",
    "src/sandbox/launcher.py",
    "src/sandbox/worker.py",
    "src/sandbox/client.py",
    "scripts/validate_phase3_agent.py",
)
POLICY_NAMES = (
    "mode",
    "local_search_first",
    "no_web_while_off",
    "completed",
    "stage_order",
    "check_status",
    "web_when_expected",
)


def policy_checks(case: dict, row: dict, *, web: bool = False) -> dict:
    """Label-derived policy outcomes for one case; absent keys are not applicable."""
    stages = row.get("stages", [])
    used_web = "optional_web" in stages
    checks = {
        "mode": row.get("mode") == case.get("expected_mode"),
        "local_search_first": len(stages) > 1 and stages[1] == "local_search",
        "completed": row.get("completed") is True,
    }
    if not web:
        checks["no_web_while_off"] = not used_web
    else:
        expected = case.get("web_expected")
        if expected is False:
            checks["web_when_expected"] = not used_web
        elif expected == "only_when_enabled":
            checks["web_when_expected"] = used_web
        elif expected == "only_on_failure_when_enabled":
            verified = (row.get("verification") or {}).get("status") == "verified"
            checks["web_when_expected"] = used_web != verified
    expected = case.get("expected_stage_order")
    if expected:
        checks["stage_order"] = [s for s in stages if s in expected] == list(expected)
    status = case.get("expected_check_status")
    if status:
        checks["check_status"] = (row.get("verification") or {}).get("status") == status
    return checks


def summarize(policy: list[dict]) -> dict:
    return {
        name: {
            "passed": sum(1 for p in policy if p.get(name) is True),
            "applicable": sum(1 for p in policy if name in p),
        }
        for name in POLICY_NAMES
    }


async def _run_case(case, book, client, sandbox, counter, web) -> dict:
    from src.agent import AgentRequest, run_agent
    from src.query_rewriting import rewrite_followup
    from src.tools.web import WebTools

    history = list(case["history"])
    started = time.monotonic()
    rewritten = await rewrite_followup(
        case["question"],
        exchanges_for(history, book),
        book,
        model=config.OLLAMA_MODEL,
        client=client,
    )
    request = AgentRequest(
        f"eval-{case['case_id']}",
        case["question"],
        book,
        config.OLLAMA_MODEL,
        history=tuple(history),
        web_enabled=web,
        retrieval_query=rewritten.retrieval_query,
    )
    row = {
        "case_id": case["case_id"],
        "kind": case["kind"],
        "question": case["question"],
        "physical_pages": [],
    }
    stage_times = []
    try:
        async for event in run_agent(
            request,
            client=client,
            sandbox=sandbox,
            counter=counter,
            # The production web tools, as the TUI wires them; None is "not configured".
            web=WebTools() if web else None,
        ):
            if event.kind == "step":
                stage_times.append((event.data["stage"], time.monotonic() - started))
                if event.data["stage"] == "route":
                    row["mode"] = event.data["mode"]
                elif event.data["stage"] == "assessment":
                    row["assessment"] = {
                        "sufficient": event.data["sufficient"],
                        "reason": event.data["reason"],
                    }
                elif event.data["stage"] == "compute_done":
                    row["computed"] = event.data["result"]
            elif event.kind == "sources":
                # Retrieved evidence counts even if a later stage hits a limit.
                row["physical_pages"] = sorted(
                    {
                        p
                        for s in event.data["sources"]
                        if s.kind == "book"
                        for p in s.physical_pages
                    }
                )
            elif event.kind == "result":
                data = event.data
                row.update(
                    mode=data["mode"],
                    stages=data["stages"],
                    verification=data["verification"],
                    completed=data["complete"],
                    done_reason=data["done_reason"],
                    answer=data["answer"],
                    trace=data["trace"],
                    model_requests=data["model_requests"],
                    tool_calls=data["tool_calls"],
                    physical_pages=sorted(
                        {
                            p
                            for s in data["sources"]
                            if s.kind == "book"
                            for p in s.physical_pages
                        }
                    ),
                    web_sources=[s.url for s in data["sources"] if s.kind == "web"],
                    evidence_ids=[s.source_id for s in data["sources"]],
                )
            elif event.kind == "error":
                row.update(
                    completed=False,
                    error=event.data["message"],
                    stages=event.data["stages"],
                    trace=event.data["trace"],
                )
    finally:
        row["seconds"] = time.monotonic() - started
        row["stage_seconds"] = stage_times
        await asyncio.to_thread(client.unload_and_wait)
    row["policy"] = policy_checks(case, row, web=web)
    return row


def _leftover_scopes() -> list[str]:
    out = subprocess.run(
        [
            "systemctl",
            "--user",
            "list-units",
            "--plain",
            "--no-legend",
            "axiom-python-*",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return [line.split()[0] for line in out.splitlines() if line.strip()]


async def _evaluate(report, cases, book, store, web):
    from src import index_store
    from src.chat_tokens import load_chat_counter
    from src.llm import OllamaClient
    from src.sandbox.client import SandboxClient

    client = OllamaClient()
    counter = await asyncio.to_thread(load_chat_counter, config.OLLAMA_MODEL)
    sandbox = SandboxClient(request_prefix="phase3-eval")
    capability = await sandbox.start()
    report["sandbox"] = {"available": capability.available, "reason": capability.reason}
    resolve = index_store.resolve_index
    try:
        with patch.object(
            index_store, "resolve_index", lambda stem: resolve(stem, root=store)
        ):
            for case in cases:
                row = await _run_case(case, book, client, sandbox, counter, web)
                report["rows"].append(row)
                print(f"{case['case_id']}: {row.get('stages')}", flush=True)
    finally:
        await sandbox.aclose()


def evaluate(book: str, *, baseline: Path, output: Path, web: bool = False) -> dict:
    before = json.loads(baseline.read_text())
    if before.get("status") != "complete" or before.get("labels_sha256") != digest(
        LABELS
    ):
        raise ValueError("Baseline is incomplete or used different labels")
    if web and not config.WEB_SEARCH_URL:
        raise ValueError("--web needs a configured WEB_SEARCH_URL")
    stream = reserve(output)
    report = {
        "schema_version": 1,
        "kind": "phase3_after_agent_online" if web else "phase3_after_agent_offline",
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
        "book_stem": book,
        "labels_sha256": digest(LABELS),
        "baseline": str(baseline),
        "baseline_sha256": digest(baseline),
        "web": (
            {"enabled": True, "search_url": config.WEB_SEARCH_URL}
            if web
            else {
                "enabled": False,
                "online_path": "not evaluated: run again with --web",
            }
        ),
        "answer_review": {
            "status": "unreviewed",
            "reviewer": None,
            "citation_accuracy": None,
            "hallucination_rate": None,
        },
        "rows": [],
    }
    store_before = tree_identity(config.VECTOR_STORE)
    try:
        labels, cases = load_cases()
        if labels["book_stem"] != book:
            raise ValueError("Labels are frozen for a different book")
        # Phase 2 cases carry their web expectation on the frozen reference.
        for case in cases:
            if case["case_id"] in labels["phase2_reference"].get("web_expected", {}):
                case.setdefault("web_expected", "only_when_enabled")
        report.update(identities(CODE))
        with tempfile.TemporaryDirectory(prefix="axiom-phase3-agent-") as temp:
            store = Path(temp) / "vector_store"
            shutil.copytree(config.VECTOR_STORE, store)
            asyncio.run(_evaluate(report, cases, book, store, web))
        rows = report["rows"]
        report["retrieval_metrics"] = {
            "after": page_metrics(cases, rows),
            "before": before.get("retrieval_metrics"),
        }
        report["completion"] = {
            "after": {
                "cases": len(rows),
                "completed": sum(1 for r in rows if r.get("completed")),
            },
            "before": before.get("completion"),
        }
        report["policy_summary"] = summarize([r["policy"] for r in rows])
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        if not isinstance(exc, Exception):
            raise
    finally:
        try:
            with httpx.Client(trust_env=False, timeout=10) as http:
                loaded = (
                    http.get(config.OLLAMA_URL + "/api/ps").json().get("models", [])
                )
        except httpx.HTTPError:
            loaded = ["unknown"]
        report["cleanup"] = {
            "loaded_models": loaded,
            "sandbox_scopes": _leftover_scopes(),
            "active_store_unchanged": tree_identity(config.VECTOR_STORE)
            == store_before,
        }
        report["passed"] = (
            report["status"] == "complete"
            and not loaded
            and not report["cleanup"]["sandbox_scopes"]
            and report["cleanup"]["active_store_unchanged"]
            and all(
                v["passed"] == v["applicable"]
                for v in report.get("policy_summary", {}).values()
            )
        )
        report["finished_utc"] = datetime.now(UTC).isoformat()
        finish(stream, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("book")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--web", action="store_true", help="grant web permission (online path)"
    )
    args = parser.parse_args()
    result = evaluate(
        args.book, baseline=args.baseline, output=args.output, web=args.web
    )
    keys = (
        "status",
        "passed",
        "error",
        "policy_summary",
        "retrieval_metrics",
        "completion",
        "cleanup",
    )
    print(json.dumps({k: result.get(k) for k in keys}, default=str))
    raise SystemExit(0 if result.get("passed") else 1)
