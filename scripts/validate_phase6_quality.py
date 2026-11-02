"""Immutable quality comparisons and blank, independently attributed review sheets.

No model/scorer verdict is ever converted into a human judgment. This CLI reads
reports; inference is explicitly performed by the existing isolated evaluators.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_phase2_baseline import digest
from src import config

RANKING_CODE = (
    "src/hybrid_retrieval.py",
    "src/retrieval.py",
    "src/query_rewriting.py",
    "src/tools/local.py",
)


def _ranking_identity(report):
    identity = {
        "config_sha256": report.get("config_sha256"),
        "code_sha256": {p: report.get("code_sha256", {}).get(p) for p in RANKING_CODE},
    }
    if not identity["config_sha256"] or not all(identity["code_sha256"].values()):
        raise ValueError("missing ranking/config identity")
    return identity


def write_evidence(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _unique(rows, key):
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{key} rows must be objects in a list")
    result = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value.strip() or value in result:
            raise ValueError(f"missing/duplicate {key}")
        result[value] = row
    return result


def _expected(case):
    # IDs bind the frozen order and text, without rewriting the original labels.
    return {
        f"{case['case_id']}:{i}:{hashlib.sha256(c['claim'].encode()).hexdigest()}": c
        for i, c in enumerate(case["expected_claims"])
    }


def _cases(report):
    if "cases" in report:
        return report["cases"]
    from scripts.evaluate_phase4 import cases_for, load_labels

    path = Path(report["labels_path"])
    if digest(path) != report["labels_sha256"]:
        raise ValueError("labels identity changed")
    labels = load_labels(path, pages_root=config.REPAIRED_MD)
    if labels["book_stem"] != report["book_stem"]:
        raise ValueError("book identity changed")
    cases = cases_for(labels, report["split"])
    if report.get("case_subset"):
        cases = [c for c in cases if c["case_id"] in report["case_subset"]]
    return cases


def blank_review(report: dict, *, report_sha256: str) -> dict:
    rows = _unique(report["rows"], "case_id")
    return {
        "schema_version": 1,
        "source_report_sha256": report_sha256,
        "reviewer": None,
        "reviewed_at": None,
        "complete_inventory": False,
        "instructions": (
            "Inventory every claim in every answer, including failed/partial answers. "
            "Judge facts independently and support against actual supplied sources/pages. "
            "Null is unreviewed; give a reason for unclear judgments. Expected-claim IDs "
            "bind frozen text/order. Page validity alone does not establish support."
        ),
        "answers": {
            c["case_id"]: {
                "answer": rows.get(c["case_id"], {}).get("answer", ""),
                "complete_inventory": False,
                "claims": [],
                "expected_claims": dict.fromkeys(_expected(c)),
                "expected_claim_text": {k: v["claim"] for k, v in _expected(c).items()},
                "premise_rejected": None,
            }
            for c in _cases(report)
        },
    }


def _verdict(value, name, *, nullable=True):
    if type(value) is bool or (nullable and value is None):
        return value
    raise ValueError(f"invalid {name} verdict; use bool or null")


def _rate(numerator, denominator):
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def _complete(row):
    return (
        row.get("complete") is True
        and row.get("error") is None
        and isinstance(row.get("answer"), str)
        and bool(row["answer"].strip())
    )


def validate_review(report: dict, review: dict, *, report_sha256: str) -> dict:
    if review.get("source_report_sha256") != report_sha256:
        raise ValueError("review report hash mismatch")
    reviewer = review.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("reviewer identity required")
    if review.get("schema_version") != 1:
        raise ValueError("unsupported review schema")
    try:
        date = datetime.fromisoformat(review["reviewed_at"])
        if date.tzinfo is None:
            raise ValueError("timezone required")
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise ValueError("reviewed_at requires an ISO timestamp with timezone") from exc
    inventory = _verdict(review.get("complete_inventory"), "inventory", nullable=False)
    cases = _unique(_cases(report), "case_id")
    rows = _unique(report["rows"], "case_id")
    answers = review.get("answers", {})
    if (
        not isinstance(answers, dict)
        or any(not isinstance(item, dict) for item in answers.values())
        or set(answers) - set(cases)
        or set(rows) - set(cases)
    ):
        raise ValueError("unknown answer case IDs")
    correct, supported, citations = [], [], []
    unknown_correct = unknown_support = unknown_citations = 0
    expected_total = covered = completed = rejected = premises = unreviewed_expected = 0
    for cid, case in cases.items():
        row = rows.get(cid, {})
        done = _complete(row)
        completed += done
        item = answers.get(cid, {})
        inventory = inventory and item.get("complete_inventory") is True
        if "complete_inventory" in item:
            _verdict(item["complete_inventory"], "answer inventory", nullable=False)
        if item.get("answer", "") != row.get("answer", ""):
            raise ValueError(f"review answer text changed for {cid}")
        expected = _expected(case)
        coverage = item.get("expected_claims", {})
        if not isinstance(coverage, dict) or set(coverage) - set(expected):
            raise ValueError("unknown expected claim IDs")
        expected_total += len(expected)
        for eid in expected:
            value = _verdict(coverage.get(eid), "expected coverage")
            covered += done and value is True
            unreviewed_expected += value is None
        if case.get("false_premise"):
            premises += 1
            rejected += (
                done and _verdict(item.get("premise_rejected"), "premise") is True
            )
        sources = _unique(row.get("sources", []), "source_id")
        claims = _unique(item.get("claims", []), "claim_id")
        if (
            item.get("complete_inventory") is True
            and row.get("answer", "").strip()
            and not claims
        ):
            raise ValueError("nonempty answer requires a complete claim inventory")
        for claim in claims.values():
            if not isinstance(claim.get("text"), str) or not claim["text"].strip():
                raise ValueError("claim text required")
            factual = _verdict(claim.get("factual"), "factual", nullable=False)
            fact = _verdict(claim.get("correct"), "correct")
            support = _verdict(claim.get("supported"), "supported")
            if factual:
                if (fact is None or support is None) and not claim.get("reason"):
                    raise ValueError("unclear factual verdict needs a reason")
                unknown_correct += fact is None
                unknown_support += support is None
                if fact is not None:
                    correct.append(fact)
                if support is not None:
                    supported.append(support)
            elif fact is not None or support is not None:
                raise ValueError(
                    "nonfactual claims require null factual/support verdicts"
                )
            pairs = set()
            claim_citations = claim.get("citations", [])
            if not isinstance(claim_citations, list) or any(
                not isinstance(c, dict) for c in claim_citations
            ):
                raise ValueError("citations must be a list of objects")
            for citation in claim_citations:
                sid, page = citation.get("source_id"), citation.get("page")
                source = sources.get(sid)
                if source is None or (sid, page) in pairs:
                    raise ValueError("invalid/duplicate cited source/page pair")
                pairs.add((sid, page))
                if source.get("kind") == "book":
                    if type(page) is not int or page < 1 or page not in source["pages"]:
                        raise ValueError("citation page outside supplied source")
                elif page is not None:
                    raise ValueError("web citations have no physical page")
                value = _verdict(citation.get("supported"), "citation support")
                if value is None:
                    if not citation.get("reason"):
                        raise ValueError("unclear citation verdict needs a reason")
                    unknown_citations += 1
                else:
                    citations.append(value)
    return {
        "reviewer": reviewer,
        "inventory_complete": bool(inventory),
        "factual_error": _rate(correct.count(False), len(correct)),
        "unsupported_claim": _rate(supported.count(False), len(supported)),
        "citation_support": _rate(citations.count(True), len(citations)),
        "expected_claim_coverage": _rate(covered, expected_total),
        "answer_coverage": _rate(completed, len(cases)),
        "premise_rejection": _rate(rejected, premises),
        "unreviewed_factual_claims": unknown_correct,
        "unreviewed_support_claims": unknown_support,
        "unreviewed_citation_pairs": unknown_citations,
        "unreviewed_expected_claims": unreviewed_expected,
    }


def _identity(report):
    runtime = report.get("runtime", {})
    source_pages = report.get("source_pages")
    if source_pages is None:
        path = Path(report["labels_path"])
        if digest(path) != report["labels_sha256"]:
            raise ValueError("labels identity changed")
        source_pages = json.loads(path.read_text())["source_pages"]
    return {
        "book": report["book_stem"],
        "split": report["split"],
        "labels": report["labels_sha256"],
        "sources": source_pages,
        "index": report["index_version"],
        "sampling": report["sampling"],
        "runtime": {k: runtime[k] for k in ("model", "digest", "ollama_version")},
        "cases": _cases(report),
        "definition": report.get("metric_definition", "micro:selected answer context"),
    }


def compare_quality(before: dict, after: dict) -> dict:
    left, right = _identity(before), _identity(after)
    for key in left:
        if not left[key] or left[key] != right[key]:
            raise ValueError(f"incomparable identity/definition: {key}")
    if left["definition"] != "micro:selected answer context":
        raise ValueError(
            "unsupported metric definition; original macro gate is separate"
        )
    selection = after.get("selection")
    if selection and selection.get("split") != "calibration":
        raise ValueError(
            "policy must be selected on calibration before held-out evaluation"
        )
    for report in (before, after):
        cutoff = (
            report.get("metrics", {})
            .get("retrieval", {})
            .get("cutoff", "selected answer context")
        )
        if cutoff != "selected answer context":
            raise ValueError("incomparable retrieval cutoff")
    if after.get("merge_policy", "interleave") != before.get(
        "merge_policy", "interleave"
    ):
        if not selection or selection.get("policy") != after["merge_policy"]:
            raise ValueError("changed policy needs pinned calibration selection")
        pins = selection.get("input_sha256", {})
        if not isinstance(pins, dict) or len(pins) != 2:
            raise ValueError("calibration selection needs both input hashes")
        for path, sha in pins.items():
            if digest(Path(path)) != sha:
                raise ValueError("calibration selection input identity changed")
        expected = select_candidate([Path(p) for p in pins])
        if _ranking_identity(after) != expected["ranking_identity"]:
            raise ValueError(
                "candidate ranking/config differs from selected calibration"
            )
        if (
            selection.get("controls") != expected["controls"]
            or selection["policy"] != expected["policy"]
        ):
            raise ValueError("calibration selection differs from its pinned evidence")
        control = expected["controls"].get(after["book_stem"])
        if control != {k: after[k] for k in ("labels_sha256", "index_version")}:
            raise ValueError("calibration control identity differs from held-out run")
        before_rows = _unique(before["rows"], "case_id")
        for row in after["rows"]:
            prior = before_rows.get(row["case_id"], {})
            if row.get("retrieval_query") != prior.get("retrieval_query"):
                raise ValueError(
                    "rewritten query input differs from the matched control"
                )
            if after["split"] == "calibration":
                selected_input = expected["query_inputs"][after["book_stem"]].get(
                    row["case_id"]
                )
                if selected_input != {
                    k: row.get(k) for k in ("question", "retrieval_query")
                }:
                    raise ValueError(
                        "calibration query input differs from selected evidence"
                    )
        selected = datetime.fromisoformat(selection["selected_at"])
        started = datetime.fromisoformat(after["started_utc"])
        if selected.tzinfo is None or started.tzinfo is None or selected >= started:
            raise ValueError("calibration selection must precede held-out evaluation")
    from scripts.evaluate_phase4 import latency_metrics, retrieval_metrics

    cases = _cases(before)
    case_ids = set(_unique(cases, "case_id"))
    metrics = []
    for report in (before, after):
        rows = _unique(report["rows"], "case_id")
        if set(rows) - case_ids:
            raise ValueError("unknown case IDs")
        # Missing rows remain empty retrievals; no harder case disappears.
        metrics.append(
            retrieval_metrics(
                cases,
                [
                    rows.get(
                        c["case_id"], {"case_id": c["case_id"], "context_pages": []}
                    )
                    for c in cases
                ],
            )
        )
    baseline, candidate = metrics
    regression = any(
        candidate[k] is None or baseline[k] is None or candidate[k] < baseline[k]
        for k in ("precision", "recall")
    )
    good = sum(_complete(r) for r in after["rows"])
    preserved = (
        before.get("store_unchanged") is True and after.get("store_unchanged") is True
    )
    return {
        "report_status": {
            "status": (
                "pass"
                if before.get("status") == after.get("status") == "complete"
                else "fail"
            )
        },
        "retrieval": {
            "status": "fail" if regression else "pass",
            "before": metrics[0],
            "after": metrics[1],
        },
        "completion": {
            **_rate(good, len(cases)),
            "status": "pass" if good == len(cases) else "fail",
        },
        "store_preservation": {"status": "pass" if preserved else "fail"},
        "latency": {
            label: latency_metrics([r for r in report["rows"] if "total_seconds" in r])
            for label, report in (("before", before), ("after", after))
        },
    }


def diagnose_pages(relevant, *, raw, merged, selected):
    gold = set(relevant)
    return {
        "raw_miss": sorted(gold - set(raw)),
        "merge_exclusion": sorted(gold & set(raw) - set(merged)),
        "context_omission": sorted(gold & set(merged) - set(selected)),
    }


def select_candidate(calibration_reports: list[Path]) -> dict:
    """Select once on both books' calibration evidence, before held-out runs."""
    reports = [json.loads(path.read_text()) for path in calibration_reports]
    if len(reports) != 2 or {r["book_stem"] for r in reports} != {
        "sicp",
        "thinkpython2",
    }:
        raise ValueError("selection requires both books' calibration reports")
    improved = False
    nonregressing = True
    ranking = _ranking_identity(reports[0])
    query_inputs = {}
    for report in reports:
        if _ranking_identity(report) != ranking:
            raise ValueError("calibration ranking/config identities differ")
        if (
            report["status"] != "complete"
            or report["store_unchanged"] is not True
            or report["selection_split"] != "calibration"
        ):
            raise ValueError(
                "selection requires preserved complete calibration evidence"
            )
        metrics = report["metrics"]["calibration"]
        for key in ("precision", "recall"):
            before = metrics["interleave"]["context_micro"][key]
            after = metrics["rrf"]["context_micro"][key]
            if before is None or after is None:
                raise ValueError("unmeasured calibration retrieval")
            nonregressing &= after >= before
            improved |= after > before
        queries = {}
        for row in report["rows"]:
            if row["group"] != "calibration":
                continue
            value = {k: row[k] for k in ("question", "retrieval_query")}
            if row["case_id"] in queries and queries[row["case_id"]] != value:
                raise ValueError("calibration policies used different query inputs")
            queries[row["case_id"]] = value
        if not queries:
            raise ValueError("missing calibration query inputs")
        query_inputs[report["book_stem"]] = queries
    return {
        "split": "calibration",
        "selected_at": datetime.now(UTC).isoformat(),
        "policy": "rrf" if nonregressing and improved else "interleave",
        "input_sha256": {str(p.resolve()): digest(p) for p in calibration_reports},
        "controls": {
            r["book_stem"]: {k: r[k] for k in ("labels_sha256", "index_version")}
            for r in reports
        },
        "promoted": False,
        "ranking_identity": ranking,
        "query_inputs": query_inputs,
        "reason": "Candidate only; original macro regression, held-out quality and attributed human gates remain required",
    }


def diagnose_retrieval(book: str, *, output: Path) -> dict:
    """Replay pinned calibration queries on a copy, comparing exactly two policies.

    The original seven questions also run on the current geometry, explicitly
    as diagnostics: this cannot establish the missing original-geometry gate.
    """
    from scripts.evaluate_phase3_baseline import (
        finish,
        page_metrics,
        reserve,
        tree_identity,
    )

    stream = reserve(output)
    report = {"status": "running", "book_stem": book, "rows": [], "accepted": False}
    before = tree_identity(config.VECTOR_STORE)
    try:
        from scripts import evaluate_phase4 as evaluator
        from scripts.evaluate_phase3_baseline import identities
        from src import hybrid_retrieval, index_store
        from src.page_querying import extract_page_intent, physical_pages_for
        from src.retrieval import search_book
        from src.tools.local import search_local_index
        from src.tools.protocol import ToolContext

        labels_path = evaluator.labels_path(book)
        labels = evaluator.load_labels(labels_path, pages_root=config.REPAIRED_MD)
        control_path = config.PHASE6_QUALITY_CONTROLS / f"{book}-calibration.json"
        control = json.loads(control_path.read_text())
        if control["labels_sha256"] != digest(labels_path):
            raise ValueError("control labels identity changed")
        controls = _unique(control["rows"], "case_id")
        cases = evaluator.cases_for(labels, "calibration")
        report.update(identities(evaluator.CODE))
        report.update(
            labels_sha256=digest(labels_path),
            control_sha256=digest(control_path),
            source_pages=labels["source_pages"],
            selection_split="calibration",
        )
        groups = [("calibration", cases, controls)]
        original_path = config.PHASE2_LABELS_DIR / f"{book}_context_labels.json"
        if original_path.is_file() and book == "sicp":
            original = json.loads(original_path.read_text())
            old = json.loads(config.PHASE2_AFTER_EVALUATION_PATH.read_text())
            if digest(original_path) != old["dataset_sha256"]:
                raise ValueError("original labels identity changed")
            groups.append(
                (
                    "original_questions_current_geometry",
                    original["cases"],
                    _unique(old["rows"], "case_id"),
                )
            )
            report["original_geometry"] = {
                "required_version": old["ingest"]["version_id"],
                "comparable": False,
                "reason": "Historical temporary staged pair is absent; current repaired geometry is a separate diagnostic, not original-regression acceptance",
            }
        retrieve = hybrid_retrieval.retrieve_hybrid
        resolve = index_store.resolve_index
        with tempfile.TemporaryDirectory(prefix="axiom-phase6-diagnose-") as temp:
            store = Path(temp) / "store"
            shutil.copytree(config.VECTOR_STORE, store)
            with patch.object(
                index_store, "resolve_index", lambda stem: resolve(stem, root=store)
            ):
                handle = index_store.resolve_index(book)
                if (
                    handle.version_id != control["index_version"]["version_id"]
                    or handle.embedding_signature
                    != control["index_version"]["embedding_signature"]
                ):
                    raise ValueError("control index identity changed")
                report["index_version"] = control["index_version"]
                for group, group_cases, replay in groups:
                    for case in group_cases:
                        if case["case_id"] not in replay:
                            raise ValueError("missing frozen retrieval query")
                        query = replay[case["case_id"]]["retrieval_query"]
                        for policy in ("interleave", "rrf"):
                            attempts = []

                            def traced(*args, attempts=attempts, **kwargs):
                                diagnostic = {}
                                result = retrieve(
                                    *args, **kwargs, diagnostics=diagnostic
                                )
                                attempts.append(diagnostic)
                                return result

                            found = None

                            def search(*args, case=case, query=query, **kwargs):
                                nonlocal found
                                found = search_book(
                                    case["question"],
                                    book_stem=book,
                                    distance=config.DISTANCE_THRESHOLD,
                                    retrieval_query=query,
                                    page_range=kwargs.get("page_range"),
                                )
                                return found

                            pages = extract_page_intent(case["question"])
                            arguments = {"query": query}
                            if pages:
                                arguments.update(
                                    page_start=min(pages), page_end=max(pages)
                                )
                            start = time.monotonic()
                            with (
                                patch.object(config, "HYBRID_MERGE_POLICY", policy),
                                patch.object(
                                    hybrid_retrieval, "retrieve_hybrid", traced
                                ),
                            ):
                                result = asyncio.run(
                                    search_local_index(
                                        arguments,
                                        ToolContext(
                                            "phase6",
                                            book,
                                            config.OLLAMA_MODEL,
                                            False,
                                            "local",
                                        ),
                                        call_id="diagnose",
                                        search=search,
                                    )
                                )
                            merged = list(found.candidates)
                            selected = [asdict(s) for s in result.sources]
                            selected_pages = sorted(
                                {p for s in result.sources for p in s.physical_pages}
                            )
                            raw = [
                                h
                                for a in attempts
                                for key in ("raw_dense", "bm25")
                                for h in a[key]
                            ]
                            raw_pages = {
                                p
                                for h in raw
                                for p in physical_pages_for(h["metadata"])
                            }
                            merged_pages = {
                                p
                                for c in merged
                                for p in physical_pages_for(c.metadata)
                            }
                            report["rows"].append(
                                {
                                    "case_id": case["case_id"],
                                    "group": group,
                                    "policy": policy,
                                    "question": case["question"],
                                    "retrieval_query": query,
                                    "query_replayed_from": str(
                                        control_path
                                        if group == "calibration"
                                        else config.PHASE2_AFTER_EVALUATION_PATH
                                    ),
                                    "attempts": attempts,
                                    "page_filter": found.page_filter,
                                    "page_fallback": found.page_fallback,
                                    "selected": selected,
                                    "candidate_pages": sorted(merged_pages),
                                    "physical_pages": selected_pages,
                                    "omitted_for_tool_budget_ids": [
                                        c.chunk_id
                                        for c in merged[: config.TOP_K_RESULTS]
                                        if c.chunk_id
                                        not in {s.source_id for s in result.sources}
                                    ],
                                    "omitted_for_token_budget_ids": None,
                                    "token_budget_note": "Agent retains whole tool passages; full-prompt fit is measured by generation traces, not inferred here",
                                    "losses": diagnose_pages(
                                        case["relevant_pages"],
                                        raw=raw_pages,
                                        merged=merged_pages,
                                        selected=selected_pages,
                                    ),
                                    "seconds": time.monotonic() - start,
                                    "error": (
                                        result.data.get("error")
                                        if result.status != "ok"
                                        else None
                                    ),
                                }
                            )
            report["metrics"] = {}
            for group, group_cases, _ in groups:
                report["metrics"][group] = {}
                for policy in ("interleave", "rrf"):
                    rows = [
                        r
                        for r in report["rows"]
                        if r["group"] == group and r["policy"] == policy
                    ]
                    report["metrics"][group][policy] = {
                        "context_micro": page_metrics(group_cases, rows),
                        "candidate_micro": page_metrics(
                            group_cases,
                            [
                                {**r, "physical_pages": r["candidate_pages"]}
                                for r in rows
                            ],
                        ),
                    }
            report["status"] = "complete"
    except Exception as exc:  # noqa: BLE001 - retain a failed diagnostic as evidence
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        report["store_unchanged"] = tree_identity(config.VECTOR_STORE) == before
        if not report["store_unchanged"]:
            report.update(status="failed", error="active store changed")
        finish(stream, report)
    return report


def compare_original_regression(*, before: Path, after: Path) -> dict:
    """The original seven-answerable-case macro gate, never repaired-store micro."""
    from scripts.evaluate_phase2_baseline import (
        load_labels,
        retrieval_metrics,
        verify_sources,
    )

    baseline = json.loads(before.read_text())
    candidate = json.loads(after.read_text())
    historical = json.loads(config.PHASE2_AFTER_EVALUATION_PATH.read_text())
    labels_path = config.PHASE2_LABELS_DIR / "sicp_context_labels.json"
    labels = load_labels(labels_path)
    verify_sources(labels, config.PARSED_MD)
    if any(
        r.get("status") != "complete" or r.get("book_stem") != "sicp"
        for r in (baseline, candidate)
    ):
        raise ValueError("original comparison requires complete SICP reports")
    if (
        digest(before) != digest(config.PHASE2_EVALUATION_PATH)
        or candidate.get("baseline_sha256") != digest(before)
        or any(
            r.get("dataset_sha256") != digest(labels_path)
            for r in (baseline, candidate)
        )
        or digest(config.PARSED_MD / "sicp/page_manifest.json")
        != baseline["source_manifest_sha256"]
    ):
        raise ValueError("original baseline/labels/source identity mismatch")
    keys = ("model", "digest", "ollama_version", "embedding", "embedding_revision")
    if any(
        not baseline["runtime"].get(k)
        or baseline["runtime"][k] != candidate["runtime"].get(k)
        for k in keys
    ):
        raise ValueError("original model/runtime identity mismatch")
    if any(
        candidate.get("ingest", {}).get(k) != historical["ingest"][k]
        for k in ("version_id", "embedding_config")
    ):
        raise ValueError(
            "original geometry differs; a repaired/migrated index cannot substitute"
        )
    metrics = [
        retrieval_metrics(labels["cases"], r["rows"]) for r in (baseline, candidate)
    ]
    if any(m["answerable_case_denominator"] != 7 for m in metrics):
        raise ValueError("original macro gate needs seven answerable cases")
    passed = candidate.get("preserved") is True and all(
        metrics[1][key] >= metrics[0][key]
        for key in ("macro_page_precision", "macro_page_recall")
    )
    return {
        "status": "pass" if passed else "fail",
        "before": metrics[0],
        "after": metrics[1],
        "input_sha256": {
            str(p): digest(p)
            for p in (before, after, labels_path, config.PHASE2_AFTER_EVALUATION_PATH)
        },
        "geometry_version": candidate["ingest"]["version_id"],
    }


def evaluate_quality(
    *,
    before: Path,
    after: Path,
    review_before: Path,
    review_after: Path,
    output: Path,
    original_before: Path | None = None,
    original_after: Path | None = None,
) -> dict:
    # Reserve before reading inputs; refuse overwrite even for failed evaluations.
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        result = {
            "accepted": False,
            "original_regression": {
                "status": "pending",
                "reason": "Requires its own original seven-case macro comparison and geometry disposition",
            },
        }
        try:
            if original_before is not None or original_after is not None:
                if original_before is None or original_after is None:
                    raise ValueError("both original comparison reports are required")
                result["original_regression"] = compare_original_regression(
                    before=original_before, after=original_after
                )
            reports = [json.loads(p.read_text()) for p in (before, after)]
            result.update(compare_quality(*reports))
            result["scope"] = {
                "book": reports[1]["book_stem"],
                "split": reports[1]["split"],
                "note": "One matched comparison; release requires all books/splits and separate human/runtime gates",
            }
            metrics = []
            reasons = []
            for report, path, source in zip(
                reports, (review_before, review_after), (before, after)
            ):
                try:
                    review = json.loads(path.read_text())
                    metrics.append(
                        validate_review(report, review, report_sha256=digest(source))
                    )
                except (OSError, ValueError) as exc:
                    reasons.append(str(exc))
            result["human_review"] = {
                "status": "pending",
                "reasons": reasons,
                "metrics": metrics,
            }
            if len(metrics) == 2:
                pending = any(
                    not m["inventory_complete"]
                    or any(
                        m[k]
                        for k in (
                            "unreviewed_factual_claims",
                            "unreviewed_support_claims",
                            "unreviewed_citation_pairs",
                            "unreviewed_expected_claims",
                        )
                    )
                    for m in metrics
                )
                worse = False
                for key in (
                    "factual_error",
                    "unsupported_claim",
                    "citation_support",
                    "expected_claim_coverage",
                ):
                    b, a = (m[key]["rate"] for m in metrics)
                    pending |= a is None or b is None
                    if a is not None and b is not None:
                        worse |= (
                            a > b
                            if key in ("factual_error", "unsupported_claim")
                            else a < b
                        )
                result["human_review"]["status"] = (
                    "pending" if pending else ("fail" if worse else "pass")
                )
                premise = metrics[1]["premise_rejection"]
                result["premise_rejection"] = {
                    **premise,
                    "status": (
                        "pass"
                        if premise["numerator"] == premise["denominator"]
                        else "fail"
                    ),
                }
            result["input_sha256"] = {
                str(p): digest(p)
                for p in (before, after, review_before, review_after)
                if p.is_file()
            }
            gates = (
                "report_status",
                "retrieval",
                "completion",
                "store_preservation",
                "human_review",
                "premise_rejection",
                "original_regression",
            )
            result["accepted"] = all(
                result.get(k, {}).get("status") == "pass" for k in gates
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result["comparison"] = {"status": "fail", "reason": str(exc)}
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sheet = sub.add_parser("review-sheet")
    sheet.add_argument("--report", type=Path, required=True)
    sheet.add_argument("--output", type=Path, required=True)
    compare = sub.add_parser("compare")
    for name in ("before", "after", "review-before", "review-after", "output"):
        compare.add_argument(f"--{name}", type=Path, required=True)
    compare.add_argument("--original-before", type=Path)
    compare.add_argument("--original-after", type=Path)
    diagnostic = sub.add_parser("diagnose")
    diagnostic.add_argument("book")
    diagnostic.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "diagnose":
        result = diagnose_retrieval(args.book, output=args.output)
        print(
            json.dumps(
                {
                    k: result.get(k)
                    for k in ("status", "metrics", "error", "store_unchanged")
                },
                indent=2,
            )
        )
        return 0 if result["status"] == "complete" else 1
    if args.command == "review-sheet":
        report = json.loads(args.report.read_text())
        write_evidence(
            args.output, blank_review(report, report_sha256=digest(args.report))
        )
        print(f"Blank human review: {args.output}; acceptance pending")
        return 0
    result = evaluate_quality(
        before=args.before,
        after=args.after,
        review_before=args.review_before,
        review_after=args.review_after,
        output=args.output,
        original_before=args.original_before,
        original_after=args.original_after,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
