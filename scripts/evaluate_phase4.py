"""Phase 4 per-book evaluation harness over frozen, hash-pinned label splits.

Labels live in tests/fixtures/phase4/<book>_labels.json. Every label page is a
one-based physical PDF page whose repaired text is pinned by sha256, so a page
that changed after labeling is refused instead of silently shifting ground truth.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import shutil
import sys
import tempfile
import time
from contextlib import ExitStack, asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_phase3_baseline import (
    digest,
    exchanges_for,
    finish,
    identities,
    page_metrics,
    reserve,
    tree_identity,
)
from scripts.validate_phase3_agent import _DISCLAIMER, premise_lints, sampling_record
from src import config
from src.agent import AgentRequest, run_agent
from src.index_metadata import validate_book
from src.page_querying import ABSTENTION
from src.query_rewriting import rewrite_followup

KINDS = frozenset(
    {
        "direct",
        "followup",
        "paraphrase",
        "contradiction",
        "unanswerable",
        "page",
        "broad",
    }
)
SPLITS = ("calibration", "test")
PIPELINES: tuple[str, ...] = ("baseline", "gated")
CODE = (
    "src/agent.py",
    "src/entailment.py",
    "src/citation_check.py",
    "src/config.py",
    "src/retrieval.py",
    "src/hybrid_retrieval.py",
    "src/query_rewriting.py",
    "src/page_querying.py",
    "src/tools/protocol.py",
    "src/tools/registry.py",
    "src/tools/local.py",
    "scripts/evaluate_phase4.py",
    "scripts/evaluate_phase3_baseline.py",
    "scripts/validate_phase3_agent.py",
)


def labels_path(book: str) -> Path:
    validate_book(book)
    return config.PHASE4_LABELS_DIR / f"{book}_labels.json"


def _claim_pages(case: dict) -> list[int]:
    return [p for claim in case["expected_claims"] for p in claim["source_pages"]]


def _check_case(case: dict, *, phase2: bool = False) -> None:
    """Phase 2 cases keep their original kind, so only their structure is checked."""
    cid = case["case_id"]
    if not phase2 and case["kind"] not in KINDS:
        raise ValueError(f"case {cid}: unknown kind {case['kind']!r}")
    if (case["kind"] == "unanswerable") == case["answerable"]:
        raise ValueError(f"case {cid}: only unanswerable cases may be unanswerable")
    if case["kind"] == "contradiction" and case["false_premise"] is not True:
        raise ValueError(f"case {cid}: contradiction needs false_premise true")
    if case["answerable"]:
        if not case["relevant_pages"] or not case["expected_claims"]:
            raise ValueError(
                f"case {cid}: answerable case needs relevant pages and claims"
            )
        if any(not c["source_pages"] for c in case["expected_claims"]) or not set(
            _claim_pages(case)
        ) <= set(case["relevant_pages"]):
            raise ValueError(f"case {cid}: claim pages must be nonempty relevant pages")
    # Phase 2 unanswerable cases carry one page-less behavioural claim.
    elif (
        case["relevant_pages"]
        or _claim_pages(case)
        or (case["expected_claims"] and not phase2)
    ):
        raise ValueError(f"case {cid}: unanswerable case has pages or claims")


def load_labels(path: Path, *, pages_root: Path = config.REPAIRED_MD) -> dict:
    """Validated labels with hash-pinned Phase 2 cases merged in and a split per case."""
    labels = json.loads(Path(path).read_text())
    if labels.get("schema_version") != 1:
        raise ValueError("unsupported label schema_version")
    book = labels["book_stem"]
    validate_book(book)
    for case in labels["cases"]:
        _check_case(case)
    cases = []
    reference = labels.get("phase2_reference")
    if reference:
        phase2_path = config.ROOT / reference["path"]
        if digest(phase2_path) != reference["sha256"]:
            raise ValueError(
                "Phase 2 labels changed since the Phase 4 labels were frozen"
            )
        phase2 = {c["case_id"]: c for c in json.loads(phase2_path.read_text())["cases"]}
        missing = sorted(set(reference["case_ids"]) - set(phase2))
        if missing:
            raise ValueError(f"Phase 2 reference names unknown cases {missing}")
        # Hash-pinned Phase 2 cases keep their original kind and claims.
        cases = [
            {**phase2[cid], "false_premise": False} for cid in reference["case_ids"]
        ]
        for case in cases:
            _check_case(case, phase2=True)
    cases += labels["cases"]
    ids = [c["case_id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case_id")
    splits = labels["splits"]
    if sorted(splits) != sorted(SPLITS):
        raise ValueError(f"splits must be exactly {list(SPLITS)}")
    calibration, test = set(splits["calibration"]), set(splits["test"])
    if calibration & test:
        raise ValueError(f"cases in both splits: {sorted(calibration & test)}")
    if calibration | test != set(ids):
        raise ValueError(
            f"cases not in a split: {sorted(set(ids) - calibration - test)}; "
            f"unknown split ids: {sorted((calibration | test) - set(ids))}"
        )
    stray = sorted(
        c["case_id"] for c in cases if c["case_id"] in test and c["kind"] not in KINDS
    )
    if stray:
        raise ValueError(f"test split holds non-Phase 4 kinds: {stray}")
    hashes = labels["source_pages"]
    for page in sorted(
        {p for c in cases for p in c["relevant_pages"] + _claim_pages(c)}
    ):
        if str(page) not in hashes:
            raise ValueError(f"label page {page} has no hash")
        source = Path(pages_root) / book / f"page_{page - 1:04d}.md"
        if not source.is_file() or digest(source) != hashes[str(page)]:
            raise ValueError(f"source page {page} changed")
    for case in cases:
        case["split"] = "calibration" if case["case_id"] in calibration else "test"
    return {**labels, "cases": cases}


def cases_for(labels: dict, split: str) -> list[dict]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    return [c for c in labels["cases"] if c["split"] == split]


def _rate(numerator: int, denominator: int) -> float | None:
    """A zero denominator is unmeasured, never a measured 0.0."""
    return numerator / denominator if denominator else None


def retrieval_metrics(cases: list[dict], rows: list[dict]) -> dict:
    rows = [{**r, "physical_pages": r["context_pages"]} for r in rows]
    return {**page_metrics(cases, rows), "cutoff": "selected answer context"}


def abstained(answer: str) -> bool:
    return answer.lstrip().startswith(ABSTENTION) or bool(_DISCLAIMER.search(answer))


def answer_metrics(cases: list[dict], rows: list[dict]) -> dict:
    """Missing, failed or incomplete rows stay in every denominator; never a success."""
    by_id = {r["case_id"]: r for r in rows}

    def answer(case: dict) -> str | None:
        row = by_id.get(case["case_id"])
        if row is None or row["error"] is not None or not row["complete"]:
            return None
        return row["answer"] or ""

    answerable = [c for c in cases if c["answerable"]]
    unanswerable = [c for c in cases if not c["answerable"]]
    premise = [c for c in cases if c["false_premise"]]
    answered = sum(
        answer(c) is not None and not abstained(answer(c)) for c in answerable
    )
    abstentions = sum(
        answer(c) is not None and abstained(answer(c)) for c in unanswerable
    )
    rejected = sum(
        answer(c) is not None and premise_lints(answer(c), c["question"]) == []
        for c in premise
    )
    return {
        "answerable": len(answerable),
        "answered": answered,
        "coverage": _rate(answered, len(answerable)),
        "unanswerable": len(unanswerable),
        "abstained": abstentions,
        "abstention_rate": _rate(abstentions, len(unanswerable)),
        "false_premise": len(premise),
        "rejected": rejected,
        "rejection_rate": _rate(rejected, len(premise)),
        "heuristic": ["abstained", "rejected"],
    }


def claim_metrics(rows: list[dict]) -> dict:
    """Scorer votes only; factual error/accuracy requires independent human review."""
    judged = [r for r in rows if "claims" in r]
    claims = [c for r in judged for c in r["claims"]]
    citations = [c for r in judged for c in r.get("citations", [])]
    flagged = sum(not c["supported"] for c in claims)
    supported = sum(c["supported"] for c in citations)
    return {
        "judged_rows": len(judged),
        "unjudged_rows": len(rows) - len(judged),
        "checked_claims": len(claims),
        "flagged_claims": flagged,
        "scorer_flag_rate": _rate(flagged, len(claims)),
        "checked_citations": len(citations),
        "supported_citations": supported,
        "scorer_citation_support_rate": _rate(supported, len(citations)),
    }


def expected_claim_recall(cases: list[dict], rows: list[dict]) -> dict:
    """Expected claims whose label pages overlap a supporting source's pages."""
    by_id = {r["case_id"]: r for r in rows}
    expected = cited = 0
    for case in cases:
        row = by_id.get(case["case_id"], {})
        # An unjudged row has no supporting sources; retrieved is not supported.
        keep = set(row.get("supporting_source_ids", ()))
        pages = {
            p
            for s in row.get("sources", [])
            if s["source_id"] in keep
            for p in s["pages"]
        }
        # Behavioural claims without pages (e.g. "say the book does not cover it")
        # cannot be cited, so they are outside this denominator.
        for claim in case["expected_claims"]:
            if claim["source_pages"]:
                expected += 1
                cited += bool(set(claim["source_pages"]) & pages)
    return {
        "expected_claims": expected,
        "cited_expected": cited,
        "supported_citation_recall": _rate(cited, expected),
    }


def _check_index_source(entry: dict | None, book: str) -> None:
    """Labels pin repaired page text, so only an index built from that text is scored."""
    report = config.REPAIRED_MD / book / "repair_report.json"
    if entry is None or entry.get("source") != "repaired":
        raise ValueError(f"Active index for '{book}' was not built from repaired pages")
    if not report.is_file() or entry.get("repair_report_sha256") != digest(report):
        raise ValueError(
            f"Active index for '{book}' predates the current repair report"
        )


def _book_pages(sources) -> list[int]:
    return sorted({p for s in sources if s.kind == "book" for p in s.physical_pages})


async def run_case(case: dict, book: str, *, client, sandbox, counter) -> dict:
    """One case through the unchanged agent path; any failure is kept in the row."""
    history = list(case["history"])
    row = {
        "case_id": case["case_id"],
        "split": case["split"],
        "kind": case["kind"],
        "question": case["question"],
        "retrieval_query": None,
        "context_pages": [],
        "candidate_ids": [],
        "answer": "",
        "complete": False,
        "abstained": False,
        "sources": [],
        "stage_seconds": {},
        "model_requests": None,
        "tool_calls": None,
        "error": None,
    }
    from src.retrieval import search_book

    def search(*args, **kwargs):
        found = search_book(*args, **kwargs)
        row.update(
            candidate_ids=[c.chunk_id for c in found.candidates],
            gated=found.gated,
            dropped=found.dropped,
            scoring_seconds=found.scoring_seconds,
        )
        return found

    started = time.monotonic()
    try:
        try:
            # The frozen history goes through rewriting exactly as in Phase 3.
            rewritten = await rewrite_followup(
                case["question"],
                exchanges_for(history, book),
                book,
                model=config.OLLAMA_MODEL,
                client=client,
            )
            row["retrieval_query"] = rewritten.retrieval_query
            request = AgentRequest(
                f"phase4-{case['case_id']}",
                case["question"],
                book,
                config.OLLAMA_MODEL,
                history=tuple(history),
                web_enabled=False,
                retrieval_query=rewritten.retrieval_query,
                today=datetime.now().astimezone().date().isoformat(),
            )
            async for event in run_agent(
                request,
                client=client,
                sandbox=sandbox,
                counter=counter,
                web=None,
                search=search,
            ):
                if event.kind == "step":
                    row["stage_seconds"][event.data["stage"]] = (
                        time.monotonic() - started
                    )
                elif event.kind == "sources":
                    if not row["candidate_ids"]:
                        row["candidate_ids"] = [
                            s.source_id for s in event.data["sources"]
                        ]
                elif event.kind == "result":
                    data = event.data
                    row.update(
                        answer=data["answer"],
                        complete=data["complete"],
                        done_reason=data.get("done_reason"),
                        trace=data.get("trace", []),
                        abstained=abstained(data["answer"]),
                        context_pages=_book_pages(data["sources"]),
                        sources=[
                            {
                                "source_id": s.source_id,
                                "kind": s.kind,
                                "pages": list(s.physical_pages),
                                "text": s.text,
                            }
                            for s in data["sources"]
                        ],
                        model_requests=data["model_requests"],
                        tool_calls=data["tool_calls"],
                    )
                    if data.get("citations"):
                        payload = data["citations"]
                        row.update(
                            citation_status=payload["status"],
                            check_seconds=payload.get("check_seconds"),
                        )
                        for key in (
                            "claims",
                            "citations",
                            "supporting_source_ids",
                            "unchecked",
                            "reason",
                        ):
                            if key in payload:
                                row[key] = payload[key]
                elif event.kind == "error":
                    kind = event.data.get("kind", "error")
                    row["error"] = f"Agent{kind.title()}: {event.data['message']}"
        finally:
            await asyncio.to_thread(client.unload_and_wait)
    except Exception as exc:  # noqa: BLE001 - a failed case is a row, not a stop
        row["error"] = row["error"] or f"{type(exc).__name__}: {exc}"
    if row["error"] is not None:
        row["complete"] = False
    row["total_seconds"] = time.monotonic() - started
    return row


def _percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile; None when nothing was timed."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(math.ceil(q * len(ordered)) - 1, 0)]


def latency_metrics(rows: list[dict]) -> dict:
    seconds = [r["total_seconds"] for r in rows]
    return {
        "n": len(seconds),
        "p50": _percentile(seconds, 0.5),
        "p95": _percentile(seconds, 0.95),
        "max": max(seconds, default=None),
    }


@asynccontextmanager
async def _services(report: dict):
    """The Ollama client, chat counter and sandbox, wired as validate_phase3_agent does."""
    from src.chat_tokens import load_chat_counter
    from src.llm import OllamaClient
    from src.sandbox.client import SandboxClient

    client = OllamaClient()
    counter = await asyncio.to_thread(load_chat_counter, config.OLLAMA_MODEL)
    sandbox = SandboxClient(request_prefix="phase4-eval")
    capability = await sandbox.start()
    report["sandbox"] = {"available": capability.available, "reason": capability.reason}
    try:
        yield {"client": client, "sandbox": sandbox, "counter": counter}
    finally:
        await sandbox.aclose()


async def _evaluate(report: dict, cases: list[dict], book: str, store: Path) -> None:
    from src import index_store

    resolve = index_store.resolve_index
    with ExitStack() as stack:
        if report["pipeline"] == "baseline":
            stack.enter_context(patch.object(config, "ENTAILMENT_GATE_ENABLED", False))
            stack.enter_context(patch.object(config, "CITATION_CHECK_ENABLED", False))
        stack.enter_context(
            patch.object(
                index_store,
                "resolve_index",
                lambda stem, **_: resolve(stem, root=store),
            )
        )
        handle = index_store.resolve_index(book)
        entry = index_store.read_manifest(store).get(book)
        report["index_version"] = {
            "version_id": handle.version_id,
            "legacy": handle.legacy,
            "embedding_signature": handle.embedding_signature,
            "source": entry and entry.get("source"),
            "repair_report_sha256": entry and entry.get("repair_report_sha256"),
        }
        _check_index_source(None if handle.legacy else entry, book)
        async with _services(report) as services:
            for case in cases:
                row = await run_case(case, book, **services)
                report["rows"].append(row)
                print(f"{case['case_id']}: {row['error'] or 'ok'}", flush=True)


def evaluate(
    book: str,
    *,
    split: str,
    pipeline: str,
    output: Path,
    case_ids: list[str] | None = None,
    merge_policy: str | None = None,
    selection: dict | None = None,
) -> dict:
    if pipeline not in PIPELINES:
        raise ValueError(f"unknown pipeline {pipeline!r}")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    if merge_policy is not None and merge_policy not in {"interleave", "rrf"}:
        raise ValueError("unknown merge policy")
    path = labels_path(book)
    stream = reserve(output)
    report = {
        "schema_version": 2,
        "kind": "phase4_eval",
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
        "book_stem": book,
        "split": split,
        "pipeline": pipeline,
        "labels_path": str(path),
        "labels_sha256": None,
        "labels_review_status": None,
        "sampling": None,
        "index_version": None,
        "rows": [],
        "failures": [],
        "metrics": None,
        "merge_policy": merge_policy or config.HYBRID_MERGE_POLICY,
        "selection": selection,
    }
    store_before = tree_identity(config.VECTOR_STORE)
    try:
        report["labels_sha256"] = digest(path)
        labels = load_labels(path, pages_root=config.REPAIRED_MD)
        if labels["book_stem"] != book:
            raise ValueError("Labels are frozen for a different book")
        report["labels_review_status"] = labels["review"]["status"]
        cases = cases_for(labels, split)
        if case_ids:
            unknown = sorted(set(case_ids) - {c["case_id"] for c in cases})
            if unknown:
                raise ValueError(f"cases not in the {split} split: {unknown}")
            cases = [c for c in cases if c["case_id"] in set(case_ids)]
            report["case_subset"] = case_ids
        report["sampling"] = sampling_record()
        report.update(identities(CODE))
        report["trust_config"] = {
            "gate_enabled": (
                config.ENTAILMENT_GATE_ENABLED if pipeline == "gated" else False
            ),
            "check_enabled": (
                config.CITATION_CHECK_ENABLED if pipeline == "gated" else False
            ),
            **judge_identity(),
        }
        with tempfile.TemporaryDirectory(prefix="axiom-phase4-eval-") as temp:
            store = Path(temp) / "vector_store"
            shutil.copytree(config.VECTOR_STORE, store)
            with patch.object(config, "HYBRID_MERGE_POLICY", report["merge_policy"]):
                asyncio.run(_evaluate(report, cases, book, store))
        rows = report["rows"]
        report["failures"] = [
            {"case_id": r["case_id"], "error": r["error"]} for r in rows if r["error"]
        ]
        report["metrics"] = {
            "retrieval": retrieval_metrics(cases, rows),
            "answer": answer_metrics(cases, rows),
            "latency": latency_metrics(rows),
            **trust_metrics(rows),
            "claims": claim_metrics(rows),
            "expected_claim_recall": expected_claim_recall(cases, rows),
        }
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        if not isinstance(exc, Exception):
            raise
    finally:
        report["store_unchanged"] = tree_identity(config.VECTOR_STORE) == store_before
        if not report["store_unchanged"] and report["status"] == "complete":
            report.update(
                status="failed", error="active vector store changed during evaluation"
            )
        report["finished_utc"] = datetime.now(UTC).isoformat()
        finish(stream, report)
    return report


def trust_metrics(rows):
    def latency(key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return {
            "n": len(values),
            "p50": _percentile(values, 0.5),
            "p95": _percentile(values, 0.95),
        }

    timed_out = sum(r.get("citation_status") == "unchecked" for r in rows)
    return {
        "scoring_latency": latency("scoring_seconds"),
        "check_latency": latency("check_seconds"),
        "check_timeouts": {
            "k": timed_out,
            "n": len(rows),
            "rate": _rate(timed_out, len(rows)),
        },
    }


def judge_identity():
    return {
        "model": config.ENTAILMENT_MODEL,
        "revision": config.ENTAILMENT_REVISION,
        "formulation": config.ENTAILMENT_FORMULATION,
        "claim_threshold": config.ENTAILMENT_CLAIM_THRESHOLD,
    }


def rescore(report: Path, *, output: Path) -> dict:
    from src.citation_check import check_answer, report_payload
    from src.entailment import get_scorer
    from src.tools.protocol import EvidenceRef

    if output.exists():
        raise FileExistsError(output)
    saved = json.loads(Path(report).read_text())
    if (
        saved.get("status") != "complete"
        or not saved.get("rows")
        or any(not r.get("complete") or r.get("error") for r in saved["rows"])
    ):
        raise ValueError("Cannot rescore incomplete report")
    path = Path(saved["labels_path"])
    if digest(path) != saved["labels_sha256"]:
        raise ValueError("Labels changed since saved report")
    labels = load_labels(path, pages_root=config.REPAIRED_MD)
    cases = cases_for(labels, saved["split"])
    if saved.get("case_subset"):
        cases = [c for c in cases if c["case_id"] in saved["case_subset"]]
    if {c["case_id"] for c in cases} != {r["case_id"] for r in saved["rows"]}:
        raise ValueError("Cannot rescore incomplete case coverage")
    stream = reserve(output)
    try:
        scorer = get_scorer()
        for row in saved["rows"]:
            sources = [
                EvidenceRef(
                    s["source_id"],
                    s["kind"],
                    s["text"],
                    physical_pages=tuple(s["pages"]),
                )
                for s in row["sources"]
            ]
            started = time.monotonic()
            payload = report_payload(check_answer(row["answer"], sources, scorer.score))
            row.update(
                {
                    k: payload[k]
                    for k in (
                        "claims",
                        "citations",
                        "supporting_source_ids",
                        "unchecked",
                    )
                }
            )
            row.update(
                citation_status=payload["status"],
                check_seconds=time.monotonic() - started,
            )
        saved.update(
            schema_version=2,
            judge=judge_identity(),
            judge_note="same model as the gated pipeline; not an independent judge",
            rescored_from=str(report),
            rescored_from_sha256=digest(report),
            rescored_utc=datetime.now(UTC).isoformat(),
        )
        saved["metrics"].update(
            claims=claim_metrics(saved["rows"]),
            expected_claim_recall=expected_claim_recall(cases, saved["rows"]),
            **trust_metrics(saved["rows"]),
        )
    except BaseException as exc:
        saved.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        finish(stream, saved)
    return saved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("book", nargs="?")
    parser.add_argument("--rescore", type=Path)
    parser.add_argument("--split", choices=SPLITS)
    parser.add_argument("--pipeline", choices=PIPELINES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", help="only these case ids")
    args = parser.parse_args(argv)
    if args.rescore:
        report = rescore(args.rescore, output=args.output)
        print(json.dumps({"status": report["status"]}))
        return 0
    if not args.book or not args.split or not args.pipeline:
        parser.error(
            "book, --split and --pipeline are required unless --rescore is used"
        )
    report = evaluate(
        args.book,
        split=args.split,
        pipeline=args.pipeline,
        output=args.output,
        case_ids=args.cases,
    )
    print(json.dumps({k: report.get(k) for k in ("status", "store_unchanged")}))
    return 0 if report["status"] == "complete" and report["store_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
