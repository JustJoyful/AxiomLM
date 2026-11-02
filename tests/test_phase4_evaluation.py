import copy
import hashlib
import json
from contextlib import asynccontextmanager

import pytest

from scripts import evaluate_phase4 as ev


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _labels(tmp_path):
    """A valid two-page label set whose pages live under tmp_path/pages/book."""
    folder = tmp_path / "pages" / "book"
    folder.mkdir(parents=True, exist_ok=True)
    pages = {1: "first page\n", 2: "second page\n"}
    for page, text in pages.items():
        (folder / f"page_{page - 1:04d}.md").write_text(text)
    return {
        "schema_version": 1,
        "book_stem": "book",
        "frozen_utc": "2026-10-28T00:00:00+00:00",
        "review": {"status": "unreviewed", "drafted_by": "test", "method": "test"},
        "source_pages": {str(p): _sha(t) for p, t in pages.items()},
        "splits": {"calibration": ["a"], "test": ["b", "c", "d"]},
        "cases": [
            {
                "case_id": "a",
                "kind": "direct",
                "question": "What is on page one?",
                "history": [],
                "answerable": True,
                "false_premise": False,
                "relevant_pages": [1],
                "expected_claims": [{"claim": "First.", "source_pages": [1]}],
            },
            {
                "case_id": "b",
                "kind": "paraphrase",
                "question": "Tell me page two.",
                "history": [],
                "answerable": True,
                "false_premise": False,
                "relevant_pages": [1, 2],
                "expected_claims": [{"claim": "Second.", "source_pages": [2]}],
            },
            {
                "case_id": "c",
                "kind": "contradiction",
                "question": "Why does page two say the opposite?",
                "history": [],
                "answerable": True,
                "false_premise": True,
                "relevant_pages": [2],
                "expected_claims": [{"claim": "It does not.", "source_pages": [2]}],
            },
            {
                "case_id": "d",
                "kind": "unanswerable",
                "question": "What was the weather?",
                "history": [],
                "answerable": False,
                "false_premise": False,
                "relevant_pages": [],
                "expected_claims": [],
            },
        ],
    }


def _load(tmp_path, labels):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(labels))
    return ev.load_labels(path, pages_root=tmp_path / "pages")


def _case(labels, case_id):
    return next(c for c in labels["cases"] if c["case_id"] == case_id)


def test_valid_labels_load_and_split(tmp_path):
    labels = _load(tmp_path, _labels(tmp_path))
    assert [c["case_id"] for c in ev.cases_for(labels, "test")] == ["b", "c", "d"]
    assert [c["case_id"] for c in ev.cases_for(labels, "calibration")] == ["a"]
    assert {c["split"] for c in ev.cases_for(labels, "test")} == {"test"}


def test_split_overlap_is_rejected(tmp_path):
    labels = _labels(tmp_path)
    labels["splits"]["calibration"].append("b")
    with pytest.raises(ValueError, match="both splits"):
        _load(tmp_path, labels)


def test_unsplit_case_is_rejected(tmp_path):
    labels = _labels(tmp_path)
    labels["splits"]["test"].remove("d")
    with pytest.raises(ValueError, match="not in a split"):
        _load(tmp_path, labels)


def test_unknown_split_id_is_named_alone(tmp_path):
    labels = _labels(tmp_path)
    labels["splits"]["test"].append("typo")
    with pytest.raises(ValueError, match=r"unknown split ids: \['typo'\]$"):
        _load(tmp_path, labels)


def test_answerable_unanswerable_kind_is_rejected(tmp_path):
    labels = _labels(tmp_path)
    case = _case(labels, "d")
    case.update(answerable=True, relevant_pages=[1])
    case["expected_claims"] = [{"claim": "x", "source_pages": [1]}]
    with pytest.raises(ValueError, match="only unanswerable cases"):
        _load(tmp_path, labels)


def test_unknown_kind_is_rejected(tmp_path):
    labels = _labels(tmp_path)
    _case(labels, "a")["kind"] = "trivia"
    with pytest.raises(ValueError, match="unknown kind"):
        _load(tmp_path, labels)


def test_answerable_case_needs_pages_and_claims(tmp_path):
    for field in ("relevant_pages", "expected_claims"):
        labels = _labels(tmp_path)
        _case(labels, "a")[field] = []
        with pytest.raises(ValueError, match="needs relevant pages and claims"):
            _load(tmp_path, labels)


def test_claim_pages_must_be_relevant_pages(tmp_path):
    labels = _labels(tmp_path)
    _case(labels, "a")["expected_claims"][0]["source_pages"] = [2]
    with pytest.raises(ValueError, match="claim pages"):
        _load(tmp_path, labels)


def test_unanswerable_case_has_no_pages_or_claims(tmp_path):
    labels = _labels(tmp_path)
    _case(labels, "d")["relevant_pages"] = [1]
    with pytest.raises(ValueError, match="unanswerable"):
        _load(tmp_path, labels)
    labels = _labels(tmp_path)
    _case(labels, "d")["expected_claims"] = [{"claim": "x", "source_pages": []}]
    with pytest.raises(ValueError, match="unanswerable"):
        _load(tmp_path, labels)


def test_contradiction_case_requires_false_premise(tmp_path):
    labels = _labels(tmp_path)
    _case(labels, "c")["false_premise"] = False
    with pytest.raises(ValueError, match="false_premise"):
        _load(tmp_path, labels)


def test_changed_source_page_is_rejected(tmp_path):
    labels = _labels(tmp_path)
    (tmp_path / "pages" / "book" / "page_0001.md").write_text("edited\n")
    with pytest.raises(ValueError, match="source page 2 changed"):
        _load(tmp_path, labels)


def test_label_page_without_hash_is_rejected(tmp_path):
    labels = _labels(tmp_path)
    del labels["source_pages"]["2"]
    with pytest.raises(ValueError, match="page 2 has no hash"):
        _load(tmp_path, labels)


def _phase2_reference(tmp_path, labels):
    phase2 = {
        "cases": [
            {
                "case_id": "old",
                "kind": "page_range",
                "question": "What is on page one?",
                "history": [],
                "answerable": True,
                "relevant_pages": [1],
                "expected_claims": [{"claim": "First.", "source_pages": [1]}],
            }
        ]
    }
    path = tmp_path / "phase2.json"
    path.write_text(json.dumps(phase2))
    labels["phase2_reference"] = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "case_ids": ["old"],
    }
    labels["splits"]["calibration"].append("old")
    return path


def test_phase2_reference_cases_are_merged(tmp_path):
    labels = _labels(tmp_path)
    _phase2_reference(tmp_path, labels)
    old = _case(_load(tmp_path, labels), "old")
    assert old["kind"] == "page_range" and old["false_premise"] is False
    assert old["split"] == "calibration"


def test_phase2_case_outside_phase4_kinds_cannot_enter_test_split(tmp_path):
    labels = _labels(tmp_path)
    _phase2_reference(tmp_path, labels)
    labels["splits"]["calibration"].remove("old")
    labels["splits"]["test"].append("old")
    with pytest.raises(ValueError, match="non-Phase 4 kinds"):
        _load(tmp_path, labels)


def test_malformed_phase2_case_is_rejected(tmp_path):
    labels = _labels(tmp_path)
    path = _phase2_reference(tmp_path, labels)
    phase2 = json.loads(path.read_text())
    phase2["cases"][0]["expected_claims"][0]["source_pages"] = [2]
    path.write_text(json.dumps(phase2))
    labels["phase2_reference"]["sha256"] = ev.digest(path)
    with pytest.raises(ValueError, match="case old: claim pages"):
        _load(tmp_path, labels)


def test_changed_phase2_reference_is_rejected(tmp_path):
    labels = _labels(tmp_path)
    path = _phase2_reference(tmp_path, labels)
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="Phase 2 labels changed"):
        _load(tmp_path, labels)


def test_unknown_split_name_is_rejected(tmp_path):
    labels = _load(tmp_path, _labels(tmp_path))
    with pytest.raises(ValueError, match="unknown split"):
        ev.cases_for(labels, "dev")
    bad = copy.deepcopy(_labels(tmp_path))
    bad["splits"]["dev"] = []
    with pytest.raises(ValueError, match="splits must be"):
        _load(tmp_path, bad)


def _metric_case(case_id, *, answerable=True, false_premise=False, pages=(1,)):
    return {
        "case_id": case_id,
        "question": "Why does the book say interpreters never evaluate expressions?",
        "answerable": answerable,
        "false_premise": false_premise,
        "relevant_pages": list(pages) if answerable else [],
        "expected_claims": (
            [{"claim": "x", "source_pages": list(pages)}] if answerable else []
        ),
    }


def _row(case_id, answer="An answer.", *, complete=True, error=None, **extra):
    return {
        "case_id": case_id,
        "answer": answer,
        "complete": complete,
        "error": error,
        "context_pages": [],
        "sources": [],
        **extra,
    }


def test_zero_denominators_are_none():
    m = ev.answer_metrics([], [])
    assert m["coverage"] is None and m["answerable"] == 0
    assert m["abstention_rate"] is None and m["rejection_rate"] is None
    c = ev.claim_metrics([])
    assert c["hallucination_rate"] is None and c["citation_accuracy"] is None
    r = ev.expected_claim_recall([], [])
    assert r["supported_citation_recall"] is None and r["expected_claims"] == 0


def test_abstained_matches_canonical_sentence_and_disclaimer():
    from src.page_querying import ABSTENTION

    assert ev.abstained(ABSTENTION + " Try another chapter.")
    assert ev.abstained("\n\n" + ABSTENTION)
    assert ev.abstained("The book does not cover the weather.")
    assert not ev.abstained("An interpreter evaluates expressions.")
    assert not ev.abstained("")


def test_answer_metrics_counts_abstention_and_premise():
    from src.page_querying import ABSTENTION

    cases = [
        _metric_case("ok"),
        _metric_case("cut"),
        _metric_case("none", answerable=False),
        _metric_case("premise", false_premise=True),
    ]
    rows = [
        _row("ok"),
        _row("cut", complete=False),
        _row("none", ABSTENTION),
        _row("premise", "That premise is incorrect: interpreters do evaluate them."),
    ]
    m = ev.answer_metrics(cases[:3], rows)
    assert (m["answerable"], m["answered"], m["coverage"]) == (2, 1, 0.5)
    assert (m["unanswerable"], m["abstained"], m["abstention_rate"]) == (1, 1, 1.0)
    m = ev.answer_metrics(cases, rows)
    assert (m["false_premise"], m["rejected"], m["rejection_rate"]) == (1, 1, 1.0)
    assert m["heuristic"] == ["abstained", "rejected"]


def test_incomplete_answers_never_count_as_abstained_or_rejected():
    from src.page_querying import ABSTENTION

    cases = [
        _metric_case("none", answerable=False),
        _metric_case("premise", false_premise=True),
    ]
    rows = [
        _row("none", ABSTENTION, complete=False),
        _row("premise", "That is incorrect", complete=False),
    ]
    m = ev.answer_metrics(cases, rows)
    assert (m["abstained"], m["rejected"]) == (0, 0)


def test_claim_metrics_skip_unjudged_rows():
    judged = _row(
        "a",
        claims=[{"supported": True}, {"supported": False}, {"supported": True}],
        citations=[{"supported": True}, {"supported": False}],
    )
    m = ev.claim_metrics([judged, _row("b")])
    assert (m["checked_claims"], m["unsupported_claims"]) == (3, 1)
    assert m["hallucination_rate"] == 1 / 3
    assert (m["checked_citations"], m["supported_citations"]) == (2, 1)
    assert m["citation_accuracy"] == 0.5
    assert m["unjudged_rows"] == 1


def test_expected_claim_recall_uses_supporting_sources_only():
    cases = [_metric_case("a", pages=(5,)), _metric_case("b", pages=(9,))]
    sources = [
        {"source_id": "s1", "kind": "book", "pages": [5], "text": "t"},
        {"source_id": "s2", "kind": "book", "pages": [9], "text": "t"},
    ]
    rows = [
        _row("a", sources=sources, supporting_source_ids=["s2"]),
        _row("b", sources=sources, supporting_source_ids=["s2"]),
    ]
    m = ev.expected_claim_recall(cases, rows)
    assert (m["expected_claims"], m["cited_expected"]) == (2, 1)
    assert m["supported_citation_recall"] == 0.5
    # Retrieved pages of an unjudged row are not supporting citations.
    unjudged = ev.expected_claim_recall(cases, [_row("b", sources=sources)])
    assert unjudged["cited_expected"] == 0


def test_retrieval_metrics_use_context_pages():
    cases = [_metric_case("a", pages=(1, 2))]
    m = ev.retrieval_metrics(cases, [_row("a", context_pages=[2, 3])])
    assert (m["hits"], m["relevant_pages"], m["retrieved_pages"]) == (1, 2, 2)
    assert m["precision"] == 0.5 and m["recall"] == 0.5
    assert m["cutoff"] == "selected answer context"


class _Client:
    def __init__(self):
        self.unloads = 0

    def unload_and_wait(self):
        self.unloads += 1


def _book_ref(source_id, pages):
    from src.tools.protocol import EvidenceRef

    return EvidenceRef(source_id, "book", "text", "book", tuple(pages))


def _fake_agent(monkeypatch, *, fail=(), requests=None):
    from src.agent import AgentEvent

    async def run_agent(request, **_):
        if requests is not None:
            requests.append(request)
        yield AgentEvent("step", {"stage": "route", "mode": "qa"})
        if request.question in fail:
            raise TimeoutError("ollama timed out")
        yield AgentEvent(
            "result",
            {
                "answer": "An answer.",
                "complete": True,
                "sources": [_book_ref("s1", [2, 1])],
                "model_requests": 2,
                "tool_calls": 1,
            },
        )

    async def rewrite(question, exchanges, book, *, model, client):
        from src.query_rewriting import RewriteResult

        rewrite.calls.append(exchanges)
        return RewriteResult(question, f"rewritten {question}", bool(exchanges), None)

    rewrite.calls = []
    monkeypatch.setattr(ev, "run_agent", run_agent)
    monkeypatch.setattr(ev, "rewrite_followup", rewrite)
    return rewrite


def _followup_case():
    return {
        **_metric_case("follow"),
        "kind": "followup",
        "split": "test",
        "question": "Give an example?",
        "history": [
            {"role": "user", "content": "What is an interpreter?"},
            {"role": "assistant", "content": "A program that evaluates programs."},
        ],
    }


async def test_followup_history_reaches_rewrite(monkeypatch):
    from scripts.evaluate_phase3_baseline import exchanges_for

    requests = []
    rewrite = _fake_agent(monkeypatch, requests=requests)
    case = _followup_case()
    row = await ev.run_case(case, "book", client=_Client(), sandbox=None, counter=None)
    assert rewrite.calls == [exchanges_for(case["history"], "book")]
    assert requests[0].history == tuple(case["history"])
    assert requests[0].retrieval_query == "rewritten Give an example?"
    assert requests[0].web_enabled is False
    assert row["retrieval_query"] == "rewritten Give an example?"
    assert row["context_pages"] == [1, 2] and row["complete"] is True
    assert row["sources"] == [
        {"source_id": "s1", "kind": "book", "pages": [2, 1], "text": "text"}
    ]
    assert row["error"] is None and row["split"] == "test"
    assert set(row["stage_seconds"]) == {"route"}


def _eval_setup(tmp_path, monkeypatch, *, provenance=None):
    """A real one-book store built from repaired pages plus a valid label file."""
    from src import index_store
    from tests.test_index_store import chunks

    (tmp_path / "book_labels.json").write_text(json.dumps(_labels(tmp_path)))
    report = tmp_path / "pages" / "book" / "repair_report.json"
    report.write_text(json.dumps({"book_stem": "book", "pages": 2}))
    if provenance is None:
        provenance = {"source": "repaired", "repair_report_sha256": ev.digest(report)}
    store = tmp_path / "store"
    index_store.build_index_version("book", chunks(), root=store, provenance=provenance)
    monkeypatch.setattr(ev.config, "VECTOR_STORE", store)
    monkeypatch.setattr(ev.config, "REPAIRED_MD", tmp_path / "pages")
    monkeypatch.setattr(ev.config, "PHASE4_LABELS_DIR", tmp_path)
    monkeypatch.setattr(ev, "identities", lambda paths: {"code_sha256": list(paths)})
    monkeypatch.setattr(ev, "sampling_record", lambda: {"final": {}})

    @asynccontextmanager
    async def services(report):
        report["sandbox"] = {"available": False, "reason": "test"}
        yield {"client": _Client(), "sandbox": None, "counter": None}

    monkeypatch.setattr(ev, "_services", services)
    return store


async def _evaluate_in_thread(**kwargs):
    import asyncio

    return await asyncio.to_thread(ev.evaluate, "book", **kwargs)


async def test_failed_case_counts_as_unanswered(tmp_path, monkeypatch):
    _eval_setup(tmp_path, monkeypatch)
    _fake_agent(monkeypatch, fail={"Tell me page two."})
    report = await _evaluate_in_thread(
        split="test", pipeline="baseline", output=tmp_path / "out.json"
    )
    assert report["status"] == "complete"
    assert report["failures"] == [
        {"case_id": "b", "error": "TimeoutError: ollama timed out"}
    ]
    answer = report["metrics"]["answer"]
    assert (answer["answerable"], answer["answered"]) == (2, 1)
    row = next(r for r in report["rows"] if r["case_id"] == "b")
    assert row["complete"] is False and row["context_pages"] == []


def test_existing_output_is_refused(tmp_path):
    output = tmp_path / "out.json"
    output.write_text("prior evidence")
    with pytest.raises(FileExistsError):
        ev.evaluate("book", split="test", pipeline="baseline", output=output)
    assert output.read_text() == "prior evidence"


async def test_report_records_identities_and_review_status(tmp_path, monkeypatch):
    _eval_setup(tmp_path, monkeypatch)
    _fake_agent(monkeypatch)
    output = tmp_path / "out.json"
    report = await _evaluate_in_thread(
        split="calibration", pipeline="baseline", output=output
    )
    assert json.loads(output.read_text()) == report
    assert report["kind"] == "phase4_eval" and report["pipeline"] == "baseline"
    assert report["labels_review_status"] == "unreviewed"
    assert report["labels_sha256"] == ev.digest(tmp_path / "book_labels.json")
    assert "scripts/evaluate_phase4.py" in report["code_sha256"]
    assert report["index_version"]["legacy"] is False
    assert [r["case_id"] for r in report["rows"]] == ["a"]
    assert report["metrics"]["latency"]["n"] == 1
    assert report["metrics"]["retrieval"]["cutoff"] == "selected answer context"


async def test_active_store_is_not_modified(tmp_path, monkeypatch):
    from scripts.evaluate_phase3_baseline import tree_identity

    store = _eval_setup(tmp_path, monkeypatch)
    before = tree_identity(store)
    _fake_agent(monkeypatch)
    report = await _evaluate_in_thread(
        split="test", pipeline="baseline", output=tmp_path / "out.json"
    )
    assert report["store_unchanged"] is True and tree_identity(store) == before


@pytest.mark.parametrize(
    ("provenance", "message"),
    [
        ({"source": "checkpoints"}, "not built from repaired pages"),
        ({"source": "repaired", "repair_report_sha256": "0" * 64}, "predates"),
    ],
)
async def test_index_not_built_from_current_repair_fails(
    tmp_path, monkeypatch, provenance, message
):
    _eval_setup(tmp_path, monkeypatch, provenance=provenance)
    _fake_agent(monkeypatch)
    report = await _evaluate_in_thread(
        split="test", pipeline="baseline", output=tmp_path / "out.json"
    )
    assert report["status"] == "failed" and message in report["error"]
    assert report["rows"] == [] and report["metrics"] is None
    assert report["index_version"]["source"] == provenance["source"]


async def test_store_changed_during_run_fails_report(tmp_path, monkeypatch):
    _eval_setup(tmp_path, monkeypatch)
    _fake_agent(monkeypatch)
    identities = iter(["before", "after"])
    monkeypatch.setattr(ev, "tree_identity", lambda _: next(identities))
    report = await _evaluate_in_thread(
        split="test", pipeline="baseline", output=tmp_path / "out.json"
    )
    assert report["store_unchanged"] is False and report["status"] == "failed"
    assert "store changed" in report["error"]


def test_unknown_pipeline_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown pipeline"):
        ev.evaluate("book", split="test", pipeline="gated", output=tmp_path / "o")
    assert not (tmp_path / "o").exists()


@pytest.mark.parametrize("book", ["sicp", "thinkpython2"])
def test_phase4_label_set_is_complete(book):
    if not (ev.config.REPAIRED_MD / book).is_dir():
        pytest.skip(f"repaired pages for {book} not present")
    labels = ev.load_labels(ev.labels_path(book))
    test = ev.cases_for(labels, "test")
    assert {c["kind"] for c in test} == ev.KINDS
    assert len(labels["cases"]) >= 12
    assert labels["review"]["status"] == "maintainer-reviewed"
