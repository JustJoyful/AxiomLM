import json

import pytest

from scripts import evaluate_phase3_baseline as baseline
from scripts import validate_phase3_agent as after


def test_frozen_labels_pin_phase2_cases_and_policy_fields():
    labels, cases = baseline.load_cases()
    assert labels["review"]["status"] == "unreviewed"
    assert {c["source"] for c in cases} == {"phase2", "phase3"}
    solve = [c for c in cases if c.get("expected_mode") == "solve"]
    assert solve and all(
        c["expected_stage_order"].index("compute")
        < c["expected_stage_order"].index("verify")
        for c in solve
    )


def test_changed_phase2_labels_are_rejected(tmp_path):
    labels = json.loads(baseline.LABELS.read_text())
    labels["phase2_reference"]["sha256"] = "0" * 64
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(labels))
    with pytest.raises(ValueError):
        baseline.load_cases(path)


def test_page_metrics_report_denominators():
    cases = [
        {"case_id": "a", "relevant_pages": [1, 2]},
        {"case_id": "b", "relevant_pages": []},
    ]
    rows = [
        {"case_id": "a", "physical_pages": [2, 3]},
        {"case_id": "b", "physical_pages": [9]},
    ]
    metrics = baseline.page_metrics(cases, rows)
    assert metrics == {
        "labelled_cases": 1,
        "relevant_pages": 2,
        "retrieved_pages": 2,
        "hits": 1,
        "precision": 0.5,
        "recall": 0.5,
    }


def test_policy_checks_detect_order_mode_web_and_incomplete():
    case = {
        "expected_mode": "solve",
        "expected_stage_order": [
            "classify",
            "local_search",
            "local_assessment",
            "compute",
            "verify",
            "final",
        ],
        "expected_check_status": "verified",
    }
    good = {
        "mode": "solve",
        "stages": [
            "classify",
            "local_search",
            "local_assessment",
            "compute",
            "verify",
            "final",
        ],
        "verification": {"status": "verified"},
        "completed": True,
    }
    assert all(after.policy_checks(case, good).values())
    skipped = {
        **good,
        "stages": ["classify", "local_search", "local_assessment", "final"],
    }
    web = {**good, "stages": good["stages"][:5] + ["optional_web", "final"]}
    relabelled = {**good, "verification": {"status": "unverified"}}
    assert after.policy_checks(case, skipped)["stage_order"] is False
    assert after.policy_checks(case, web)["no_web_while_off"] is False
    assert after.policy_checks(case, relabelled)["check_status"] is False
    assert after.policy_checks(case, {**good, "completed": False})["completed"] is False


def test_online_policy_follows_frozen_web_expectation():
    used = {"mode": "qa", "stages": ["classify", "local_search", "optional_web"]}
    unused = {"mode": "qa", "stages": ["classify", "local_search", "final"]}

    def ok(expected, row):
        case = {"expected_mode": "qa", "web_expected": expected}
        return after.policy_checks(case, row, web=True)["web_when_expected"]

    assert "no_web_while_off" not in after.policy_checks({}, used, web=True)
    # A missing label is visible: applicable, not passed, counted as missing.
    missing = after.policy_checks({}, used, web=True)
    assert "web_when_expected" in missing and missing["web_when_expected"] is None
    summary = after.summarize([missing])["web_when_expected"]
    assert summary == {"passed": 0, "applicable": 1, "missing_label": 1}
    assert ok(False, unused) and not ok(False, used)
    assert ok("only_when_enabled", used) and not ok("only_when_enabled", unused)
    failed = {**used, "verification": {"status": "unverified"}}
    verified = {**unused, "verification": {"status": "verified"}}
    assert ok("only_on_failure_when_enabled", failed)
    assert ok("only_on_failure_when_enabled", verified)
    assert not ok(
        "only_on_failure_when_enabled", {**failed, "stages": unused["stages"]}
    )


def test_book_answerable_phase2_cases_default_to_no_web():
    labels, cases = baseline.load_cases()
    after.apply_web_expectations(labels, cases)
    by_id = {c["case_id"]: c for c in cases}
    assert by_id["outside_book"]["web_expected"] == "only_when_enabled"
    for cid in ("interpreter", "interpreter_example", "procedures_and_data"):
        assert by_id[cid]["web_expected"] is False
    assert all("web_expected" in c for c in cases)
    used = {"mode": "qa", "stages": ["classify", "local_search", "optional_web"]}
    checks = after.policy_checks(by_id["interpreter_example"], used, web=True)
    assert checks["web_when_expected"] is False


async def test_run_case_records_the_followup_rewrite(monkeypatch):
    from src import agent, query_rewriting
    from src.agent import AgentEvent

    async def rewrite(question, exchanges, book, *, model, client):
        return query_rewriting.RewriteResult(
            question, "What is another example of an interpreter?", True, None
        )

    seen = []

    async def run(request, **kwargs):
        seen.append(request.retrieval_query)
        assert request.today == row_today()
        yield AgentEvent("error", {"message": "stop", "stages": [], "trace": []})

    class Client:
        def unload_and_wait(self):
            pass

    monkeypatch.setattr(query_rewriting, "rewrite_followup", rewrite)
    monkeypatch.setattr(agent, "run_agent", run)
    case = {
        "case_id": "interpreter_example",
        "kind": "followup",
        "question": "Give another example of that.",
        "history": [],
    }
    row = await after._run_case(case, "sicp", Client(), None, None, False)
    assert row["rewrite"] == {
        "applied": True,
        "retrieval_query": "What is another example of an interpreter?",
        "fallback_reason": None,
    }
    assert seen == ["What is another example of an interpreter?"]
    assert row["today"] == row_today()


def row_today():
    from datetime import datetime

    return datetime.now().astimezone().date().isoformat()


def test_reports_refuse_overwrite(tmp_path):
    output = tmp_path / "before.json"
    output.write_text("previous")
    with pytest.raises(FileExistsError):
        baseline.reserve(output)
    assert output.read_text() == "previous"


def test_after_refuses_incomplete_or_stale_baseline(tmp_path):
    stale = tmp_path / "before.json"
    stale.write_text(json.dumps({"status": "complete", "labels_sha256": "old"}))
    with pytest.raises(ValueError):
        after.evaluate("sicp", baseline=stale, output=tmp_path / "after.json")
    assert not (tmp_path / "after.json").exists()


def test_review_sheet_shows_the_rewrite():
    from pathlib import Path

    from scripts import review_phase3_agent as review

    row = {
        "case_id": "interpreter_example",
        "kind": "followup",
        "question": "Give another example of that.",
        "rewrite": {
            "applied": True,
            "retrieval_query": "What is another example of an interpreter?",
            "fallback_reason": None,
        },
        "policy": {"mode": True},
        "answer": "An example.",
    }
    report = {
        "passed": True,
        "git_head": "abc",
        "sampling": after.sampling_record(),
        "rows": [row],
    }
    text = review.build(report, report, Path("after.json"), Path("online.json"))
    assert '"temperature": 0.3' in text and "temperature 1" not in text
    assert (
        "Retrieval query (rewritten): What is another example of an interpreter?"
        in text
    )
    assert "web expected: False" in text
    assert "- Lints (not gates): labelled pages not retrieved: [35, 36, 37]" in text


def test_report_records_the_sampling_actually_sent():
    from src import config

    record = after.sampling_record()
    assert record["final"]["temperature"] == config.OLLAMA_TEMPERATURE
    assert record["json_calls"]["temperature"] == 0
    assert record["json_calls"]["seed"] == config.AGENT_JSON_SEED


def test_stability_reruns_select_labelled_cases_only():
    _, cases = baseline.load_cases()
    picked = after.select_cases(cases, ["sqrt_newton", "interpreter_example"])
    assert [c["case_id"] for c in picked] == ["interpreter_example", "sqrt_newton"]
    with pytest.raises(ValueError, match="Unknown case ids"):
        after.select_cases(cases, ["sqrt_newton", "nope"])


def lint(answer, **row):
    case = row.pop("case", {})
    return after.answer_lints({"answer": answer, **row}, case)


def test_lint_internal_terms():
    found = lint("The sandbox trace from the local search tool says 3.")
    assert found == [
        "internal term: local search",
        "internal term: sandbox",
        "internal term: tool",
        "internal term: trace",
    ]
    assert lint("An interpreter is a program.") == []


def test_lint_page_number_used_as_section():
    assert lint("See Section 523 and Section 1.3.4.") == [
        "page as section: Section 523"
    ]


def test_lint_disclaimer_when_book_evidence_covers_the_question():
    covered = {"assessment": {"sufficient": True}}
    assert lint("The evidence does not contain an example.", **covered) == [
        "disclaimer although the book evidence covers the question"
    ]
    case = {"relevant_pages": [35]}
    assert lint("It does not cover this.", physical_pages=[35], case=case) == [
        "disclaimer although the book evidence covers the question"
    ]
    assert lint("The book does not cover this.", physical_pages=[300]) == []


def test_lint_unused_web_sources_on_a_no_web_case():
    row = {"web_sources": ["https://a.org"], "case": {"web_expected": False}}
    assert lint("An interpreter runs programs.", **row) == [
        "web sources fetched but not used in the answer"
    ]
    assert lint("The web says an interpreter runs programs.", **row) == []


def test_lint_reports_missed_labelled_pages():
    case = {"relevant_pages": [796, 797, 855]}
    assert lint("ok", physical_pages=[796], case=case) == [
        "labelled pages not retrieved: [797, 855]"
    ]


def test_lint_cited_pages_must_be_retrieved_pages():
    # Run 07 case 8: "p. 33" came from the table of contents text on p. 4.
    row = {"physical_pages": [4, 136, 139, 140, 141]}
    answer = "In Chapter 1 (p. 33), a procedure is a black box; Chapter 2 (p. 140)."
    assert lint(answer, **row) == ["cited pages not retrieved: [33]"]
    assert after.cited_pages("See pp. 139-141, page 4 and pages 136 and 140.") == [
        139,
        140,
        141,
        4,
        136,
        140,
    ]
    assert lint("See pp. 139–141 and p. 4.", **row) == []
    assert lint("See pp. 139–142.", **row) == ["cited pages not retrieved: [142]"]
    assert lint("It has 33 pages of exercises.", **row) == []


def test_lint_verification_claimed_to_match_the_book():
    # Run 07 case 9, offline and online.
    for answer in (
        (
            "The verification status is verified, matching the book's theoretical "
            "derivation for this polynomial."
        ),
        "It was verified as identical to the book's symbolic differentiation approach.",
    ):
        assert lint(answer) == ["verification claimed to match the book"]
    assert lint("It is verified. The book describes deriv.") == []
    assert lint("This matches the book's definition of a derivative.") == []


def test_lint_contradiction_answers_must_reject_the_premise():
    question = (
        "Since the book says interpreters never evaluate expressions, why does it "
        "call them evaluators?"
    )
    case = {"kind": "contradiction"}
    rejected = (
        "The premise is incorrect: the book says the interpreter evaluates "
        "expressions in a read-eval-print loop."
    )
    assert lint(rejected, question=question, case=case) == []
    quoted = "The claim that interpreters never evaluate expressions is not true."
    assert lint(quoted, question=question, case=case) == []
    # Run 07 offline and online case 12.
    accepted = (
        "The book clarifies that while the interpreter itself never evaluates "
        "expressions in the sense of executing them directly, it acts as an evaluator."
    )
    assert lint(accepted, question=question, case=case) == [
        "premise not rejected",
        "premise restated without negation: evaluate expressions",
    ]
    softened = "Interpreters do not evaluate expressions in that sense. It is wrong."
    assert lint(softened, question=question, case=case) == [
        "premise restated without negation: evaluate expressions"
    ]
    # Stability run r1: normal order is a true negation of a different claim.
    normal_order = (
        "The premise is wrong. In normal order the interpreter does not evaluate "
        "operands until their values are needed."
    )
    assert lint(normal_order, question=question, case=case) == []
    # Other case kinds are not checked.
    assert lint(accepted, question=question) == []


def test_lints_never_change_passed():
    import inspect

    source = inspect.getsource(after.evaluate)
    passed = source[source.index('report["passed"] = (') :]
    passed = passed[: passed.index('report["finished_utc"]')]
    assert "lint" not in passed
