import copy
import json

import pytest

from scripts import validate_phase6_quality as quality


@pytest.fixture
def report():
    return {
        "status": "complete",
        "book_stem": "book",
        "split": "test",
        "labels_sha256": "a" * 64,
        "source_pages": {"1": "b" * 64},
        "index_version": {"version_id": "v1", "embedding_signature": "e1"},
        "runtime": {"model": "qwen", "digest": "d1", "ollama_version": "1"},
        "sampling": {"final": {"temperature": 0.3}},
        "store_unchanged": True,
        "metric_definition": "micro:selected answer context",
        "cases": [
            {
                "case_id": "a",
                "answerable": True,
                "false_premise": False,
                "relevant_pages": [1],
                "expected_claims": [{"claim": "First", "source_pages": [1]}],
            },
            {
                "case_id": "b",
                "answerable": True,
                "false_premise": True,
                "relevant_pages": [1],
                "expected_claims": [{"claim": "Second", "source_pages": [1]}],
            },
        ],
        "rows": [
            {
                "case_id": "a",
                "answer": "First. Second. Try this.",
                "complete": True,
                "error": None,
                "context_pages": [1],
                "total_seconds": 1.0,
                "sources": [
                    {"source_id": "s1", "pages": [1], "kind": "book", "text": "First"}
                ],
            }
        ],
    }


def review_for(report):
    review = quality.blank_review(report, report_sha256="c" * 64)
    review.update(
        reviewer="Human Reviewer",
        reviewed_at="2026-11-01T10:00:00+00:00",
        complete_inventory=True,
    )
    a = review["answers"]["a"]
    a["complete_inventory"] = True
    a["expected_claims"] = {k: True for k in a["expected_claims"]}
    a["claims"] = [
        {
            "claim_id": "a1",
            "text": "First.",
            "factual": True,
            "correct": True,
            "supported": True,
            "citations": [{"source_id": "s1", "page": 1, "supported": True}],
        },
        {
            "claim_id": "a2",
            "text": "Second.",
            "factual": True,
            "correct": False,
            "supported": False,
            "citations": [{"source_id": "s1", "page": 1, "supported": False}],
        },
        {
            "claim_id": "a3",
            "text": "Try this.",
            "factual": False,
            "correct": None,
            "supported": None,
            "citations": [],
        },
    ]
    return review


def test_wrong_report_hash_or_missing_reviewer_is_rejected(report):
    r = review_for(report)
    with pytest.raises(ValueError, match="hash"):
        quality.validate_review(report, r, report_sha256="d" * 64)
    r["reviewer"] = None
    with pytest.raises(ValueError, match="reviewer"):
        quality.validate_review(report, r, report_sha256="c" * 64)


def test_null_verdict_is_not_correct(report):
    r = review_for(report)
    r["answers"]["a"]["claims"][0].update(
        correct=None, supported=None, reason="unclear"
    )
    metrics = quality.validate_review(report, r, report_sha256="c" * 64)
    assert metrics["factual_error"] == {"numerator": 1, "denominator": 1, "rate": 1.0}
    assert metrics["unreviewed_factual_claims"] == 1


def test_complete_inventory_required(report):
    r = review_for(report)
    r["complete_inventory"] = False
    metrics = quality.validate_review(report, r, report_sha256="c" * 64)
    assert metrics["inventory_complete"] is False


def test_incomplete_and_missing_answers_stay_in_coverage_denominator(report):
    r = review_for(report)
    report["rows"][0]["complete"] = False
    metrics = quality.validate_review(report, r, report_sha256="c" * 64)
    assert metrics["answer_coverage"]["denominator"] == 2
    assert metrics["answer_coverage"]["numerator"] == 0
    assert metrics["expected_claim_coverage"] == {
        "numerator": 0,
        "denominator": 2,
        "rate": 0.0,
    }


def test_page_validity_is_not_claim_support(report):
    m = quality.validate_review(report, review_for(report), report_sha256="c" * 64)
    for key in ("factual_error", "unsupported_claim", "citation_support"):
        assert m[key] == {"numerator": 1, "denominator": 2, "rate": 0.5}
    r = review_for(report)
    r["answers"]["a"]["claims"][0]["citations"][0]["page"] = 2
    with pytest.raises(ValueError, match="source|page"):
        quality.validate_review(report, r, report_sha256="c" * 64)


@pytest.mark.parametrize(
    "key,value",
    [
        ("metric_definition", "macro:selected answer context"),
        ("labels_sha256", "changed"),
        ("index_version", {"version_id": "v2"}),
        ("source_pages", {"1": "changed"}),
        ("sampling", {}),
    ],
)
def test_macro_and_micro_controls_cannot_be_mixed(report, key, value):
    after = copy.deepcopy(report)
    after[key] = value
    with pytest.raises(ValueError, match="identity|definition"):
        quality.compare_quality(report, after)


@pytest.mark.parametrize(
    "field,value", [("factual", "yes"), ("correct", 1), ("supported", "unclear")]
)
def test_closed_verdict_values(report, field, value):
    r = review_for(report)
    r["answers"]["a"]["claims"][0][field] = value
    with pytest.raises(ValueError):
        quality.validate_review(report, r, report_sha256="c" * 64)


def test_unique_claims_and_unknown_expected_ids_rejected(report):
    r = review_for(report)
    r["answers"]["a"]["claims"].append(r["answers"]["a"]["claims"][0])
    with pytest.raises(ValueError, match="duplicate"):
        quality.validate_review(report, r, report_sha256="c" * 64)
    r = review_for(report)
    r["answers"]["a"]["expected_claims"]["invented"] = True
    with pytest.raises(ValueError, match="expected"):
        quality.validate_review(report, r, report_sha256="c" * 64)


def test_blank_sheet_never_supplies_human_verdicts(report):
    r = quality.blank_review(report, report_sha256="c" * 64)
    assert r["reviewer"] is None and r["reviewed_at"] is None
    assert not r["complete_inventory"]
    assert set(r["answers"]) == {"a", "b"}
    assert all(a["claims"] == [] for a in r["answers"].values())


def test_nonempty_answer_cannot_declare_empty_complete_inventory(report):
    r = review_for(report)
    r["answers"]["a"]["claims"] = []
    with pytest.raises(ValueError, match="inventory"):
        quality.validate_review(report, r, report_sha256="c" * 64)


def test_malformed_claims_raise_validation_error(report):
    r = review_for(report)
    r["answers"]["a"]["claims"] = [None]
    with pytest.raises(ValueError, match="rows|claim"):
        quality.validate_review(report, r, report_sha256="c" * 64)


def test_existing_evidence_is_not_overwritten(tmp_path):
    path = tmp_path / "report.json"
    path.write_text("old")
    with pytest.raises(FileExistsError):
        quality.write_evidence(path, {"new": True})
    assert path.read_text() == "old"


def test_failed_case_remains_in_denominator(report):
    result = quality.compare_quality(report, report)
    assert result["completion"]["denominator"] == 2
    assert result["completion"]["numerator"] == 1
    assert result["completion"]["status"] == "fail"


def test_empty_answer_and_failed_report_cannot_pass_completion(report):
    report["rows"][0]["answer"] = ""
    assert quality.compare_quality(report, report)["completion"]["numerator"] == 0
    report["status"] = "failed"
    assert quality.compare_quality(report, report)["report_status"]["status"] == "fail"


def test_test_selected_policy_is_rejected(report):
    after = copy.deepcopy(report)
    after["selection"] = {"split": "test", "policy": "rrf"}
    with pytest.raises(ValueError, match="calibration"):
        quality.compare_quality(report, after)


def test_changed_policy_cannot_use_an_empty_selection_pin(report):
    after = copy.deepcopy(report)
    after.update(
        merge_policy="rrf",
        started_utc="2026-11-01T12:00:00+00:00",
        selection={
            "split": "calibration",
            "policy": "rrf",
            "input_sha256": {},
            "selected_at": "2026-11-01T10:00:00+00:00",
        },
    )
    with pytest.raises(ValueError, match="calibration"):
        quality.compare_quality(report, after)


@pytest.mark.parametrize("bad", [{"a": None}, {"a": []}])
def test_malformed_answer_review_is_rejected(report, bad):
    r = review_for(report)
    r["answers"] = bad
    with pytest.raises(ValueError, match="answer"):
        quality.validate_review(report, r, report_sha256="c" * 64)


def test_actual_metric_cutoff_mismatch_is_rejected(report):
    before, after = copy.deepcopy(report), copy.deepcopy(report)
    before["metrics"] = {"retrieval": {"cutoff": "selected answer context"}}
    after["metrics"] = {"retrieval": {"cutoff": "merged top-15"}}
    with pytest.raises(ValueError, match="cutoff"):
        quality.compare_quality(before, after)


def calibration_paths(tmp_path):
    paths = []
    for book in ("sicp", "thinkpython2"):
        path = tmp_path / f"{book}.json"
        path.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "book_stem": book,
                    "store_unchanged": True,
                    "selection_split": "calibration",
                    "index_version": {"version_id": "v1", "embedding_signature": "e1"},
                    "labels_sha256": "a" * 64,
                    "config_sha256": "config",
                    "code_sha256": {
                        "src/hybrid_retrieval.py": "hybrid",
                        "src/retrieval.py": "retrieval",
                        "src/query_rewriting.py": "rewrite",
                        "src/tools/local.py": "local",
                    },
                    "rows": [
                        {
                            "group": "calibration",
                            "case_id": "a",
                            "question": "Question?",
                            "retrieval_query": "query",
                            "policy": p,
                        }
                        for p in ("interleave", "rrf")
                    ],
                    "metrics": {
                        "calibration": {
                            "interleave": {
                                "context_micro": {"precision": 0.5, "recall": 0.6}
                            },
                            "rrf": {"context_micro": {"precision": 0.6, "recall": 0.7}},
                        }
                    },
                }
            )
        )
        paths.append(path)
    return paths


def test_selection_requires_preserved_calibration_evidence(tmp_path):
    paths = calibration_paths(tmp_path)
    selection = quality.select_candidate(paths)
    assert selection["policy"] == "rrf" and selection["split"] == "calibration"
    assert set(selection["input_sha256"]) == {str(p.resolve()) for p in paths}
    bad = json.loads(paths[0].read_text())
    bad["selection_split"] = "test"
    paths[0].write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="calibration"):
        quality.select_candidate(paths)


@pytest.mark.parametrize("drift", ["config", "ranking", "query"])
def test_selection_rejects_unselected_ranking_config_or_inputs(tmp_path, report, drift):
    from datetime import UTC, datetime, timedelta

    paths = calibration_paths(tmp_path)
    selection = quality.select_candidate(paths)
    report.update(
        book_stem="sicp",
        config_sha256="config",
        code_sha256=json.loads(paths[0].read_text())["code_sha256"],
    )
    report["rows"][0].update(question="Question?", retrieval_query="query")
    after = copy.deepcopy(report)
    after.update(
        merge_policy="rrf",
        selection=selection,
        started_utc=(datetime.now(UTC) + timedelta(seconds=1)).isoformat(),
    )
    if drift == "config":
        after["config_sha256"] = "changed-RRF-K"
    elif drift == "ranking":
        after["code_sha256"]["src/hybrid_retrieval.py"] = "changed"
    else:
        after["rows"][0]["retrieval_query"] = "different query"
    with pytest.raises(ValueError, match="ranking|config|query|input"):
        quality.compare_quality(report, after)


def test_instrumentation_only_change_keeps_selection_valid(tmp_path, report):
    from datetime import UTC, datetime, timedelta

    paths = calibration_paths(tmp_path)
    report.update(
        book_stem="sicp",
        config_sha256="config",
        code_sha256=json.loads(paths[0].read_text())["code_sha256"],
    )
    report["rows"][0].update(question="Question?", retrieval_query="query")
    after = copy.deepcopy(report)
    after.update(
        merge_policy="rrf",
        selection=quality.select_candidate(paths),
        started_utc=(datetime.now(UTC) + timedelta(seconds=1)).isoformat(),
    )
    after["code_sha256"]["src/agent.py"] = "new-trace-fields"
    assert quality.compare_quality(report, after)["retrieval"]["status"] == "pass"


def test_malformed_citation_retains_readable_failed_review(tmp_path, report):
    before, after, rb, ra = [
        tmp_path / n for n in ("b.json", "a.json", "rb.json", "ra.json")
    ]
    for p in (before, after):
        p.write_text(json.dumps(report))
    review = review_for(report)
    review["source_report_sha256"] = quality.digest(before)
    review["answers"]["a"]["claims"][0]["citations"] = [None]
    for p in (rb, ra):
        p.write_text(json.dumps(review))
    output = tmp_path / "result.json"
    result = quality.evaluate_quality(
        before=before, after=after, review_before=rb, review_after=ra, output=output
    )
    assert not result["accepted"]
    assert json.loads(output.read_text())["human_review"]["reasons"]


def test_unreviewed_or_changed_sources_cannot_pass_quality(tmp_path, report):
    paths = [tmp_path / n for n in ("before.json", "after.json", "rb.json", "ra.json")]
    for p, value in zip(
        paths,
        [report, report, quality.blank_review(report, report_sha256="c" * 64), {}],
    ):
        p.write_text(json.dumps(value))
    result = quality.evaluate_quality(
        before=paths[0],
        after=paths[1],
        review_before=paths[2],
        review_after=paths[3],
        output=tmp_path / "result.json",
    )
    assert result["accepted"] is False
    assert result["human_review"]["status"] == "pending"
    assert result["original_regression"]["status"] == "pending"


def test_complete_attributed_comparison_can_pass_its_own_scope(
    tmp_path, monkeypatch, report
):
    report["rows"].append({**copy.deepcopy(report["rows"][0]), "case_id": "b"})
    before, after, rb, ra = [
        tmp_path / n for n in ("b.json", "a.json", "rb.json", "ra.json")
    ]
    for p in (before, after):
        p.write_text(json.dumps(report))
    review = review_for(report)
    b = review["answers"]["b"]
    b.update(
        complete_inventory=True,
        premise_rejected=True,
        claims=copy.deepcopy(review["answers"]["a"]["claims"]),
    )
    b["expected_claims"] = dict.fromkeys(b["expected_claims"], True)
    for p, source in ((rb, before), (ra, after)):
        review["source_report_sha256"] = quality.digest(source)
        p.write_text(json.dumps(review))
    # Original-macro validation has its own real file/source/geometry test below.
    monkeypatch.setattr(
        quality, "compare_original_regression", lambda **kw: {"status": "pass"}
    )
    result = quality.evaluate_quality(
        before=before,
        after=after,
        review_before=rb,
        review_after=ra,
        original_before=before,
        original_after=after,
        output=tmp_path / "result.json",
    )
    assert result["accepted"] and result["scope"]["book"] == "book"


def test_diagnostics_distinguish_candidate_miss_from_context_omission():
    stages = quality.diagnose_pages([1, 2, 3], raw=[1, 2], merged=[1, 2], selected=[1])
    assert stages == {"raw_miss": [3], "merge_exclusion": [], "context_omission": [2]}


def test_diagnostic_refuses_overwrite_before_loading_models(tmp_path):
    path = tmp_path / "report.json"
    path.write_text("old")
    with pytest.raises(FileExistsError):
        quality.diagnose_retrieval("book", output=path)
    assert path.read_text() == "old"


def test_diagnostic_uses_copy_and_records_both_policies(tmp_path, monkeypatch):
    from scripts import evaluate_phase3_baseline as baseline
    from src import db, index_store
    from tests.test_phase4_evaluation import _eval_setup

    store = _eval_setup(tmp_path, monkeypatch)
    control_dir = tmp_path / "controls"
    control_dir.mkdir()
    entry = index_store.read_manifest(store)["book"]
    control = {
        "labels_sha256": quality.digest(tmp_path / "book_labels.json"),
        "index_version": {k: entry[k] for k in ("version_id", "embedding_signature")},
        "rows": [{"case_id": "a", "retrieval_query": "one"}],
    }
    (control_dir / "book-calibration.json").write_text(json.dumps(control))
    monkeypatch.setattr(quality.config, "PHASE6_QUALITY_CONTROLS", control_dir)
    monkeypatch.setattr(baseline, "identities", lambda _: {})
    original_resolve = index_store.resolve_index
    roots = []

    def resolve(book, *, root):
        roots.append(root)
        assert root != store
        return original_resolve(book, root=root)

    monkeypatch.setattr(index_store, "resolve_index", resolve)
    monkeypatch.setattr(db, "embed_query", lambda *a, **k: [1.0] + [0.0] * 767)
    report = quality.diagnose_retrieval("book", output=tmp_path / "diagnostic.json")
    assert report["status"] == "complete", report.get("error")
    assert report["store_unchanged"] and roots
    assert {r["policy"] for r in report["rows"]} == {"interleave", "rrf"}
    assert all(r["attempts"] and r["selected"] for r in report["rows"])
    assert not report["accepted"]


def test_original_regression_needs_its_own_comparable_result(tmp_path, monkeypatch):
    labels = {
        "schema_version": 1,
        "book_stem": "sicp",
        "source_pages": {"1": quality.hashlib.sha256(b"page").hexdigest()},
        "cases": [
            {
                "case_id": str(i),
                "book_stem": "sicp",
                "kind": "direct",
                "question": "Question?",
                "history": [],
                "answerable": True,
                "relevant_pages": [1],
                "expected_claims": [{"claim": "Fact", "source_pages": [1]}],
            }
            for i in range(7)
        ],
    }
    path = tmp_path / "labels" / "sicp_context_labels.json"
    path.parent.mkdir()
    path.write_text(json.dumps(labels))
    monkeypatch.setattr(quality.config, "PHASE2_LABELS_DIR", path.parent)
    parsed = tmp_path / "parsed"
    (parsed / "checkpoints/sicp").mkdir(parents=True)
    (parsed / "checkpoints/sicp/page_0000.md").write_text("page")
    manifest = parsed / "sicp/page_manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "book_stem": "sicp",
                "pages": [
                    {
                        "physical_page": 1,
                        "page_idx": 0,
                        "checkpoint_file": "page_0000.md",
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(quality.config, "PARSED_MD", parsed)
    runtime = {
        "model": "qwen",
        "digest": "d1",
        "ollama_version": "1",
        "embedding": "jina",
        "embedding_revision": "e1",
    }
    geometry = {"version_id": "v1", "embedding_config": {"geometry": "original"}}
    historical = tmp_path / "historical.json"
    historical.write_text(json.dumps({"ingest": geometry}))
    monkeypatch.setattr(quality.config, "PHASE2_AFTER_EVALUATION_PATH", historical)
    before = tmp_path / "original-before.json"
    before.write_text(
        json.dumps(
            {
                "status": "complete",
                "book_stem": "sicp",
                "runtime": runtime,
                "dataset_sha256": quality.digest(path),
                "source_manifest_sha256": quality.digest(manifest),
                "rows": [{"case_id": str(i), "physical_pages": [1]} for i in range(7)],
            }
        )
    )
    monkeypatch.setattr(quality.config, "PHASE2_EVALUATION_PATH", before)
    after = tmp_path / "original-after.json"
    candidate = {
        "status": "complete",
        "book_stem": "sicp",
        "runtime": runtime,
        "preserved": True,
        "baseline_sha256": quality.digest(before),
        "dataset_sha256": quality.digest(path),
        "ingest": geometry,
        "rows": [{"case_id": str(i), "physical_pages": [1]} for i in range(7)],
    }
    after.write_text(json.dumps(candidate))
    assert (
        quality.compare_original_regression(before=before, after=after)["status"]
        == "pass"
    )
    relocated = tmp_path / "exact-copy.json"
    relocated.write_bytes(before.read_bytes())
    assert (
        quality.compare_original_regression(before=relocated, after=after)["status"]
        == "pass"
    )
    fake = json.loads(before.read_text())
    for row in fake["rows"]:
        row["physical_pages"] = []
    substituted = tmp_path / "substituted-before.json"
    substituted.write_text(json.dumps(fake))
    replacement_candidate = copy.deepcopy(candidate)
    replacement_candidate["baseline_sha256"] = quality.digest(substituted)
    after.write_text(json.dumps(replacement_candidate))
    with pytest.raises(ValueError, match="frozen|original baseline"):
        quality.compare_original_regression(before=substituted, after=after)
    after.write_text(json.dumps(candidate))
    candidate["rows"][0]["physical_pages"] = [2]
    after.write_text(json.dumps(candidate))
    assert (
        quality.compare_original_regression(before=before, after=after)["status"]
        == "fail"
    )
    candidate["ingest"]["version_id"] = "new-repaired-index"
    after.write_text(json.dumps(candidate))
    with pytest.raises(ValueError, match="geometry"):
        quality.compare_original_regression(before=before, after=after)
