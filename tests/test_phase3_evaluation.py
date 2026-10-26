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
    assert "web_when_expected" not in after.policy_checks({}, used, web=True)
    assert ok(False, unused) and not ok(False, used)
    assert ok("only_when_enabled", used) and not ok("only_when_enabled", unused)
    failed = {**used, "verification": {"status": "unverified"}}
    verified = {**unused, "verification": {"status": "verified"}}
    assert ok("only_on_failure_when_enabled", failed)
    assert ok("only_on_failure_when_enabled", verified)
    assert not ok(
        "only_on_failure_when_enabled", {**failed, "stages": unused["stages"]}
    )


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
