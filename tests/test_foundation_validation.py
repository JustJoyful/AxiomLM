import weakref
import json

from scripts.validate_foundation import (
    _run_tui_workload_probe,
    _valid_before_baseline,
    _probe_ingest_failure_cleanup,
    _should_run_ingest,
    evaluate_report,
)
from src import db


def valid_report():
    return {
        "model": {
            "name": "qwen3.5:4b-q4_K_M",
            "digest": "d" * 64,
            "ollama_version": "0.20.0",
            "quantization": "Q4_K_M",
        },
        "local_service": True,
        "tools": {"passed": True, "unload": "all Ollama models unloaded and confirmed"},
        "query": {"device": "cpu", "parameters_on_cpu": True},
        "ingest": {
            "device": "cuda",
            "encoded": True,
            "released": True,
            "failure_cleanup": True,
        },
        "generation": {
            "prompt_eval_count": 4096,
            "oom": False,
            "vram_sampled": True,
            "ollama_vram_peak_bytes": 1024,
            "unload": "all Ollama models unloaded and confirmed",
        },
        "baseline": {
            "before": True,
            "after": True,
            "question_count": 5,
            "before_schema_valid": True,
            "book_identity_matches": True,
            "index_identity": "index-sha256",
            "index_identity_matches": True,
            "labels_complete": True,
            "after_review": {
                "all_questions_reviewed": True,
                "unsupported_claims_checked": True,
                "citation_accuracy": 1.0,
                "hallucination_rate": 0.0,
            },
            "retrieval_metrics": {
                "before": {
                    "precision": 1.0,
                    "recall": 1.0,
                    "answerable_case_denominator": 4,
                },
                "after": {
                    "precision": 1.0,
                    "recall": 1.0,
                    "answerable_case_denominator": 4,
                },
            },
        },
        "workload_serialization": True,
        "workload_evidence": {
            "tui_dispatch_verified": True,
            "cancellation_cleanup_verified": True,
        },
        "local_only": True,
        "local_failure_behavior": True,
    }


def test_complete_evidence_passes_all_foundation_gates():
    verdict = evaluate_report(valid_report())
    assert verdict["passed"] is True
    assert all(verdict["gates"].values())


def test_unresolved_query_device_fails_named_gate():
    report = valid_report()
    report["query"]["device"] = "cuda"
    verdict = evaluate_report(report)
    assert verdict["passed"] is False
    assert verdict["gates"]["query_cpu"] is False


def test_tool_failure_fails_named_gate():
    report = valid_report()
    report["tools"]["passed"] = False
    verdict = evaluate_report(report)
    assert verdict["passed"] is False
    assert verdict["gates"]["tool_roundtrip"] is False


def test_cleanup_failure_fails_named_gate():
    report = valid_report()
    report["ingest"]["released"] = False
    verdict = evaluate_report(report)
    assert verdict["passed"] is False
    assert verdict["gates"]["ingest_release"] is False


def test_oom_fails_named_context_gate():
    report = valid_report()
    report["generation"]["oom"] = True
    verdict = evaluate_report(report)
    assert verdict["passed"] is False
    assert verdict["gates"]["context_fit"] is False


def test_context_without_observed_ollama_vram_fails_fit_gate():
    report = valid_report()
    report["generation"]["ollama_vram_peak_bytes"] = 0
    verdict = evaluate_report(report)
    assert verdict["passed"] is False
    assert verdict["gates"]["context_fit"] is False


def test_local_service_failure_fails_named_gate():
    report = valid_report()
    report["local_service"] = False
    verdict = evaluate_report(report)
    assert verdict["passed"] is False
    assert verdict["gates"]["local_service"] is False


def test_missing_prechange_baseline_fails_named_gate():
    report = valid_report()
    report["baseline"]["before"] = False
    verdict = evaluate_report(report)
    assert verdict["passed"] is False
    assert verdict["gates"]["baseline_comparison"] is False


def test_failed_ollama_unload_skips_gpu_ingest():
    assert _should_run_ingest({"unload": "failed: timeout"}) is False


def test_confirmed_ollama_unload_allows_gpu_ingest():
    assert _should_run_ingest(
        {"unload": "all Ollama models unloaded and confirmed"}
    ) is True


def test_generation_unload_failure_fails_named_gate():
    report = valid_report()
    report["generation"]["unload"] = "failed: timeout"
    verdict = evaluate_report(report)
    assert verdict["passed"] is False
    assert verdict["gates"]["generation_unload"] is False


def test_wrong_model_quantization_fails_identity_gate():
    report = valid_report()
    report["model"]["quantization"] = "Q4_0"
    verdict = evaluate_report(report)
    assert verdict["passed"] is False
    assert verdict["gates"]["model_identity"] is False


def test_coordinator_smoke_without_tui_evidence_fails_workload_gate():
    report = valid_report()
    report["workload_evidence"]["tui_dispatch_verified"] = False
    verdict = evaluate_report(report)
    assert verdict["passed"] is False
    assert verdict["gates"]["workload_serialization"] is False


def test_tui_probe_collects_serialization_evidence(monkeypatch):
    expected = {
        "tui_dispatch_verified": True,
        "cancellation_cleanup_verified": True,
        "admission_held_through_unload": True,
    }

    class Result:
        returncode = 0
        stdout = json.dumps(expected)
        stderr = ""

    monkeypatch.setattr("scripts.validate_foundation.subprocess.run", lambda *a, **k: Result())

    assert _run_tui_workload_probe() == expected


def test_baseline_without_reviewed_claim_and_retrieval_metrics_fails_gate():
    report = valid_report()
    report["baseline"].pop("after_review")
    verdict = evaluate_report(report)
    assert verdict["passed"] is False
    assert verdict["gates"]["baseline_comparison"] is False


def test_injected_ingest_failure_drops_all_local_model_owners(monkeypatch):
    class FakeModel:
        def encode(self, *_args, **_kwargs):
            return []

    fake_model = FakeModel()
    model_ref = weakref.ref(fake_model)
    monkeypatch.setattr(db, "_INGEST_MODEL", fake_model)
    del fake_model

    assert _probe_ingest_failure_cleanup("sample") is True
    assert db._INGEST_MODEL is None
    assert model_ref() is None


def test_baseline_requires_matching_book_labels_and_runtime_identity(tmp_path):
    path = tmp_path / "before.json"
    path.write_text(
        json.dumps(
            {
                "questions": [
                    {"kind": kind, "answer": "an answer"}
                    for kind, _, _ in __import__(
                        "scripts.validate_foundation", fromlist=["BASELINE_CASES"]
                    ).BASELINE_CASES
                ]
            }
        ),
        encoding="utf-8",
    )
    assert _valid_before_baseline(path, "sicp") is False
