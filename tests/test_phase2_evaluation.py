import json
from pathlib import Path

import pytest

from scripts import validate_phase2_context as evaluation
from scripts.evaluate_phase2_baseline import digest
from src import config


def test_requires_separate_book_labels_and_frozen_baseline_identity(
    tmp_path, monkeypatch
):
    labels = Path("tests/fixtures/phase2/sicp_context_labels.json")
    local = tmp_path / labels.name
    local.write_bytes(labels.read_bytes())
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "status": "complete",
                "book_stem": "sicp",
                "dataset_sha256": digest(local),
                "runtime": {
                    "embedding": config.EMBED_MODEL,
                    "embedding_revision": config.EMBED_REVISION,
                },
            }
        )
    )
    monkeypatch.setattr(config, "PHASE2_LABELS_DIR", tmp_path)
    monkeypatch.setattr(config, "PHASE2_EVALUATION_PATH", baseline)
    evaluation.load_inputs("sicp")
    with pytest.raises(FileNotFoundError):
        evaluation.load_inputs("other")
    local.write_text(local.read_text() + " ")
    with pytest.raises(ValueError, match="identity"):
        evaluation.load_inputs("sicp")


def test_after_path_never_overwrites_baseline(tmp_path, monkeypatch):
    path = tmp_path / "baseline.json"
    path.write_text("original")
    monkeypatch.setattr(config, "PHASE2_EVALUATION_PATH", path)
    monkeypatch.setattr(config, "PHASE2_AFTER_EVALUATION_PATH", path)
    with pytest.raises(ValueError):
        evaluation.save_report({"status": "complete"})
    assert path.read_text() == "original"


def test_nonstaging_evaluation_rejected_before_any_index_work():
    with pytest.raises(ValueError, match="staging"):
        evaluation.validate("sicp", staging=False)


def test_staging_build_cannot_change_active_manifest_or_baseline(tmp_path, monkeypatch):
    from src import index_store
    from src.late_chunking import EmbeddedChunk

    active = tmp_path / "active"
    active.mkdir()
    (active / "active_indexes.json").write_text("original active manifest")
    parsed = tmp_path / "parsed" / "sicp"
    parsed.mkdir(parents=True)
    manifest = parsed / "page_manifest.json"
    manifest.write_text("{}")
    baseline = tmp_path / "baseline.json"
    baseline.write_text("original baseline")
    labels = tmp_path / "sicp_context_labels.json"
    labels.write_text("{}")
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "passed": True,
                "compatible_methods": ["mean"],
                "revision": config.EMBED_REVISION,
                "protocol": {"chunk_content_token_ceiling": 300},
            }
        )
    )
    monkeypatch.setattr(config, "VECTOR_STORE", active)
    monkeypatch.setattr(config, "PARSED_MD", parsed.parent)
    monkeypatch.setattr(config, "PHASE2_EVALUATION_PATH", baseline)
    monkeypatch.setattr(config, "PHASE2_AFTER_EVALUATION_PATH", tmp_path / "after.json")
    monkeypatch.setattr(config, "PHASE2_LABELS_DIR", tmp_path)
    monkeypatch.setattr(config, "PHASE2_GEOMETRY_REPORT_PATH", gate)
    monkeypatch.setattr(
        evaluation,
        "load_inputs",
        lambda book: (
            {"book_stem": book},
            {"source_manifest_sha256": digest(manifest)},
        ),
    )
    monkeypatch.setattr(evaluation, "verify_sources", lambda *_: None)

    def staged(book, labels, baseline, root, report):
        assert root != active
        row = EmbeddedChunk(
            "id", "text", {"book_stem": book, "physical_page": 1}, [1.0] + [0.0] * 767
        )
        index_store.build_index_version(book, [row], root=root / "versioned")

    monkeypatch.setattr(evaluation, "_run_staged", staged)
    result = evaluation.validate("sicp")
    assert result["preserved"] and result["status"] == "complete"
    assert (active / "active_indexes.json").read_text() == "original active manifest"
    assert baseline.read_text() == "original baseline"


def test_protected_hashes_follow_requested_book(tmp_path, monkeypatch):
    folder = tmp_path / "labels"
    folder.mkdir()
    (folder / "other_context_labels.json").write_text("other book")
    for attribute in ["VECTOR_STORE", "PARSED_MD"]:
        root = tmp_path / attribute
        root.mkdir()
        (root / "marker").write_text("unchanged")
        monkeypatch.setattr(config, attribute, root)
    for attribute in ["PHASE2_EVALUATION_PATH", "PHASE2_GEOMETRY_REPORT_PATH"]:
        path = tmp_path / attribute
        path.write_text("frozen")
        monkeypatch.setattr(config, attribute, path)
    monkeypatch.setattr(config, "PHASE2_LABELS_DIR", folder)
    assert evaluation.protected_inputs("other")["labels"] == digest(
        folder / "other_context_labels.json"
    )


def test_integrity_failure_returns_failed_result_for_cli(tmp_path, monkeypatch):
    """An audit failure cannot look like successful evaluation to automation."""
    parsed = tmp_path / "parsed" / "sicp"
    parsed.mkdir(parents=True)
    manifest = parsed / "page_manifest.json"
    manifest.write_text("{}")
    monkeypatch.setattr(config, "PARSED_MD", parsed.parent)
    monkeypatch.setattr(config, "PHASE2_AFTER_EVALUATION_PATH", tmp_path / "after.json")
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "passed": True,
                "compatible_methods": ["mean"],
                "revision": config.EMBED_REVISION,
                "protocol": {
                    "chunk_content_token_ceiling": config.RETRIEVAL_CHUNK_TOKENS
                },
            }
        )
    )
    monkeypatch.setattr(config, "PHASE2_GEOMETRY_REPORT_PATH", gate)
    monkeypatch.setattr(
        evaluation,
        "load_inputs",
        lambda book: ({}, {"source_manifest_sha256": digest(manifest)}),
    )
    monkeypatch.setattr(evaluation, "verify_sources", lambda *_: None)
    monkeypatch.setattr(
        evaluation, "protected_inputs", lambda book: {"baseline": "frozen"}
    )
    monkeypatch.setattr(
        evaluation,
        "digest",
        lambda path: digest(manifest) if Path(path) == manifest else "frozen",
    )

    def staged(*args):
        monkeypatch.setattr(
            evaluation, "protected_inputs", lambda book: {"baseline": "changed"}
        )

    monkeypatch.setattr(evaluation, "_run_staged", staged)
    with pytest.raises(RuntimeError, match="integrity"):
        evaluation.validate("sicp")
    result = json.loads((tmp_path / "after.json").read_text())
    assert result["status"] == "failed" and result["preserved"] is False
