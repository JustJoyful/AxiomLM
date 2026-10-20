import json
from pathlib import Path

import pytest

from scripts import evaluate_phase2_baseline as baseline
from src import config

LABELS = config.ROOT / "tests/fixtures/phase2/sicp_context_labels.json"


def test_frozen_labels_cover_eight_independent_cases():
    labels = baseline.load_labels(LABELS)
    assert len(labels["cases"]) == 8
    followup = next(row for row in labels["cases"] if row["kind"] == "followup")
    assert [turn["role"] for turn in followup["history"]] == ["user", "assistant"]
    assert all(row["relevant_pages"] for row in labels["cases"] if row["answerable"])


@pytest.mark.parametrize(
    "field,value",
    [("relevant_pages", [0]), ("answerable", "yes"), ("expected_claims", [])],
)
def test_invalid_gold_labels_are_rejected(tmp_path, field, value):
    labels = json.loads(LABELS.read_text())
    labels["cases"][0][field] = value
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(labels))
    with pytest.raises(ValueError):
        baseline.load_labels(path)


def test_claim_evidence_cannot_escape_labeled_pages(tmp_path):
    labels = json.loads(LABELS.read_text())
    labels["cases"][0]["expected_claims"][0]["source_pages"] = [999]
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(labels))
    with pytest.raises(ValueError):
        baseline.load_labels(path)


def test_metrics_deduplicate_pages_and_exclude_unanswerable():
    cases = [
        {"case_id": "a", "answerable": True, "relevant_pages": [1, 2]},
        {"case_id": "b", "answerable": True, "relevant_pages": [3]},
        {"case_id": "c", "answerable": False, "relevant_pages": []},
    ]
    rows = [
        {"case_id": "a", "physical_pages": [1, 1, 9]},
        {"case_id": "b", "physical_pages": []},
        {"case_id": "c", "physical_pages": [5]},
    ]
    result = baseline.retrieval_metrics(cases, rows)
    assert result["macro_page_precision"] == 0.25
    assert result["macro_page_recall"] == 0.25
    assert result["answerable_case_denominator"] == 2
    assert result["unanswerable_cases_returning_evidence"] == 1
    with pytest.raises(ValueError):
        baseline.retrieval_metrics(cases, rows[:-1])


def test_source_manifest_and_hashes_detect_drift(tmp_path):
    import hashlib

    text = b"original page"
    (tmp_path / "checkpoints/book").mkdir(parents=True)
    (tmp_path / "book").mkdir()
    (tmp_path / "checkpoints/book/page_0000.md").write_bytes(text)
    (tmp_path / "book/page_manifest.json").write_text(
        json.dumps(
            {
                "book_stem": "book",
                "pages": [
                    {
                        "page_idx": 0,
                        "physical_page": 1,
                        "checkpoint_file": "page_0000.md",
                    }
                ],
            }
        )
    )
    labels = {
        "book_stem": "book",
        "source_pages": {"1": hashlib.sha256(text).hexdigest()},
    }
    baseline.verify_sources(labels, tmp_path)
    (tmp_path / "checkpoints/book/page_0000.md").write_text("modified")
    with pytest.raises(ValueError, match="changed"):
        baseline.verify_sources(labels, tmp_path)


def test_snapshot_is_separate_and_active_files_are_untouched(tmp_path):
    source = tmp_path / "active"
    source.mkdir()
    (source / "index").write_bytes(b"persisted index")
    before = baseline.tree_identity(source)
    snapshot = tmp_path / "snapshot"
    baseline.snapshot_store(source, snapshot)
    (snapshot / "index").write_bytes(b"local query side effect")
    assert baseline.tree_identity(source) == before


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt])
def test_generation_failure_saves_partial_report_and_keeps_active_index(
    tmp_path, monkeypatch, failure_type
):
    from types import SimpleNamespace

    import chromadb
    import huggingface_hub

    from scripts import capture_runtime_manifest
    from src import db, llm, tui

    active = tmp_path / "active"
    active.mkdir()
    (active / "index").write_bytes(b"original index")
    before = baseline.tree_identity(active)
    parsed = tmp_path / "parsed/sicp"
    parsed.mkdir(parents=True)
    (parsed / "page_manifest.json").write_text("{}")
    output = tmp_path / "report.json"
    monkeypatch.setattr(config, "VECTOR_STORE", active)
    monkeypatch.setattr(config, "PARSED_MD", parsed.parent)
    monkeypatch.setattr(config, "PHASE2_EVALUATION_PATH", output)
    monkeypatch.setattr(baseline, "verify_sources", lambda *_: None)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **_: str(tmp_path))
    monkeypatch.setattr(
        capture_runtime_manifest,
        "capture_manifest",
        lambda: {"model": config.OLLAMA_MODEL},
    )
    collection = SimpleNamespace(
        get=lambda **_: {"ids": ["id"]}, count=lambda: 1, metadata={}
    )
    monkeypatch.setattr(
        chromadb,
        "PersistentClient",
        lambda path: SimpleNamespace(
            path=Path(path), get_collection=lambda **_: collection
        ),
    )
    monkeypatch.setattr(db, "embed_query", lambda _: [1.0])
    monkeypatch.setattr(
        db,
        "get_query_model",
        lambda: SimpleNamespace(
            device="cpu",
            prompts={"document": "Document: "},
            default_prompt_name="document",
        ),
    )
    released = []

    class App:
        def _format_history(self):
            return ""

        def _prepare_legacy_query(self, question, *_args):
            assert db._CLIENT.path != active
            tui.embed_query(question)
            return "prompt", [{"physical_page": 1}], [1], ["id"]

        def _try_acquire_workload(self, kind):
            return kind

        def _release_workload(self, token):
            released.append(token)

    class Client:
        @staticmethod
        def _options():
            return {}

        def generate(self, *_args, **_kwargs):
            raise failure_type("injected local model failure")

        def unload_and_wait(self):
            released.append("unloaded")

    monkeypatch.setattr(tui, "AxiomLMApp", App)
    monkeypatch.setattr(llm, "OllamaClient", Client)
    with pytest.raises(failure_type, match="injected local model failure"):
        baseline.run()
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert len(report["rows"]) == 1
    assert report["active_store_unchanged"] is True
    assert baseline.tree_identity(active) == before
    assert released == ["unloaded", "generation"]
    with pytest.raises(FileExistsError):
        baseline.run()
