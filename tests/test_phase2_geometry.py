import itertools

import pytest

from scripts import probe_phase2_geometry as geometry
from tests.test_phase2_probe import Tokenizer


def test_geometry_keeps_exact_source_once_and_chapter_boundaries():
    pages = [
        (1, "## **1.1 One**\n\nabc\n\ndef\n"),
        (2, "ghi\n\n" + "z" * 80),
        (3, "## **2.1 Two**\n\nend"),
    ]
    chunks, blocks = geometry.geometry(pages, Tokenizer(), 30, 90)
    for page, text in pages:
        ranges = [r for c in chunks for r in c["sources"] if r["page"] == page]
        assert "".join(text[r["start"] : r["stop"]] for r in ranges) == text
        assert ranges[0]["start"] == 0 and ranges[-1]["stop"] == len(text)
        assert all(a["stop"] == b["start"] for a, b in itertools.pairwise(ranges))
    assert all(len(c["text"]) <= 30 for c in chunks)
    assert all(len(b["input_ids"]) <= 90 for b in blocks)
    assert all(
        len({chunks[s["page"]]["section"] for s in b["spans"]}) == 1 for b in blocks
    )
    assert geometry.geometry(pages, Tokenizer(), 30, 90) == (chunks, blocks)


def test_shared_chunk_pages_rank_deduplicates_evidence_and_filters_intersection():
    import torch

    chunks = [{"pages": [1, 2]}, {"pages": [2]}, {"pages": [3]}]
    cases = {"cases": [{"case_id": "v", "question": "Explain physical PDF page 2."}]}
    rows = geometry.rank(
        torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]]),
        torch.tensor([[1.0, 0.0]]),
        chunks,
        cases,
    )
    assert rows[0]["physical_pages"] == [1, 2]
    assert rows[0]["chunk_ids"] == [0, 1]


def test_frozen_validation_rejects_label_drift(tmp_path):
    p = tmp_path / "labels.json"
    p.write_text("{}")
    with pytest.raises(ValueError, match="frozen"):
        geometry.check_protocol(p, {"label_sha256": "wrong"})


def test_unnumbered_sicp_chapter_title_starts_a_fresh_context():
    pages = [
        (1, "## **1.3.4 Last**\n\nold"),
        (2, "## **Building Abstractions with Data**\n\nnew"),
    ]
    chunks, blocks = geometry.geometry(pages, Tokenizer(), 80, 200)
    assert [c["chapter"] for c in chunks] == ["1", "2"]
    assert len(blocks) == 2


def test_serialization_check_preserves_punctuation_spaces_with_real_tokenizer():
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    from transformers import AutoTokenizer

    from src import config

    try:
        snapshot = snapshot_download(
            config.EMBED_MODEL, revision=config.EMBED_REVISION, local_files_only=True
        )
    except LocalEntryNotFoundError:
        pytest.skip("Pinned local tokenizer not installed")
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=True
    )
    text = "Table of contents . . .\n"
    chunks, blocks = geometry.geometry([(1, text)], tokenizer, 300, 8192)
    geometry.verify_serialization(chunks, blocks, tokenizer)


@pytest.mark.parametrize("prior_reason", [None, "original experiment failure"])
@pytest.mark.parametrize(
    "error", [OSError("audit read failed"), ValueError("empty store")]
)
def test_integrity_read_failure_persists_failed_gate(
    tmp_path, monkeypatch, error, prior_reason
):
    import json

    def fail(_):
        raise error

    monkeypatch.setattr(geometry, "tree_identity", fail)
    report = {"status": "complete", "passed": True, "blocked_reason": prior_reason}
    output = tmp_path / "report.json"
    geometry.finalize_report(
        report, output, {}, tmp_path / "labels", tmp_path / "protocol"
    )
    saved = json.loads(output.read_text())
    assert saved["status"] == "failed"
    assert saved["passed"] is False and saved["preserved"] is False
    assert str(error) in saved["integrity_error"]
    assert saved["blocked_reason"] == (prior_reason or saved["integrity_error"])
