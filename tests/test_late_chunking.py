import pytest
import torch

from src import late_chunking as late
from src.index_metadata import SourceSegment
from tests.test_phase2_probe import Tokenizer


def segment(text, page=1, chapter="one", section="a"):
    return SourceSegment(
        text, "book", (page,), chapter, section, None, f"page_{page-1:04d}.md"
    )


def test_chunks_keep_exact_source_and_markers_with_large_paragraph_and_fences():
    text = "```python\n" + "x" * 8200 + "\n```\n\n$$a=b$$\n"
    blocks = late.split_context_blocks(
        [segment(text)], Tokenizer(), max_tokens=8192, chunk_tokens=300
    )
    assert "".join(c for b in blocks for c in b.chunk_texts) == text
    assert all(
        b.token_count <= 8192 and b.token_count == len(b.token_ids) for b in blocks
    )
    assert all(len(t) <= 300 for b in blocks for t in b.chunk_texts)
    assert all(b.token_ids[eos] == 9 for b in blocks for eos in b.chunk_eos_positions)
    assert blocks == late.split_context_blocks([segment(text)], Tokenizer(), 8192, 300)


def test_chunk_crosses_pages_but_blocks_stop_at_chapters():
    blocks = late.split_context_blocks(
        [segment("abc", 1), segment("def", 2), segment("new", 3, "two")],
        Tokenizer(),
        100,
        30,
    )
    assert len(blocks) == 2 and blocks[0].chunk_texts == ("abcdef",)
    assert {p for s in blocks[0].chunk_segments[0] for p in s.source_pages} == {1, 2}


def test_mean_pooling_uses_only_content_and_returns_exact_provenance(monkeypatch):
    block = late.split_context_blocks([segment("abc")], Tokenizer(), 50, 10)[0]

    def states(model, ids):
        h = torch.zeros(len(ids), 768)
        h[:, 0] = 1
        h[block.chunk_spans[0][0] : block.chunk_spans[0][1], 1] = 2
        return h

    monkeypatch.setattr(late, "token_states", states)
    row = late.embed_context_block(block, object(), "mean")[0]
    assert row.text == "abc" and row.metadata["physical_pages"] == "[1]"
    assert row.embedding[:2] == pytest.approx([1 / 5**0.5, 2 / 5**0.5])
    with pytest.raises(ValueError):
        late.embed_context_block(block, object(), "last")


def test_8192_includes_document_prefix_and_eos():
    b = late.split_context_blocks([segment("x" * 8181)], Tokenizer(), 8192, 8181)
    assert b[0].token_count == 8192
    with pytest.raises(ValueError):
        late.split_context_blocks([segment("x" * 8182)], Tokenizer(), 8192, 8182)


def test_full_sicp_geometry_equals_frozen_probe():
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    from transformers import AutoTokenizer

    from scripts.probe_phase2_geometry import geometry
    from src import config
    from src.index_metadata import source_segments

    if not (config.CHECKPOINTS / "sicp").exists():
        pytest.skip("SICP artifacts unavailable")
    try:
        path = snapshot_download(
            config.EMBED_MODEL, revision=config.EMBED_REVISION, local_files_only=True
        )
    except LocalEntryNotFoundError:
        pytest.skip("Pinned tokenizer unavailable")
    tokenizer = AutoTokenizer.from_pretrained(
        path, local_files_only=True, trust_remote_code=True
    )
    pages = [
        (int(p.stem.split("_")[1]) + 1, p.read_text())
        for p in sorted((config.CHECKPOINTS / "sicp").glob("page_*.md"))
    ]
    expected, old_blocks = geometry(pages, tokenizer, 300, 8192)
    actual = late.split_context_blocks(source_segments("sicp"), tokenizer)
    assert [t for b in actual for t in b.chunk_texts] == [c["text"] for c in expected]
    assert [list(b.token_ids) for b in actual] == [b["input_ids"] for b in old_blocks]
