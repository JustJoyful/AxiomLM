import json
from dataclasses import FrozenInstanceError

import pytest

from src import index_metadata as m


def test_heading_inheritance_clears_children_and_preserves_sources(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(m.config, "CHECKPOINTS", tmp_path / "checkpoints")
    monkeypatch.setattr(m.config, "PARSED_MD", tmp_path)
    folder = tmp_path / "checkpoints" / "book"
    folder.mkdir(parents=True)
    texts = [
        "before\n\n# Chapter One\n\n## Section A\n\n### Detail\n\nbody\n",
        "",
        "# Chapter Two\n\ncontinued\n",
    ]
    for i, text in enumerate(texts):
        (folder / f"page_{i:04d}.md").write_text(text)
    rows = m.source_segments("book")
    assert "".join(r.text for r in rows) == "".join(texts)
    assert rows[0].chapter == "Unknown Chapter"
    assert rows[-1].chapter == "Chapter Two"
    assert rows[-1].section == "Unknown Section" and rows[-1].subsection is None
    assert rows[-1].source_pages == (3,)
    assert rows[-1].source_path == "page_0002.md"
    assert rows == m.source_segments("book")
    with pytest.raises(FrozenInstanceError):
        rows[0].text = "changed"


@pytest.mark.parametrize(
    "manifest",
    [
        None,
        {"pages": []},
        {
            "pages": [
                {"page_idx": 0, "physical_page": 1, "checkpoint_file": "page_0000.md"}
            ]
        },
    ],
)
def test_manifest_missing_empty_and_zero_index(tmp_path, monkeypatch, manifest):
    monkeypatch.setattr(m.config, "CHECKPOINTS", tmp_path / "checkpoints")
    monkeypatch.setattr(m.config, "PARSED_MD", tmp_path)
    folder = tmp_path / "checkpoints" / "book"
    folder.mkdir(parents=True)
    (folder / "page_0000.md").write_text("text")
    if manifest is not None:
        (tmp_path / "book").mkdir()
        (tmp_path / "book" / "page_manifest.json").write_text(json.dumps(manifest))
    assert m.source_segments("book")[0].source_pages == (1,)


def test_cross_page_metadata_and_distinct_source_identity():
    segment = m.SourceSegment(
        "text", "book", (2, 4), "Chapter", "Section", None, "page_0001.md"
    )
    meta = m.normalize_metadata(segment)
    assert meta["physical_page_start"] == 2 and meta["physical_page_end"] == 4
    assert json.loads(meta["physical_pages"]) == [2, 4] and meta["physical_page"] == 2
    kwargs = {
        "book_stem": "book",
        "source_paths": ["page_0001.md"],
        "source_pages": [2, 4],
        "chapter": "Chapter",
        "section": "Section",
        "ordinal": 0,
        "text": "text",
    }
    assert m.stable_chunk_id(**kwargs) == m.stable_chunk_id(**kwargs)
    assert (
        len(
            {
                m.stable_chunk_id(**kwargs),
                m.stable_chunk_id(**dict(kwargs, ordinal=1)),
                m.stable_chunk_id(**dict(kwargs, book_stem="other")),
            }
        )
        == 3
    )


def test_book_path_escape_and_invalid_page_rejected():
    with pytest.raises(ValueError):
        m.source_segments("../escape")
    with pytest.raises(ValueError):
        m.normalize_metadata(m.SourceSegment("x", "book", (0,), "c", "s", None, "p"))


def test_adjacent_headings_and_fenced_heading_text_keep_exact_offsets(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(m.config, "CHECKPOINTS", tmp_path)
    monkeypatch.setattr(m.config, "PARSED_MD", tmp_path)
    folder = tmp_path / "book"
    folder.mkdir()
    text = "# Parent\n## Child\n### Detail\nbody\n```md\n# literal code\n```\n# New Parent\nnew body\n"
    (folder / "page_0000.md").write_text(text)
    rows = m.source_segments("book")
    assert "".join(r.text for r in rows) == text
    assert rows[-1].chapter == "New Parent" and rows[-1].section == "Unknown Section"
    literal = next(r for r in rows if "# literal code" in r.text)
    assert (
        literal.chapter == "Parent"
        and literal.section == "Child"
        and literal.subsection == "Detail"
    )
    assert all(
        text[r.source_start : r.source_start + len(r.text)] == r.text for r in rows
    )
