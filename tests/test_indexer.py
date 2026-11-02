from src import index_metadata as m


def test_sicp_unnumbered_chapter_and_numbered_section_reset(tmp_path, monkeypatch):
    monkeypatch.setattr(m.config, "CHECKPOINTS", tmp_path)
    monkeypatch.setattr(m.config, "PARSED_MD", tmp_path)
    folder = tmp_path / "sicp"
    folder.mkdir()
    (folder / "page_0000.md").write_text(
        "## **1.3.4 Old**\n\ntext\n\n## **Building Abstractions with Data**\n\nnew\n\n## **2.1 Introduction**\n\nbody"
    )
    rows = m.source_segments("sicp")
    assert rows[-1].chapter == "2" and rows[-1].section == "2.1"
    assert rows[-3].chapter == "2" and rows[-3].section == "Unknown Section"


def test_contextual_ingest_failure_releases_all_model_owners(monkeypatch):
    import gc
    import weakref

    import pytest

    from src import db, indexer
    from tests.test_phase2_probe import Tokenizer

    class Model:
        def __getitem__(self, _):
            return self

        tokenizer = Tokenizer()

    model = Model()
    reference = weakref.ref(model)
    monkeypatch.setattr(db, "_INGEST_MODEL", model)
    del model
    monkeypatch.setattr(
        indexer,
        "source_segments",
        lambda _, **__: [
            m.SourceSegment("abc", "book", (1,), "c", "s", None, "page_0000.md")
        ],
    )
    monkeypatch.setattr(db, "get_ingest_model", lambda **_: db._INGEST_MODEL)

    def fail(block, model, pooling):
        raise RuntimeError("injected")

    monkeypatch.setattr(indexer, "embed_context_block", fail)
    with pytest.raises(RuntimeError):
        indexer.contextual_chunks("book")
    gc.collect()
    assert db._INGEST_MODEL is None and reference() is None


def _source_tree(tmp_path, monkeypatch, *, repaired_pages, report=True):
    import json

    from src import config

    monkeypatch.setattr(config, "CHECKPOINTS", tmp_path / "checkpoints")
    monkeypatch.setattr(config, "REPAIRED_MD", tmp_path / "repaired")
    checkpoints = tmp_path / "checkpoints" / "book"
    checkpoints.mkdir(parents=True)
    for i in range(2):
        (checkpoints / f"page_{i:04d}.md").write_text(f"original {i}\n")
    repaired = tmp_path / "repaired" / "book"
    repaired.mkdir(parents=True)
    for i in range(repaired_pages):
        (repaired / f"page_{i:04d}.md").write_text(f"repaired {i}\n")
    if report:
        (repaired / "repair_report.json").write_text(
            json.dumps({"book_stem": "book", "pages": 2})
        )
    return repaired


def _fake_chunks(monkeypatch, calls):
    from src import indexer
    from src.late_chunking import EmbeddedChunk

    def fake(book, *, allow_cpu_embeddings=False, pages_dir=None):
        calls.append(pages_dir)
        return [
            EmbeddedChunk(
                "id-one",
                "one",
                {"book_stem": book, "physical_page": 1, "physical_pages": "[1]"},
                [1.0] + [0.0] * 767,
            )
        ]

    monkeypatch.setattr(indexer, "contextual_chunks", fake)


def test_repaired_source_requires_matching_page_files(tmp_path, monkeypatch):
    import pytest

    from src import indexer

    _source_tree(tmp_path, monkeypatch, repaired_pages=1)
    _fake_chunks(monkeypatch, calls := [])
    with pytest.raises(ValueError, match="page files"):
        indexer.index_book("book", source="repaired", root=tmp_path / "store")
    assert calls == []


def test_repaired_source_requires_repair_report(tmp_path, monkeypatch):
    import pytest

    from src import indexer

    _source_tree(tmp_path, monkeypatch, repaired_pages=2, report=False)
    _fake_chunks(monkeypatch, calls := [])
    with pytest.raises(ValueError, match="repair_report.json"):
        indexer.index_book("book", source="repaired", root=tmp_path / "store")
    assert calls == []


def test_repaired_source_requires_report_describing_its_pages(tmp_path, monkeypatch):
    import json

    import pytest

    from src import indexer

    repaired = _source_tree(tmp_path, monkeypatch, repaired_pages=2)
    _fake_chunks(monkeypatch, calls := [])
    for report in (
        {},
        {"book_stem": "other", "pages": 2},
        {"book_stem": "book", "pages": 3},
    ):
        (repaired / "repair_report.json").write_text(json.dumps(report))
        with pytest.raises(ValueError, match="does not describe"):
            indexer.index_book("book", source="repaired", root=tmp_path / "store")
    assert calls == []


def test_index_manifest_records_source_and_report_hash(tmp_path, monkeypatch):
    import hashlib

    from src import index_store, indexer

    repaired = _source_tree(tmp_path, monkeypatch, repaired_pages=2)
    _fake_chunks(monkeypatch, calls := [])
    indexer.index_book("book", source="repaired", root=tmp_path / "store")
    entry = index_store.read_manifest(tmp_path / "store")["book"]
    assert calls == [repaired]
    assert entry["source"] == "repaired"
    assert (
        entry["repair_report_sha256"]
        == hashlib.sha256((repaired / "repair_report.json").read_bytes()).hexdigest()
    )
    assert index_store.resolve_index("book", root=tmp_path / "store").legacy is False


def test_default_source_is_checkpoints(tmp_path, monkeypatch):
    from src import index_store, indexer

    _source_tree(tmp_path, monkeypatch, repaired_pages=2)
    _fake_chunks(monkeypatch, calls := [])
    indexer.index_book("book", root=tmp_path / "store")
    entry = index_store.read_manifest(tmp_path / "store")["book"]
    assert calls == [None]
    assert entry["source"] == "checkpoints" and "repair_report_sha256" not in entry
    assert indexer._build_parser().parse_args(["book"]).source == "checkpoints"
    assert (
        indexer._build_parser().parse_args(["book", "--source", "repaired"]).source
        == "repaired"
    )
