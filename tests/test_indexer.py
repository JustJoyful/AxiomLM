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
        lambda _: [
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
