import json
import sqlite3

import pytest

from src import study_state as s


def section(version="v1", chapter="1", book="sicp"):
    return s.SectionRef(
        s.section_identity(book, version, chapter, "Intro"),
        book,
        version,
        chapter,
        "Intro",
        (1, 2),
        ("a",),
    )


def test_duplicate_turn_is_idempotent_and_versions_are_separate(tmp_path):
    first, rebuilt = section(), section("v2")
    path = tmp_path / "study.sqlite3"
    store = s.StudyStore(path)
    store.record_question("session-a", "turn-1", [first, first])
    store.record_question("session-a", "turn-1", [first])
    reopened = s.StudyStore(path)
    assert reopened.counts([first], session_id="session-a")[first.section_id] == 1
    assert reopened.counts([first], session_id="session-b")[first.section_id] == 0
    assert reopened.counts([rebuilt])[rebuilt.section_id] == 0
    assert reopened.counts([section(chapter="2"), section(book="other")]) == {
        section(chapter="2").section_id: 0,
        section(book="other").section_id: 0,
    }


def test_corrupt_state_is_not_reset_or_overwritten(tmp_path):
    path = tmp_path / "state"
    path.write_bytes(b"corrupt original")
    with pytest.raises(sqlite3.DatabaseError):
        s.StudyStore(path).counts([section()])
    assert path.read_bytes() == b"corrupt original"


def test_unknown_schema_is_not_migrated(tmp_path):
    path = tmp_path / "state"
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version=99")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="schema"):
        s.StudyStore(path).counts([section()])
    assert path.read_bytes() == before


def catalog_root(tmp_path):
    from src.index_store import pair_path

    version = "a" * 64
    (tmp_path / "active_indexes.json").write_text(
        json.dumps({"schema": 2, "books": {"sicp": {"version_id": version}}})
    )
    path = pair_path(tmp_path, "sicp", version)
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE chunks(chunk_id, document, metadata_json)")
        for cid, chapter, title, pages in [
            ("a", "1", "Intro", [1, 2]),
            ("b", "1", "Intro", [3]),
            ("c", "2", "Intro", [4]),
            ("d", "Unknown Chapter", "Intro", [5]),
        ]:
            db.execute(
                "INSERT INTO chunks VALUES(?,?,?)",
                (
                    cid,
                    "passage " + cid,
                    json.dumps(
                        {
                            "book_stem": "sicp",
                            "chapter": chapter,
                            "section": title,
                            "physical_pages": json.dumps(pages),
                        }
                    ),
                ),
            )
    return path


def test_catalog_groups_stable_chunks_and_preserves_pages(tmp_path):
    path = catalog_root(tmp_path)
    before = path.read_bytes()
    catalog = s.load_sections("sicp", root=tmp_path)
    assert len(catalog.sections) == 2 and catalog.unassigned_chunks == 1
    assert catalog.sections[0].physical_pages == (1, 2, 3)
    assert catalog.sections[0].chunk_ids == ("a", "b")
    assert catalog.sections[0].section_id != catalog.sections[1].section_id
    assert path.read_bytes() == before


def test_legacy_catalog_is_unavailable(tmp_path):
    with pytest.raises(ValueError, match="v2"):
        s.load_sections("sicp", root=tmp_path)
