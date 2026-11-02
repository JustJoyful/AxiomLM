import pytest

from src.study_state import SectionCatalog, StudyStore
from src.ui.stream import SourceRef, TurnRecord
from tests.test_study_state import section


@pytest.mark.asyncio
async def test_clear_keeps_session_new_replaces_it(app_factory):
    app = app_factory()
    async with app.run_test():
        original = app._study_session_id
        await app.handle_command("/clear", [])
        assert app._study_session_id == original
        await app.handle_command("/new", [])
        assert app._study_session_id != original


@pytest.mark.asyncio
async def test_practice_and_coverage_list_sections_without_visits(
    app_factory, monkeypatch, tmp_path
):
    from src import study_state

    sec = section()
    monkeypatch.setattr(
        study_state, "load_sections", lambda book: SectionCatalog(book, "v1", (sec,), 2)
    )
    app = app_factory()
    app._study_store = StudyStore(tmp_path / "study.sqlite3")
    async with app.run_test():
        app.active_collection_name = "sicp"
        await app.handle_command("/practice", [])
        await app.handle_command("/coverage", [])
        assert any("1." in text and "Intro" in text for text in app.notices)
        assert any("Study coverage, not mastery." in text for text in app.notices)
        assert any("Not yet asked about" in text for text in app.notices)
        assert app._study_store.counts([sec])[sec.section_id] == 0


@pytest.mark.asyncio
async def test_complete_qa_records_local_source_sections_once(
    app_factory, monkeypatch, tmp_path
):
    from src import study_state

    sec = section()
    monkeypatch.setattr(
        study_state, "load_sections", lambda book: SectionCatalog(book, "v1", (sec,), 0)
    )
    app = app_factory()
    app._study_store = StudyStore(tmp_path / "state")
    record = TurnRecord("r", "q", "sicp", "m", phase="DONE")
    record.index_version = "v1"
    record.sources = [SourceRef("sicp", source_id="a")]
    async with app.run_test():
        await app._record_study_observation(record)
        await app._record_study_observation(record)
        assert app._study_store.counts([sec])[sec.section_id] == 1
        record.request_id = "failed"
        record.phase = "ERROR"
        await app._record_study_observation(record)
        assert app._study_store.counts([sec])[sec.section_id] == 1


@pytest.mark.asyncio
async def test_practice_sources_can_be_expanded_and_copied(app_factory):
    from src.ui.stream import StudyStream, TurnView

    app = app_factory()
    record = TurnRecord("practice", "Practice", "sicp", "m", phase="PRACTICE")
    record.set_answer("Why? [sources 1]")
    record.sources = [
        SourceRef("sicp", physical_page=2, chapter="1", section="Intro", source_id="a")
    ]
    async with app.run_test():
        view = TurnView(record)
        app.turns.append(view)
        await app.query_one(StudyStream).add_turn(view)
        await view.finalize("PRACTICE")
        await app.handle_command("/sources", [])
        assert "Intro" in str(view.source_line.render())
        assert not any("no completed answer" in line for line in app.notices)


@pytest.mark.asyncio
async def test_practice_history_can_be_trimmed(app_factory, monkeypatch):
    from src.ui import stream

    monkeypatch.setattr(stream, "UI_SCROLLBACK_BLOCKS", 1)
    app = app_factory()
    async with app.run_test():
        surface = app.query_one(stream.StudyStream)
        for number in range(3):
            record = TurnRecord(str(number), "Practice", "sicp", "m", phase="PRACTICE")
            await surface.add_turn(stream.TurnView(record))
        assert len(surface.query(stream.TurnView)) == 1


def test_practice_coverage_snapshot(app_factory, snap_compare, monkeypatch, tmp_path):
    from src import study_state

    sec = section()
    monkeypatch.setattr(
        study_state, "load_sections", lambda book: SectionCatalog(book, "v1", (sec,), 2)
    )
    app = app_factory()
    app._study_store = StudyStore(tmp_path / "state")

    async def before(pilot):
        await app.workers.wait_for_complete()
        app._hide_home()
        app.active_collection_name = "sicp"
        await app.handle_command("/coverage", [])
        await pilot.pause()

    assert snap_compare(app, terminal_size=(100, 30), run_before=before)
