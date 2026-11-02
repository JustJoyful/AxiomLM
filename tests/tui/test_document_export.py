import pytest

from src.ui.stream import SourceRef, TurnRecord, TurnView


@pytest.mark.asyncio
async def test_export_n_captures_completed_turn_before_await(app_factory):
    app = app_factory()
    record = TurnRecord("old", "q", "book", "model", phase="DONE")
    record.set_answer("answer")
    record.sources = [SourceRef("book", 3)]
    record.citation_status = "unchecked"
    app.turns = [TurnView(record)]
    snapshot = app._export_input(1)
    record.set_answer("changed")
    app.turns.clear()
    assert snapshot.turn_id == "old" and snapshot.text.startswith("answer")
    assert "Citations not checked" in snapshot.notices
    assert snapshot.sources[0].physical_page == 3


@pytest.mark.asyncio
async def test_export_rejects_active_or_incomplete_turn(app_factory):
    app = app_factory()
    app.turns = [TurnView(TurnRecord("r", "q", "b", "m", phase="ERROR"))]
    with pytest.raises(ValueError):
        app._export_input(1)
    with pytest.raises(ValueError):
        app._export_input(0)


def test_export_result_snapshot(app_factory, snap_compare):
    app = app_factory()

    async def before(pilot):
        await app.workers.wait_for_complete()
        app._hide_home()
        await app.write_line(
            "○ export unavailable: /private/export/job/answer.txt · typst is not installed"
        )
        await pilot.pause()

    assert snap_compare(app, terminal_size=(100, 30), run_before=before)
