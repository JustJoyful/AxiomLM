from src.ui.stream import SourceRef, TurnRecord, TurnView


def test_copy_payload_includes_unsupported_list():
    record = TurnRecord("r", "question", "book", "m")
    record.set_answer("An unsupported factual sentence remains.")
    record.unsupported = ["An unsupported factual sentence remains."]
    assert "Not supported by the retrieved evidence" in record.copy_payload()


async def test_tui_shows_unsupported_block_and_marks_supporting_sources(app_factory):
    app = app_factory()
    from src.ui.stream import StudyStream

    record = TurnRecord("r", "Explain environments", "sicp", "m", phase="DONE")
    record.set_answer("Environments bind names to values.")
    record.unsupported = [record.raw_answer]
    record.citation_status = "checked"
    record.sources = [
        SourceRef("sicp", 3, supports_claim=True),
        SourceRef("sicp", 4, supports_claim=False),
    ]
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        app._hide_home()
        view = TurnView(record)
        await app.query_one(StudyStream).add_turn(view)
        await view.finalize()
        view.show_sources()
        await pilot.pause()
        assert "Not supported by the retrieved evidence" in str(
            view.unsupported_line.render()
        )
        assert "p. 3" in str(view.source_line.render()) and "p. 4" in str(
            view.source_line.render()
        )
        assert "supports a claim" in str(view.source_line.render())


async def test_check_timeout_marks_unchecked(monkeypatch, app_factory):
    app = app_factory()
    import asyncio
    from src import config
    from src import citation_check

    record = TurnRecord("r", "question", "book", "m")
    monkeypatch.setattr(config, "CITATION_CHECK_ENABLED", True)

    async def interrupted(*args, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(citation_check, "run_check", interrupted)
    import pytest

    with pytest.raises(asyncio.CancelledError):
        await app._apply_citation_check(record, "An answer with enough words.", [])
    assert record.citation_status == "unchecked"


def test_unsupported_claim_snapshot(app_factory, snap_compare):
    app = app_factory()

    async def before(pilot):
        from src.ui.stream import StudyStream

        await app.workers.wait_for_complete()
        app._hide_home()
        record = TurnRecord("r", "Explain environments", "sicp", "m", phase="DONE")
        record.set_answer("Environments bind names to values.")
        record.apply_citations(
            {
                "status": "checked",
                "answer": record.raw_answer,
                "claims": [{"text": record.raw_answer, "supported": False}],
                "supporting_source_ids": [],
            }
        )
        view = TurnView(record)
        await app.query_one(StudyStream).add_turn(view)
        await view.finalize()

    assert snap_compare(app, terminal_size=(100, 30), run_before=before)


async def test_phase2_incomplete_answer_skips_check(app_factory, monkeypatch):
    import asyncio
    from src import config, tui
    from src import citation_check

    app = app_factory()
    app._discover_books = lambda: {"sicp": 1}
    monkeypatch.setattr(config, "CITATION_CHECK_ENABLED", True)
    monkeypatch.setattr(app, "_prepare_query", lambda *args: ("prompt", [], [], []))

    async def forbidden(*args, **kwargs):
        raise AssertionError("incomplete answer must not be checked")

    monkeypatch.setattr(citation_check, "run_check", forbidden)

    class Client:
        async def astream_generate(self, *args, **kwargs):
            yield {
                "response": "A partial answer with several words.",
                "done": True,
                "done_reason": "length",
            }

        def unload_and_wait(self):
            pass

    monkeypatch.setattr(tui, "OllamaClient", Client)
    async with app.run_test():
        await app.workers.wait_for_complete()
        await app._submit_question("Explain environments", "sicp", app.selected_model)
        for _ in range(300):
            if app._active_query_request_id is None:
                break
            await asyncio.sleep(0.01)
        record = app.turns[-1].record
        assert (
            record.citation_status == "skipped"
            and record.citation_reason == "incomplete"
        )
        assert record.phase != "DONE"


def test_zero_claims_do_not_judge_retrieved_sources():
    record = TurnRecord("r", "question", "book", "m")
    record.sources = [SourceRef("book", 1, source_id="s")]
    record.apply_citations(
        {"status": "checked", "claims": [], "supporting_source_ids": []}
    )
    assert record.sources[0].supports_claim is None
