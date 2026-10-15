import pytest

from src.ui.stream import SourceRef, StudyStream, TurnRecord, TurnView
from src.ui.themes import THEMES


async def prepare_scene(app, scene):
    await app.workers.wait_for_complete()
    if scene == "home":
        return
    app._hide_home()
    app.active_collection_name = "linear_algebra"
    app._book_counts["linear_algebra"] = 640
    app._status = "● ready"
    app._refresh_condition()
    app._set_header()
    if scene == "commands":
        await app.action_commands()
        return
    if scene in {"settings", "theme"}:
        await app.handle_command("/" + scene, [])
        return
    question = (
        "/parse ~/books/linear_algebra.pdf"
        if scene == "ingest"
        else "Explain the relation between A and B."
    )
    record = TurnRecord(
        "snapshot", question, "linear_algebra", app.selected_model, phase="DONE"
    )
    if scene in {"chat", "narrow"}:
        record.set_answer(
            '## Conditional probability\n\nP(A | B) = P(A ∩ B) / P(B).\n\n```python\nanswer = "λ"\n```\n\n| Event | Meaning |\n|---|---|\n| A | First event |'
        )
        record.sources = [SourceRef("linear_algebra", 112, "Chapter 4")]
    elif scene == "midstream":
        record.set_answer('- First item\n- Second item\n\n```python\nx = "λ"')
        record.phase = "STREAMING"
    else:
        record.phase = "INGEST" if scene == "ingest" else "ERROR"
    view = TurnView(record)
    app.turns.append(view)
    await app.query_one(StudyStream).add_turn(view)
    if scene == "ingest":
        view.step_line.update(
            "↳ parse ✓ pymupdf4llm\n↳ chunk ✓ 640 chunks\n↳ embed ━━━━━━━━▌─────── 2/4 50%\n↳ index ○"
        )
    elif scene == "failure":
        view.step_line.add_class("failed")
        view.step_line.update(
            "✗ can't reach Ollama at localhost:11434 · start it, then /retry"
        )
    else:
        view.collapse_sources()
        if scene == "midstream":
            view.step_line.update(
                "↳ search_local_index ✓ 6 chunks · 0.4s\n↳ thinking ✓ 1.1s"
            )


@pytest.mark.parametrize(
    "scene",
    [
        "home",
        "chat",
        "midstream",
        "failure",
        "ingest",
        "commands",
        "settings",
        "theme",
        "narrow",
    ],
)
def test_scenes_match_reviewed_snapshots(app_factory, snap_compare, scene):
    app = app_factory()

    async def before(pilot):
        await prepare_scene(app, scene)
        if scene == "midstream":
            await pilot.pause()
            app.turns[-1].tail.display = True
            app._position_tail(app.turns[-1])

    assert snap_compare(
        app,
        terminal_size=(40, 20) if scene == "narrow" else (100, 30),
        run_before=before,
    )


@pytest.mark.parametrize("theme", ["terminal", "mono", *THEMES])
def test_builtin_theme_snapshot(app_factory, snap_compare, theme):
    app = app_factory()
    app.settings.theme = theme
    assert snap_compare(app, terminal_size=(80, 24))


async def test_theme_layout_is_identical_without_color(app_factory):
    frames = []
    for theme in ["terminal", "mono", *THEMES]:
        app = app_factory()
        app.settings.theme = theme
        async with app.run_test(size=(80, 30)) as pilot:
            await prepare_scene(app, "chat")
            await pilot.pause()
            frames.append(
                "\n".join(
                    strip.text for strip in app.screen._compositor.render_strips()
                )
            )
    import difflib

    for theme, frame in zip(["terminal", "mono", *THEMES], frames):
        assert frame == frames[0], (
            theme
            + "\n"
            + "\n".join(
                difflib.unified_diff(frames[0].splitlines(), frame.splitlines())
            )
        )
