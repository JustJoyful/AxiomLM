import pytest

from src.ui.motion import MotionClock, render_progress


def test_progress_uses_observed_eighth_blocks():
    assert render_progress(1, 16, 8) == "▌───────"
    assert render_progress(8, 16, 8) == "━━━━────"
    assert render_progress(16, 16, 8) == "━━━━━━━━"
    assert render_progress(1, 0, 8) == "────────"


def test_shared_clock_deadlines_budget_and_idle_pause(monkeypatch):
    from src.ui import motion

    monkeypatch.setattr(motion.time, "monotonic", lambda: 1.0)
    monkeypatch.delenv("NO_MOTION", raising=False)
    monkeypatch.delenv("AXIOM_REDUCE_MOTION", raising=False)
    frames = []

    class Timer:
        def resume(self):
            pass

        def pause(self):
            pass

    class App:
        _terminal_focused = True
        active = True

        def set_interval(self, interval, callback, **kwargs):
            assert interval == pytest.approx(1 / 30)
            return Timer()

        def _motion_active(self):
            return self.active

        def _motion_frame(self, now, clock):
            frames.append((now, clock.spinner))

    app = App()
    clock = MotionClock(app)
    clock.start()
    clock.tick(1.0)
    clock.tick(1.01)
    clock.tick(1.08)
    assert frames == [(1.0, "⠋"), (1.08, "⠙")]
    assert clock.elapsed(1.0, 1.49) == "0.4s"
    assert clock.elapsed(1.0, 62.0) == "1:01"
    app.active = False
    clock.tick(1.2)
    assert not clock.running
    app.active = True
    clock.start()
    app._terminal_focused = False
    clock.tick(1.3)
    assert not clock.running


def test_reduced_motion_is_static_and_readable(monkeypatch):
    monkeypatch.setenv("AXIOM_REDUCE_MOTION", "1")

    class App:
        pass

    clock = MotionClock(App())
    assert clock.reduced
    assert clock.label("thinking", 1.0).plain == "thinking"
    assert clock.spinner == "…"


async def test_copy_confirmation_expires_on_shared_clock(app_factory, monkeypatch):
    from src import tui

    app = app_factory()
    monkeypatch.setattr(
        tui, "copy_text", lambda app, text: (True, f"copied {len(text)} chars")
    )
    async with app.run_test():
        await app._copy("text")
        confirmation = app._copy_confirmation
        assert confirmation.is_attached
        deadline = app._copy_expiry
        app.motion.tick(deadline - 0.01)
        assert app._copy_confirmation is confirmation
        app.motion.tick(deadline + 0.04)
        assert app._copy_confirmation is None
        assert not app.motion.running


def test_shimmer_and_blink_use_shared_deadlines(monkeypatch):
    from src.ui import motion

    monkeypatch.delenv("NO_MOTION", raising=False)
    monkeypatch.delenv("AXIOM_REDUCE_MOTION", raising=False)
    monkeypatch.setattr(motion.time, "monotonic", lambda: 0)
    clock = MotionClock(object())
    assert clock.label("thinking", 0.00).spans != clock.label("thinking", 0.06).spans
    assert clock.label("thinking", 0.72).spans == clock.label("thinking", 0.90).spans
    assert clock.blink(0.49)
    assert not clock.blink(0.5)
    assert clock.blink(1)
