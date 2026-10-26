import asyncio

import pytest

from src.sandbox import launcher


@pytest.mark.asyncio
async def test_cancel_waits_for_empty_job_cgroup(tmp_path):
    (tmp_path / "cgroup.kill").write_text("")
    (tmp_path / "cgroup.events").write_text("populated 1\n")

    async def empty():
        await asyncio.sleep(0.02)
        (tmp_path / "cgroup.events").write_text("populated 0\n")

    empty_task = asyncio.create_task(empty())
    await launcher.kill_job(tmp_path)
    assert (tmp_path / "cgroup.kill").read_text() == "1"
    assert empty_task.done()


def test_guard_requires_stopped_child_identity():
    parent = launcher.ProcessIdentity(100, 10, 2, 90, "S")
    valid = launcher.ProcessIdentity(101, 20, 3, 100, "T")
    launcher.validate_child(valid, parent, 3)
    for bad in [
        launcher.ProcessIdentity(101, 20, 3, 99, "T"),
        launcher.ProcessIdentity(101, 20, 4, 100, "T"),
        launcher.ProcessIdentity(101, 20, 3, 100, "R"),
    ]:
        with pytest.raises(ValueError):
            launcher.validate_child(bad, parent, 3)


@pytest.mark.asyncio
async def test_populated_job_blocks_reuse(tmp_path, monkeypatch):
    from src import config

    monkeypatch.setattr(config, "PYTHON_CLEANUP_SECONDS", 0.02)
    (tmp_path / "cgroup.kill").write_text("")
    (tmp_path / "cgroup.events").write_text("populated 1\n")
    with pytest.raises(TimeoutError):
        await launcher.kill_job(tmp_path)
