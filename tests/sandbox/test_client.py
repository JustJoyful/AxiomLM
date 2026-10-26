import asyncio
from dataclasses import asdict

import pytest

from src.sandbox.client import SandboxClient, parse_result
from src.sandbox.protocol import PythonResult


def test_stale_reply_cannot_complete_next_request():
    row = {"event": "result", **asdict(PythonResult("old", "ok", value="5"))}
    with pytest.raises(ValueError):
        parse_result(row, "new")


@pytest.mark.parametrize("status", ["verified", True, None])
def test_invalid_worker_status_is_not_success(status):
    row = {"event": "result", **asdict(PythonResult("r", "ok", value="5"))}
    row["status"] = status
    with pytest.raises(ValueError):
        parse_result(row, "r")


@pytest.mark.asyncio
async def test_oversized_code_rejected_before_start(monkeypatch):
    client = SandboxClient(request_prefix="test")

    async def forbidden():
        raise AssertionError("must not start")

    monkeypatch.setattr(client, "start", forbidden)
    with pytest.raises(ValueError):
        await client.execute("x" * 8193)


@pytest.mark.asyncio
async def test_timeout_cancels_job_not_warm_worker(monkeypatch):
    from src.sandbox import launcher

    calls = []

    async def kill(group):
        calls.append(group)

    monkeypatch.setattr(launcher, "kill_job", kill)
    cancel = asyncio.Event()

    async def response():
        await asyncio.sleep(0.03)
        return {"event": "result", "id": "r", "status": "error"}

    status, result = await launcher.await_call(
        asyncio.create_task(response()), cancel, "private-job", deadline=0.01
    )
    assert status == "timeout" and calls == ["private-job"]
    assert result["id"] == "r"
