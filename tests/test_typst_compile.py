import os

import pytest

from src import typst_compile as tc

# A finite workload: Typst rejects constant/infinite while loops before execution.
SLOW_TYPST = "#{ for i in range(10000) { for j in range(10000) { let n = i + j } } }"


def test_command_isolated_and_paths_fixed(tmp_path, monkeypatch):
    monkeypatch.setattr(tc.shutil, "which", lambda name: "/usr/bin/" + name)
    argv = tc.compiler_command(tmp_path, "axiom-typst-test")
    assert "--unshare-net" in argv and "--die-with-parent" in argv
    assert "MemoryMax=512M" in argv and "KillMode=control-group" in argv
    assert argv[-6:] == [
        "/bin/typst",
        "compile",
        "--root",
        "/job",
        "/job/document.typ",
        "/job/document.pdf",
    ]
    assert not any("/home" in arg for arg in argv if arg != str(tmp_path))


def test_missing_typst_is_unavailable(monkeypatch):
    monkeypatch.setattr(tc.shutil, "which", lambda _: None)
    assert tc.compiler_capability()[0] is False


@pytest.mark.asyncio
async def test_unavailable_does_not_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "compiler_capability", lambda: (False, "missing"))
    result = await tc.compile_typst(tmp_path, remaining=1)
    assert result.status == "unavailable" and result.pdf_path is None


def test_invalid_or_oversized_pdf_is_error(tmp_path, monkeypatch):
    path = tmp_path / "document.pdf"
    path.write_bytes(b"%PDF fake")
    assert not tc.valid_pdf(path)
    monkeypatch.setattr(tc.config, "TYPST_MAX_PDF_BYTES", 2)
    assert not tc.valid_pdf(path)


@pytest.mark.asyncio
async def test_syntax_failure_returns_bounded_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "compiler_capability", lambda: (True, None))
    (tmp_path / "document.typ").write_text("bad")

    async def run(*args):
        return 1, "error: " + "e" * 10000

    monkeypatch.setattr(tc, "_run_compile", run)
    result = await tc.compile_typst(tmp_path, remaining=1)
    assert result.status == "syntax_error" and result.pdf_path is None
    assert len(result.diagnostics.encode()) <= 4096


@pytest.mark.skipif(
    os.environ.get("AXIOM_RUN_TYPST_TESTS") != "1", reason="opt-in Typst host gate"
)
@pytest.mark.asyncio
async def test_host_file_read_and_package_fetch_denied(tmp_path):
    assert tc.compiler_capability()[0], tc.compiler_capability()[1]
    for text in ('#read("/etc/passwd")', '#import "@preview/cetz:0.3.0": *'):
        (tmp_path / "document.typ").write_text(text)
        result = await tc.compile_typst(tmp_path, remaining=10)
        assert result.status != "ok" and result.pdf_path is None


host = pytest.mark.skipif(
    os.environ.get("AXIOM_RUN_TYPST_TESTS") != "1",
    reason="opt-in Typst security gate; missing compiler is not acceptance",
)


@host
@pytest.mark.asyncio
async def test_timeout_kills_unit_and_next_compile_succeeds(tmp_path):
    assert tc.compiler_capability()[0], tc.compiler_capability()[1]
    (tmp_path / "document.typ").write_text(SLOW_TYPST)
    result = await tc.compile_typst(tmp_path, remaining=0.5)
    assert result.status == "timeout"
    assert not (tmp_path / "document.pdf").exists()
    (tmp_path / "document.typ").write_text("Recovered")
    result = await tc.compile_typst(tmp_path, remaining=10)
    assert result.status == "ok", result.diagnostics
    assert tc.valid_pdf(result.pdf_path)


async def unit_state(unit):
    import asyncio

    proc = await asyncio.create_subprocess_exec(
        "systemctl",
        "--user",
        "show",
        unit,
        "--property=ActiveState,ControlGroup,MemoryMax,TasksMax,LimitFSIZE",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await asyncio.wait_for(
        proc.communicate(), tc.config.TYPST_CLEANUP_SECONDS
    )
    return dict(
        line.split("=", 1) for line in stdout.decode().splitlines() if "=" in line
    )


@host
@pytest.mark.asyncio
async def test_compile_memory_and_file_limits(tmp_path, monkeypatch):
    import asyncio

    assert tc.compiler_capability()[0], tc.compiler_capability()[1]
    command = tc.compiler_command
    units = []

    def capture(job, unit):
        units.append(unit)
        return command(job, unit)

    monkeypatch.setattr(tc, "compiler_command", capture)
    (tmp_path / "document.typ").write_text(SLOW_TYPST)
    task = asyncio.create_task(tc.compile_typst(tmp_path, remaining=10))
    state = {}
    try:
        async with asyncio.timeout(5):
            while state.get("ActiveState") != "active":
                if units:
                    state = await unit_state(units[0])
                await asyncio.sleep(0.05)
        assert int(state["MemoryMax"]) == 536870912
        assert int(state["TasksMax"]) == 16
        assert int(state["LimitFSIZE"]) == 8388608
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert (await unit_state(units[0])).get("ActiveState") != "active"
    assert not (tmp_path / "document.pdf").exists()
    cgroup = state.get("ControlGroup")
    if cgroup:
        from pathlib import Path

        procs = Path("/sys/fs/cgroup") / cgroup.lstrip("/") / "cgroup.procs"
        assert not procs.exists() or not procs.read_text().strip()


@host
@pytest.mark.asyncio
async def test_parent_death_has_bounded_cleanup(tmp_path):
    import asyncio
    import sys

    assert tc.compiler_capability()[0], tc.compiler_capability()[1]
    (tmp_path / "document.typ").write_text(SLOW_TYPST)
    # This real parent dies without executing Python cleanup; systemd owns the bound.
    script = """
import asyncio, sys
from pathlib import Path
from src import typst_compile as tc
job=Path(sys.argv[1])
original=tc.compiler_command
def capture(job, unit):
    (job/'unit-name').write_text(unit)
    return original(job, unit)
tc.compiler_command=capture
asyncio.run(tc.compile_typst(job, remaining=30))
"""
    parent = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        str(tmp_path),
        cwd=tc.config.ROOT,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    unit = None
    try:
        async with asyncio.timeout(5):
            while unit is None:
                marker = tmp_path / "unit-name"
                if marker.exists():
                    name = marker.read_text()
                    if (await unit_state(name)).get("ActiveState") == "active":
                        unit = name
                await asyncio.sleep(0.05)
        parent.kill()
        await parent.wait()
        async with asyncio.timeout(
            tc.config.TYPST_COMPILE_SECONDS + tc.config.TYPST_CLEANUP_SECONDS + 2
        ):
            while (await unit_state(unit)).get("ActiveState") == "active":
                await asyncio.sleep(0.1)
        assert not tc.valid_pdf(tmp_path / "document.pdf")
    finally:
        if parent.returncode is None:
            parent.kill()
            await parent.wait()
        if unit:
            await tc._stop(unit)


def test_service_stop_deadline_uses_config(tmp_path, monkeypatch):
    monkeypatch.setattr(tc.config, "TYPST_CLEANUP_SECONDS", 2.5)
    monkeypatch.setattr(tc.shutil, "which", lambda name: "/usr/bin/" + name)
    assert "TimeoutStopSec=2.5" in tc.compiler_command(tmp_path, "axiom-typst-test")
