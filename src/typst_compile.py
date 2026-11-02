"""CPU-only Typst compilation in a bounded, network-isolated user service."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import fitz

from src import config


@dataclass(frozen=True)
class CompileResult:
    status: Literal["ok", "syntax_error", "unavailable", "timeout", "resource_error"]
    diagnostics: str
    pdf_path: Path | None = None


def compiler_capability() -> tuple[bool, str | None]:
    for name in (config.TYPST_BINARY, "bwrap", "systemd-run", "systemctl"):
        if not shutil.which(name):
            return False, f"{name} is not installed"
    if not Path("/sys/fs/cgroup/cgroup.controllers").exists():
        return False, "cgroups v2 is required"
    try:
        probe = subprocess.run(
            ["systemctl", "--user", "show-environment"],
            capture_output=True,
            timeout=config.TYPST_CLEANUP_SECONDS,
            check=False,
        )
        if probe.returncode:
            return False, "User systemd manager unavailable"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    return True, None


def compiler_command(job: Path, unit: str) -> list[str]:
    binary = Path(shutil.which(config.TYPST_BINARY) or config.TYPST_BINARY).resolve()
    command = [
        "systemd-run",
        "--user",
        "--wait",
        "--pipe",
        "--collect",
        "--unit",
        unit,
        "-p",
        f"MemoryMax={config.TYPST_MEMORY_MAX}",
        "-p",
        "MemorySwapMax=0",
        "-p",
        f"TasksMax={config.TYPST_TASKS_MAX}",
        "-p",
        f"RuntimeMaxSec={config.TYPST_COMPILE_SECONDS}",
        "-p",
        f"LimitFSIZE={config.TYPST_MAX_PDF_BYTES}",
        "-p",
        "KillMode=control-group",
        "-p",
        f"TimeoutStopSec={config.TYPST_CLEANUP_SECONDS}",
        "--",
        "bwrap",
        "--unshare-all",
        "--unshare-net",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--dir",
        "/bin",
        "--ro-bind",
        str(binary),
        "/bin/typst",
    ]
    for mount in config.TYPST_RUNTIME_MOUNTS:
        if Path(mount).exists():
            command += ["--ro-bind", mount, mount]
    return command + [
        "--bind",
        str(job.resolve()),
        "/job",
        "--chdir",
        "/job",
        "--setenv",
        "HOME",
        "/tmp",
        "--",
        "/bin/typst",
        "compile",
        "--root",
        "/job",
        "/job/document.typ",
        "/job/document.pdf",
    ]


def valid_pdf(path: Path) -> bool:
    try:
        if (
            path.is_symlink()
            or not 0 < path.stat().st_size <= config.TYPST_MAX_PDF_BYTES
        ):
            return False
        with fitz.open(path) as pdf:
            return pdf.is_pdf and pdf.page_count > 0 and not pdf.needs_pass
    except (OSError, RuntimeError, ValueError):
        return False


async def _stop(unit):
    process = await asyncio.create_subprocess_exec(
        "systemctl",
        "--user",
        "stop",
        unit,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        await asyncio.wait_for(process.wait(), config.TYPST_CLEANUP_SECONDS)
    except TimeoutError:
        process.kill()
        await process.wait()


async def _run_compile(job, unit, timeout):
    process = await asyncio.create_subprocess_exec(
        *compiler_command(job, unit),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    diagnostic = bytearray()

    async def drain():
        while block := await process.stderr.read(4096):
            diagnostic.extend(
                block[: max(0, config.TYPST_DIAGNOSTIC_BYTES - len(diagnostic))]
            )
        await process.wait()

    try:
        await asyncio.wait_for(drain(), timeout)
    except (asyncio.CancelledError, TimeoutError):
        await asyncio.shield(_stop(unit))
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise
    return process.returncode, diagnostic.decode("utf-8", errors="replace")


async def compile_typst(job: Path, *, remaining: float) -> CompileResult:
    available, reason = await asyncio.to_thread(compiler_capability)
    if not available:
        return CompileResult("unavailable", reason or "Compiler unavailable")
    pdf = job / "document.pdf"
    pdf.unlink(missing_ok=True)
    try:
        source = job / "document.typ"
        if source.is_symlink() or source.stat().st_size > config.DOCUMENT_MAX_TYP_BYTES:
            return CompileResult("resource_error", "Typst source exceeds limit")
        code, diagnostics = await _run_compile(
            job,
            f"axiom-typst-{uuid.uuid4().hex}",
            min(remaining, config.TYPST_COMPILE_SECONDS),
        )
        diagnostics = diagnostics.encode()[: config.TYPST_DIAGNOSTIC_BYTES].decode(
            "utf-8", errors="ignore"
        )
        if code == 0 and valid_pdf(pdf):
            return CompileResult("ok", diagnostics, pdf)
        # Only ordinary syntax diagnostics earn a model correction.
        status = (
            "syntax_error"
            if code == 1 and "error:" in diagnostics
            else "resource_error"
        )
        return CompileResult(
            status, diagnostics or "Invalid PDF or compiler resource failure"
        )
    except TimeoutError:
        return CompileResult("timeout", "Compiler deadline exceeded")
    except OSError as exc:
        return CompileResult("unavailable", str(exc))
    finally:
        if not valid_pdf(pdf):
            pdf.unlink(missing_ok=True)
