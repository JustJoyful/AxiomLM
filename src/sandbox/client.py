"""Serialized async ownership of the native sandbox guard and its user scope."""

from __future__ import annotations

import ast
import asyncio
import json
import sys
import uuid
from dataclasses import fields
from typing import TYPE_CHECKING

from src import config

from . import launcher
from .protocol import PythonResult, encode_frame

if TYPE_CHECKING:
    from src.tools.math_checks import CheckResult


def _backstop() -> float:
    """Outer bound if the guard stalls: execution deadline plus every cleanup phase."""
    return config.PYTHON_RESPONSE_SECONDS + config.PYTHON_CLEANUP_SECONDS * 4


def parse_result(row: dict, ident: str) -> PythonResult:
    names = {f.name for f in fields(PythonResult)}
    if (
        set(row) != names | {"event"}
        or row["event"] != "result"
        or row["request_id"] != ident
    ):
        raise ValueError("Stale or malformed sandbox result")
    if row["status"] not in (
        "ok",
        "error",
        "timeout",
        "cancelled",
        "unavailable",
        "worker_died",
    ):
        raise ValueError("Invalid sandbox result status")
    if (
        type(row["truncated"]) is not bool
        or any(not isinstance(row[key], str) for key in ("stdout", "stderr"))
        or (row["value"] is not None and not isinstance(row["value"], str))
    ):
        raise ValueError("Invalid sandbox output fields")
    if (
        sum(len((row[key] or "").encode()) for key in ("stdout", "stderr", "value"))
        > config.PYTHON_MAX_OUTPUT_BYTES
    ):
        raise ValueError("Sandbox output exceeds limit")
    if row["worker_pid"] is not None and (
        type(row["worker_pid"]) is not int or row["worker_pid"] <= 0
    ):
        raise ValueError("Invalid worker host PID")
    if row["worker_instance_id"] is not None and not isinstance(
        row["worker_instance_id"], str
    ):
        raise ValueError("Invalid worker instance ID")
    return PythonResult(**{name: row[name] for name in names})


class SandboxClient:
    def __init__(self, *, request_prefix: str):
        self.request_prefix = request_prefix[:32]
        self.unit = None
        self.process = None
        self.worker_pid = None
        self.worker_instance_id = None
        self._lock = asyncio.Lock()

    async def start(self) -> launcher.SandboxCapabilities:
        if self.process is not None and self.process.returncode is None:
            return launcher.SandboxCapabilities(True, None)
        capabilities = await asyncio.to_thread(launcher.check_capabilities)
        if not capabilities.available:
            return capabilities
        self.unit = f"axiom-python-{uuid.uuid4().hex}.scope"
        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "src.sandbox.launcher",
            "--guard",
            self.unit,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            limit=config.PYTHON_MAX_FRAME_BYTES,
        )
        try:
            row = await asyncio.wait_for(
                launcher._read_json(self.process.stdout), config.PYTHON_STARTUP_SECONDS
            )
            if (
                set(row) != {"event", "worker_pid", "worker_instance_id"}
                or row["event"] != "ready"
                or type(row["worker_pid"]) is not int
            ):
                raise ValueError(
                    f'Sandbox guard startup failed: {row.get("stderr", "invalid readiness")}'
                )
            self.worker_pid, self.worker_instance_id = (
                row["worker_pid"],
                row["worker_instance_id"],
            )
        except (OSError, ValueError, EOFError, TimeoutError) as exc:
            await self.aclose()
            return launcher.SandboxCapabilities(False, str(exc))
        except asyncio.CancelledError:
            await asyncio.shield(self.aclose())
            raise
        return capabilities

    async def verify(self, check) -> CheckResult:
        """Run the trusted symbolic checker in a fresh isolated child."""
        from dataclasses import asdict

        from src.tools.math_checks import CheckResult as Result

        result = await self._call({"op": "verify", "check": asdict(check)})
        if result.status != "ok" or result.value is None:
            reason = (result.stderr or result.status)[-300:]
            return Result(
                "unverified" if result.status == "unavailable" else "error",
                f"Verification sandbox {result.status}: {reason}",
            )
        try:
            payload = json.loads(ast.literal_eval(result.value))
            if set(payload) != {"status", "reason"} or payload["status"] not in (
                "verified",
                "disagreed",
                "unverified",
                "error",
            ):
                raise ValueError("Unexpected checker payload")
        except (ValueError, SyntaxError, TypeError) as exc:
            return Result("error", f"Malformed checker result: {type(exc).__name__}")
        return Result(payload["status"], str(payload["reason"]))

    async def execute(self, code: str) -> PythonResult:
        return await self._call({"op": "execute", "code": code})

    async def _call(self, body: dict) -> PythonResult:
        ident = f"{self.request_prefix}-{uuid.uuid4().hex}"
        frame = encode_frame({"id": ident, **body})
        async with self._lock:
            capabilities = await self.start()
            if not capabilities.available:
                return PythonResult(
                    ident,
                    "unavailable",
                    stderr=capabilities.reason or "Sandbox unavailable",
                )
            response = None
            try:
                self.process.stdin.write(frame)
                await self.process.stdin.drain()
                response = asyncio.create_task(launcher._read_json(self.process.stdout))
                row = await asyncio.wait_for(
                    asyncio.shield(response),
                    _backstop(),
                )
                result = parse_result(row, ident)
                if result.status == "worker_died":
                    await self.aclose()
                return result
            except asyncio.CancelledError:
                try:
                    self.process.stdin.write(
                        encode_frame(
                            {
                                "id": uuid.uuid4().hex,
                                "op": "cancel",
                                "request_id": ident,
                            }
                        )
                    )
                    await self.process.stdin.drain()
                    await asyncio.wait_for(
                        asyncio.shield(response),
                        _backstop(),
                    )
                finally:
                    if (
                        response is None
                        or not response.done()
                        or response.cancelled()
                        or response.exception() is not None
                    ):
                        await asyncio.shield(self.aclose())
                raise
            except (OSError, EOFError, ValueError, TimeoutError) as exc:
                await self.aclose()
                return PythonResult(
                    ident,
                    "worker_died",
                    stderr=str(exc)[: config.PYTHON_MAX_OUTPUT_BYTES],
                )
            finally:
                if response is not None and not response.done():
                    response.cancel()
                    await asyncio.gather(response, return_exceptions=True)

    async def aclose(self) -> None:
        process, unit = self.process, self.unit
        self.process = self.unit = None
        if process is not None and process.returncode is None:
            try:
                process.stdin.write(
                    encode_frame({"id": uuid.uuid4().hex, "op": "shutdown"})
                )
                await process.stdin.drain()
                process.stdin.close()
                await asyncio.wait_for(process.wait(), config.PYTHON_CLEANUP_SECONDS)
            except (OSError, TimeoutError):
                if unit is not None:
                    await launcher.stop_unit(unit)
                if process.returncode is None:
                    process.kill()
                    await process.wait()
        if unit is not None:
            await launcher.stop_unit(unit)
