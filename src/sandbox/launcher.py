"""Fail-closed native runtime manifest, host probe, scope entry and request guard.

Request placement never writes a numeric PID. The host guard opens a fresh request
leaf's ``cgroup.procs`` and passes that descriptor to the warm supervisor; the call
child writes ``0`` (the kernel's "this process") before any user code exists, then
closes it. Migration permission is checked against the guard's open-time
credentials, so the sandbox needs no cgroup mount or writable cgroup path.
"""

from __future__ import annotations

import asyncio
import ctypes
import importlib.util
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import sysconfig
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

# Executed directly by the fixed host probe/entry/guard, never from an untrusted call.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config

CGROUP_ROOT = Path("/sys/fs/cgroup")
UNIT_PATTERN = r"axiom-python-[a-zA-Z0-9-]+\.scope"
# The worker finds its private placement socket here; bwrap preserves inherited fds.
CONTROL_FD = 3
SUPERVISOR_LEAF = "supervisor"
OOM_NOTICE = "Sandbox memory limit exceeded; request processes were OOM-killed ({count} kernel OOM kill events)."


@dataclass(frozen=True)
class RuntimeMount:
    source: Path
    target: str


@dataclass(frozen=True)
class SandboxCapabilities:
    available: bool
    reason: str | None
    mounts: tuple[RuntimeMount, ...] = ()


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=config.PYTHON_STARTUP_SECONDS,
        check=True,
    )


def runtime_mounts() -> tuple[RuntimeMount, ...]:
    """Resolve only the interpreter, stdlib, package and shared-library closure."""
    executable = Path(sys.executable).resolve(strict=True)
    stdlib = Path(sysconfig.get_path("stdlib")).resolve(strict=True)
    mounts = [RuntimeMount(executable, "/runtime/bin/python")]
    # Avoid binding a stdlib parent that also exposes host site-packages.
    for child in sorted(stdlib.iterdir()):
        if child.name in {
            "site-packages",
            "__pycache__",
            "lib-dynload",
            "tkinter",
            "idlelib",
            "turtledemo",
        }:
            continue
        mounts.append(
            RuntimeMount(
                child.resolve(strict=True),
                f"/runtime/lib/python{sys.version_info.major}.{sys.version_info.minor}/{child.name}",
            )
        )
    for name in ("sympy", "mpmath"):
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError(f"Missing runtime package: {name}")
        source = Path(next(iter(spec.submodule_search_locations))).resolve(strict=True)
        mounts.append(RuntimeMount(source, f"/packages/{name}"))
    for name in ("launcher.py", "protocol.py", "worker.py"):
        source = Path(__file__).with_name(name)
        if source.exists():
            mounts.append(RuntimeMount(source.resolve(strict=True), f"/{name}"))
    # The only extra module: the trusted symbolic checker run by verify requests.
    checker = Path(__file__).resolve().parents[1] / "tools" / "math_checks.py"
    mounts.append(RuntimeMount(checker.resolve(strict=True), "/trusted/math_checks.py"))
    extensions = [
        path
        for path in sorted((stdlib / "lib-dynload").glob("*.so"))
        if path.name.split(".")[0] in config.PYTHON_EXTENSION_MODULES
    ]
    mounts.extend(
        RuntimeMount(
            path.resolve(strict=True),
            f"/runtime/lib/python{sys.version_info.major}.{sys.version_info.minor}/lib-dynload/{path.name}",
        )
        for path in extensions
    )
    binaries = [executable, *extensions]
    libraries = set()
    for binary in binaries:
        output = _run(["ldd", str(binary)]).stdout
        if "not found" in output:
            raise RuntimeError(f"Unresolved shared library in {binary.name}")
        for line in output.splitlines():
            for name in re.findall(r"/[^\s()]+", line):
                path = Path(name)
                libraries.add((path.resolve(strict=True), str(path)))
    mounts.extend(RuntimeMount(source, target) for source, target in sorted(libraries))
    return tuple(mounts)


def _scope_args(unit: str) -> list[str]:
    return [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        f"--unit={unit}",
        "--property",
        "MemoryMax=250M",
        "--property",
        "MemorySwapMax=0",
        "--property",
        "TasksMax=32",
        "--property",
        "Delegate=yes",
        "--property",
        "KillMode=control-group",
        "--",
    ]


def _limits() -> str:
    return json.dumps(
        {
            "output": config.PYTHON_MAX_OUTPUT_BYTES,
            "frame": config.PYTHON_MAX_FRAME_BYTES,
            "code": config.PYTHON_MAX_CODE_BYTES,
            "id": config.PYTHON_MAX_ID_CHARS,
            "poll": config.PYTHON_POLL_SECONDS,
            "placement": config.PYTHON_CLEANUP_SECONDS,
            "control_fd": CONTROL_FD,
            "math": {
                "chars": config.MATH_MAX_EXPRESSION_CHARS,
                "nodes": config.MATH_MAX_AST_NODES,
                "exponent": config.MATH_MAX_INTEGER_EXPONENT,
                "digits": config.MATH_MAX_NUMERIC_DIGITS,
            },
        }
    )


def build_launch(
    unit: str, mounts: tuple[RuntimeMount, ...], *, control_fd: int = CONTROL_FD
) -> tuple[str, ...]:
    """Scope → trusted host entry (self-placement) → Bubblewrap → warm worker."""
    if not re.fullmatch(UNIT_PATTERN, unit):
        raise ValueError("Invalid sandbox unit")
    approved = runtime_mounts()
    if mounts != approved:
        raise ValueError("Unapproved runtime manifest")
    args = _scope_args(unit) + [
        str(Path(sys.executable).resolve()),
        "-I",
        str(Path(__file__).resolve()),
        "--enter",
        unit,
        str(control_fd),
        "--",
        "bwrap",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-net",
        "--unshare-ipc",
        "--unshare-uts",
        "--new-session",
        "--die-with-parent",
        "--cap-drop",
        "ALL",
        "--clearenv",
        "--setenv",
        "AXIOM_PYTHON_LIMITS",
        _limits(),
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--size",
        str(config.PYTHON_SCRATCH_BYTES),
        "--tmpfs",
        "/scratch",
        "--chdir",
        "/scratch",
        "--setenv",
        "SYMPY_CACHE_SIZE",
        str(config.SYMPY_CACHE_SIZE),
    ]
    for mount in mounts:
        args.extend(("--ro-bind", str(mount.source), mount.target))
    args.extend(
        (
            "--remount-ro",
            "/dev",
            "--remount-ro",
            "/",
            "/runtime/bin/python",
            "-I",
            "-S",
            "/worker.py",
        )
    )
    return tuple(args)


def own_cgroup() -> Path:
    relative = Path("/proc/self/cgroup").read_text().strip().split("0::", 1)[1]
    return CGROUP_ROOT / relative.lstrip("/")


def check_scope_limits(group: Path) -> dict:
    observed = {
        name: (group / name).read_text().strip()
        for name in ("memory.max", "memory.swap.max", "pids.max", "cgroup.controllers")
    }
    if (
        observed["memory.max"] != str(config.PYTHON_MEMORY_BYTES)
        or observed["memory.swap.max"] != "0"
        or observed["pids.max"] != str(config.PYTHON_MAX_TASKS)
    ):
        raise RuntimeError(f"Effective scope limits differ: {observed}")
    if not {"memory", "pids"} <= set(observed["cgroup.controllers"].split()):
        raise RuntimeError("Missing delegated memory/pids controllers")
    if not (group / "cgroup.kill").exists():
        raise RuntimeError("Missing cgroup.kill")
    return observed


def _probe_scope() -> dict:
    group = own_cgroup()
    observed = check_scope_limits(group)
    leaf = group / "capability-probe"
    leaf.mkdir()
    try:
        if not os.access(leaf / "cgroup.procs", os.W_OK) or not os.access(
            leaf / "cgroup.kill", os.W_OK
        ):
            raise RuntimeError("Request cgroup is not writable")
    finally:
        leaf.rmdir()
    return observed


def enter_scope(unit: str, control_fd: int, command: list[str]) -> int:
    """Trusted scope entry: verify limits, self-place, delegate, then exec bwrap."""
    try:
        group = own_cgroup()
        if group.name != unit or not re.fullmatch(UNIT_PATTERN, unit):
            raise RuntimeError("Scope entry is not running in its named unit")
        check_scope_limits(group)
        supervisor = group / SUPERVISOR_LEAF
        supervisor.mkdir()
        # "0" migrates exactly the writing process; no PID can be confused here.
        (supervisor / "cgroup.procs").write_text("0")
        (group / "cgroup.subtree_control").write_text("+memory +pids")
        if control_fd != CONTROL_FD:
            os.dup2(control_fd, CONTROL_FD, inheritable=True)
            os.close(control_fd)
        else:
            os.set_inheritable(CONTROL_FD, True)
        print(
            json.dumps(
                {"event": "entered", "cgroup": str(group.relative_to(CGROUP_ROOT))}
            ),
            flush=True,
        )
        os.execvp(command[0], command)
    except (OSError, RuntimeError, ValueError, IndexError) as exc:
        print(json.dumps({"event": "enter_failed", "reason": str(exc)}), flush=True)
    return 1


def open_pidfd(pid: int) -> int:
    """Use the same libc pidfd API on portable builds lacking os.pidfd_open."""
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        function = libc.pidfd_open
    except AttributeError as exc:
        raise RuntimeError("Native pidfd_open unavailable") from exc
    function.argtypes = (ctypes.c_int, ctypes.c_uint)
    function.restype = ctypes.c_int
    fd = function(pid, 0)
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def probe_host() -> dict:
    """Fixed trusted probe; no user/generated Python executes here."""
    if sys.platform != "linux" or not hasattr(os, "fork"):
        raise RuntimeError("Linux fork support required")
    os.close(open_pidfd(os.getpid()))
    if not (CGROUP_ROOT / "cgroup.controllers").is_file():
        raise RuntimeError("cgroups v2 unavailable")
    unit = f"axiom-python-probe-{uuid.uuid4().hex}.scope"
    result = _run(
        _scope_args(unit)
        + [sys.executable, "-I", str(Path(__file__).resolve()), "--probe-host"]
    )
    return json.loads(result.stdout)


def check_capabilities() -> SandboxCapabilities:
    for program in ("bwrap", "systemd-run", "systemctl", "ldd"):
        if shutil.which(program) is None:
            return SandboxCapabilities(False, f"Missing required executable: {program}")
    try:
        probe_host()
        mounts = runtime_mounts()
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        detail = (
            exc.stderr.strip()
            if isinstance(exc, subprocess.CalledProcessError)
            else str(exc)
        )
        return SandboxCapabilities(False, f"Sandbox prerequisite failed: {detail}")
    return SandboxCapabilities(True, None, mounts)


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    start_time: int
    namespace_pid: int
    parent_pid: int
    state: str


def _nspids(pid: int) -> list[int]:
    status = (Path("/proc") / str(pid) / "status").read_text()
    line = next(line for line in status.splitlines() if line.startswith("NSpid:"))
    return [int(value) for value in line.split()[1:]]


def process_identity(pid: int) -> ProcessIdentity:
    fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    return ProcessIdentity(
        pid, int(fields[19]), _nspids(pid)[-1], int(fields[1]), fields[0]
    )


def validate_child(
    child: ProcessIdentity, parent: ProcessIdentity, namespace_pid: int
) -> None:
    if (
        child.parent_pid != parent.pid
        or child.namespace_pid != namespace_pid
        or child.state not in ("T", "t")
        or child.start_time < parent.start_time
    ):
        raise ValueError("Stopped child identity mismatch")


def cgroup_pids(group: Path) -> list[int]:
    return [int(pid) for pid in (group / "cgroup.procs").read_text().split()]


def oom_kills(group: Path) -> int:
    rows = dict(
        line.split()
        for line in (group / "memory.events").read_text().splitlines()
        if line.strip()
    )
    return int(rows.get("oom_kill", 0)) + int(rows.get("oom_group_kill", 0))


def oom_kills_safe(group: Path | None) -> int:
    try:
        return oom_kills(group) if group is not None else 0
    except (OSError, ValueError):
        return 0


async def kill_job(group: Path) -> None:
    (group / "cgroup.kill").write_text("1")
    async with asyncio.timeout(config.PYTHON_CLEANUP_SECONDS):
        while "populated 0" not in (group / "cgroup.events").read_text().splitlines():
            await asyncio.sleep(config.PYTHON_POLL_SECONDS)


async def await_call(response, cancel, group, *, deadline):
    cancellation = asyncio.create_task(cancel.wait())
    try:
        done, _ = await asyncio.wait(
            (response, cancellation),
            timeout=deadline,
            return_when=asyncio.FIRST_COMPLETED,
        )
        status = (
            "ok"
            if response in done
            else "cancelled" if cancellation in done else "timeout"
        )
        if status != "ok":
            await kill_job(group)
        result = await asyncio.wait_for(response, config.PYTHON_CLEANUP_SECONDS)
        return status, result
    finally:
        cancellation.cancel()
        await asyncio.gather(cancellation, return_exceptions=True)
        if not response.done():
            response.cancel()
            await asyncio.gather(response, return_exceptions=True)


async def stop_unit(unit: str) -> None:
    if not re.fullmatch(UNIT_PATTERN, unit):
        raise ValueError("Invalid sandbox unit")
    proc = await asyncio.create_subprocess_exec(
        "systemctl",
        "--user",
        "stop",
        unit,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        await asyncio.wait_for(proc.wait(), config.PYTHON_STARTUP_SECONDS)
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()


async def _read_json(reader):
    from .protocol import _constant, _pairs

    line = await reader.readline()
    if not line:
        raise EOFError("Worker pipe closed")
    if len(line) > config.PYTHON_MAX_FRAME_BYTES or not line.endswith(b"\n"):
        raise ValueError("Invalid guard frame")
    row = json.loads(line, object_pairs_hook=_pairs, parse_constant=_constant)
    if not isinstance(row, dict):
        raise ValueError("Invalid guard envelope")  # noqa: TRY004
    return row


def bounded_fields(stdout: str, stderr: str, value: str | None, notice: str):
    """Append a guard notice to stderr while keeping the combined output bound."""
    limit = config.PYTHON_MAX_OUTPUT_BYTES
    notice = notice.encode()[:limit].decode("utf-8", errors="ignore")
    room = limit - len(notice.encode()) - len((value or "").encode())
    if room < 0:
        value, room = None, limit - len(notice.encode())
    out = stdout.encode()[:room].decode("utf-8", errors="ignore")
    room -= len(out.encode())
    err = stderr.encode()[: max(0, room - 1)].decode("utf-8", errors="ignore")
    truncated = len(out) < len(stdout) or len(err) < len(stderr)
    return out, (err + "\n" if err else "") + notice, value, truncated


class PlacementError(RuntimeError):
    """The request child was not provably alone and stopped in its leaf."""


class Guard:
    """Host owner of one scope: request leaves, deadlines and fail-closed cleanup."""

    def __init__(self, unit: str):
        if not re.fullmatch(UNIT_PATTERN, unit):
            raise ValueError("Invalid sandbox unit")
        self.unit = unit
        self.process = None
        self.scope: Path | None = None
        self.control: socket.socket | None = None
        self.supervisor: ProcessIdentity | None = None
        self.instance_id: str | None = None
        self.import_identity = None
        self.broken: str | None = None
        self._sequence = 0

    async def start(self) -> dict:
        mounts = await asyncio.to_thread(runtime_mounts)
        ours, theirs = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        self.control = ours
        try:
            self.process = await asyncio.create_subprocess_exec(
                *build_launch(self.unit, mounts, control_fd=theirs.fileno()),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                pass_fds=(theirs.fileno(),),
                limit=config.PYTHON_MAX_FRAME_BYTES,
            )
        finally:
            theirs.close()
        async with asyncio.timeout(config.PYTHON_STARTUP_SECONDS):
            entered = await _read_json(self.process.stdout)
            if entered.get("event") != "entered" or set(entered) != {
                "event",
                "cgroup",
            }:
                raise RuntimeError(
                    f"Scope entry failed: {entered.get('reason', 'invalid entry')}"
                )
            scope = (CGROUP_ROOT / entered["cgroup"]).resolve()
            if scope.name != self.unit or CGROUP_ROOT not in scope.parents:
                raise RuntimeError("Scope entry reported an unexpected cgroup")
            self.scope = scope
            ready = await _read_json(self.process.stdout)
        if set(ready) != {
            "event",
            "namespace_pid",
            "worker_instance_id",
            "import_identity",
        } or (ready["event"] != "ready" or type(ready["namespace_pid"]) is not int):
            raise RuntimeError("Invalid worker readiness")
        self.supervisor = self._find_supervisor(ready["namespace_pid"])
        self.instance_id = ready["worker_instance_id"]
        self.import_identity = ready["import_identity"]
        return {
            "event": "ready",
            "worker_pid": self.supervisor.pid,
            "worker_instance_id": self.instance_id,
        }

    def _find_supervisor(self, namespace_pid: int) -> ProcessIdentity:
        """Read-only lookup: the one nested-namespace member with the ready PID."""
        matches = []
        for pid in cgroup_pids(self.scope / SUPERVISOR_LEAF):
            try:
                nested = len(_nspids(pid)) > 1
                identity = process_identity(pid)
            except (FileNotFoundError, StopIteration):
                continue
            if nested and identity.namespace_pid == namespace_pid:
                matches.append(identity)
        if len(matches) != 1:
            raise RuntimeError("Cannot identify the warm supervisor on the host")
        return matches[0]

    async def _send(self, message: dict) -> None:
        line = (json.dumps(message, separators=(",", ":")) + "\n").encode()
        self.process.stdin.write(line)
        await self.process.stdin.drain()

    def _verify_placement(self, leaf: Path, ident: str, row: dict) -> ProcessIdentity:
        if set(row) != {"id", "event", "namespace_pid"} or (
            row["id"] != ident
            or row["event"] != "child_stopped"
            or type(row["namespace_pid"]) is not int
        ):
            raise PlacementError("Invalid child_stopped frame")
        members = cgroup_pids(leaf)
        if len(members) != 1:
            raise PlacementError(f"Request leaf has {len(members)} members")
        child = process_identity(members[0])
        try:
            validate_child(child, self.supervisor, row["namespace_pid"])
        except ValueError as exc:
            raise PlacementError(str(exc)) from exc
        return child

    def _result(self, ident: str, status: str, row: dict | None = None, notice=""):
        from .protocol import PythonResult

        row = row or {}
        stdout, stderr, value = (
            row.get("stdout", ""),
            row.get("stderr", ""),
            row.get("value"),
        )
        truncated = bool(row.get("truncated", False))
        if notice:
            stdout, stderr, value, cut = bounded_fields(stdout, stderr, value, notice)
            truncated |= cut
        return PythonResult(
            ident,
            status,
            stdout=stdout,
            stderr=stderr,
            value=value,
            truncated=truncated,
            worker_pid=self.supervisor.pid if self.supervisor else None,
            worker_instance_id=self.instance_id,
        )

    def _worker_row(self, ident: str, row: dict) -> dict:
        expected = {
            "id",
            "event",
            "status",
            "stdout",
            "stderr",
            "value",
            "truncated",
            "worker_instance_id",
            "import_identity",
        }
        if (
            set(row) != expected
            or row["id"] != ident
            or row["event"] != "result"
            or row["status"] not in ("ok", "error")
            or row["worker_instance_id"] != self.instance_id
            or row["import_identity"] != self.import_identity
            or not isinstance(row["stdout"], str)
            or not isinstance(row["stderr"], str)
            or (row["value"] is not None and not isinstance(row["value"], str))
            or type(row["truncated"]) is not bool
        ):
            raise ValueError("Malformed worker result")
        return row

    async def execute(
        self, ident: str, code: str, cancel: asyncio.Event, *, check: dict | None = None
    ):
        if self.broken:
            return self._result(ident, "worker_died", notice=self.broken)
        self._sequence += 1
        leaf = self.scope / f"request-{self._sequence}"
        scope_ooms = oom_kills_safe(self.scope)
        try:
            leaf.mkdir()
            (leaf / "memory.oom.group").write_text("1")
            procs = os.open(leaf / "cgroup.procs", os.O_WRONLY | os.O_CLOEXEC)
            try:
                socket.send_fds(self.control, [b"P"], [procs])
            finally:
                os.close(procs)
            await self._send(
                {"id": ident, "op": "verify", "check": check}
                if check is not None
                else {"id": ident, "op": "execute", "code": code}
            )
            async with asyncio.timeout(config.PYTHON_RESPONSE_SECONDS):
                stopped = await _read_json(self.process.stdout)
            self._verify_placement(leaf, ident, stopped)
            await self._send({"id": ident, "op": "resume"})
            response = asyncio.create_task(_read_json(self.process.stdout))
            status, row = await await_call(
                response, cancel, leaf, deadline=config.PYTHON_RESPONSE_SECONDS
            )
            row = self._worker_row(ident, row)
            # Background descendants of a returned call never outlive their request.
            await kill_job(leaf)
            killed = oom_kills(leaf)
            leaf.rmdir()
        except (EOFError, OSError, ValueError, RuntimeError, TimeoutError) as exc:
            reason = f"{type(exc).__name__}: {exc}"
            ooms = oom_kills_safe(self.scope) - scope_ooms
            if ooms > 0:
                reason = OOM_NOTICE.format(count=ooms) + " " + reason
            await self.fail(reason)
            return self._result(ident, "worker_died", notice=reason)
        if killed:
            return self._result(ident, "error", row, OOM_NOTICE.format(count=killed))
        return self._result(ident, status if status != "ok" else row["status"], row)

    async def fail(self, reason: str) -> None:
        """Fail closed: kill every scope member and refuse further calls."""
        self.broken = self.broken or reason
        await self.close(graceful=False)

    async def close(self, *, graceful: bool = True) -> None:
        process = self.process
        if process is not None and process.returncode is None and graceful:
            try:
                await self._send({"id": uuid.uuid4().hex, "op": "shutdown"})
                await asyncio.wait_for(process.wait(), config.PYTHON_CLEANUP_SECONDS)
            except (OSError, TimeoutError):
                pass
        if self.scope is not None and self.scope.exists():
            try:
                await kill_job(self.scope)
            except (OSError, TimeoutError):
                await stop_unit(self.unit)
        if process is not None and process.returncode is None:
            try:
                await asyncio.wait_for(process.wait(), config.PYTHON_CLEANUP_SECONDS)
            except TimeoutError:
                await stop_unit(self.unit)
                process.kill()
                await process.wait()
        if self.control is not None:
            self.control.close()
            self.control = None


async def _guard_loop(unit: str) -> int:
    from .protocol import PythonResult, decode_frame

    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=config.PYTHON_MAX_FRAME_BYTES)
    await loop.connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader), os.fdopen(0, "rb", buffering=0)
    )

    def emit(row: dict) -> bool:
        try:
            sys.stdout.write(json.dumps(row, separators=(",", ":")) + "\n")
            sys.stdout.flush()
        except OSError:
            return False  # Client is gone; scope cleanup still runs below.
        return True

    guard = Guard(unit)
    try:
        emit(await guard.start())
    except (OSError, RuntimeError, ValueError, EOFError, TimeoutError) as exc:
        await guard.close(graceful=False)
        result = PythonResult("startup", "unavailable", stderr=str(exc)[:1024])
        emit({"event": "result", **asdict(result)})
        return 1

    queue: asyncio.Queue = asyncio.Queue()
    active: dict[str, asyncio.Event] = {}
    cancelled: set[str] = set()

    async def client_frames():
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                message = decode_frame(line)
                if message["op"] == "cancel":
                    event = active.get(message["request_id"])
                    if event is not None:
                        event.set()
                    else:
                        cancelled.add(message["request_id"])
                    continue
                if message["op"] == "shutdown":
                    break
                await queue.put(message)
        except (ValueError, OSError):
            pass
        # Client EOF, shutdown or a malformed frame cancels the active call.
        for event in active.values():
            event.set()
        await queue.put(None)

    frames = asyncio.create_task(client_frames())
    try:
        while (message := await queue.get()) is not None:
            event = asyncio.Event()
            if message["id"] in cancelled or frames.done():
                event.set()
            active[message["id"]] = event
            try:
                result = await guard.execute(
                    message["id"],
                    message.get("code", ""),
                    event,
                    check=message.get("check"),
                )
            finally:
                active.pop(message["id"], None)
            if not emit({"event": "result", **asdict(result)}) or guard.broken:
                break
    finally:
        frames.cancel()
        await asyncio.gather(frames, return_exceptions=True)
        await guard.close(graceful=not guard.broken)
    return 0


def guard_main(unit: str) -> int:
    """Host guard process: client frames on stdin, results on stdout."""
    return asyncio.run(_guard_loop(unit))


if __name__ == "__main__":
    if sys.argv[1:] == ["--probe-host"]:
        print(json.dumps(_probe_scope()))
    elif len(sys.argv) == 3 and sys.argv[1] == "--guard":
        raise SystemExit(guard_main(sys.argv[2]))
    elif len(sys.argv) > 5 and sys.argv[1] == "--enter" and sys.argv[4] == "--":
        raise SystemExit(enter_scope(sys.argv[2], int(sys.argv[3]), sys.argv[5:]))
    else:
        raise SystemExit("Expected --probe-host, --guard UNIT or --enter UNIT FD --")
