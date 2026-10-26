"""Warm fork supervisor. Generated code runs only in a self-placed, resumed child."""

from __future__ import annotations

import ctypes
import json
import os
import resource
import select
import signal
import socket
import sys
import traceback
import uuid

PR_SET_DUMPABLE = 4
# Fixed trusted code for verify requests; the check arrives as data, never as code.
VERIFY_CODE = (
    "from math_checks import run_check\n"
    "result = run_check(__axiom_check, __axiom_math_limits)"
)
CHECK_FIELDS = {"kind", "expression", "candidate", "variable"}


def _send(fd: int, message: dict, limit: int) -> None:
    frame = (json.dumps(message, ensure_ascii=True, allow_nan=False) + "\n").encode()
    if len(frame) > limit:
        raise ValueError("Oversized worker frame")
    while frame:
        frame = frame[os.write(fd, frame) :]


def _read(stream, limit):
    frame = stream.readline(limit + 1)
    if not frame:
        raise EOFError
    if len(frame) > limit or not frame.endswith(b"\n"):
        raise ValueError("Invalid worker frame")
    pairs = json.loads(frame, object_pairs_hook=_unique)
    if not isinstance(pairs, dict):
        raise ValueError("Expected object")  # noqa: TRY004
    return pairs


def _unique(items):
    message = {}
    for key, value in items:
        if key in message:
            raise ValueError("Duplicate key")
        message[key] = value
    return message


def _placement(control) -> int:
    """Receive exactly one guard-opened request ``cgroup.procs`` descriptor."""
    message, fds, flags, _ = socket.recv_fds(control, 1, 1)
    if message != b"P" or len(fds) != 1 or flags & socket.MSG_CTRUNC:
        for fd in fds:
            os.close(fd)
        raise ValueError("Invalid request placement")
    return fds[0]


def _child(code, namespace, placement, output, error, value, limit):
    # Self-placement first: "0" is this process, so no PID can be confused, and
    # every later descendant inherits the request cgroup.
    os.write(placement, b"0")
    os.close(placement)
    # Prefer request processes over the warm supervisor as OOM victims. The inherited
    # non-dumpable flag hides /proc/self, so lift it only for this trusted write;
    # no untrusted process exists until the guard resumes this child.
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl(PR_SET_DUMPABLE, 1, 0, 0, 0)
    with open("/proc/self/oom_score_adj", "w") as stream:
        stream.write("1000")
    libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0)
    os.dup2(os.open("/dev/null", os.O_RDONLY), 0)
    os.dup2(output, 1)
    os.dup2(error, 2)
    os.dup2(value, 3)
    os.closerange(4, resource.getrlimit(resource.RLIMIT_NOFILE)[0])
    sys.stdout = os.fdopen(1, "w", encoding="utf-8", closefd=False)
    sys.stderr = os.fdopen(2, "w", encoding="utf-8", closefd=False)
    # No code runs before the host guard has placed this child in its job cgroup.
    os.kill(os.getpid(), signal.SIGSTOP)
    result = {"status": "ok", "value": None}
    try:
        local = dict(namespace)
        exec(code, local)  # noqa: S102 — only the isolated call child executes code.
        if "result" in local:
            result["value"] = repr(local["result"])[:limit]
    except BaseException:  # noqa: BLE001
        # SystemExit and KeyboardInterrupt are call errors, not supervisor exits.
        result["status"] = "error"
        traceback.print_exc()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        _send(3, result, limit * 6 + 128)
    finally:
        os._exit(0)


def _execute(code, namespace, ident, stream, output_fd, control, limits):
    placement = _placement(control)
    pipes = [os.pipe() for _ in range(3)]
    pid = os.fork()
    if pid == 0:
        try:
            _child(code, namespace, placement, *(p[1] for p in pipes), limits["output"])
        except BaseException:  # noqa: BLE001
            traceback.print_exc()  # Pre-stop setup failure; supervisor stderr only.
        finally:
            os._exit(1)
    os.close(placement)
    reads = [p[0] for p in pipes]
    for _, write in pipes:
        os.close(write)
    buffers = {fd: bytearray() for fd in reads}
    truncated = False
    try:
        _, status = os.waitpid(pid, os.WUNTRACED)
        if not os.WIFSTOPPED(status):
            raise RuntimeError("Call child did not stop")
        _send(
            output_fd,
            {"id": ident, "event": "child_stopped", "namespace_pid": pid},
            limits["frame"],
        )
        resume = _read(stream, limits["frame"])
        if resume != {"id": ident, "op": "resume"}:
            raise ValueError("Resume identity mismatch")
        os.kill(pid, signal.SIGCONT)
        for fd in reads:
            os.set_blocking(fd, False)
        exited = False
        while True:
            ready, _, _ = select.select(reads + [stream], [], [], limits["poll"])
            if stream in ready:
                # Only guard EOF/shutdown is legal while a call is active.
                raise EOFError
            for fd in ready:
                if fd is stream:
                    continue
                data = os.read(fd, limits["output"] + 1)
                remaining = (
                    limits["output"] if fd != reads[2] else limits["output"] * 6 + 128
                )
                used = len(buffers[fd])
                buffers[fd].extend(data[: max(0, remaining - used)])
                truncated |= used + len(data) > remaining
            if not exited:
                exited = os.waitpid(pid, os.WNOHANG)[0] == pid
            if exited:
                # A background descendant must not hold the result open indefinitely.
                for fd in reads:
                    try:
                        data = os.read(fd, limits["frame"])
                    except BlockingIOError:
                        continue
                    cap = (
                        limits["output"] * 6 + 128
                        if fd == reads[2]
                        else limits["output"]
                    )
                    used = len(buffers[fd])
                    buffers[fd].extend(data[: max(0, cap - used)])
                    truncated |= used + len(data) > cap
                break
        try:
            result = json.loads(buffers[reads[2]])
            if (
                set(result) != {"status", "value"}
                or result["status"] not in ("ok", "error")
                or (
                    result["value"] is not None and not isinstance(result["value"], str)
                )
            ):
                raise ValueError("Invalid child result")
            if result["value"] is not None:
                result["value"].encode("utf-8")
        except (ValueError, TypeError, RecursionError, UnicodeError):
            result = {"status": "error", "value": None}
        remaining = limits["output"]
        fields = {}
        for key, data in (
            ("stdout", bytes(buffers[reads[0]])),
            ("stderr", bytes(buffers[reads[1]])),
            ("value", (result["value"] or "").encode()),
        ):
            part = data[:remaining].decode("utf-8", errors="ignore")
            fields[key] = (
                part if key != "value" or result["value"] is not None else None
            )
            truncated |= len(data) > remaining
            remaining -= len(part.encode())
        return {"status": result["status"], **fields, "truncated": truncated}
    finally:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
        for fd in reads:
            os.close(fd)


def serve(input_fd: int, output_fd: int) -> None:
    limits = json.loads(os.environ["AXIOM_PYTHON_LIMITS"])
    import sympy

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:  # Protects supervisor memory.
        raise OSError(ctypes.get_errno(), "Cannot protect supervisor")
    control = socket.socket(fileno=limits["control_fd"])
    control.settimeout(limits["placement"])
    namespace = {"sympy": sympy, "__builtins__": __builtins__}
    identity = {"worker_instance_id": uuid.uuid4().hex, "import_identity": id(sympy)}
    _send(
        output_fd,
        {"event": "ready", "namespace_pid": os.getpid(), **identity},
        limits["frame"],
    )
    with os.fdopen(os.dup(input_fd), "rb", buffering=0) as stream:
        while True:
            try:
                message = _read(stream, limits["frame"])
            except EOFError:
                return
            ident = message.get("id")
            if not isinstance(ident, str) or not ident or len(ident) > limits["id"]:
                raise ValueError("Invalid worker request ID")
            if message == {"id": ident, "op": "shutdown"}:
                return
            if set(message) == {"id", "op", "check"} and message["op"] == "verify":
                check = message["check"]
                if not isinstance(check, dict) or set(check) != CHECK_FIELDS:
                    raise ValueError("Invalid verify request")
                call_namespace = {
                    **namespace,
                    "__axiom_check": dict(check),
                    "__axiom_math_limits": dict(limits["math"]),
                }
                result = _execute(
                    VERIFY_CODE,
                    call_namespace,
                    ident,
                    stream,
                    output_fd,
                    control,
                    limits,
                )
                _send(
                    output_fd,
                    {"id": ident, "event": "result", **result, **identity},
                    limits["frame"],
                )
                continue
            if (
                set(message) != {"id", "op", "code"}
                or message["op"] != "execute"
                or not isinstance(message["code"], str)
                or len(message["code"].encode()) > limits["code"]
            ):
                raise ValueError("Invalid execute request")
            result = _execute(
                message["code"], namespace, ident, stream, output_fd, control, limits
            )
            _send(
                output_fd,
                {"id": ident, "event": "result", **result, **identity},
                limits["frame"],
            )


if __name__ == "__main__":
    sys.path.insert(0, "/packages")
    sys.path.insert(1, "/trusted")
    serve(0, 1)
