"""Fixed adversarial diagnostics, exclusively through the native sandbox client."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import signal
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src.sandbox import launcher
from src.sandbox.client import SandboxClient

REQUIRED_CHECKS = (
    "scope_limits",
    "filesystem",
    "network",
    "supervisor_protection",
    "scratch",
    "fresh_warm_cache",
    "timeout_recovery",
    "native_timeout",
    "sympy_timeout",
    "success_descendants",
    "timeout_descendants",
    "cancel_descendants",
    "fork_bomb",
    "protocol_output",
    "malformed_child_output",
    "request_handles",
    "oom_recovery",
    "descendant_oom",
    "worker_death",
    "guard_death",
    "client_eof",
    "tui_sigkill",
    "preservation",
    "final_cleanup",
)


DESCENDANTS = """import os,time
if os.fork()==0:
 os.setsid()
 if os.fork()==0:
  while True:time.sleep(1)
 os._exit(0)
time.sleep(.1)
"""


def all_passed(checks: dict) -> bool:
    return all(checks.get(name, {}).get("passed") is True for name in REQUIRED_CHECKS)


def _hashes(output: Path) -> dict:
    files = []
    for root in (config.VECTOR_STORE, config.PARSED_MD, config.LOGS_DIR):
        if root.exists():
            files.extend(p for p in root.rglob("*") if p.is_file() and p != output)
    files.extend([config.ROOT / "src/config.py", config.ROOT / "uv.lock"])
    return {
        str(p.relative_to(config.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(set(files))
    }


def _group(client):
    identity = (
        Path(f"/proc/{client.worker_pid}/cgroup").read_text().strip().split("0::")[1]
    )
    path = Path("/sys/fs/cgroup") / identity.lstrip("/")
    if path.name != "supervisor" or path.parent.name != client.unit:
        raise RuntimeError("Unexpected supervisor cgroup")
    return path.parent


def _members(group):
    result = []
    if group.exists():
        for path in group.rglob("cgroup.procs"):
            for pid in path.read_text().split():
                try:
                    result.append(asdict(launcher.process_identity(int(pid))))
                except FileNotFoundError:
                    pass
    return result


def _alive(rows):
    alive = []
    for row in rows:
        try:
            current = launcher.process_identity(row["pid"])
            if current.start_time == row["start_time"]:
                alive.append(row["pid"])
        except FileNotFoundError:
            pass
    return sorted(set(alive))


async def _guard_death(client) -> dict:
    """SIGKILL the guard during a call with setsid descendants.

    Client emergency cleanup (``stop_unit``) is replaced by a recorder for the whole
    check, so teardown can only come from the Bubblewrap/pid-namespace parent-death
    chain, not from client code or Python finally blocks.
    """
    capability = await client.start()
    if not capability.available:
        raise RuntimeError(capability.reason)
    group = _group(client)
    unit = client.unit
    recorded = []
    original = launcher.stop_unit

    async def record(name):
        recorded.append({"unit": name, "seconds": time.monotonic() - killed})

    killed = time.monotonic()
    launcher.stop_unit = record
    task = asyncio.create_task(client.execute(DESCENDANTS + "while True:time.sleep(1)"))
    try:
        async with asyncio.timeout(config.PYTHON_RESPONSE_SECONDS):
            while True:
                leaves = list(group.glob("request-*"))
                if (
                    leaves
                    and len((leaves[0] / "cgroup.procs").read_text().split()) >= 2
                ):
                    break
                await asyncio.sleep(config.PYTHON_POLL_SECONDS)
        members = _members(group)
        guard = asdict(launcher.process_identity(client.process.pid))
        killed = time.monotonic()
        os.kill(guard["pid"], signal.SIGKILL)
        async with asyncio.timeout(config.PYTHON_STARTUP_SECONDS):
            while group.exists() or _alive(members + [guard]):
                await asyncio.sleep(config.PYTHON_POLL_SECONDS)
        teardown = time.monotonic() - killed
        result = await asyncio.wait_for(task, config.PYTHON_STARTUP_SECONDS)
    finally:
        launcher.stop_unit = original
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    remaining = _alive(members + [guard])
    return {
        "passed": result.status == "worker_died"
        and not remaining
        and not group.exists()
        and len(members) >= 5,
        "unit": unit,
        "guard": guard,
        "observed_pids": members,
        "remaining_pids": remaining,
        "teardown_seconds": teardown,
        "emergency_cleanup_disabled": True,
        "suppressed_stop_unit_calls": recorded,
        "result": asdict(result),
    }


async def _harness(metadata: Path, mode: str):
    client = SandboxClient(request_prefix="parent-death-fixture")
    capability = await client.start()
    if not capability.available:
        raise RuntimeError(capability.reason)
    metadata.write_text(
        json.dumps(
            {
                "unit": client.unit,
                "worker_pid": client.worker_pid,
                "guard_pid": client.process.pid,
            }
        )
    )
    task = asyncio.create_task(
        client.execute(
            "import os,time\nif os.fork()==0:\n os.setsid()\n while True: time.sleep(1)\nwhile True: time.sleep(1)"
        )
    )
    if mode == "eof":
        await asyncio.sleep(0.3)
        os._exit(0)
    await task


async def _parent_death(output, mode):
    metadata = output.parent / (mode + "-harness.json")
    harness = await asyncio.create_subprocess_exec(
        sys.executable,
        str(Path(__file__).resolve()),
        "--harness",
        str(metadata),
        "--mode",
        mode,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        async with asyncio.timeout(config.PYTHON_STARTUP_SECONDS * 2):
            while not metadata.exists():
                if harness.returncode is not None:
                    raise RuntimeError("Harness startup failed")
                await asyncio.sleep(config.PYTHON_POLL_SECONDS)
            data = json.loads(metadata.read_text())
            group = Path("/sys/fs/cgroup") / Path(
                f'/proc/{data["worker_pid"]}/cgroup'
            ).read_text().strip().split("0::")[1].lstrip("/")
            group = group.parent
            while not list(group.glob("request-*")):
                await asyncio.sleep(config.PYTHON_POLL_SECONDS)
            await asyncio.sleep(0.1)
            members = _members(group)
            members.append(asdict(launcher.process_identity(data["guard_pid"])))
            if mode == "sigkill":
                harness.kill()
            await harness.wait()
            async with asyncio.timeout(config.PYTHON_STARTUP_SECONDS):
                while group.exists() or _alive(members):
                    await asyncio.sleep(config.PYTHON_POLL_SECONDS)
            return {
                "passed": True,
                "unit": data["unit"],
                "observed_pids": members,
                "remaining_pids": _alive(members),
            }
    finally:
        if harness.returncode is None:
            harness.kill()
            await harness.wait()
        if metadata.exists():
            await launcher.stop_unit(json.loads(metadata.read_text())["unit"])


async def _validate(output: Path) -> dict:
    report = {
        "started_utc": datetime.now(UTC).isoformat(),
        "passed": False,
        "blocked_reason": None,
        "checks": {},
        "rows": [],
    }
    protected = _hashes(output)
    report["protected_before"] = protected
    client = SandboxClient(request_prefix="host-diagnostic")
    tracked = []
    groups = set()
    started = time.monotonic()

    def check(name, passed, **observed):
        report["checks"][name] = {"passed": bool(passed), **observed}
        if not passed:
            raise RuntimeError("Mandatory host check failed: " + name)

    async def run(code):
        stamp = time.monotonic()
        row = await client.execute(code)
        report["rows"].append(
            {"code": code, "result": asdict(row), "seconds": time.monotonic() - stamp}
        )
        if client.process is not None and client.process.returncode is None:
            group = _group(client)
            groups.add(group)
            tracked.extend(_members(group))
            if list(group.glob("request-*")):
                raise RuntimeError("Request cgroup survived terminal result")
        return row

    async def recovering_timeout(name, code):
        before = await run("result=2+3")
        bad = await run(code)
        after = await run("result=2+3")
        same = (
            before.worker_pid == bad.worker_pid == after.worker_pid
            and before.worker_instance_id
            == bad.worker_instance_id
            == after.worker_instance_id
        )
        check(
            name,
            bad.status == "timeout"
            and after.status == "ok"
            and after.value == "5"
            and same,
            same_worker=same,
            before=asdict(before),
            timeout=asdict(bad),
            after=asdict(after),
        )

    try:
        capability = await client.start()
        if not capability.available:
            raise RuntimeError(capability.reason)
        group = _group(client)
        groups.add(group)
        tracked.extend(_members(group))
        limits = {
            name: (group / name).read_text().strip()
            for name in (
                "memory.max",
                "memory.swap.max",
                "pids.max",
                "cgroup.subtree_control",
            )
        }
        report["effective_memory_max_bytes"] = int(limits["memory.max"])
        report["mounts"] = [
            {"source": str(m.source), "target": m.target} for m in capability.mounts
        ]
        report["runtime"] = {
            "python": sys.version,
            "executable": sys.executable,
            "model": config.OLLAMA_MODEL,
            "jina_revision": config.EMBED_REVISION,
        }
        check(
            "scope_limits",
            limits["memory.max"] == "262144000"
            and limits["memory.swap.max"] == "0"
            and limits["pids.max"] == "32"
            and {"memory", "pids"} <= set(limits["cgroup.subtree_control"].split()),
            unit=client.unit,
            observed=limits,
        )
        row = await run("""import os
paths=['/home','/root','/etc/passwd','/sys/fs/cgroup','/dev/nvidia0','/dev/host-write','/worker.py']
result={}
for path in paths:
 try:
  mode='w' if path in ('/worker.py','/dev/host-write') else 'r'
  with open(path,mode) as f: f.read(1) if mode=='r' else f.write('x')
  result[path]=False
 except OSError: result[path]=True
""")
        facts = (
            json.loads(
                row.value.replace("'", '"')
                .replace("True", "true")
                .replace("False", "false")
            )
            if row.status == "ok"
            else {}
        )
        check("filesystem", len(facts) == 7 and all(facts.values()), result=asdict(row))
        row = await run("""import socket
result=[]
for address in [('127.0.0.1',80),('1.1.1.1',80),('169.254.169.254',80)]:
 s=socket.socket();s.settimeout(.2)
 try: s.connect(address);result.append(False)
 except OSError: result.append(True)
 finally:s.close()
try:socket.getaddrinfo('example.com',443);result.append(False)
except OSError:result.append(True)
""")
        check(
            "network",
            row.status == "ok" and row.value == "[True, True, True, True]",
            result=asdict(row),
        )
        row = await run("""import os
result=[]
for path in [f'/proc/{os.getppid()}/mem',f'/proc/{os.getppid()}/fd/0']:
 try:fd=os.open(path,os.O_RDWR);os.close(fd);result.append(False)
 except OSError:result.append(True)
""")
        check(
            "supervisor_protection",
            row.status == "ok" and row.value == "[True, True]",
            result=asdict(row),
        )
        row = await run("""import errno,os
try:
 with open('/scratch/full','wb') as f:
  for i in range(40):f.write(b'x'*1048576)
 result=False
except OSError as e:result=e.errno==errno.ENOSPC
finally:
 try:os.unlink('/scratch/full')
 except FileNotFoundError:pass
""")
        check("scratch", row.status == "ok" and row.value == "True", result=asdict(row))
        a = await run("x=7;print(x);result=sympy.__version__")
        b = await run(
            "import importlib; cache=importlib.import_module('sympy.core.cache');result=('x' not in globals(),cache.SYMPY_CACHE_SIZE)"
        )
        growth = []
        for _ in range(4):
            growth.append(
                asdict(
                    await run(
                        "x=sympy.Symbol('x');result=[sympy.expand((x+i)**8) for i in range(128)][-1]"
                    )
                )
            )
        check(
            "fresh_warm_cache",
            a.status == b.status == "ok"
            and a.stdout == "7\n"
            and b.value == "(True, 128)"
            and a.worker_instance_id == b.worker_instance_id
            and all(r["status"] == "ok" for r in growth),
            first=asdict(a),
            second=asdict(b),
            growth=growth,
            memory_current=(group / "memory.current").read_text().strip(),
        )
        await recovering_timeout("timeout_recovery", "while True: pass")
        await recovering_timeout(
            "native_timeout", "import ctypes;ctypes.CDLL(None).sleep(60)"
        )
        await recovering_timeout(
            "sympy_timeout",
            "x=sympy.Symbol('x')\nwhile True: sympy.integrate(sympy.sin(x),x)",
        )
        descendants = DESCENDANTS
        row = await run(descendants + "result=5")
        check(
            "success_descendants",
            row.status == "ok" and row.value == "5",
            result=asdict(row),
            remaining_request_groups=[p.name for p in group.glob("request-*")],
        )
        await recovering_timeout(
            "timeout_descendants", descendants + "while True:time.sleep(1)"
        )
        task = asyncio.create_task(
            client.execute(descendants + "while True:time.sleep(1)")
        )
        async with asyncio.timeout(config.PYTHON_RESPONSE_SECONDS):
            while not list(group.glob("request-*")):
                await asyncio.sleep(config.PYTHON_POLL_SECONDS)
        await asyncio.sleep(0.1)
        observed = _members(group)
        parent_id = client.worker_instance_id
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        row = await run("result=2+3")
        check(
            "cancel_descendants",
            row.status == "ok"
            and row.worker_instance_id == parent_id
            and not list(group.glob("request-*")),
            result=asdict(row),
            observed_pids=observed,
        )
        await recovering_timeout(
            "fork_bomb",
            """import os,time
while True:
 try:
  if os.fork()==0:
   while True:time.sleep(1)
 except OSError:time.sleep(.01)
""",
        )
        row = await run(
            'import os;os.write(1,b\'{"event":"ready"}\\n\');print("x"*100000);result=5'
        )
        check(
            "protocol_output",
            row.status == "ok" and row.truncated and len(row.stdout.encode()) <= 8192,
            result=asdict(row),
        )
        row = await run('import os;os.write(3,b"bad frame")')
        after = await run("result=5")
        check(
            "malformed_child_output",
            row.status == "error" and after.status == "ok" and after.value == "5",
            result=asdict(row),
            after=asdict(after),
        )
        row = await run("""import os
result=[]
for n in os.listdir('/proc/self/fd'):
 try:result.append(os.readlink(f'/proc/self/fd/{n}'))
 except FileNotFoundError:pass  # the listing descriptor itself
result.sort()
""")
        check(
            "request_handles",
            row.status == "ok"
            and "cgroup" not in (row.value or "cgroup")
            and "socket:" not in (row.value or "socket:"),
            result=asdict(row),
        )
        oom_before = (group / "memory.events").read_text()
        row = await run("result=bytearray(300*1024*1024)")
        oom_after = (
            (group / "memory.events").read_text() if group.exists() else "scope removed"
        )
        after = await run("result=5")
        check(
            "oom_recovery",
            row.status in ("error", "worker_died")
            and "OOM" in row.stderr
            and after.status == "ok"
            and after.value == "5"
            and oom_before != oom_after,
            result=asdict(row),
            before=oom_before,
            after_events=oom_after,
            after=asdict(after),
        )
        parent_id = client.worker_instance_id
        row = await run(
            "import os\npid=os.fork()\nif pid==0:\n b=b'x'*(300*1024*1024)\n os._exit(0)\nos.waitpid(pid,0)\nresult=5"
        )
        after = await run("result=5")
        check(
            "descendant_oom",
            row.status == "error"
            and "OOM" in row.stderr
            and after.status == "ok"
            and after.worker_instance_id == parent_id,
            result=asdict(row),
            after=asdict(after),
        )
        old = client.worker_instance_id
        row = await run("import os,signal;os.kill(os.getppid(),signal.SIGKILL)")
        after = await run("result=5")
        check(
            "worker_death",
            row.status == "worker_died"
            and after.status == "ok"
            and after.worker_instance_id != old,
            result=asdict(row),
            after=asdict(after),
        )
        report["checks"]["guard_death"] = await _guard_death(client)
        if not report["checks"]["guard_death"]["passed"]:
            raise RuntimeError("Mandatory host check failed: guard_death")
        report["checks"]["client_eof"] = await _parent_death(output, "eof")
        report["checks"]["tui_sigkill"] = await _parent_death(output, "sigkill")
    except (OSError, ValueError, RuntimeError, TimeoutError, EOFError) as exc:
        report["blocked_reason"] = str(exc)
    finally:
        cleanup_error = None
        try:
            await client.aclose()
        except (OSError, RuntimeError, TimeoutError) as exc:
            cleanup_error = str(exc)
        protected_after = _hashes(output)
        report["protected_after"] = protected_after
        report["checks"]["preservation"] = {
            "passed": all(
                protected_after.get(name) == digest
                for name, digest in protected.items()
            )
        }
        try:
            async with asyncio.timeout(config.PYTHON_STARTUP_SECONDS):
                while _alive(tracked):
                    await asyncio.sleep(config.PYTHON_POLL_SECONDS)
        except TimeoutError:
            cleanup_error = cleanup_error or "Cleanup verification timed out"
        remaining = _alive(tracked)
        report["checks"]["final_cleanup"] = {
            "passed": not cleanup_error
            and not remaining
            and not any(g.exists() for g in groups),
            "remaining_pids": remaining,
            "observed_pids": tracked,
            "error": cleanup_error,
        }
        report["passed"] = all_passed(report["checks"])
        if not report["passed"] and report["blocked_reason"] is None:
            report["blocked_reason"] = "Required host check missing or failed"
        report["seconds"] = time.monotonic() - started
        report["finished_utc"] = datetime.now(UTC).isoformat()
    return report


def validate(*, output: Path) -> dict:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve exclusively before launching anything; prior evidence is never overwritten.
    with output.open("x") as stream:
        stream.write('{"status":"running"}\n')
        stream.flush()
        report = asyncio.run(_validate(output))
        stream.seek(0)
        json.dump(report, stream, indent=2)
        stream.truncate()
        stream.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--harness", type=Path)
    parser.add_argument("--mode", choices=("eof", "sigkill"))
    args = parser.parse_args()
    if args.harness:
        asyncio.run(_harness(args.harness, args.mode))
    elif args.output:
        result = validate(output=args.output)
        print(
            json.dumps(
                {
                    "passed": result["passed"],
                    "blocked_reason": result["blocked_reason"],
                    "checks": {k: v["passed"] for k, v in result["checks"].items()},
                }
            )
        )
        raise SystemExit(0 if result["passed"] else 1)
    else:
        parser.error("--output required")
