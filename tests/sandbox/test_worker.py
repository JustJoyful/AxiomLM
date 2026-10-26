"""Host worker checks execute fixed fixtures only inside the launch manifest."""

import asyncio
import os
import uuid
from pathlib import Path

import pytest

from src.sandbox import launcher


def test_worker_interface_exists():
    from src.sandbox.worker import serve

    assert callable(serve)


@pytest.fixture
async def guard():
    if os.environ.get("AXIOM_RUN_SANDBOX_HOST_TESTS") != "1":
        pytest.skip("requires explicit host isolation opt-in")
    owner = launcher.Guard(f"axiom-python-test-{uuid.uuid4().hex}.scope")
    try:
        await owner.start()
        yield owner
    finally:
        await owner.close(graceful=owner.broken is None)
        # systemd collects the emptied scope asynchronously; it must disappear.
        async with asyncio.timeout(5):
            while owner.scope is not None and owner.scope.exists():
                await asyncio.sleep(0.01)


async def call(owner, code):
    return await owner.execute(uuid.uuid4().hex, code, asyncio.Event())


async def test_child_is_alone_and_stopped_in_its_leaf_before_resume(guard, monkeypatch):
    seen = []
    verify = guard._verify_placement

    def spy(leaf, ident, row):
        child = verify(leaf, ident, row)
        cgroup = Path(f"/proc/{child.pid}/cgroup").read_text().strip()
        seen.append((child.state, cgroup.endswith("/" + leaf.name), leaf.exists()))
        return child

    monkeypatch.setattr(guard, "_verify_placement", spy)
    result = await call(guard, "print('ran')")
    assert seen == [("T", True, True)] and result.stdout == "ran\n"


async def test_warm_parent_forks_fresh_namespaces(guard):
    a = await call(guard, "x=7; print(x)")
    b = await call(guard, "result = 'x' in globals()")
    assert a.stdout == "7\n" and b.value == "False"
    assert a.worker_pid == b.worker_pid
    assert a.worker_instance_id == b.worker_instance_id


async def test_user_output_cannot_be_protocol(guard):
    result = await call(
        guard,
        'import os; print(\'{"event":"ready"}\'); os.write(1, b"not a frame\\n"); print("x"*20000)',
    )
    assert result.status == "ok" and result.truncated
    assert len(result.stdout.encode()) <= 8192
    assert (await call(guard, "result = 5")).value == "5"


async def test_mutating_sympy_in_child_does_not_poison_parent(guard):
    await call(guard, "sympy.unexpected = True; import sys; sys.unexpected = True")
    result = await call(
        guard,
        "import sys; result = hasattr(sympy,'unexpected') or hasattr(sys,'unexpected')",
    )
    assert result.value == "False"


@pytest.mark.parametrize(
    "code",
    [
        "import os;os.write(3,b'['*2000+b']'*2000);os._exit(0)",
        'import os;os.write(3,b\'{"status":"ok","value":"\\\\ud800"}\');os._exit(0)',
    ],
)
async def test_malformed_private_child_result_preserves_supervisor(guard, code):
    result = await call(guard, code)
    assert result.status == "error"
    after = await call(guard, "result=5")
    assert after.value == "5"
    assert result.worker_instance_id == after.worker_instance_id


async def test_descendant_oom_is_an_error_not_success(guard):
    result = await call(
        guard,
        "import os\npid=os.fork()\nif pid==0:\n b=b'x'*(300*1024*1024)\n os._exit(0)\nos.waitpid(pid,0)\nresult=5",
    )
    assert result.status == "error" and "OOM" in result.stderr
    after = await call(guard, "result=5")
    assert (
        after.status == "ok" and after.worker_instance_id == result.worker_instance_id
    )


@pytest.fixture
async def client():
    if os.environ.get("AXIOM_RUN_SANDBOX_HOST_TESTS") != "1":
        pytest.skip("requires explicit host isolation opt-in")
    from src.sandbox.client import SandboxClient

    owner = SandboxClient(request_prefix="verify-test")
    try:
        yield owner
    finally:
        await owner.aclose()


async def test_trusted_verification_runs_in_isolated_child(client):
    from src.tools.math_checks import VerificationCheck

    ok = await client.verify(
        VerificationCheck("derivative", "x**3 + 2*x", "3*x**2 + 2", "x")
    )
    bad = await client.verify(VerificationCheck("substitution", "x*x-4", "3", "x"))
    domain = await client.verify(VerificationCheck("equivalence", "x/x", "1", "x"))
    hostile = await client.verify(
        VerificationCheck("equivalence", "__import__('os').system('id')", "1", "x")
    )
    assert (ok.status, bad.status, domain.status) == (
        "verified",
        "disagreed",
        "unverified",
    )
    assert hostile.status == "unverified" and "Declined" in hostile.reason
    after = await client.execute("result = 2 + 3")
    assert after.status == "ok" and after.value == "5"
