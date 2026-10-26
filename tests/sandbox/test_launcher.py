from pathlib import Path

import pytest

from src.sandbox import launcher


def test_launch_is_closed_when_required_boundary_missing(monkeypatch):
    monkeypatch.setattr(launcher.shutil, "which", lambda name: None)
    result = launcher.check_capabilities()
    assert not result.available and "bwrap" in result.reason
    assert result.mounts == ()


def test_missing_delegation_disables_launch(monkeypatch):
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(
        launcher,
        "probe_host",
        lambda: (_ for _ in ()).throw(RuntimeError("cgroup delegation unavailable")),
    )
    result = launcher.check_capabilities()
    assert not result.available and "delegation" in result.reason


def test_mount_manifest_excludes_host_data_and_resolves_symlinks():
    mounts = launcher.runtime_mounts()
    assert mounts and all(
        m.source.is_absolute() and m.source == m.source.resolve() for m in mounts
    )
    for mount in mounts:
        assert mount.source.exists()
        assert str(mount.source) not in (
            str(Path.home()),
            str(Path.cwd()),
            "/",
            "/usr",
            "/lib",
            "/sys/fs/cgroup",
        )
        assert "site-packages" not in str(mount.source) or mount.target in (
            "/packages/sympy",
            "/packages/mpmath",
        )
        assert not mount.target.startswith(("/home", "/sys", "/dev"))
    targets = {m.target for m in mounts}
    assert {
        "/runtime/bin/python",
        "/packages/sympy",
        "/packages/mpmath",
        "/protocol.py",
        "/launcher.py",
    } <= targets


def test_launch_has_mandatory_boundaries_and_bounded_scratch():
    mounts = launcher.runtime_mounts()
    args = launcher.build_launch("axiom-python-test.scope", mounts)
    assert args[:3] == ("systemd-run", "--user", "--scope")
    for value in (
        "MemoryMax=250M",
        "MemorySwapMax=0",
        "TasksMax=32",
        "Delegate=yes",
        "KillMode=control-group",
    ):
        assert value in args
    for flag in (
        "--unshare-user",
        "--unshare-pid",
        "--unshare-net",
        "--unshare-ipc",
        "--unshare-uts",
        "--new-session",
        "--die-with-parent",
        "--clearenv",
    ):
        assert flag in args
    assert ("--size", "33554432", "--tmpfs", "/scratch") == args[
        args.index("--size") : args.index("--size") + 4
    ]
    assert args[-4:] == ("/runtime/bin/python", "-I", "-S", "/worker.py")


def test_unapproved_mount_or_unit_is_rejected(tmp_path):
    for mount in (
        launcher.RuntimeMount(Path.home(), "/home"),
        launcher.RuntimeMount(tmp_path, "/packages/sympy"),
    ):
        with pytest.raises(ValueError):
            launcher.build_launch("axiom-python-test.scope", (mount,))
    with pytest.raises(ValueError):
        launcher.build_launch("unrelated.scope", launcher.runtime_mounts())


def test_pidfd_identity_works_with_portable_cpython():
    import os
    import select

    # The selected portable build lacks os.pidfd_open; libc still exposes it.
    fd = launcher.open_pidfd(os.getpid())
    try:
        assert select.select([fd], [], [], 0)[0] == []
    finally:
        os.close(fd)


def test_capabilities_available_only_after_host_probe(monkeypatch):
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(launcher, "probe_host", dict)
    monkeypatch.setattr(launcher, "runtime_mounts", lambda: ())
    result = launcher.check_capabilities()
    assert result.available and result.reason is None


def test_launch_enters_scope_through_trusted_self_placement():
    args = launcher.build_launch(
        "axiom-python-test.scope", launcher.runtime_mounts(), control_fd=9
    )
    entry = args.index("--enter")
    assert args[entry + 1 : entry + 5] == (
        "axiom-python-test.scope",
        "9",
        "--",
        "bwrap",
    )
    assert args[entry - 2 : entry] == ("-I", str(Path(launcher.__file__).resolve()))


def test_no_numeric_pid_is_ever_written_to_cgroup_procs():
    import re

    source = Path(launcher.__file__).read_text()
    writes = re.findall(r'"cgroup\.procs"\)\.write_text\(([^)]*)\)', source)
    assert writes == ['"0"']
    assert "socket.send_fds" in source


def _fake_scope(root, unit):
    scope = root / "user.slice" / unit
    scope.mkdir(parents=True)
    for name, value in {
        "memory.max": "262144000",
        "memory.swap.max": "0",
        "pids.max": "32",
        "cgroup.controllers": "cpu memory pids",
        "cgroup.kill": "",
        "cgroup.subtree_control": "",
    }.items():
        (scope / name).write_text(value + "\n")
    return scope


def test_scope_entry_self_places_then_delegates(tmp_path, monkeypatch, capsys):
    unit = "axiom-python-test.scope"
    scope = _fake_scope(tmp_path, unit)
    real_mkdir = Path.mkdir

    def mkdir(self, *args, **kwargs):
        real_mkdir(self, *args, **kwargs)
        (self / "cgroup.procs").write_text("")

    executed = []
    monkeypatch.setattr(Path, "mkdir", mkdir)
    monkeypatch.setattr(launcher, "CGROUP_ROOT", tmp_path)
    monkeypatch.setattr(launcher, "own_cgroup", lambda: scope)
    monkeypatch.setattr(launcher.os, "execvp", lambda *a: executed.append(a))
    monkeypatch.setattr(launcher.os, "set_inheritable", lambda *a: None)
    launcher.enter_scope(unit, launcher.CONTROL_FD, ["bwrap", "--x"])
    assert (scope / "supervisor/cgroup.procs").read_text() == "0"
    assert (scope / "cgroup.subtree_control").read_text() == "+memory +pids"
    assert executed == [("bwrap", ["bwrap", "--x"])]
    import json

    assert json.loads(capsys.readouterr().out)["cgroup"] == f"user.slice/{unit}"


def test_scope_entry_refuses_wrong_unit_or_limits(tmp_path, monkeypatch, capsys):
    import json

    scope = _fake_scope(tmp_path, "axiom-python-other.scope")
    monkeypatch.setattr(launcher, "own_cgroup", lambda: scope)
    monkeypatch.setattr(
        launcher.os, "execvp", lambda *a: (_ for _ in ()).throw(AssertionError)
    )
    assert launcher.enter_scope("axiom-python-test.scope", 3, ["bwrap"]) == 1
    assert json.loads(capsys.readouterr().out)["event"] == "enter_failed"
    (scope / "memory.max").write_text("max\n")
    assert launcher.enter_scope("axiom-python-other.scope", 3, ["bwrap"]) == 1
    assert "limits" in json.loads(capsys.readouterr().out)["reason"]
    assert not (scope / "supervisor").exists()


def _guard_with_leaf(tmp_path, pids):
    guard = launcher.Guard("axiom-python-test.scope")
    guard.supervisor = launcher.ProcessIdentity(100, 10, 2, 90, "S")
    leaf = tmp_path / "request-1"
    leaf.mkdir()
    (leaf / "cgroup.procs").write_text("".join(f"{pid}\n" for pid in pids))
    return guard, leaf


@pytest.mark.parametrize("pids", [[], [101, 102]])
def test_placement_requires_exactly_one_member(tmp_path, pids):
    guard, leaf = _guard_with_leaf(tmp_path, pids)
    row = {"id": "r", "event": "child_stopped", "namespace_pid": 3}
    with pytest.raises(launcher.PlacementError):
        guard._verify_placement(leaf, "r", row)


def test_placement_verifies_stopped_child_of_supervisor(tmp_path, monkeypatch):
    guard, leaf = _guard_with_leaf(tmp_path, [101])
    identities = {101: launcher.ProcessIdentity(101, 20, 3, 100, "T")}
    monkeypatch.setattr(launcher, "process_identity", identities.__getitem__)
    row = {"id": "r", "event": "child_stopped", "namespace_pid": 3}
    assert guard._verify_placement(leaf, "r", row).pid == 101
    for bad in (
        {**row, "id": "other"},
        {**row, "namespace_pid": 4},
        {**row, "extra": 1},
    ):
        with pytest.raises(launcher.PlacementError):
            guard._verify_placement(leaf, "r", bad)
    identities[101] = launcher.ProcessIdentity(101, 20, 3, 100, "R")
    with pytest.raises(launcher.PlacementError):
        guard._verify_placement(leaf, "r", row)


def test_oom_events_are_counted(tmp_path):
    (tmp_path / "memory.events").write_text(
        "low 0\nhigh 0\nmax 0\noom 0\noom_kill 2\noom_group_kill 1\n"
    )
    assert launcher.oom_kills(tmp_path) == 3


def test_guard_notice_keeps_output_bound():
    from src import config

    out, err, value, truncated = launcher.bounded_fields(
        "o" * 9000, "e" * 9000, "v" * 100, "memory notice"
    )
    total = sum(len(x.encode()) for x in (out, err, value or ""))
    assert total <= config.PYTHON_MAX_OUTPUT_BYTES and truncated
    assert err.endswith("memory notice") and value == "v" * 100


def test_worker_result_cannot_claim_foreign_identity_or_status():
    guard = launcher.Guard("axiom-python-test.scope")
    guard.instance_id, guard.import_identity = "boot", 1
    row = {
        "id": "r",
        "event": "result",
        "status": "ok",
        "stdout": "",
        "stderr": "",
        "value": "5",
        "truncated": False,
        "worker_instance_id": "boot",
        "import_identity": 1,
    }
    assert guard._worker_row("r", row) == row
    for bad in (
        {**row, "worker_instance_id": "forged"},
        {**row, "import_identity": 2},
        {**row, "status": "timeout"},
        {**row, "id": "stale"},
    ):
        with pytest.raises(ValueError):
            guard._worker_row("r", bad)
