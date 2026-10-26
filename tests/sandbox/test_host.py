import os

import pytest

from scripts import validate_python_sandbox as diagnostic


def test_report_refuses_existing_evidence_before_host_probe(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    output.write_text("previous evidence")
    monkeypatch.setattr(
        diagnostic,
        "SandboxClient",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not launch")),
    )
    with pytest.raises(FileExistsError):
        diagnostic.validate(output=output)
    assert output.read_text() == "previous evidence"


def test_missing_mandatory_check_cannot_pass():
    assert diagnostic.all_passed({"timeout_recovery": {"passed": True}}) is False
    assert (
        diagnostic.all_passed(
            {name: {"passed": True} for name in diagnostic.REQUIRED_CHECKS}
        )
        is True
    )


@pytest.mark.skipif(
    os.environ.get("AXIOM_RUN_SANDBOX_HOST_TESTS") != "1",
    reason="explicit host opt-in required",
)
def test_host_boundaries_recovery_and_actual_parent_death(tmp_path):
    report = diagnostic.validate(output=tmp_path / "host/report.json")
    assert report["passed"], report["blocked_reason"]
    assert report["checks"]["timeout_recovery"]["same_worker"]
    assert report["checks"]["tui_sigkill"]["remaining_pids"] == []
    assert report["effective_memory_max_bytes"] == 262144000


def test_cleanup_failure_is_persisted_as_failed_report(tmp_path, monkeypatch):
    from src.sandbox.launcher import SandboxCapabilities

    class Client:
        def __init__(self, **kwargs):
            pass

        async def start(self):
            return SandboxCapabilities(False, "backend unavailable")

        async def aclose(self):
            raise RuntimeError("cleanup unconfirmed")

    monkeypatch.setattr(diagnostic, "SandboxClient", Client)
    monkeypatch.setattr(diagnostic, "_hashes", lambda output: {})
    output = tmp_path / "report.json"
    report = diagnostic.validate(output=output)
    import json

    assert not report["passed"]
    assert not report["checks"]["final_cleanup"]["passed"]
    assert "cleanup unconfirmed" in report["checks"]["final_cleanup"]["error"]
    assert json.loads(output.read_text()) == report


def test_cleanup_timeout_retains_surviving_identities(tmp_path, monkeypatch):
    from src import config
    from src.sandbox.launcher import SandboxCapabilities

    class Client:
        def __init__(self, **kwargs):
            pass

        async def start(self):
            return SandboxCapabilities(False, "backend unavailable")

        async def aclose(self):
            pass

    monkeypatch.setattr(diagnostic, "SandboxClient", Client)
    monkeypatch.setattr(diagnostic, "_hashes", lambda output: {})
    monkeypatch.setattr(diagnostic, "_alive", lambda rows: [123])
    monkeypatch.setattr(config, "PYTHON_STARTUP_SECONDS", 0.02)
    report = diagnostic.validate(output=tmp_path / "report.json")
    assert not report["passed"]
    assert report["checks"]["final_cleanup"]["remaining_pids"] == [123]
