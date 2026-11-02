import pytest

from scripts import validate_phase5_documents as v


@pytest.mark.asyncio
async def test_refuses_existing_report(tmp_path):
    output = tmp_path / "report.json"
    output.write_text("original")
    with pytest.raises(FileExistsError):
        await v.validate_documents(output=output)
    assert output.read_text() == "original"


@pytest.mark.asyncio
async def test_missing_typst_is_unavailable_not_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(v, "compiler_capability", lambda: (False, "missing"))
    report = await v.validate_documents(output=tmp_path / "report.json")
    assert report["accepted"] is False
    assert report["compiler"]["available"] is False
    assert report["human_review"] is None


def test_cli_exits_nonzero_and_reports_failed_host_checks(
    tmp_path, monkeypatch, capsys
):
    import sys

    async def failed(*, output):
        return {
            "compiler": {"available": True},
            "host_security_tests": {"returncode": 1},
            "checks_passed": False,
        }

    monkeypatch.setattr(v, "validate_documents", failed)
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_phase5_documents.py", "--output", str(tmp_path / "report.json")],
    )
    with pytest.raises(SystemExit) as error:
        v.main()
    assert error.value.code == 1
    assert '"checks_passed": false' in capsys.readouterr().out
