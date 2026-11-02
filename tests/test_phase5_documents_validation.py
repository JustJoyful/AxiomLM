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
