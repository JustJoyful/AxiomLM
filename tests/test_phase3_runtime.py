import pytest

from scripts import probe_phase3_runtime as probe
from src.chat_tokens import render_chat


def test_every_required_gate_must_pass():
    assert probe.all_passed({name: {"passed": True} for name in probe.REQUIRED})
    partial = {name: {"passed": True} for name in probe.REQUIRED[:-1]}
    assert not probe.all_passed(partial)
    assert not probe.all_passed({**partial, probe.REQUIRED[-1]: {"passed": False}})


def test_held_out_cases_exist_and_render():
    assert len(probe.CASES["held_out"]) >= 4
    for messages, tools, think in probe.CASES["held_out"].values():
        assert render_chat(messages, tools, think=think).startswith("<|im_start|>")


def test_probe_refuses_existing_report(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    output.write_text("prior evidence")
    monkeypatch.setattr(
        probe, "_identities", lambda: (_ for _ in ()).throw(AssertionError("no calls"))
    )
    with pytest.raises(FileExistsError):
        probe.probe(output=output)
    assert output.read_text() == "prior evidence"
