from types import SimpleNamespace

import pytest

from src import documents as d
from src.typst_compile import CompileResult


class Client:
    def __init__(self, replies=None):
        self.replies = iter(replies or ["= Answer"] * 3)
        self.requests = []

    async def achat(self, messages, **kwargs):
        self.requests.append((messages, kwargs))
        return {"complete": True, "message": {"content": next(self.replies)}}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(d.config, "EXPORT_DIR", tmp_path)
    monkeypatch.setattr(d, "compiler_capability", lambda: (True, None))
    return d.ExportInput(
        "turn",
        "book",
        d.config.OLLAMA_MODEL,
        'Differentiate x^3 + 2x.\n```python\n#read("secret")\n```',
        "Citations not checked: disabled",
        (),
    )


@pytest.mark.asyncio
async def test_two_repairs_then_raw_text(setup):
    calls = []

    async def syntax(job, **kwargs):
        calls.append(job)
        return CompileResult("syntax_error", "error: invalid")

    client = Client()
    result = await d.export_document(
        setup,
        client=client,
        counter=SimpleNamespace(count=lambda *a, **k: 10),
        compile=syntax,
    )
    assert result.status == "raw_text" and result.compile_attempts == 3
    assert result.pdf_path is None and result.text_path.read_text() == setup.text
    assert len(calls) == len(client.requests) == 3
    text = result.typ_path.read_text()
    assert "Citations not checked: disabled" in text
    assert '#raw("#read(\\"secret\\")\\n"' in text
    assert all(k["tools"] is None and k["think"] is False for _, k in client.requests)


@pytest.mark.asyncio
async def test_missing_compiler_saves_source_without_generation(setup, monkeypatch):
    monkeypatch.setattr(d, "compiler_capability", lambda: (False, "missing typst"))
    client = Client()
    result = await d.export_document(setup, client=client, counter=None)
    assert result.status == "unavailable" and result.text_path.read_text() == setup.text
    assert not client.requests


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["timeout", "resource_error", "unavailable"])
async def test_terminal_compile_failures_do_not_retry(setup, status):
    async def compiler(*a, **k):
        return CompileResult(status, "failure")

    client = Client()
    result = await d.export_document(
        setup,
        client=client,
        counter=SimpleNamespace(count=lambda *a, **k: 10),
        compile=compiler,
    )
    assert result.compile_attempts == 1 and result.pdf_path is None
    assert len(client.requests) == 1


@pytest.mark.asyncio
async def test_input_or_prompt_over_budget_is_not_truncated(setup):
    client = Client()
    result = await d.export_document(
        setup, client=client, counter=SimpleNamespace(count=lambda *a, **k: 4096)
    )
    assert not client.requests and result.text_path.read_text() == setup.text


def test_literal_rejects_control_characters():
    with pytest.raises(ValueError):
        d.typst_literal("bad\x00")
    assert d.typst_literal('数学 "\\') == '"数学 \\"\\\\"'


@pytest.mark.asyncio
@pytest.mark.parametrize("failures", [0, 1])
async def test_success_stops_after_validated_pdf(setup, failures):
    import fitz

    calls = []

    async def compiler(job, **kwargs):
        calls.append(job)
        if len(calls) <= failures:
            return CompileResult("syntax_error", "error: invalid")
        path = job / "document.pdf"
        with fitz.open() as pdf:
            pdf.new_page().insert_text((30, 30), "Answer")
            pdf.save(path)
        return CompileResult("ok", "", path)

    result = await d.export_document(
        setup,
        client=Client(),
        counter=SimpleNamespace(count=lambda *a, **k: 10),
        compile=compiler,
    )
    assert result.status == "pdf" and result.compile_attempts == failures + 1
    assert d.valid_pdf(result.pdf_path)


@pytest.mark.asyncio
async def test_cancel_keeps_source_and_removes_partial_pdf(setup):
    import asyncio

    async def compiler(job, **kwargs):
        (job / "document.pdf").write_bytes(b"partial")
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await d.export_document(
            setup,
            client=Client(),
            counter=SimpleNamespace(count=lambda *a, **k: 10),
            compile=compiler,
        )
    jobs = list(d.config.EXPORT_DIR.iterdir())
    assert len(jobs) == 1
    assert (jobs[0] / "answer.txt").read_text() == setup.text
    assert not (jobs[0] / "document.pdf").exists()
    assert "cancelled" in (jobs[0] / "report.json").read_text()


@pytest.mark.asyncio
async def test_unavailable_export_keeps_host_source_provenance(setup, monkeypatch):
    import json
    from dataclasses import replace

    from src.ui.stream import SourceRef

    monkeypatch.setattr(d, "compiler_capability", lambda: (False, "missing"))
    source = replace(
        setup, sources=(SourceRef("book", physical_page=42, source_id="chunk"),)
    )
    result = await d.export_document(source, client=Client(), counter=None)
    report = json.loads((result.job / "report.json").read_text())
    assert report["sources"][0]["physical_page"] == 42
    assert report["notices"] == source.notices


@pytest.mark.skipif(
    __import__("os").environ.get("AXIOM_RUN_TYPST_TESTS") != "1",
    reason="opt-in real Typst math gate",
)
@pytest.mark.asyncio
@pytest.mark.parametrize("example", d.MATH_EXAMPLES)
async def test_host_worked_math_examples_compile_and_open(tmp_path, example):
    from src.typst_compile import compile_typst, compiler_capability

    assert compiler_capability()[0], compiler_capability()[1]
    (tmp_path / "document.typ").write_text(example)
    result = await compile_typst(tmp_path, remaining=10)
    assert result.status == "ok", result.diagnostics
    assert d.valid_pdf(result.pdf_path)
