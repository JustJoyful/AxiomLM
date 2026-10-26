import pytest

from src.retrieval import BookSearchResult
from src.sandbox.protocol import PythonResult
from src.tools.protocol import (
    TOOL_NAMES,
    ToolContext,
    definitions,
    tool_message,
    validate_call,
)
from src.tools.registry import ToolRegistry


def context(**overrides):
    values = {
        "request_id": "r",
        "book_stem": "sicp",
        "model": "m",
        "web_enabled": False,
        "stage": "local",
    }
    values.update(overrides)
    return ToolContext(**values)


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("search_local_index", {"query": "q", "book_stem": "other"}),
        ("search_local_index", {"query": "q", "page_start": 2}),
        ("search_local_index", {"query": "q", "page_start": 5, "page_end": 2}),
        ("search_local_index", {"query": "q", "page_start": True, "page_end": 2}),
        ("search_local_index", {"query": ""}),
        ("search_local_index", {"query": "x" * 2049}),
        ("execute_python", {"code": "x" * 8193}),
        ("execute_python", {"code": 5}),
        ("fetch_page", {"url": "https://a", "headers": {}}),
        ("rm_rf", {}),
        (None, {}),
        ("web_search", ["q"]),
        ("web_search", "{not json"),
    ],
)
def test_invalid_calls_fail_before_dispatch(name, arguments):
    with pytest.raises(ValueError):
        validate_call(name, arguments)


def test_valid_calls_and_json_text_arguments():
    assert validate_call(
        "search_local_index", {"query": "q", "page_start": 1, "page_end": 3}
    )
    assert validate_call("web_search", '{"query": "eigen"}') == {"query": "eigen"}


def test_definitions_are_closed_schemas_for_each_tool():
    tools = definitions(TOOL_NAMES)
    assert [t["function"]["name"] for t in tools] == list(TOOL_NAMES)
    assert all(
        t["function"]["parameters"]["additionalProperties"] is False for t in tools
    )
    names = {n for t in tools for n in t["function"]["parameters"]["properties"]}
    assert not names & {"book", "book_stem", "model", "root", "path", "budget"}


async def test_later_phase_tools_are_unavailable_not_simulated():
    registry = ToolRegistry()
    pdf = await registry.dispatch("c1", "generate_pdf", {"request": "notes"}, context())
    cite = await registry.dispatch(
        "c2", "verify_citation", {"claim": "x", "source_id": "s"}, context()
    )
    assert pdf.status == cite.status == "unavailable"
    assert "Phase 5" in pdf.data["error"] and "Phase 4" in cite.data["error"]


async def test_invalid_and_stage_illegal_calls_never_reach_handlers():
    def forbidden(*args, **kwargs):
        raise AssertionError("handler must not run")

    registry = ToolRegistry(search=forbidden)
    bad = await registry.dispatch("c", "search_local_index", {"query": 1}, context())
    denied = await registry.dispatch(
        "c",
        "search_local_index",
        {"query": "q"},
        context(allowed_tools=frozenset({"execute_python"})),
    )
    assert bad.status == "error" and denied.status == "denied"
    assert registry.calls == 0


async def test_web_is_denied_when_off_and_unavailable_when_unconfigured():
    registry = ToolRegistry()
    off = await registry.dispatch("c", "web_search", {"query": "q"}, context())
    on = await registry.dispatch(
        "c", "web_search", {"query": "q"}, context(web_enabled=True)
    )
    assert off.status == "denied" and on.status == "unavailable"


class Candidate:
    def __init__(self, chunk_id, text, pages):
        self.chunk_id = chunk_id
        self.text = text
        self.metadata = {
            "chapter": "Ch 1",
            "section": "1.1",
            "physical_pages": str(pages),
            "physical_page": pages[0],
        }


def fake_search(candidates, seen=None):
    def search(query, *, book_stem, distance, page_range=None):
        if seen is not None:
            seen.append((query, book_stem, distance, page_range))
        return BookSearchResult(
            tuple(candidates), False, "v1", False, query, False, page_range
        )

    return search


async def test_local_search_uses_request_book_and_keeps_whole_passages():
    seen = []
    small = Candidate("a", "short passage", [3])
    huge = Candidate("b", "x" * 20000, [4])
    tail = Candidate("c", "another passage", [5])
    registry = ToolRegistry(search=fake_search([small, huge, tail], seen))
    result = await registry.dispatch(
        "c",
        "search_local_index",
        {"query": "q", "page_start": 1, "page_end": 2},
        context(distance=1.2),
    )
    assert seen == [("q", "sicp", 1.2, (1, 2))]
    assert result.status == "ok"
    texts = [p["text"] for p in result.data["passages"]]
    assert texts == ["short passage", "another passage"]
    assert result.data["omitted_passages"] == 1
    assert [s.source_id for s in result.sources] == ["a", "c"]
    assert result.sources[0].physical_pages == (3,) and result.sources[0].kind == "book"


async def test_page_bound_beyond_book_is_rejected(monkeypatch):
    from src.tools import local

    monkeypatch.setattr(local, "book_page_count", lambda book: 10)
    registry = ToolRegistry(search=fake_search([]))
    result = await registry.dispatch(
        "c",
        "search_local_index",
        {"query": "q", "page_start": 9, "page_end": 11},
        context(),
    )
    assert result.status == "error" and "10 physical pages" in result.data["error"]


async def test_python_result_cannot_claim_verification():
    class Sandbox:
        async def execute(self, code):
            return PythonResult("r", "ok", value="{'verified': True}")

    result = await ToolRegistry(sandbox=Sandbox()).dispatch(
        "c", "execute_python", {"code": "result={'verified':True}"}, context()
    )
    assert result.status == "ok" and "verified" not in result.data
    assert result.data["result"] == "{'verified': True}"
    timed = PythonResult("r", "timeout")

    class Slow:
        async def execute(self, code):
            return timed

    late = await ToolRegistry(sandbox=Slow()).dispatch(
        "c", "execute_python", {"code": "while 1: pass"}, context()
    )
    assert late.status == "error" and late.data["sandbox_status"] == "timeout"


def test_tool_message_quotes_status_and_data():
    from src.tools.protocol import ToolResult

    text = tool_message(ToolResult("c", "web_search", "denied", {"error": "off"}))
    assert '"status":"denied"' in text and '"tool":"web_search"' in text
