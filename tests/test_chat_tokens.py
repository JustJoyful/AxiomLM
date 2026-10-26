import pytest

from src import chat_tokens
from src.chat_tokens import ChatTokenCounter, render_chat, served_tool

TOOL = {
    "type": "function",
    "function": {
        "name": "search_local_index",
        "description": "Search the selected book.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text", "maxLength": 9},
                "page_start": {"type": "integer", "minimum": 1},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def test_served_tool_drops_schema_keywords_go_structs_do_not_model():
    served = served_tool(TOOL)["function"]["parameters"]
    assert list(served) == ["type", "required", "properties"]
    assert served["properties"]["query"] == {"type": "string", "description": "Text"}
    assert served["properties"]["page_start"] == {"type": "integer"}


def test_chat_render_covers_all_roles_tools_and_thinking():
    messages = [
        {"role": "system", "content": "Policy"},
        {"role": "user", "content": "数学 ∫"},
        {
            "role": "assistant",
            "content": "",
            "thinking": "search first",
            "tool_calls": [
                {"function": {"name": "f", "arguments": {"q": "x", "n": 2}}}
            ],
        },
        {"role": "tool", "content": "r1"},
        {"role": "tool", "content": "r2"},
    ]
    text = render_chat(messages, [TOOL], think=True)
    assert text.startswith("<|im_start|>system\n# Tools")
    assert "\n</IMPORTANT>\n\nPolicy<|im_end|>" in text
    assert "<think>\nsearch first\n</think>\n\n<tool_call>\n<function=f>" in text
    assert "<parameter=n>\n2\n</parameter>" in text
    assert text.count("<|im_start|>user") == 2  # consecutive tool results share a turn
    assert text.endswith("<|im_start|>assistant\n<think>\n")
    assert render_chat(messages[1:2], None, think=False).endswith(
        "<think>\n\n</think>\n\n"
    )


def test_old_turn_reasoning_and_empty_reasoning_are_not_rendered():
    text = render_chat(
        [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1", "thinking": "stale"},
            {"role": "user", "content": "Q2"},
        ],
        None,
        think=True,
    )
    assert "stale" not in text
    tail = render_chat(
        [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}],
        None,
        think=False,
    )
    assert "<|im_start|>assistant\nA<|im_end|>" in tail


def test_counter_counts_rendered_text_and_rejects_unknown_roles():
    counter = ChatTokenCounter(len)
    messages = [{"role": "user", "content": "hi"}]
    assert counter.count(messages, None, think=False) == len(
        render_chat(messages, None, think=False)
    )
    with pytest.raises(ValueError):
        counter.count([{"role": "narrator", "content": "x"}], None, think=False)


def test_unvalidated_model_has_no_chat_counter():
    with pytest.raises(ValueError):
        chat_tokens.load_chat_counter("qwen3:4b")
