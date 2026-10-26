import asyncio

import pytest

from src import query_rewriting as rewrite


def history(book="book"):
    return [
        rewrite.ConversationExchange(
            "Explain interpreters.", "They execute programs.", book
        )
    ]


@pytest.mark.parametrize(
    "question,expected",
    [
        ("What is it?", True),
        ("Give another example of that.", True),
        ("What is Scheme?", False),
        ("Summarize page 5.", False),
        ("What is this book about?", False),
        ("What about that?", True),
    ],
)
def test_eligibility(question, expected):
    assert rewrite.should_rewrite_followup(question, history(), "book") is expected
    assert not rewrite.should_rewrite_followup(question, [], "book")
    assert not rewrite.should_rewrite_followup(question, history("other"), "book")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output,applied",
    [
        ("Give an interpreter example?", True),
        ("", False),
        ("Answer: interpreters execute code.", False),
        ("```question```", False),
    ],
)
async def test_rewrite_preserves_original_and_only_uses_last_two_exchanges(
    output, applied
):
    calls = []

    class Client:
        async def agenerate(self, prompt, **kwargs):
            calls.append((prompt, kwargs))
            return {"response": output, "done": True}

        def unload_and_wait(self):
            calls.append("unloaded")

    exchanges = (
        [rewrite.ConversationExchange("PRIVATE OLD", "OLD ANSWER", "book")]
        + history()
        + history()
    )
    result = await rewrite.rewrite_followup(
        "Give another example of that.",
        exchanges,
        "book",
        model="local",
        client=Client(),
    )
    assert (
        result.original_question == "Give another example of that."
        and result.applied is applied
    )
    assert result.retrieval_query == (output if applied else result.original_question)
    assert "PRIVATE OLD" not in calls[0][0]
    assert calls[0][1]["num_predict"] == 96 and calls[0][1]["timeout"] == 15
    assert calls[-1] == "unloaded"


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [TimeoutError(), asyncio.CancelledError()])
async def test_timeout_falls_back_and_cancellation_propagates_after_cleanup(error):
    cleaned = []

    class Client:
        async def agenerate(self, *args, **kwargs):
            raise error

        def unload_and_wait(self):
            cleaned.append(True)

    if isinstance(error, asyncio.CancelledError):
        with pytest.raises(asyncio.CancelledError):
            await rewrite.rewrite_followup(
                "What is it?", history(), "book", model="m", client=Client()
            )
    else:
        result = await rewrite.rewrite_followup(
            "What is it?", history(), "book", model="m", client=Client()
        )
        assert result.retrieval_query == "What is it?" and result.fallback_reason
    assert cleaned == [True]


@pytest.mark.asyncio
async def test_book_overview_query_is_traced_without_retrieved_evidence():
    class Client:
        async def agenerate(self, *args, **kwargs):
            raise AssertionError(
                "Recognized overview intent does not need a model rewrite"
            )

    result = await rewrite.rewrite_followup(
        "What this book about brocacho", [], "book", model="local", client=Client()
    )
    assert result.original_question == "What this book about brocacho"
    assert result.applied
    assert "preface" in result.retrieval_query and "purpose" in result.retrieval_query
