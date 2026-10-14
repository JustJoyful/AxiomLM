from src.page_querying import strip_model_footnotes


def test_strip_model_footnotes_preserves_answer_text_and_app_citation():
    response = (
        "Supported claim [^1].\n\n"
        "[^1]: Source section, p.12\n"
        "[sicp · p.12]"
    )

    assert strip_model_footnotes(response) == "Supported claim.\n\n[sicp · p.12]"


def test_abstention_discards_unsupported_followup_after_refusal():
    response = (
        "The indexed material does not provide enough evidence to answer this question.\n\n"
        "The conversation history mentioned an example, but the book has no other examples."
    )

    assert strip_model_footnotes(response) == (
        "The indexed material does not provide enough evidence to answer this question."
    )
