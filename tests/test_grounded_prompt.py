from src.tui import PROMPT_TEMPLATE


def test_rag_prompt_requires_evidence_grounded_answers_and_app_citation():
    assert "Every factual sentence must be supported by a retrieved excerpt" in PROMPT_TEMPLATE
    assert "Conversation history is never evidence" in PROMPT_TEMPLATE
    assert "Never infer book content from its title" in PROMPT_TEMPLATE
    assert "Do not invent or alter examples" in PROMPT_TEMPLATE
    assert "say only: 'The indexed material does not provide enough evidence to answer this question.'" in PROMPT_TEMPLATE
    assert "Do not write citations; the application adds citations" in PROMPT_TEMPLATE
