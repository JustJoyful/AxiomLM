from src.tui import PROMPT_TEMPLATE


def test_rag_prompt_requires_evidence_grounded_answers_and_app_citation():
    assert (
        "Every factual sentence must be supported by a retrieved excerpt"
        in PROMPT_TEMPLATE
    )
    assert "Conversation history is never evidence" in PROMPT_TEMPLATE
    assert "Never infer book content from its title" in PROMPT_TEMPLATE
    assert "Do not invent or alter examples" in PROMPT_TEMPLATE
    assert (
        "say only: 'The indexed material does not provide enough evidence to answer this question.'"
        in PROMPT_TEMPLATE
    )
    assert "Do not write citations; the application adds citations" in PROMPT_TEMPLATE


async def test_agent_qa_rules_answer_directly_and_reach_the_final_prompt():
    from src import agent
    from tests.test_agent import (
        Candidate,
        Client,
        Counter,
        Sandbox,
        collect,
        qa_script,
        search_with,
    )

    client = Client(qa_script(sufficient=True))
    await collect(
        request=agent.AgentRequest("r", "Summarize physical PDF page 1.", "sicp", "m"),
        client=client,
        sandbox=Sandbox(),
        counter=Counter(),
        search=search_with([Candidate("c", "Structure and Interpretation", 1)], []),
    )
    final = client.requests[-1]["messages"][-1]["content"]
    for phrase in (
        "starting with the answer",
        "even a title page",
        "resolve what it refers to from the conversation",
        "do not invent reasons for the false premise",
        "only if no passage addresses it",
    ):
        assert phrase in final
    # Measured with the configured model's chat counter: 133 tokens. A char bound
    # keeps later edits from silently eating the 4096-token final budget.
    assert len(agent.QA_RULES) <= 600
