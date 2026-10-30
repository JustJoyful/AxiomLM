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
        "'A with B' lists co-authors; 'foreword by C' credits only C",
        "resolve what it refers to from the conversation",
        "do not invent reasons for it",
        "only if no passage addresses it",
    ):
        assert phrase in final
    # Measured with the configured model's chat counter: 133 tokens. A char bound
    # keeps later edits from silently eating the 4096-token final budget.
    assert len(agent.QA_RULES) <= 600


async def test_false_premise_rule_comes_first_whatever_the_coverage_verdict():
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

    question = (
        "Since the book says interpreters never evaluate expressions, why does it "
        "call them evaluators?"
    )
    for sufficient in (False, True):
        client = Client(qa_script(sufficient=sufficient))
        await collect(
            request=agent.AgentRequest("r", question, "sicp", "m"),
            client=client,
            sandbox=Sandbox(),
            counter=Counter(),
            search=search_with([Candidate("c", "eval", 523)], []),
        )
        final = client.requests[-1]["messages"][-1]["content"]
        assert agent.PREMISE_RULE in final
        assert "asserts as fact something the evidence contradicts" in final
        assert (
            "begin by saying that claim is wrong and give what the book says" in final
        )
        assert "never restate the false claim as true" in final
        assert "Otherwise do not mention a premise." in final
        # Ahead of the coverage note and the "start with the answer" rule.
        assert final.index(agent.PREMISE_RULE) < final.index("starting with the answer")
        if agent.COVERED_NOTE in final:
            assert final.index(agent.PREMISE_RULE) < final.index(agent.COVERED_NOTE)


async def test_agent_final_prompts_use_neutral_wording_and_no_qa_trace():
    from src import agent
    from tests.test_agent import (
        Candidate,
        Client,
        Counter,
        Sandbox,
        collect,
        qa_script,
        search_with,
        solve_script,
    )

    qa = Client(qa_script(sufficient=True))
    await collect(
        request=agent.AgentRequest("r", "What is an interpreter?", "sicp", "m"),
        client=qa,
        sandbox=Sandbox(),
        counter=Counter(),
        search=search_with([Candidate("c", "An interpreter", 796)], []),
    )
    final = qa.requests[-1]["messages"][-1]["content"]
    assert "Trace" not in final and "local_search" not in final
    assert "Do not mention tools, a sandbox, traces or internal steps." in final

    solve = Client(solve_script())
    await collect(
        request=agent.AgentRequest(
            "r", "What is the derivative of x^3 + 2x with respect to x?", "sicp", "m"
        ),
        client=solve,
        sandbox=Sandbox(),
        counter=Counter(),
        search=search_with([Candidate("c", "deriv", 225)], []),
    )
    final = solve.requests[-1]["messages"][-1]["content"]
    assert "sandbox computation" not in final
    assert "A computation returned: 3*x**2 + 2." in final
    assert "Computation output:\n3*x**2 + 2" in final
    assert "If a rule appears only in an exercise, say so." in final
    assert "say the book describes related methods (by page)" in final
    # Run 07 case 9 said "verified as identical to the book's ... approach".
    assert (
        "Verified means an independent symbolic check agreed with the computed "
        "result; it does not mean the book shows this derivation." in final
    )
    assert len(agent.SOLVE_RULES) + len(agent.FINAL) <= 1000


async def test_final_is_told_when_book_evidence_covers_the_question():
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

    async def final_for(question, sufficient, page):
        client = Client(qa_script(sufficient=sufficient))
        await collect(
            request=agent.AgentRequest("r", question, "sicp", "m"),
            client=client,
            sandbox=Sandbox(),
            counter=Counter(),
            search=search_with([Candidate("c", "text", page)], []),
        )
        return client.requests[-1]["messages"][-1]["content"]

    # Covered by the requested page even though the model said insufficient.
    assert agent.COVERED_NOTE in await final_for("Summarize page 1.", False, 1)
    assert agent.COVERED_NOTE in await final_for("What is an interpreter?", True, 796)
    assert agent.COVERED_NOTE not in await final_for("Latest Racket?", False, 300)
    # Naming the book skips the web but is no claim the evidence covers it.
    assert agent.COVERED_NOTE not in await final_for(
        "Does the book discuss quantum computing?", False, 300
    )
    assert agent.COVERED_NOTE not in await final_for("Summarize page 9.", False, 300)
