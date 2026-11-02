import json
import os

import pytest

from src import config
from src import citation_check as cc
from src.page_querying import ABSTENTION
from src.tools.protocol import EvidenceRef


def source(text="evidence", pages=(57,), cid="s"):
    return EvidenceRef(cid, "book", text, physical_pages=pages)


def test_split_skips_code_math_headings_and_short_sentences():
    answer = "# Heading of many words\n```python\nThis code has many words.\n```\n$$This math has many words.$$\nYes.\n- Interpreters can evaluate program expressions."
    claims = cc.split_claims(answer)
    assert [c.text for c in claims] == [
        "Interpreters can evaluate program expressions."
    ]
    assert all(answer[c.start : c.end] == c.text for c in claims)


def test_split_parses_page_mentions():
    claims = cc.split_claims(
        "Interpreters can evaluate code as on pp. 35–37. They can also compile programs (p. 57)."
    )
    assert [c.pages for c in claims] == [(35, 36, 37), (57,)]


def test_unsupported_claim_is_flagged_not_removed():
    answer = "Interpreters can evaluate program expressions."
    report = cc.check_answer(
        answer, [source()], lambda pairs: [0.01] * len(pairs), threshold=0.5
    )
    assert not report.verdicts[0].supported and report.answer == answer


def test_failed_page_mention_is_reported_without_changing_answer():
    answer = "Interpreters evaluate code (p. 57)."
    report = cc.check_answer(
        answer, [source()], lambda pairs: [0.01] * len(pairs), threshold=0.5
    )
    assert report.answer == "Interpreters evaluate code (p. 57)."
    assert report.verdicts[0].failed_pages == (57,)
    assert not report.verdicts[0].supported
    assert cc.report_payload(report)["citations"] == [{"page": 57, "supported": False}]


def test_claim_supported_elsewhere_preserves_original_page_mention():
    report = cc.check_answer(
        "Interpreters evaluate code (p. 57).",
        [source(pages=(58,))],
        lambda pairs: [0.9] * len(pairs),
        threshold=0.5,
    )
    assert report.verdicts[0].supported
    assert report.answer == "Interpreters evaluate code (p. 57)."
    assert report.supporting_source_ids == ("s",)


def test_sources_supporting_nothing_are_not_marked_supporting():
    refs = [source()]
    report = cc.check_answer(
        "Interpreters can evaluate program expressions.",
        refs,
        lambda pairs: [0.01] * len(pairs),
        threshold=0.5,
    )
    assert not report.supporting_source_ids and len(refs) == 1


def test_abstention_checks_nothing():
    def forbidden(pairs):
        raise AssertionError("no scoring for abstention")

    report = cc.check_answer(ABSTENTION, [source()], forbidden)
    assert report.status == "checked" and not report.verdicts


def test_claims_over_cap_are_counted_unchecked():
    report = cc.check_answer(
        "Interpreters can evaluate program expressions. Compilers can translate program expressions.",
        [source()],
        lambda pairs: [0.9] * len(pairs),
        max_claims=1,
    )
    assert len(report.verdicts) == 1 and report.unchecked == 1


def test_check_scores_all_sources_once_and_validates_output():
    calls = []

    def score(pairs):
        calls.append(pairs)
        return [0.9, 0.1]

    report = cc.check_answer(
        "Interpreters can evaluate program expressions.",
        [source(), source("other", cid="t")],
        score,
        threshold=0.5,
    )
    assert len(calls) == 1 and len(calls[0]) == 2
    assert report.supporting_source_ids == ("s",)
    with pytest.raises(ValueError):
        cc.check_answer(
            "Interpreters can evaluate program expressions.",
            [source()],
            lambda pairs: [],
        )


@pytest.mark.skipif(
    os.environ.get("AXIOM_RUN_SCORER_TESTS") != "1", reason="opt-in real CPU scorer"
)
def test_paraphrase_and_tamper_with_real_scorer():
    from src.entailment import get_scorer

    pairs_path = config.PHASE4_LABELS_DIR / "scorer_pairs.json"
    if not pairs_path.exists():
        pytest.skip("personal pairs absent")
    pairs = json.loads(pairs_path.read_text())
    test_ids = set(pairs["splits"]["test"])
    rows = [p for p in pairs["claim_pairs"] if p["pair_id"] in test_ids]
    scores = get_scorer().score([(p["premise"], p["claim"]) for p in rows])
    negatives = [
        (p["pair_id"], s)
        for p, s in zip(rows, scores)
        if p["relation"] in ("tamper", "contradiction")
        and s >= config.ENTAILMENT_CLAIM_THRESHOLD
    ]
    recall = sum(
        s >= config.ENTAILMENT_CLAIM_THRESHOLD
        for p, s in zip(rows, scores)
        if p["relation"] == "paraphrase"
    )
    assert not negatives, f"false supports {negatives}; paraphrase recall {recall}/5"
    assert len(rows) == 25


async def test_request_deadline_marks_unchecked_without_loading(monkeypatch):
    from src import entailment

    monkeypatch.setattr(
        entailment, "get_scorer", lambda: pytest.fail("deadline already expired")
    )
    assert (await cc.run_check("An answer with enough words.", [], remaining=0))[
        "reason"
    ] == "deadline"


async def test_scorer_failure_is_visible(monkeypatch):
    from src import entailment

    def fail():
        raise OSError("offline files missing")

    monkeypatch.setattr(entailment, "get_scorer", fail)
    payload = await cc.run_check("An answer with enough words.", [source()])
    assert (
        payload["status"] == "unavailable"
        and "offline files missing" in payload["reason"]
    )


@pytest.mark.parametrize("separator", [" ", "\n"])
def test_trailing_citation_is_reported_without_changing_answer(separator):
    answer = "Scheme supports lexical scoping." + separator + "(p. 57)"
    report = cc.check_answer(
        answer, [source(pages=(58,))], lambda pairs: [0.9] * len(pairs), threshold=0.5
    )
    assert report.answer == answer
    assert report.verdicts[0].supported
    assert cc.report_payload(report)["citations"] == [{"page": 57, "supported": False}]
    assert report.unchecked == 0


@pytest.mark.parametrize(
    "answer", [ABSTENTION, "```python\nx = 1\n```", "$$x^2 + y^2 = z^2$$"]
)
async def test_zero_claim_live_check_does_not_load_scorer(monkeypatch, answer):
    from src import entailment

    def forbidden():
        raise AssertionError("zero claims need no scorer")

    monkeypatch.setattr(entailment, "get_scorer", forbidden)
    payload = await cc.run_check(answer, [source()])
    assert payload["status"] == "checked" and payload["claims"] == []
    assert payload["answer"] == answer
