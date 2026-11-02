import json
import sqlite3
from types import SimpleNamespace

import pytest

from src import study as s
from src.study_state import load_sections
from tests.test_study_state import catalog_root


@pytest.mark.parametrize(
    "questions",
    [
        [],
        [{"text": "", "source_ids": ["a"]}],
        [{"text": "Question?", "source_ids": ["foreign"]}],
        [{"text": "Question?", "source_ids": []}],
        [{"text": "Q" * 513, "source_ids": ["a"]}],
        [{"text": "Question?", "source_ids": ["a"], "answer": "injected"}],
        [{"text": "Question?", "source_ids": ["a", "a"]}],
    ],
)
def test_invalid_questions_rejected(questions):
    with pytest.raises(ValueError):
        s.parse_questions(
            json.dumps({"questions": questions}), count=1, source_ids=frozenset({"a"})
        )


def test_question_sources_are_nonempty_subset_of_provided_ids():
    questions = s.parse_questions(
        '{"questions":[{"text":"Why?", "source_ids":["a"]}]}',
        count=1,
        source_ids=frozenset({"a", "b"}),
    )
    assert questions[0].text == "Why?" and questions[0].source_ids == ("a",)


def test_evidence_is_from_exact_captured_section_and_version(tmp_path):
    path = catalog_root(tmp_path)
    section = load_sections("sicp", root=tmp_path).sections[0]
    refs = s.section_evidence(section, root=tmp_path)
    assert [r.source_id for r in refs] == ["a", "b"]
    assert refs[0].text == "passage a" and refs[0].physical_pages == (1, 2)
    path.unlink()
    with pytest.raises(sqlite3.OperationalError):
        s.section_evidence(section, root=tmp_path)


@pytest.mark.asyncio
async def test_generation_has_no_tools_and_rejects_incomplete_reply(
    tmp_path, monkeypatch
):
    catalog_root(tmp_path)
    section = load_sections("sicp", root=tmp_path).sections[0]
    original_evidence = s.section_evidence
    monkeypatch.setattr(
        s, "section_evidence", lambda sec: tuple(original_evidence(sec, root=tmp_path))
    )

    class Client:
        async def achat(self, messages, **kwargs):
            assert kwargs["tools"] is None and kwargs["think"] is False
            assert "passage a" in messages[-1]["content"]
            return {"complete": False, "message": {"content": "{}"}}

    with pytest.raises(ValueError, match="Incomplete"):
        await s.generate_practice(
            section,
            1,
            client=Client(),
            counter=SimpleNamespace(count=lambda *a, **k: 10),
        )
