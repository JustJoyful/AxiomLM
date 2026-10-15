import pytest

from src.ui.render import extract_fenced_blocks, normalize_math
from src.ui.stream import TurnRecord


@pytest.mark.parametrize(
    ("raw", "language", "source"),
    [
        ("```python\nx = 'λ'\n\n```\n", "python", "x = 'λ'\n\n"),
        ("~~~tex\r\n\\frac{a}{b}\r\n~~~\r\n", "tex", "\\frac{a}{b}\r\n"),
        ("````python\n```\n````\n", "python", "```\n"),
        ("```py\nx = 1", "py", "x = 1"),
    ],
)
def test_fenced_source_is_exact(raw, language, source):
    blocks = extract_fenced_blocks(raw)
    assert len(blocks) == 1
    assert blocks[0].language == language
    assert blocks[0].source == source


def test_math_normalization_leaves_code_and_raw_answer_intact():
    raw = r"\frac{a}{b} \cap C \le x \ge y" + "\n```tex\n\\sum x\n```\n"
    assert normalize_math(raw) == "(a)/(b) ∩ C ≤ x ≥ y\n```tex\n\\sum x\n```\n"
    record = TurnRecord("r", "question", "book", "model")
    record.set_answer(raw)
    assert record.raw_answer == raw
    assert record.copy_text.endswith(
        "```"
    )  # existing postprocessor trims trailing whitespace
    assert record.copy_is_sanitized
    assert record.copy_payload().startswith("[AxiomLM sanitized answer]\n")
    assert "\\frac" in record.copy_text
    assert "∩" in record.display_answer


def test_unsupported_continuation_and_footnotes_do_not_become_citations():
    record = TurnRecord("r", "q", "b", "m")
    record.set_answer(
        "The indexed material does not provide enough evidence to answer this question. Invented facts."
    )
    assert (
        record.copy_text
        == "The indexed material does not provide enough evidence to answer this question."
    )
    record.set_answer("Supported text[^1]\n[^1]: invented citation")
    assert record.copy_text == "Supported text"
    assert record.sources == []
    record.set_answer("Supported text [¹]\n[¹]: invented citation")
    assert record.copy_text == "Supported text"


def test_empty_and_multiple_fences():
    assert extract_fenced_blocks("no code") == []
    assert [b.source for b in extract_fenced_blocks("```\na\n```\n~~~\nb\n~~~")] == [
        "a\n",
        "b\n",
    ]
    assert normalize_math("") == ""


@pytest.mark.parametrize(
    "raw",
    [
        "Use `array [1] x`",
        '```python\nx = "[^1]"\n```',
        "~~~python\n[^1]: literal code\n~~~",
        "Use ``x ` [1] [^2]`` then prose [3].",
    ],
)
def test_citation_sanitization_protects_code_at_every_stream_split(raw):
    from src.ui.render import SafeAnswerBuffer, sanitize_answer

    expected = raw.replace("prose [3].", "prose.")
    assert sanitize_answer(raw) == expected
    for split in range(len(raw) + 1):
        buffer = SafeAnswerBuffer()
        buffer.feed(raw[:split])
        buffer.feed(raw[split:])
        assert buffer.finish() == expected
    buffer = SafeAnswerBuffer()
    for character in raw:
        buffer.feed(character)
    assert buffer.finish() == expected
