import pymupdf
import pytest

from src import text_repair
from src.text_repair import RepairStats, code_lines, map_pua, repair_code, repair_page

# An excerpt of sicp physical p. 39 as pymupdf4llm wrote it, and the monospaced
# lines of that PDF page as code_lines() reads them.
PAGE = (
    "```\n(*5size)\n10\n```\n\n"
    "```\n(definepi3.14159)\n```\n\n"
    "```\n(defineradius10)\n```\n\n"
    "```\n(*pi(*radiusradius))\n314.159\n"
    "(definecircumference(*2piradius))\ncircumference\n62.8318\n```\n\n"
    "\ue049e interpreter makes this step-by-step program construction "
    "particularly convenient. \ue049is feature encourages the incremental "
    "development and testing of programs. \ue049is memory is called the "
    "_environment_."
)
PDF_LINES = [
    "(* 5 size)",
    "10",
    "(define pi 3.14159)",
    "(define radius 10)",
    "(* pi (* radius radius))",
    "314.159",
    "(define circumference (* 2 pi radius))",
    "circumference",
    "62.8318",
]


def fixture_page():
    return PAGE, PDF_LINES


def test_fixture_page_gets_code_spacing_and_ligatures_back():
    markdown, lines = fixture_page()
    stats = RepairStats()
    repaired = repair_page(markdown, lines, 39, stats)
    assert "(definepi3.14159)" in markdown
    for line in (
        "(* 5 size)",
        "(define pi 3.14159)",
        "(define radius 10)",
        "(* pi (* radius radius))",
        "(define circumference (* 2 pi radius))",
    ):
        assert f"\n{line}\n" in repaired
    assert "e interpreter makes" in markdown
    assert "The interpreter makes" in repaired
    assert "This feature encourages" in repaired
    assert "This memory is called" in repaired
    assert stats.code_lines_repaired == 5
    assert stats.pua_mapped[0xE049] == markdown.count("")
    assert stats.skipped == []
    # Only whitespace and PUA changed.
    assert text_repair.squash(repaired) == text_repair.squash(
        markdown.replace("", "Th")
    )


def test_code_line_with_several_or_no_pdf_lines_is_kept_and_logged():
    markdown = "```\n(+12)\n(f)\n```\nprose(+12)"
    stats = RepairStats()
    repaired = repair_code(markdown, ["(+ 1 2)", "(+ 12)", "(g)"], 7, stats)
    assert repaired == markdown
    assert stats.code_lines_repaired == 0
    assert stats.skipped == [
        {"page": 7, "line": 2, "text": "(+12)", "reason": "several PDF lines"},
        {"page": 7, "line": 3, "text": "(f)", "reason": "no PDF line"},
    ]


def test_only_fenced_lines_are_respaced():
    markdown = "(definesize2)\n```\n(definesize2)\n```"
    stats = RepairStats()
    repaired = repair_code(markdown, ["(define size 2)"], 1, stats)
    assert repaired == "(definesize2)\n```\n(define size 2)\n```"
    assert stats.code_lines_repaired == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("prey-printing", "pretty-printing"),
        ("oen", "often"),
        ("otation", "Quotation"),
        ("Anowledgments", "Acknowledgments"),
        ("Maines", "Machines"),
        (" Press", "MIT Press"),
        (".", "1992."),
        ("Wiley  Sons", "Wiley & Sons"),
        (" ", "CC BY-SA"),
        ("o)", "of)"),
    ],
)
def test_pua_glyphs_map_to_text(text, expected):
    assert map_pua(text, RepairStats()) == expected


def test_unknown_pua_is_kept_and_counted():
    stats = RepairStats()
    assert map_pua("xy", stats) == "xy"
    assert stats.pua_unmapped == {0xF8F4: 1}


def test_code_lines_keep_spaces_and_skip_inline_code():
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "(+ 21 35 12 7)", fontname="cour", fontsize=10)
    writer = pymupdf.TextWriter(page.rect)
    writer.append((72, 100), "Type ", font=pymupdf.Font("helv"), fontsize=10)
    writer.append(writer.last_point, "(f x)", font=pymupdf.Font("cour"), fontsize=10)
    writer.write_text(page)
    assert code_lines(page) == ["(+ 21 35 12 7)"]
