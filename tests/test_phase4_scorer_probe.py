"""Phase 4 scorer selection probe: the pure scoring helpers and the pairs fixture."""

import json
from typing import ClassVar

import pytest

from scripts import probe_phase4_scorer as probe
from src import config

PAIRS = config.PHASE4_LABELS_DIR / "scorer_pairs.json"
BOOKS = ("sicp", "thinkpython2")


def test_auc_handles_ties():
    assert probe.auc([0.5], [0.5]) == 0.5


def test_auc_perfect_separation():
    assert probe.auc([0.9, 0.8], [0.1]) == 1.0


def test_auc_counts_half_for_each_tie():
    assert probe.auc([0.9, 0.5], [0.5, 0.1]) == pytest.approx((1 + 1 + 0.5 + 1) / 4)


def test_best_threshold_maximises_tpr_minus_fpr():
    assert probe.best_threshold([0.9, 0.8, 0.4], [0.5, 0.45, 0.1]) == 0.8


def test_best_threshold_tie_takes_the_stricter_threshold():
    # 0.7 and 0.4 both give TPR-FPR = 2/3; the higher one marks fewer pairs supported.
    assert probe.best_threshold([0.9, 0.7, 0.4], [0.6, 0.2, 0.1]) == 0.7


def test_best_threshold_uses_calibration_only():
    rows = [
        {"pair_id": "c1", "relation": "answers"},
        {"pair_id": "c2", "relation": "topical"},
        {"pair_id": "t1", "relation": "answers"},
        {"pair_id": "t2", "relation": "topical"},
    ]
    calibration = {"c1", "c2"}
    scores = {"c1": 0.8, "c2": 0.3, "t1": 0.6, "t2": 0.4}
    shifted = {**scores, "t1": 0.01, "t2": 0.99}
    pos, neg = probe.split_scores(rows, scores, calibration, {"answers"})
    assert (pos, neg) == ([0.8], [0.3])
    assert probe.best_threshold(
        *probe.split_scores(rows, scores, calibration, {"answers"})
    ) == probe.best_threshold(
        *probe.split_scores(rows, shifted, calibration, {"answers"})
    )


def _fixture_or_skip() -> dict:
    if not PAIRS.is_file():
        pytest.skip("personal scorer pairs fixture is absent")
    if not all((config.REPAIRED_MD / book).is_dir() for book in BOOKS):
        pytest.skip("repaired pages are absent")
    return probe.load_pairs(PAIRS)


def test_pairs_fixture_premises_match_pages():
    pairs = _fixture_or_skip()
    for row in pairs["question_pairs"] + pairs["claim_pairs"]:
        page = config.REPAIRED_MD / row["book_stem"] / f"page_{row['page'] - 1:04d}.md"
        assert row["premise"] in page.read_text(), row["pair_id"]


def test_pairs_fixture_has_required_relations():
    pairs = _fixture_or_skip()
    test_ids = set(pairs["splits"]["test"])
    for key, relations in (
        ("question_pairs", probe.QUESTION_RELATIONS),
        ("claim_pairs", probe.CLAIM_RELATIONS),
    ):
        for relation in relations:
            rows = [r for r in pairs[key] if r["relation"] == relation]
            assert len(rows) >= 8, (key, relation)
            assert {r["book_stem"] for r in rows} == set(BOOKS), (key, relation)
            assert any(r["pair_id"] in test_ids for r in rows), (key, relation)


def _write_pairs(tmp_path, *, premise="Procedures are applied to arguments."):
    folder = tmp_path / "pages" / "book"
    folder.mkdir(parents=True)
    page_text = "Intro.\n\nProcedures are applied to arguments.\n"
    (folder / "page_0004.md").write_text(page_text)
    from scripts.evaluate_phase3_baseline import digest

    data = {
        "schema_version": 1,
        "review": {"status": "unreviewed"},
        "source_pages": {"book": {"5": digest(folder / "page_0004.md")}},
        "splits": {"calibration": ["q1"], "test": ["c1"]},
        "question_pairs": [
            {
                "pair_id": "q1",
                "book_stem": "book",
                "page": 5,
                "premise": premise,
                "question": "What are procedures applied to?",
                "relation": "answers",
            }
        ],
        "claim_pairs": [
            {
                "pair_id": "c1",
                "book_stem": "book",
                "page": 5,
                "premise": premise,
                "claim": "Procedures are applied to arguments.",
                "relation": "entailed",
            }
        ],
    }
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps(data))
    return path, tmp_path / "pages"


def test_load_pairs_accepts_verbatim_premises(tmp_path):
    path, root = _write_pairs(tmp_path)
    pairs = probe.load_pairs(path, pages_root=root, books=("book",))
    assert [r["split"] for r in pairs["question_pairs"]] == ["calibration"]
    assert [r["split"] for r in pairs["claim_pairs"]] == ["test"]


def test_load_pairs_rejects_premise_not_on_page(tmp_path):
    path, root = _write_pairs(tmp_path, premise="Procedures are never applied.")
    with pytest.raises(ValueError, match="q1"):
        probe.load_pairs(path, pages_root=root, books=("book",))


def test_load_pairs_rejects_changed_page(tmp_path):
    path, root = _write_pairs(tmp_path)
    (root / "book" / "page_0004.md").write_text(
        "Procedures are applied to arguments.\n"
    )
    with pytest.raises(ValueError, match="changed"):
        probe.load_pairs(path, pages_root=root, books=("book",))


def test_load_pairs_rejects_unknown_relation_and_unsplit_pair(tmp_path):
    path, root = _write_pairs(tmp_path)
    data = json.loads(path.read_text())
    data["question_pairs"][0]["relation"] = "maybe"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="relation"):
        probe.load_pairs(path, pages_root=root, books=("book",))
    data["question_pairs"][0]["relation"] = "answers"
    data["splits"]["calibration"] = []
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="split"):
        probe.load_pairs(path, pages_root=root, books=("book",))


class _Info:
    sha = "b" * 40
    card_data: ClassVar[dict] = {"license": "apache-2.0"}
    tags = ()


class _FakeScorer:
    """Scores by keyword so each relation lands where the test expects."""

    parameters = 22_000_000
    max_length = 512
    revision = "b" * 40

    def __init__(self, model_id, revision, *, max_parameters=None):
        self.model = type(
            "M",
            (),
            {
                "config": type("C", (), {"id2label": {0: "entailment", 1: "x"}})(),
                "tokenizer": type(
                    "T",
                    (),
                    {"encode": lambda self, t, add_special_tokens=False: t.split()},
                )(),
            },
        )()

    def score(self, pairs):
        table = {
            "ANS": 0.9,
            "TOP": 0.2,
            "CON": 0.7,
            "UNR": 0.1,
            "ENT": 0.95,
            "NEU": 0.3,
            "TAM": 0.8,
        }
        return [next(v for k, v in table.items() if k in hyp) for _, hyp in pairs]


def test_probe_candidate_reports_counts_at_calibration_threshold(monkeypatch):
    monkeypatch.setattr(probe, "model_info", lambda repo, revision=None: _Info())
    monkeypatch.setattr(probe, "Scorer", _FakeScorer)
    rows = lambda key, field, spec: [
        {
            "pair_id": pid,
            "book_stem": "sicp",
            "page": 1,
            "premise": "p",
            field: text,
            "relation": rel,
            "split": split,
        }
        for pid, rel, text, split in spec
    ]
    pairs = {
        "question_pairs": rows(
            "q",
            "question",
            [
                ("qa1", "answers", "ANS one", "calibration"),
                ("qt1", "topical", "TOP one", "calibration"),
                ("qa2", "answers", "ANS two", "test"),
                ("qp2", "paraphrase", "ANS para", "test"),
                ("qt2", "topical", "TOP two", "test"),
                ("qc2", "contradicts", "CON two", "test"),
                ("qu2", "unrelated", "UNR two", "test"),
            ],
        ),
        "claim_pairs": rows(
            "c",
            "claim",
            [
                ("ce1", "entailed", "ENT one", "calibration"),
                ("cn1", "neutral", "NEU one", "calibration"),
                ("ce2", "entailed", "ENT two", "test"),
                ("ct2", "tamper", "TAM two", "test"),
                ("cn2", "neutral", "NEU two", "test"),
            ],
        ),
    }
    pairs["splits"] = {
        s: [
            r["pair_id"]
            for k in ("question_pairs", "claim_pairs")
            for r in pairs[k]
            if r["split"] == s
        ]
        for s in ("calibration", "test")
    }
    report = probe.probe_candidate(
        "org/nli", None, pairs, chunk_texts={"sicp": ["a b c"]}, timed_runs=1
    )
    assert report["revision"] == "b" * 40 and report["eligible"] is True
    gate = report["formulations"]["answers"]
    assert gate["threshold"] == 0.9
    assert gate["test"]["positives_passing"] == {"k": 2, "n": 2, "rate": 1.0}
    assert gate["test"]["contradicts_passing"] == {"k": 0, "n": 1, "rate": 0.0}
    assert gate["test"]["mean"]["answers"] > gate["test"]["mean"]["topical"]
    claims = report["claims"]
    assert claims["threshold"] == 0.95
    assert claims["test"]["supported"]["tamper"] == {"k": 0, "n": 1, "rate": 0.0}
    assert claims["test"]["contradiction_or_tamper_supported"]["n"] == 1
    assert report["truncation"]["answers"]["sicp"] == {"k": 0, "n": 1, "rate": 0.0}
    assert report["cuda_initialized"] is False


def test_oversized_or_restricted_candidate_is_recorded_ineligible(monkeypatch):
    class Big(_FakeScorer):
        parameters = 100_000_000

    class Restricted(_Info):
        card_data: ClassVar[dict] = {"license": "cc-by-nc-4.0"}

    monkeypatch.setattr(probe, "model_info", lambda repo, revision=None: Restricted())
    monkeypatch.setattr(probe, "Scorer", Big)
    pairs = {
        "question_pairs": [],
        "claim_pairs": [],
        "splits": {"calibration": [], "test": []},
    }
    report = probe.probe_candidate("org/big", None, pairs, chunk_texts={}, timed_runs=1)
    assert report["eligible"] is False
    assert {"parameters", "license"} <= set(report["rejected_for"])


def test_existing_probe_output_is_refused(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    output.write_text("{}")
    monkeypatch.setattr(
        probe,
        "probe_candidate",
        lambda *a, **k: pytest.fail("probe ran despite an existing output"),
    )
    with pytest.raises(SystemExit, match="exists"):
        probe.main(["--candidate", "org/model", "--output", str(output)])
    assert output.read_text() == "{}"
