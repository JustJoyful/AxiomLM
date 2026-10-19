import pytest
import torch

from scripts import probe_phase2_late_chunking as probe


class Tokenizer:
    eos_token_id = 9

    def __call__(self, text, add_special_tokens=True, **kwargs):
        ids = [ord(c) for c in text]
        offsets = [(i, i + 1) for i in range(len(text))]
        if add_special_tokens:
            ids.append(self.eos_token_id)
            offsets.append((0, 0))
        return {"input_ids": ids, "offset_mapping": offsets}


def test_block_budget_counts_prefix_and_each_end_marker():
    pages = [(1, "abcd"), (2, "efgh"), (3, "ijkl")]
    blocks = probe.make_blocks(pages, Tokenizer(), 23)
    assert len(blocks) == 2
    assert all(len(b["input_ids"]) <= 23 for b in blocks)
    assert [p["page"] for b in blocks for p in b["spans"]] == [1, 2, 3]
    for block in blocks:
        ends = [span["eos"] for span in block["spans"]]
        assert len(set(ends)) == len(ends)
        assert all(block["input_ids"][i] == 9 for i in ends)
        assert all(
            span["start"] < span["stop"] <= span["eos"] for span in block["spans"]
        )
    with pytest.raises(ValueError, match="exceeds"):
        probe.make_blocks([(1, "x" * 50)], Tokenizer(), 23)


def test_pools_subspans_not_the_whole_context():
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]])
    pooled = probe.pool_span(hidden, {"start": 1, "stop": 3, "eos": 3})
    assert torch.allclose(pooled["mean"], torch.tensor([1.0, 2.0]) / 5**0.5)
    assert torch.allclose(pooled["last"], torch.tensor([1.0, 1.0]) / 2**0.5)
    assert torch.equal(pooled["eos"], torch.tensor([-1.0, 0.0]))


@pytest.mark.parametrize(
    "vectors",
    [torch.zeros(2, 768), torch.ones(2, 767), torch.full((2, 768), float("nan"))],
)
def test_invalid_vectors_block_compatibility(vectors):
    with pytest.raises(ValueError):
        probe.validate_vectors(vectors)


def test_gate_requires_controls_quality_fit_and_cleanup():
    good = {
        "controlled_top1": 2,
        "controlled_cases": 2,
        "recall": 0.6,
        "precision": 0.5,
    }
    assert probe.compatible_methods({"mean": good}, 0.6, 0.5, True, True) == ["mean"]
    assert not probe.compatible_methods(
        {"mean": dict(good, controlled_top1=1)}, 0.6, 0.5, True, True
    )
    assert not probe.compatible_methods(
        {"mean": dict(good, recall=0.5)}, 0.6, 0.5, True, True
    )
    assert not probe.compatible_methods({"mean": good}, 0.6, 0.5, False, True)
    assert not probe.compatible_methods({"mean": good}, 0.6, 0.5, True, False)


def test_missing_snapshot_is_visible_and_never_downloads(tmp_path):
    with pytest.raises(FileNotFoundError):
        probe.require_snapshot(tmp_path / "absent")


def test_model_cleanup_runs_on_failure_and_success():
    events = []

    class Model:
        pass

    for fail in (False, True):
        reference = None
        try:
            with probe.owned_model(
                Model, lambda: events.append("gc"), lambda: events.append("empty")
            ) as get_model:
                import weakref

                reference = weakref.ref(get_model())
                if fail:
                    raise RuntimeError("injected")
        except RuntimeError:
            pass
        assert reference() is None
    assert events == ["gc", "empty", "gc", "empty"]


def test_baseline_queries_are_matched_by_case_id_not_row_position():
    labels = {"cases": [{"case_id": "a"}, {"case_id": "b"}]}
    baseline = {
        "status": "complete",
        "rows": [
            {"case_id": "b", "retrieval_query": "query b"},
            {"case_id": "a", "retrieval_query": "query a"},
        ],
    }
    assert probe.ordered_queries(baseline, labels) == ["query a", "query b"]
    with pytest.raises(ValueError):
        probe.ordered_queries(dict(baseline, status="running"), labels)
    with pytest.raises(ValueError):
        probe.ordered_queries(dict(baseline, rows=[baseline["rows"][0]] * 2), labels)


def test_failed_source_verification_still_persists_original_evidence(
    tmp_path, monkeypatch
):
    import json

    from src import config

    path = tmp_path / "report.json"
    monkeypatch.setattr(config, "PHASE2_PROBE_PATH", path)
    monkeypatch.setattr(config, "CHECKPOINTS", tmp_path)
    monkeypatch.setattr(probe, "tree_identity", lambda _: "original")
    report = {
        "status": "failed",
        "passed": False,
        "blocked_reason": "original model error",
        "capacity": [{"passed": True}],
    }
    probe.finalize_report(
        report, {"book_stem": "sicp"}, "original", {"missing.md": "expectedhash"}
    )
    saved = json.loads(path.read_text())
    assert saved["status"] == "failed" and saved["passed"] is False
    assert saved["capacity"] == [{"passed": True}]
    assert saved["blocked_reason"] == "original model error"
    assert "FileNotFoundError" in saved["integrity_error"]
