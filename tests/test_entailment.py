"""Phase 4 entailment scorer: label lookup, size cap, windowing and CPU loading."""

import numpy as np
import pytest

from src import config, entailment
from src.entailment import FORMULATIONS, Scorer, hypothesis_for, premise_windows


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class FakeParam:
    def __init__(self, count):
        self.count = count

    def numel(self):
        return self.count


class FakeConfig:
    def __init__(self, id2label):
        self.id2label = id2label


class FakeModel:
    def __init__(self, id2label=None, parameters=1000, max_length=512, rows=None):
        self.tokenizer = WordTokenizer()
        self.config = FakeConfig(
            id2label or {0: "contradiction", 1: "entailment", 2: "neutral"}
        )
        self._parameters = parameters
        self.max_length = max_length
        self.rows = rows
        self.calls = []

    def parameters(self):
        return iter([FakeParam(self._parameters)])

    def predict(self, pairs, apply_softmax=True, batch_size=32):
        self.calls.append((list(pairs), apply_softmax, batch_size))
        return np.array(self.rows(pairs))


def test_entailment_column_is_found_by_label_name():
    scorer = Scorer("m", "r", model=FakeModel())
    assert scorer.entailment_index == 1
    assert scorer.parameters == 1000
    assert scorer.max_length == 512


def test_binary_model_label_is_supported():
    scorer = Scorer("m", "r", model=FakeModel({0: "entailment", 1: "not_entailment"}))
    assert scorer.entailment_index == 0


def test_model_without_entailment_label_is_rejected():
    with pytest.raises(ValueError, match="entail"):
        Scorer("m", "r", model=FakeModel({0: "LABEL_0", 1: "LABEL_1"}))


def test_oversized_model_is_rejected():
    with pytest.raises(ValueError, match="parameters"):
        Scorer("m", "r", model=FakeModel(parameters=100_000_000))


def test_premise_windows_cover_text_without_overlap():
    premise = (
        "One two three four. Five six seven eight.\n\n"
        "Nine ten. " + " ".join(f"w{i}" for i in range(30)) + "."
    )
    windows = premise_windows(premise, "is it", WordTokenizer(), max_length=15)
    budget = 15 - 3 - 2
    assert len(windows) > 1
    assert all(len(window.split()) <= budget for window in windows)
    assert " ".join(windows).split() == premise.split()
    assert all(window in premise for window in windows)
    assert sum(len(window) for window in windows) <= len(premise)


def test_support_in_last_window_is_found():
    premise = (
        "Alpha beta gamma delta.\n\nEpsilon zeta eta theta.\n\nIota kappa lambda mu."
    )
    model = FakeModel(
        max_length=9,
        rows=lambda pairs: [
            [0.05, 0.9 if premise_text.startswith("Iota") else 0.1, 0.05]
            for premise_text, _ in pairs
        ],
    )
    scorer = Scorer("m", "r", model=model)
    assert scorer.score([(premise, "Claim here")]) == [pytest.approx(0.9)]
    assert len(model.calls) == 1
    assert len(model.calls[0][0]) == 3
    assert model.calls[0][1] is True
    assert scorer.score([]) == []


def test_hypothesis_formulations():
    assert set(FORMULATIONS) == {"question", "answers", "relevant"}
    assert hypothesis_for("  What is X?  ", "question") == "What is X?"
    assert (
        hypothesis_for("What is X?", "answers")
        == "This text answers the question: What is X?"
    )
    assert (
        hypothesis_for("What is X?", "relevant")
        == "This text is relevant to the question: What is X?"
    )
    with pytest.raises(ValueError):
        hypothesis_for("What is X?", "nope")


def test_unconfigured_scorer_raises(monkeypatch):
    monkeypatch.setattr(entailment, "_SCORER", None)
    monkeypatch.setattr(config, "ENTAILMENT_MODEL", "")
    with pytest.raises(RuntimeError, match="Entailment scorer is not configured"):
        entailment.get_scorer()


def test_scorer_stays_on_cpu_when_cuda_exists(monkeypatch):
    import torch

    seen = {}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(entailment, "_SCORER", None)
    monkeypatch.setattr(config, "ENTAILMENT_MODEL", "org/nli")
    monkeypatch.setattr(config, "ENTAILMENT_REVISION", "a" * 40)

    def fake_download(repo_id, revision):
        seen["download"] = (repo_id, revision)
        return "/snapshot"

    def fake_cross_encoder(path, device):
        seen["load"] = (path, device)
        return FakeModel()

    monkeypatch.setattr(entailment, "snapshot_download", fake_download)
    monkeypatch.setattr(entailment, "CrossEncoder", fake_cross_encoder)
    scorer = entailment.get_scorer()
    assert seen == {"download": ("org/nli", "a" * 40), "load": ("/snapshot", "cpu")}
    assert entailment.get_scorer() is scorer
