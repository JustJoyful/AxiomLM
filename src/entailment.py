"""Phase 4 entailment scorer: one pinned NLI cross-encoder, always on CPU.

Long premises are split into non-overlapping windows that fit the model's
max length beside the hypothesis; a premise's score is its best window, so a
supporting sentence past the truncation point is still seen.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from huggingface_hub import snapshot_download
from sentence_transformers import CrossEncoder

from src import config

FORMULATIONS = {
    "question": "{question}",
    "answers": "This text answers the question: {question}",
    "relevant": "This text is relevant to the question: {question}",
}

# [CLS] premise [SEP] hypothesis [SEP]
_SPECIAL_TOKENS = 3
_SPLITS = (re.compile(r"\n\s*\n"), re.compile(r"(?<=[.!?])\s+"), re.compile(r"\s+"))

_SCORER: Scorer | None = None


def hypothesis_for(
    question: str, formulation: str = config.ENTAILMENT_FORMULATION
) -> str:
    if formulation not in FORMULATIONS:
        raise ValueError(f"Unknown entailment formulation: {formulation!r}")
    return FORMULATIONS[formulation].format(question=question.strip())


def _spans(text: str, start: int, end: int, level: int, fits) -> list[tuple[int, int]]:
    """Split text[start:end] at the coarsest boundary whose parts fit."""
    if fits(text[start:end]):
        return [(start, end)]
    if level == len(_SPLITS):
        spans = []
        while start < end:
            low, high, stop = start + 1, end, start
            while low <= high:
                middle = (low + high) // 2
                if fits(text[start:middle]):
                    stop, low = middle, middle + 1
                else:
                    high = middle - 1
            if stop == start:
                raise ValueError("A premise character exceeds the token budget")
            spans.append((start, stop))
            start = stop
        return spans
    spans = []
    cursor = start
    for match in _SPLITS[level].finditer(text, start, end):
        if text[cursor : match.start()].strip():
            spans.extend(_spans(text, cursor, match.start(), level + 1, fits))
        cursor = match.end()
    if text[cursor:end].strip():
        spans.extend(_spans(text, cursor, end, level + 1, fits))
    return spans


def premise_windows(
    premise: str, hypothesis: str, tokenizer, max_length: int
) -> list[str]:
    budget = (
        max_length
        - _SPECIAL_TOKENS
        - len(tokenizer.encode(hypothesis, add_special_tokens=False))
    )
    if budget < 1:
        raise ValueError("Hypothesis leaves no room for the premise")

    def fits(text: str) -> bool:
        return len(tokenizer.encode(text, add_special_tokens=False)) <= budget

    windows: list[tuple[int, int]] = []
    for start, end in _spans(premise, 0, len(premise), 0, fits):
        if windows and fits(premise[windows[-1][0] : end]):
            windows[-1] = (windows[-1][0], end)
        else:
            windows.append((start, end))
    return [premise[start:end] for start, end in windows] or [premise]


class Scorer:
    def __init__(
        self,
        model_id: str,
        revision: str,
        *,
        max_parameters: int | None = config.ENTAILMENT_MAX_PARAMETERS,
        model=None,
    ):
        if model is None:
            if config.ENTAILMENT_DEVICE != "cpu":
                raise ValueError(
                    f"Entailment scorer must run on CPU, not {config.ENTAILMENT_DEVICE!r}"
                )
            path = snapshot_download(repo_id=model_id, revision=revision)
            model = CrossEncoder(path, device=config.ENTAILMENT_DEVICE)
        self.model_id = model_id
        self.revision = revision
        self.model = model
        self.parameters = sum(p.numel() for p in model.parameters())
        if max_parameters is not None and self.parameters >= max_parameters:
            raise ValueError(
                f"{model_id} has {self.parameters} parameters; limit is {max_parameters}"
            )
        labels = {
            int(i): str(name).lower() for i, name in model.config.id2label.items()
        }
        matches = [i for i, name in labels.items() if name.startswith("entail")]
        if not matches:
            raise ValueError(f"{model_id} has no entailment label: {labels}")
        self.entailment_index = matches[0]
        self.max_length = int(model.max_length)

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """P(entailment) per (premise, hypothesis): the max over premise windows."""
        flat: list[tuple[str, str]] = []
        owners: list[int] = []
        for index, (premise, hypothesis) in enumerate(pairs):
            for window in premise_windows(
                premise, hypothesis, self.model.tokenizer, self.max_length
            ):
                flat.append((window, hypothesis))
                owners.append(index)
        if not flat:
            return []
        rows = self.model.predict(
            flat, apply_softmax=True, batch_size=config.ENTAILMENT_BATCH_SIZE
        )
        best = [0.0] * len(pairs)
        for owner, row in zip(owners, rows):
            best[owner] = max(best[owner], float(row[self.entailment_index]))
        return best


def get_scorer() -> Scorer:
    """Return the configured scorer, loading it once from the pinned snapshot."""
    global _SCORER
    if _SCORER is None:
        if not config.ENTAILMENT_MODEL:
            raise RuntimeError("Entailment scorer is not configured")
        _SCORER = Scorer(config.ENTAILMENT_MODEL, config.ENTAILMENT_REVISION)
    return _SCORER
