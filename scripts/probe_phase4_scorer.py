"""Phase 4 scorer selection probe over hash-pinned question and claim pairs.

Pairs live in tests/fixtures/phase4/scorer_pairs.json. Every premise must be a
verbatim substring of its one-based physical page in the repaired text, and
that page is pinned by sha256, so a page that changed after the pairs were
written is refused instead of silently shifting the selection evidence.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
import time
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from huggingface_hub import model_info

from scripts.evaluate_phase3_baseline import digest
from src import config
from src.entailment import FORMULATIONS, Scorer, hypothesis_for, premise_windows
from src.index_store import pair_path, read_manifest

QUESTION_RELATIONS = ("answers", "paraphrase", "topical", "contradicts", "unrelated")
CLAIM_RELATIONS = ("entailed", "paraphrase", "neutral", "contradiction", "tamper")
GATE_POSITIVES = {"answers", "paraphrase"}
CLAIM_POSITIVES = {"entailed", "paraphrase"}
SPLITS = ("calibration", "test")
BOOKS = ("sicp", "thinkpython2")
PERMISSIVE_LICENSES = frozenset(
    {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "cc-by-4.0", "cc-by-sa-4.0"}
)
GATE_BATCH = config.HYBRID_CANDIDATE_LIMIT
WARMUP_RUNS = 3
TIMED_RUNS = 20
_PAIR_KINDS = (
    ("question_pairs", "question", QUESTION_RELATIONS),
    ("claim_pairs", "claim", CLAIM_RELATIONS),
)


def load_pairs(
    path: Path, *, pages_root: Path = config.REPAIRED_MD, books=BOOKS
) -> dict:
    """Validated pairs, each annotated with its split."""
    pairs = json.loads(Path(path).read_text())
    if pairs.get("schema_version") != 1:
        raise ValueError("unsupported pairs schema_version")
    hashes = pairs["source_pages"]
    texts: dict[tuple[str, int], str] = {}
    ids: list[str] = []
    for key, field, relations in _PAIR_KINDS:
        for row in pairs[key]:
            pid = row["pair_id"]
            ids.append(pid)
            expected = {"pair_id", "book_stem", "page", "premise", field, "relation"}
            if set(row) != expected:
                raise ValueError(f"pair {pid}: fields must be {sorted(expected)}")
            if row["relation"] not in relations:
                raise ValueError(f"pair {pid}: unknown relation {row['relation']!r}")
            book, page = row["book_stem"], row["page"]
            if book not in books or not isinstance(page, int) or page < 1:
                raise ValueError(f"pair {pid}: bad book/page {book!r}/{page!r}")
            if not row["premise"].strip() or not row[field].strip():
                raise ValueError(f"pair {pid}: empty premise or {field}")
            if (book, page) not in texts:
                source = Path(pages_root) / book / f"page_{page - 1:04d}.md"
                pinned = hashes.get(book, {}).get(str(page))
                if pinned is None:
                    raise ValueError(f"{book} page {page} has no hash")
                if not source.is_file() or digest(source) != pinned:
                    raise ValueError(f"{book} source page {page} changed")
                texts[book, page] = source.read_text()
            if row["premise"] not in texts[book, page]:
                raise ValueError(
                    f"pair {pid}: premise is not verbatim on {book} page {page}"
                )
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate pair_id")
    splits = pairs["splits"]
    if sorted(splits) != sorted(SPLITS):
        raise ValueError(f"splits must be exactly {list(SPLITS)}")
    calibration, test = set(splits["calibration"]), set(splits["test"])
    if calibration & test:
        raise ValueError(f"pairs in both splits: {sorted(calibration & test)}")
    if calibration | test != set(ids):
        raise ValueError(
            f"pairs not in a split: {sorted(set(ids) - calibration - test)}; "
            f"unknown split ids: {sorted((calibration | test) - set(ids))}"
        )
    for key, _, _ in _PAIR_KINDS:
        for row in pairs[key]:
            row["split"] = "calibration" if row["pair_id"] in calibration else "test"
    return pairs


def auc(pos: list[float], neg: list[float]) -> float | None:
    """Mann–Whitney AUC: P(positive > negative), ties count one half."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def best_threshold(pos: list[float], neg: list[float]) -> float | None:
    """The observed score maximising TPR−FPR for `score >= t`; ties take the higher t."""
    if not pos or not neg:
        return None
    best, best_j = None, Fraction(-2)
    for t in sorted(set(pos + neg), reverse=True):
        # Exact arithmetic: float rounding would break genuine ties arbitrarily.
        j = Fraction(sum(p >= t for p in pos), len(pos)) - Fraction(
            sum(n >= t for n in neg), len(neg)
        )
        if j > best_j:
            best, best_j = t, j
    return best


def split_scores(rows, scores, ids, positives) -> tuple[list[float], list[float]]:
    """Positive and negative scores of the rows whose pair_id is in ids."""
    chosen = [r for r in rows if r["pair_id"] in ids]
    pos = [scores[r["pair_id"]] for r in chosen if r["relation"] in positives]
    neg = [scores[r["pair_id"]] for r in chosen if r["relation"] not in positives]
    return pos, neg


def _rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": k / n if n else None}


def _passing(rows, scores, threshold) -> dict:
    return _rate(sum(scores[r["pair_id"]] >= threshold for r in rows), len(rows))


def _split_metrics(rows, scores, ids, positives, relations, threshold) -> dict:
    chosen = [r for r in rows if r["pair_id"] in ids]
    by_relation = {
        rel: [r for r in chosen if r["relation"] == rel] for rel in relations
    }
    pos, neg = split_scores(rows, scores, ids, positives)
    metrics = {
        "mean": {
            rel: statistics.fmean(scores[r["pair_id"]] for r in rs) if rs else None
            for rel, rs in by_relation.items()
        },
        "auc_positive_vs_rest": auc(pos, neg),
    }
    if threshold is not None:
        metrics["passing"] = {
            rel: _passing(rs, scores, threshold) for rel, rs in by_relation.items()
        }
        metrics["positives_passing"] = _passing(
            [r for r in chosen if r["relation"] in positives], scores, threshold
        )
        metrics["negatives_passing"] = _passing(
            [r for r in chosen if r["relation"] not in positives], scores, threshold
        )
    return metrics


def _timed(fn, runs: int) -> dict:
    for _ in range(min(WARMUP_RUNS, runs)):
        fn()
    seconds = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        seconds.append(time.perf_counter() - start)
    seconds.sort()
    return {
        "runs": runs,
        "p50": statistics.median(seconds),
        "p95": seconds[min(len(seconds) - 1, round(0.95 * (len(seconds) - 1)))],
    }


def _rss_bytes() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _license(info) -> str | None:
    card = info.card_data or {}
    value = card.get("license") if hasattr(card, "get") else None
    if value is None:
        value = next(
            (t.split(":", 1)[1] for t in info.tags or [] if t.startswith("license:")),
            None,
        )
    return value


def chunk_texts(books=BOOKS, *, root: Path = config.VECTOR_STORE) -> dict:
    """Chunk texts of each book's active index, read-only from its BM25 table."""
    manifest = read_manifest(root)
    texts = {}
    for book in books:
        entry = manifest.get(book)
        if entry is None:
            raise ValueError(f"'{book}' has no active versioned index")
        path = pair_path(root, book, entry["version_id"])
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
            texts[book] = [
                row[0]
                for row in db.execute("SELECT document FROM chunks ORDER BY chunk_id")
            ]
    return texts


def probe_candidate(
    repo: str,
    revision: str | None,
    pairs: dict,
    *,
    chunk_texts: dict,
    timed_runs: int = TIMED_RUNS,
) -> dict:
    """Size, license, truncation, CPU latency and per-relation behavior of one model."""
    info = model_info(repo, revision=revision)
    license_id = _license(info)
    rss_before = _rss_bytes()
    start = time.perf_counter()
    try:
        scorer = Scorer(repo, info.sha, max_parameters=None)
    except ValueError as exc:
        return {
            "repo": repo,
            "revision": info.sha,
            "license": license_id,
            "eligible": False,
            "rejected_for": ["label"],
            "error": str(exc),
        }
    load_seconds = time.perf_counter() - start
    rss_after = _rss_bytes()
    rejected = []
    if scorer.parameters >= config.ENTAILMENT_MAX_PARAMETERS:
        rejected.append("parameters")
    if license_id not in PERMISSIVE_LICENSES:
        rejected.append("license")
    report = {
        "repo": repo,
        "requested_revision": revision,
        "revision": info.sha,
        "license": license_id,
        "parameters": scorer.parameters,
        "id2label": {str(k): v for k, v in scorer.model.config.id2label.items()},
        "max_length": scorer.max_length,
        "eligible": not rejected,
        "rejected_for": rejected,
        "load_seconds": load_seconds,
        "rss_delta_bytes": (
            rss_after - rss_before if None not in (rss_before, rss_after) else None
        ),
        "torch_threads": torch.get_num_threads(),
    }
    tokenizer = scorer.model.tokenizer
    questions = pairs["question_pairs"]
    claims = pairs["claim_pairs"]
    calibration = set(pairs["splits"]["calibration"])
    test = set(pairs["splits"]["test"])
    answers = [r["question"] for r in questions if r["relation"] == "answers"]
    # Worst case: the longest answers question leaves the least room for the premise.
    longest = max(answers, key=lambda q: len(tokenizer.encode(q)), default=None)

    report["truncation"] = {}
    for name in FORMULATIONS:
        if longest is None:
            break
        hypothesis = hypothesis_for(longest, name)
        report["truncation"][name] = {
            book: _rate(
                sum(
                    len(premise_windows(t, hypothesis, tokenizer, scorer.max_length))
                    > 1
                    for t in texts
                ),
                len(texts),
            )
            for book, texts in chunk_texts.items()
        }

    report["latency"] = {}
    gate_texts = [t for texts in chunk_texts.values() for t in texts][:GATE_BATCH]
    if gate_texts and longest is not None:
        hypothesis = hypothesis_for(longest, config.ENTAILMENT_FORMULATION)
        batch = [(t, hypothesis) for t in gate_texts]
        windows = sum(
            len(premise_windows(t, hypothesis, tokenizer, scorer.max_length))
            for t in gate_texts
        )
        report["latency"]["gate_batch"] = {
            "pairs": len(batch),
            "windows": windows,
            "seconds": _timed(lambda: scorer.score(batch), timed_runs),
        }
    if claims:
        claim_batch = [(r["premise"], r["claim"]) for r in claims]
        timing = _timed(lambda: scorer.score(claim_batch), timed_runs)
        report["latency"]["claim_pair"] = {
            "pairs": len(claim_batch),
            "seconds_per_pair": {
                k: timing[k] / len(claim_batch) for k in ("p50", "p95")
            },
        }

    report["formulations"] = {}
    for name in FORMULATIONS:
        values = scorer.score(
            [(r["premise"], hypothesis_for(r["question"], name)) for r in questions]
        )
        scores = dict(zip((r["pair_id"] for r in questions), values))
        threshold = best_threshold(
            *split_scores(questions, scores, calibration, GATE_POSITIVES)
        )
        result = {"threshold": threshold}
        for split, ids in (("calibration", calibration), ("test", test)):
            result[split] = _split_metrics(
                questions, scores, ids, GATE_POSITIVES, QUESTION_RELATIONS, threshold
            )
            result[split]["auc_answers_vs_topical"] = auc(
                *split_scores(
                    [r for r in questions if r["relation"] in ("answers", "topical")],
                    scores,
                    ids,
                    {"answers"},
                )
            )
            if threshold is not None:
                result[split]["contradicts_passing"] = result[split]["passing"][
                    "contradicts"
                ]
        result["scores"] = scores
        report["formulations"][name] = result

    values = scorer.score([(r["premise"], r["claim"]) for r in claims])
    scores = dict(zip((r["pair_id"] for r in claims), values))
    threshold = best_threshold(
        *split_scores(claims, scores, calibration, CLAIM_POSITIVES)
    )
    report["claims"] = {"threshold": threshold, "scores": scores}
    for split, ids in (("calibration", calibration), ("test", test)):
        metrics = _split_metrics(
            claims, scores, ids, CLAIM_POSITIVES, CLAIM_RELATIONS, threshold
        )
        if threshold is not None:
            metrics["supported"] = metrics.pop("passing")
            rows = [
                r
                for r in claims
                if r["pair_id"] in ids and r["relation"] in ("contradiction", "tamper")
            ]
            metrics["contradiction_or_tamper_supported"] = _passing(
                rows, scores, threshold
            )
        report["claims"][split] = metrics
    report["cuda_initialized"] = torch.cuda.is_initialized()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate", action="append", required=True, help="REPO or REPO@REVISION"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pairs", type=Path, default=config.PHASE4_LABELS_DIR / "scorer_pairs.json"
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; refusing to overwrite")
    pairs = load_pairs(args.pairs)
    if pairs["review"].get("status") != "maintainer-reviewed":
        raise SystemExit("scorer pairs are not maintainer-reviewed; refusing to probe")
    texts = chunk_texts()
    candidates = []
    for spec in args.candidate:
        repo, _, revision = spec.partition("@")
        candidates.append(
            probe_candidate(repo, revision or None, pairs, chunk_texts=texts)
        )
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "pairs": {
            "path": str(args.pairs),
            "sha256": digest(args.pairs),
            "review": pairs["review"].get("status"),
        },
        "torch": torch.__version__,
        "gate_formulation_for_latency": config.ENTAILMENT_FORMULATION,
        "candidates": candidates,
        "cuda_initialized": torch.cuda.is_initialized(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    for c in candidates:
        print(
            json.dumps(
                {k: c.get(k) for k in ("repo", "revision", "parameters", "eligible")}
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
