"""Advisory claim verdicts from the shared CPU scorer; never rewrite answers."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from src import config
from src.page_querying import ABSTENTION

_PAGE = re.compile(r"\bpp?\.\s*(\d+)(?:\s*[–-]\s*(\d+))?")
_MENTION = re.compile(
    r"(?:\b(?:on|see)\s+)?\(?\bpp?\.\s*\d+(?:\s*[–-]\s*\d+)?\)?", re.IGNORECASE
)
_PROTECTED = re.compile(
    r"```[^\n]*\n[\s\S]*?(?:```|\Z)|~~~[^\n]*\n[\s\S]*?(?:~~~|\Z)|\$\$[\s\S]*?(?:\$\$|\Z)|^\s*#{1,6}\s+[^\n]*",
    re.MULTILINE,
)


@dataclass(frozen=True)
class Claim:
    text: str
    start: int
    end: int
    pages: tuple[int, ...]


@dataclass(frozen=True)
class ClaimVerdict:
    claim: Claim
    supported: bool
    score: float
    source_id: str | None
    failed_pages: tuple[int, ...]


@dataclass(frozen=True)
class CitationReport:
    status: str
    verdicts: tuple[ClaimVerdict, ...]
    unchecked: int
    supporting_source_ids: tuple[str, ...]
    answer: str


def _pages(match):
    first = int(match[1])
    last = int(match[2] or first)
    # Malformed or huge ranges are not expanded into an unbounded allocation.
    if first < 1 or last < first or last - first > config.TOOL_QUERY_MAX_CHARS:
        return ()
    return tuple(range(first, last + 1))


def split_claims(answer: str) -> list[Claim]:
    masked = _PROTECTED.sub(
        lambda m: "".join("\n" if c == "\n" else " " for c in m[0]), answer
    )
    masked = re.sub(r"\bpp?\.", lambda m: m[0][:-1] + "\u2024", masked)
    claims = []
    for match in re.finditer(r".+?(?:[.!?](?=\s|$)|(?=\n|$))", masked):
        if not match[0].strip():
            continue
        start, end = match.span()
        raw = answer[start:end]
        prefix = re.match(r"\s*(?:[-*+]\s+|\d+[.)]\s+)?", raw)[0]
        start += len(prefix)
        end = start + len(answer[start:end].rstrip())
        text = answer[start:end]
        pages = tuple(p for m in _PAGE.finditer(text) for p in _pages(m))
        if pages and _strip_mentions(text) in ("", ".") and claims:
            previous = claims[-1]
            if not answer[previous.end : start].strip():
                claims[-1] = Claim(
                    answer[previous.start : end],
                    previous.start,
                    end,
                    previous.pages + pages,
                )
                continue
        if text == ABSTENTION or len(text.split()) < config.CITATION_MIN_CLAIM_WORDS:
            continue
        claims.append(Claim(text, start, end, pages))
    return claims


def _strip_mentions(text, failed=None):
    def replace(match):
        page_match = _PAGE.search(match[0])
        if failed is not None and not set(_pages(page_match)) & set(failed):
            return match[0]
        return ""

    cleaned = _MENTION.sub(replace, text)
    return re.sub(r"\s+([.!?,;:])", r"\1", re.sub(r" {2,}", " ", cleaned)).strip()


def _scores(pairs, score):
    values = list(score(pairs)) if pairs else []
    if len(values) != len(pairs) or any(
        not math.isfinite(v) or not 0 <= v <= 1 for v in values
    ):
        raise ValueError("Invalid entailment probabilities")
    return values


def check_answer(answer, sources, score, *, threshold=None, max_claims=None):
    threshold = config.ENTAILMENT_CLAIM_THRESHOLD if threshold is None else threshold
    max_claims = config.CITATION_MAX_CLAIMS if max_claims is None else max_claims
    claims = split_claims(answer)
    selected = claims[:max_claims]
    values = _scores(
        [(s.text, _strip_mentions(c.text)) for c in selected for s in sources], score
    )
    verdicts, supporting = [], set()
    for index, claim in enumerate(selected):
        scores = values[index * len(sources) : (index + 1) * len(sources)]
        best = max(range(len(scores)), key=scores.__getitem__) if scores else None
        supported = [s for s, value in zip(sources, scores) if value >= threshold]
        supporting.update(s.source_id for s in supported)
        failed = tuple(
            p
            for p in claim.pages
            if not any(s.kind == "book" and p in s.physical_pages for s in supported)
        )
        verdicts.append(
            ClaimVerdict(
                claim,
                bool(supported),
                scores[best] if best is not None else 0.0,
                sources[best].source_id if best is not None else None,
                failed,
            )
        )
    return CitationReport(
        "checked",
        tuple(verdicts),
        len(claims) - len(selected),
        tuple(s.source_id for s in sources if s.source_id in supporting),
        answer,
    )


def verify(claim, source, score, threshold=None):
    threshold = config.ENTAILMENT_CLAIM_THRESHOLD if threshold is None else threshold
    value = _scores([(source.text, claim)], score)[0]
    return value >= threshold, value


def report_payload(report):
    return {
        "status": report.status,
        "answer": report.answer,
        "unchecked": report.unchecked,
        "supporting_source_ids": list(report.supporting_source_ids),
        "claims": [
            {
                "text": v.claim.text,
                "supported": v.supported,
                "score": v.score,
                "source_id": v.source_id,
                "failed_pages": list(v.failed_pages),
            }
            for v in report.verdicts
        ],
        "citations": [
            {"page": p, "supported": p not in v.failed_pages}
            for v in report.verdicts
            for p in v.claim.pages
        ],
    }


async def run_check(answer, sources, *, remaining=None):
    """Bound host-side checking without spending agent model/tool calls."""
    import asyncio
    import time

    from src.entailment import get_scorer

    budget = config.CITATION_CHECK_SECONDS
    deadline = remaining is not None and remaining <= budget
    if remaining is not None:
        budget = min(budget, remaining)
    if budget <= 0:
        return {"status": "unchecked", "reason": "deadline", "check_seconds": 0.0}
    started = time.monotonic()

    def check():
        return report_payload(
            check_answer(answer, sources, lambda pairs: get_scorer().score(pairs))
        )

    try:
        payload = await asyncio.wait_for(asyncio.to_thread(check), budget)
    except TimeoutError:
        payload = {
            "status": "unchecked",
            "reason": "deadline" if deadline else "time budget",
        }
    except Exception as exc:  # noqa: BLE001 - surface scorer failures
        payload = {
            "status": "unavailable",
            "reason": f"Entailment scorer unavailable: {exc}",
        }
    return {**payload, "check_seconds": time.monotonic() - started}
