"""Regenerate answers from saved retrieval evidence to review a prompt change."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import httpx

from src.config import (
    FOUNDATION_REPORT_PATH,
    OLLAMA_CONTEXT_TOKENS,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_OUTPUT_TOKENS,
    OLLAMA_TEMPERATURE,
    OLLAMA_TOP_P,
    OLLAMA_URL,
)
from src.db import get_existing_collection
from src.page_querying import build_citation, strip_model_footnotes
from src.tui import PROMPT_TEMPLATE


def regenerate_answers(source: Path, output: Path) -> dict[str, Any]:
    if not OLLAMA_URL.startswith(("http://localhost:", "http://127.0.0.1:")):
        raise ValueError("Prompt review must use the configured local Ollama service.")
    prior = json.loads(source.read_text(encoding="utf-8"))
    book_stem = prior["book_stem"]
    collection = get_existing_collection(book_stem)
    rows = prior["baseline"]["after_results"]
    history: list[dict[str, str]] = []
    results: list[dict[str, Any]] = []
    client = httpx.Client(timeout=120.0)

    try:
        for prior_row in rows:
            ids = prior_row["evidence_ids"]
            if ids:
                fetched = collection.get(
                    ids=ids, include=["documents", "metadatas"]
                )
                fetched_by_id = dict(
                    zip(
                        fetched["ids"],
                        zip(fetched["documents"], fetched["metadatas"]),
                    )
                )
                if set(fetched_by_id) != set(ids):
                    raise RuntimeError(
                        f"Saved evidence is missing for {prior_row['kind']}."
                    )
                context_blocks = []
                pages: list[int] = []
                metadata_rows = []
                for chunk_id in ids:
                    document, metadata = fetched_by_id[chunk_id]
                    page = metadata.get("physical_page", metadata.get("page"))
                    try:
                        page = int(page)
                    except (TypeError, ValueError):
                        page = "N/A"
                    context_blocks.append(
                        f"[{metadata.get('Header 1', 'Unknown Chapter')} · "
                        f"{metadata.get('Header 2', 'Unknown Section')} · p.{page}]\n"
                        f"{document}"
                    )
                    metadata_rows.append(metadata)
                    if isinstance(page, int) and page >= 1:
                        pages.append(page)
                context = "\n\n".join(context_blocks)
            else:
                context = "No direct context retrieved from indexed files. Use your general knowledge."
                metadata_rows = []
                pages = []

            history_text = ""
            if history:
                history_text = "\n".join(
                    ["Conversation so far:"]
                    + [
                        f"{'User' if turn['role'] == 'user' else 'Axie'}: {turn['content']}"
                        for turn in history[-6:]
                    ]
                ) + "\n"
            prompt = PROMPT_TEMPLATE.format(
                book=book_stem,
                chapter=(
                    ",".join(
                        sorted(
                            {
                                str(meta.get("Header 1", "Unknown"))
                                for meta in metadata_rows
                            }
                        )
                    )
                    if metadata_rows
                    else "General Knowledge"
                ),
                pages=(
                    ",".join(
                        sorted(
                            {
                                str(meta.get("physical_page", meta.get("page", "?")))
                                for meta in metadata_rows
                            }
                        )
                    )
                    if metadata_rows
                    else "N/A"
                ),
                history=history_text,
                context=context,
                question=prior_row["question"],
            )
            started = time.monotonic()
            response = client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "think": False,
                    "keep_alive": OLLAMA_KEEP_ALIVE,
                    "options": {
                        "temperature": OLLAMA_TEMPERATURE,
                        "top_p": OLLAMA_TOP_P,
                        "num_ctx": OLLAMA_CONTEXT_TOKENS,
                        "num_predict": OLLAMA_OUTPUT_TOKENS,
                    },
                },
            )
            response.raise_for_status()
            generated = response.json()
            answer = strip_model_footnotes(str(generated.get("response", "")))
            citation = build_citation(book_stem, pages)
            if answer and not answer.endswith(citation):
                answer = f"{answer}\n\n{citation}"
            results.append(
                {
                    "kind": prior_row["kind"],
                    "question": prior_row["question"],
                    "evidence_ids": ids,
                    "answer": answer,
                    "latency_seconds": round(time.monotonic() - started, 2),
                    "prompt_eval_count": generated.get("prompt_eval_count"),
                    "generation_count": generated.get("eval_count"),
                }
            )
            history.extend(
                [
                    {"role": "user", "content": prior_row["question"]},
                    {"role": "assistant", "content": answer},
                ]
            )
    finally:
        client.close()

    result = {
        "book_stem": book_stem,
        "model": OLLAMA_MODEL,
        "evidence_source_report": str(source),
        "retrieval_reused": True,
        "gpu_fit_measured": False,
        "output_limit": OLLAMA_OUTPUT_TOKENS,
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=FOUNDATION_REPORT_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=FOUNDATION_REPORT_PATH.with_name("foundation_prompt_review.json"),
    )
    args = parser.parse_args()
    report = regenerate_answers(args.source, args.output)
    for row in report["results"]:
        print(
            f"{row['kind']}: {row['generation_count']} tokens, "
            f"{row['latency_seconds']}s"
        )
    print(f"Saved prompt review to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
