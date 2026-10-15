"""Compare actual Phase 1 retrieval at fixed K=8 and configured K=5.

Reuses the accepted SICP labels and existing index. Never indexes or ingests.
Answer grounding/citation review remains explicit until reviewed against passages.
"""

import json
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate_foundation import _current_index_identity, _retrieval_metrics
from src import config, tui
from src.llm import OllamaClient
from src.ui.render import sanitize_answer


def validate():
    foundation = json.loads(config.FOUNDATION_REPORT_PATH.read_text())
    baseline = foundation["baseline"]
    labels = json.loads((config.ROOT / baseline["before_path"]).read_text())
    book = foundation["book_stem"]
    identity = _current_index_identity(book)
    if identity != baseline["index_identity"]:
        raise RuntimeError(
            "Index identity changed; existing labels cannot be assumed compatible."
        )
    app = tui.AxiomLMApp()
    report = {
        "book": book,
        "index_identity": identity,
        "baseline_model": baseline["after_model_identity"],
        "config": {
            "model": config.OLLAMA_MODEL,
            "distance_threshold": config.DISTANCE_THRESHOLD,
        },
        "runs": {},
    }
    client = OllamaClient()
    reference = baseline["after_results"]
    for count in (8, config.TOP_K_RESULTS):
        app._conversation = []
        rows = []
        report["runs"][str(count)] = {"rows": rows}
        for previous in reference:
            query = previous["question"]
            started = time.monotonic()
            prompt, metadata, pages, ids = app._prepare_query(
                query, app._format_history(), book, count, config.DISTANCE_THRESHOLD
            )
            retrieval_seconds = time.monotonic() - started
            started = time.monotonic()
            print(
                f'K={count} {previous["kind"]}: {len(ids)} chunks; generating',
                flush=True,
            )
            token = app._try_acquire_workload("generation")
            if token is None:
                raise RuntimeError("GPU admission unavailable")
            try:
                answer = client.generate(
                    prompt, model=config.OLLAMA_MODEL, stream=False, think=False
                )
                if not isinstance(answer, dict):
                    raise TypeError("Expected completed response")
                raw = answer.get("response", "")
            finally:
                client.unload_and_wait()
                app._release_workload(token)
            row = {
                "kind": previous["kind"],
                "question": query,
                "evidence_ids": ids,
                "evidence_metadata": metadata,
                "physical_pages": pages,
                "retrieval_seconds": retrieval_seconds,
                "generation_seconds": time.monotonic() - started,
                "raw_answer": raw,
                "copy_text": sanitize_answer(raw),
                "grounding_reviewed": False,
                "fixed_k8_ids_match": (
                    ids == previous["evidence_ids"] if count == 8 else None
                ),
            }
            rows.append(row)
            app._conversation.extend(
                [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": row["copy_text"]},
                ]
            )
            report["runs"][str(count)]["retrieval_metrics"] = (
                _retrieval_metrics(labels, rows)
                if len(rows) == len(reference)
                else None
            )
            config.UI_RETRIEVAL_REPORT_PATH.write_text(json.dumps(report, indent=2))
            print(f'K={count} {previous["kind"]}: captured', flush=True)
    return report


if __name__ == "__main__":
    result = validate()
    print(config.UI_RETRIEVAL_REPORT_PATH)
