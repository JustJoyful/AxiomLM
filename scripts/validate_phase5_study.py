"""Per-book practice evidence using copied metadata pairs and temporary study state."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import shutil
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_phase3_baseline import identities
from src import config, study
from src.chat_tokens import load_chat_counter
from src.index_store import manifest_path, pair_path, read_manifest
from src.llm import OllamaClient
from src.study_state import StudyStore, load_sections


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def validate_study(books, *, output: Path) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        report = {
            "accepted": False,
            "human_review": None,
            "books": [],
            "model": config.OLLAMA_MODEL,
            "code_sha256": {
                name: digest(config.ROOT / name)
                for name in ("src/config.py", "src/study.py", "src/study_state.py")
            },
            "definition": "Source linkage is not semantic support; coverage is not mastery.",
        }
        original_evidence = study.section_evidence
        for book in books:
            row = {"book": book, "status": "error", "cases": [], "human_review": None}
            report["books"].append(row)
            try:
                root = config.VECTOR_STORE
                entry = read_manifest(root)[book]
                original_pair = pair_path(root, book, entry["version_id"])
                originals = {
                    str(p): digest(p) for p in (manifest_path(root), original_pair)
                }
                with tempfile.TemporaryDirectory(
                    prefix="axiom-study-validation-"
                ) as folder:
                    copied = Path(folder) / "indexes"
                    target = pair_path(copied, book, entry["version_id"])
                    target.parent.mkdir(parents=True)
                    shutil.copy2(original_pair, target)
                    manifest_path(copied).write_text(
                        json.dumps({"schema": 2, "books": {book: entry}})
                    )
                    catalog = load_sections(book, root=copied)
                    sections = catalog.sections[:2]
                    row["identity"] = await asyncio.to_thread(
                        identities,
                        (
                            "src/config.py",
                            "src/study.py",
                            "src/study_state.py",
                            "scripts/validate_phase5_study.py",
                        ),
                    )
                    row["hardware_platform"] = platform.platform()
                    row.update(
                        index_version=entry["version_id"],
                        embedding_signature=entry.get("embedding_signature"),
                        sections=[asdict(s) for s in sections],
                        unassigned_chunks=catalog.unassigned_chunks,
                        original_hashes=originals,
                    )
                    # Pins reach disk before the first model call.
                    handle.seek(0)
                    json.dump(report, handle, indent=2)
                    handle.truncate()
                    handle.flush()
                    store = StudyStore(Path(folder) / "study.sqlite3")
                    store.record_question("a", "turn-1", sections)
                    store.record_question("a", "turn-1", sections)
                    row["counts"] = {
                        "session_a": store.counts(sections, session_id="a"),
                        "session_b": store.counts(sections, session_id="b"),
                        "all": store.counts(sections),
                    }
                    counter = await asyncio.to_thread(
                        load_chat_counter, config.OLLAMA_MODEL
                    )
                    client = OllamaClient()
                    native_chat = client.achat
                    replies = []

                    async def capture_chat(
                        *args, native_chat=native_chat, replies=replies, **kwargs
                    ):
                        reply = await native_chat(*args, **kwargs)
                        replies.append(reply)
                        return reply

                    client.achat = capture_chat
                    try:
                        with patch.object(
                            study,
                            "section_evidence",
                            lambda section, copied=copied: original_evidence(
                                section, root=copied
                            ),
                        ):
                            for i, section in enumerate(sections):
                                case = {
                                    "section_id": section.section_id,
                                    "count": (1, 3, 5)[(books.index(book) * 2 + i) % 3],
                                    "human_supported": None,
                                    "human_checked": None,
                                }
                                row["cases"].append(case)
                                replies.clear()
                                start = time.monotonic()
                                try:
                                    result = await study.generate_practice(
                                        section,
                                        case["count"],
                                        client=client,
                                        counter=counter,
                                    )
                                    case.update(
                                        status="generated",
                                        questions=[asdict(q) for q in result.questions],
                                        sources=[asdict(s) for s in result.sources],
                                        linked=len(result.questions),
                                        total=len(result.questions),
                                    )
                                except Exception as exc:  # noqa: BLE001
                                    case.update(
                                        status="error",
                                        error=f"{type(exc).__name__}: {exc}",
                                    )
                                case["seconds"] = time.monotonic() - start
                                case["model_replies"] = list(replies)
                    finally:
                        await asyncio.to_thread(client.unload_and_wait)
                    row["status"] = "measured"
                row["originals_unchanged"] = all(
                    digest(Path(p)) == value for p, value in originals.items()
                )
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"
        handle.seek(0)
        json.dump(report, handle, indent=2)
        handle.truncate()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("books", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(validate_study(args.books, output=args.output))
    print(
        json.dumps(
            [{"book": r["book"], "status": r["status"]} for r in result["books"]]
        )
    )


if __name__ == "__main__":
    main()
