#!/usr/bin/env python3
"""
validate_page_metadata.py — Verify Chroma page metadata integrity.

Checks:
1. Collection has metadata rows.
2. Every metadata row includes integer physical_page >= 1.
3. Collection can be re-opened and preserves count.
4. Sample where-filter on physical_page returns matching rows.
"""

from __future__ import annotations

import argparse
import sys

from src.db import get_collection, list_collections


def _validate_collection(book_stem: str, sample_limit: int = 5000) -> None:
    col = get_collection(book_stem)
    initial_count = col.count()
    if initial_count <= 0:
        raise RuntimeError(f"{book_stem}: empty collection")

    result = col.get(limit=sample_limit, include=["metadatas"])
    metadatas = result.get("metadatas") or []
    if not metadatas:
        raise RuntimeError(f"{book_stem}: no metadata rows returned")

    physical_pages: list[int] = []
    for idx, meta in enumerate(metadatas):
        if not isinstance(meta, dict):
            raise RuntimeError(f"{book_stem}: metadata row {idx} is not a dict")
        if "physical_page" not in meta:
            raise RuntimeError(f"{book_stem}: missing physical_page at row {idx}")
        page_val = meta["physical_page"]
        if not isinstance(page_val, int):
            raise RuntimeError(f"{book_stem}: non-int physical_page at row {idx}: {page_val!r}")
        if page_val < 1:
            raise RuntimeError(f"{book_stem}: invalid physical_page at row {idx}: {page_val}")
        physical_pages.append(page_val)

    # Persistence check: re-open and ensure count remains stable.
    reloaded = get_collection(book_stem)
    if reloaded.count() != initial_count:
        raise RuntimeError(
            f"{book_stem}: count changed after reload ({initial_count} -> {reloaded.count()})"
        )

    # Filter check: query one known page and assert all returned rows match.
    sample_page = physical_pages[0]
    filtered = reloaded.get(where={"physical_page": sample_page}, limit=10, include=["metadatas"])
    filtered_meta = filtered.get("metadatas") or []
    if not filtered_meta:
        raise RuntimeError(f"{book_stem}: where filter returned no rows for page {sample_page}")
    for idx, meta in enumerate(filtered_meta):
        if meta.get("physical_page") != sample_page:
            raise RuntimeError(
                f"{book_stem}: filter mismatch row {idx} expected {sample_page}, "
                f"got {meta.get('physical_page')}"
            )

    print(
        f"PASS {book_stem}: count={initial_count}, "
        f"sampled={len(metadatas)}, min_page={min(physical_pages)}, max_page={max(physical_pages)}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate physical_page metadata in Chroma collections.")
    parser.add_argument(
        "book_stems",
        nargs="*",
        help="Collection names to validate. If omitted, validates all collections.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=5000,
        help="Max metadata rows to sample per collection (default: 5000).",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    targets = args.book_stems or list_collections()
    if not targets:
        print("No collections found to validate.")
        return 0

    failures: list[str] = []
    for stem in targets:
        try:
            _validate_collection(stem, sample_limit=args.sample_limit)
        except Exception as exc:  # explicit reporting per collection
            failures.append(f"{stem}: {exc}")
            print(f"FAIL {stem}: {exc}")

    if failures:
        print("\nMetadata validation failed:")
        for msg in failures:
            print(f" - {msg}")
        return 1

    print("\nPASS: page metadata validation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
