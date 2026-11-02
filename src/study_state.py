"""Read-only section catalogs and durable, version-scoped coverage observations."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from src import config
from src.index_metadata import validate_book
from src.index_store import pair_path, read_manifest
from src.page_querying import physical_pages_for


@dataclass(frozen=True)
class SectionRef:
    section_id: str
    book_stem: str
    index_version: str
    chapter: str
    section: str
    physical_pages: tuple[int, ...]
    chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class SectionCatalog:
    book_stem: str
    index_version: str
    sections: tuple[SectionRef, ...]
    unassigned_chunks: int


def section_identity(book, version, chapter, section):
    return hashlib.sha256(
        json.dumps(
            [book, version, chapter, section], ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def section_rows(book, version, root):
    validate_book(book)
    if not isinstance(version, str) or not re.fullmatch("[a-f0-9]{64}", version):
        raise ValueError("Invalid v2 index version")
    path = pair_path(root, book, version)
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        return [
            (cid, text, json.loads(meta))
            for cid, text, meta in db.execute(
                "SELECT chunk_id, document, metadata_json FROM chunks ORDER BY chunk_id"
            )
        ]


def _known(value):
    return isinstance(value, str) and value.strip().lower() not in {
        "",
        "unknown",
        "unknown chapter",
        "unknown section",
        "none",
        "null",
        "n/a",
    }


def load_sections(
    book_stem: str, *, root: Path = config.VECTOR_STORE
) -> SectionCatalog:
    validate_book(book_stem)
    entry = read_manifest(root).get(book_stem)
    if entry is None:
        raise ValueError(
            "Study sections require an existing v2 index; ordinary chat remains available"
        )
    version = entry.get("version_id")
    groups, unassigned = {}, 0
    for cid, _, meta in section_rows(book_stem, version, root):
        chapter, section = meta.get("chapter"), meta.get("section")
        pages = physical_pages_for(meta)
        if (
            meta.get("book_stem") != book_stem
            or not _known(chapter)
            or not _known(section)
            or not pages
            or any(type(p) is not int or p < 1 for p in pages)
        ):
            unassigned += 1
            continue
        ids, page_set = groups.setdefault((chapter, section), (set(), set()))
        ids.add(cid)
        page_set.update(pages)
    if not groups:
        raise ValueError("No identified chapter/section metadata available")
    sections = [
        SectionRef(
            section_identity(book_stem, version, chapter, section),
            book_stem,
            version,
            chapter,
            section,
            tuple(sorted(pages)),
            tuple(sorted(ids)),
        )
        for (chapter, section), (ids, pages) in groups.items()
    ]
    sections.sort(
        key=lambda s: (min(s.physical_pages), s.chapter, s.section, s.section_id)
    )
    return SectionCatalog(book_stem, version, tuple(sections), unassigned)


class StudyStore:
    def __init__(self, path: Path = config.STUDY_STATE_PATH):
        self.path = path

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=config.STUDY_STORAGE_SECONDS)
        try:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ValueError("Unsupported study state schema")
            if version == 0:
                if db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall():
                    raise ValueError("Unknown study state schema; original preserved")
                with db:
                    db.execute(
                        "CREATE TABLE visits(session_id TEXT, turn_id TEXT, book_stem TEXT, index_version TEXT, section_id TEXT, PRIMARY KEY(session_id,turn_id,book_stem,index_version,section_id))"
                    )
                    db.execute("PRAGMA user_version=1")
            return db
        except Exception:
            db.close()
            raise

    def record_question(self, session_id, turn_id, sections):
        db = self._connect()
        try:
            with db:
                db.executemany(
                    "INSERT OR IGNORE INTO visits VALUES(?,?,?,?,?)",
                    [
                        (
                            session_id,
                            turn_id,
                            s.book_stem,
                            s.index_version,
                            s.section_id,
                        )
                        for s in sections
                    ],
                )
        finally:
            db.close()

    def counts(self, sections, *, session_id=None):
        db = self._connect()
        try:
            result = {}
            for s in sections:
                query = "SELECT COUNT(*) FROM visits WHERE book_stem=? AND index_version=? AND section_id=?"
                args = [s.book_stem, s.index_version, s.section_id]
                if session_id is not None:
                    query += " AND session_id=?"
                    args.append(session_id)
                result[s.section_id] = db.execute(query, args).fetchone()[0]
            return result
        finally:
            db.close()
