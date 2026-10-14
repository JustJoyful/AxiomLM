"""In-process admission control for GPU-bound generation and ingestion jobs."""

from __future__ import annotations

from threading import Lock
from typing import Literal
from uuid import uuid4

WorkloadKind = Literal["generation", "ingest"]


class WorkloadCoordinator:
    """Allow one owned GPU workload at a time; release only after it stops."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._owner: tuple[str, WorkloadKind] | None = None

    def try_acquire(self, kind: WorkloadKind) -> str | None:
        if kind not in ("generation", "ingest"):
            raise ValueError(f"Unsupported workload kind: {kind!r}")
        with self._lock:
            if self._owner is not None:
                return None
            token = uuid4().hex
            self._owner = (token, kind)
            return token

    def release(self, token: str) -> None:
        with self._lock:
            if self._owner is None or self._owner[0] != token:
                raise ValueError("Workload token does not own the active admission.")
            self._owner = None

    @property
    def active_kind(self) -> WorkloadKind | None:
        with self._lock:
            return self._owner[1] if self._owner is not None else None
