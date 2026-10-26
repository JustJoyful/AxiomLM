"""Bounded, strictly validated client frames; worker control is private."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from src import config


@dataclass(frozen=True)
class PythonResult:
    request_id: str
    status: Literal["ok", "error", "timeout", "cancelled", "unavailable", "worker_died"]
    stdout: str = ""
    stderr: str = ""
    value: str | None = None
    truncated: bool = False
    worker_pid: int | None = None
    worker_instance_id: str | None = None


CHECK_FIELDS = {"kind", "expression", "candidate", "variable"}


def _validate(message: object) -> dict:
    if not isinstance(message, dict):
        raise ValueError("Frame must be an object")  # noqa: TRY004
    ident = message.get("id")
    if (
        not isinstance(ident, str)
        or not ident
        or len(ident) > config.PYTHON_MAX_ID_CHARS
    ):
        raise ValueError("Missing or invalid request ID")
    fields = {
        "execute": {"id", "op", "code"},
        "verify": {"id", "op", "check"},
        "cancel": {"id", "op", "request_id"},
        "shutdown": {"id", "op"},
    }
    op = message.get("op")
    if not isinstance(op, str) or op not in fields or set(message) != fields[op]:
        raise ValueError("Unknown operation or fields")
    if op == "execute":
        code = message["code"]
        if (
            not isinstance(code, str)
            or len(code.encode("utf-8")) > config.PYTHON_MAX_CODE_BYTES
        ):
            raise ValueError("Invalid or oversized code")
    if op == "verify":
        check = message["check"]
        if (
            not isinstance(check, dict)
            or set(check) != CHECK_FIELDS
            or not all(isinstance(v, str) for v in check.values())
            or check["kind"] not in ("substitution", "derivative", "equivalence")
            or len(check["variable"]) != 1
            or any(
                len(check[k]) > config.MATH_MAX_EXPRESSION_CHARS
                for k in ("expression", "candidate")
            )
        ):
            raise ValueError("Invalid verification check")
    if op == "cancel":
        target = message["request_id"]
        if (
            not isinstance(target, str)
            or not target
            or len(target) > config.PYTHON_MAX_ID_CHARS
        ):
            raise ValueError("Invalid cancellation ID")
    return message


def encode_frame(message: dict) -> bytes:
    try:
        _validate(message)
        frame = (
            json.dumps(
                message, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, UnicodeError) as exc:
        raise ValueError("Invalid frame") from exc
    if len(frame) > config.PYTHON_MAX_FRAME_BYTES:
        raise ValueError("Frame too large")
    return frame


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _constant(_value):
    raise ValueError("Non-finite JSON value")


def decode_frame(frame: bytes) -> dict:
    if not isinstance(frame, bytes) or len(frame) > config.PYTHON_MAX_FRAME_BYTES:
        raise ValueError("Invalid or oversized frame")
    if not frame.endswith(b"\n") or frame.count(b"\n") != 1:
        raise ValueError("Expected one newline-terminated frame")
    try:
        message = json.loads(
            frame.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_constant
        )
        return _validate(message)
    except (UnicodeError, TypeError, RecursionError) as exc:
        raise ValueError("Invalid frame") from exc
