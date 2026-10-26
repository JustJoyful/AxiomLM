import pytest

from src.sandbox.protocol import decode_frame, encode_frame


@pytest.mark.parametrize(
    "message",
    [
        {"id": "r", "op": "execute", "code": "print(2)"},
        {"id": "cancel-1", "op": "cancel", "request_id": "r"},
        {"id": "shutdown-1", "op": "shutdown"},
    ],
)
def test_roundtrip_requests(message):
    frame = encode_frame(message)
    assert frame.endswith(b"\n") and frame.count(b"\n") == 1
    assert decode_frame(frame) == message


@pytest.mark.parametrize(
    "frame",
    [
        b'{"id":"r","id":"other","op":"shutdown"}\n',
        b'{"op":"shutdown"}\n',
        b'{"id":"","op":"shutdown"}\n',
        b'{"id":true,"op":"shutdown"}\n',
        b'{"id":"r","op":"shell"}\n',
        b'{"id":"r","op":"shutdown","extra":1}\n',
        b'{"id":"r","op":"execute","code":42}\n',
        b'{"id":"r","op":"shutdown"}\n{}\n',
        b"[]\n",
        b"\xff\n",
        b'{"id":"r","op":"execute","code":NaN}\n',
        b" " * 65537,
    ],
)
def test_protocol_rejects_unbounded_or_malformed_frames(frame):
    with pytest.raises(ValueError):
        decode_frame(frame)


def test_code_is_limited_in_utf8_bytes_before_spawn():
    with pytest.raises(ValueError):
        encode_frame({"id": "r", "op": "execute", "code": "x" * 8193})
    with pytest.raises(ValueError):
        encode_frame({"id": "r", "op": "execute", "code": "λ" * 4097})
    assert (
        decode_frame(encode_frame({"id": "r", "op": "execute", "code": "x" * 8192}))[
            "code"
        ]
        == "x" * 8192
    )


def test_worker_control_is_distinct_from_client_requests():
    with pytest.raises(ValueError):
        encode_frame({"id": "r", "op": "resume"})
    with pytest.raises(ValueError):
        decode_frame(b'{"id":"r","event":"child_stopped","namespace_pid":2}\n')


def test_verify_frames_carry_only_bounded_check_fields():
    from src.sandbox.protocol import encode_frame

    good = {
        "kind": "derivative",
        "expression": "x**3",
        "candidate": "3*x**2",
        "variable": "x",
    }
    assert encode_frame({"id": "v", "op": "verify", "check": good})
    for bad in (
        {**good, "code": "import os"},
        {**good, "kind": "execute"},
        {**good, "variable": "xy"},
        {**good, "expression": "x" * 513},
        {**good, "candidate": 3},
        "x**3",
    ):
        with pytest.raises(ValueError):
            encode_frame({"id": "v", "op": "verify", "check": bad})
