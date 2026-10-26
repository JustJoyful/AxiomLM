"""Prove the installed model's native chat contract before agent work depends on it.

Gates: exact chat token accounting against served prompt_eval_count (calibration
and held-out shapes), a real total output bound that includes thinking, a
schema-validated tool round-trip, context fit on the device, and stream
cancellation followed by confirmed unload. Every request uses keep_alive 0, so
each count is a cold, uncached evaluation. Existing reports are never overwritten.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import jsonschema

from src import config
from src.chat_tokens import load_chat_counter, render_chat
from src.llm import OllamaClient

SEARCH = {
    "type": "function",
    "function": {
        "name": "search_local_index",
        "description": "Search the selected book.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search text",
                    "maxLength": 2048,
                },
                "page_start": {"type": "integer", "minimum": 1},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}
PYTHON = {
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": "Run Python in the sandbox.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
}
NESTED = {
    "type": "function",
    "function": {
        "name": "verify_citation",
        "description": "Check a claim — against one source (Ünïcode).",
        "parameters": {
            "type": "object",
            "properties": {
                "claim": {"type": "string", "description": "Claim text"},
                "source_id": {"type": ["string", "null"]},
                "mode": {"type": "string", "enum": ["strict", "lenient"]},
                "pages": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["claim", "source_id"],
        },
    },
}
ADD = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two integers and return their sum.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First addend"},
                "b": {"type": "integer", "description": "Second addend"},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    },
}
CALL = {
    "function": {
        "name": "search_local_index",
        "arguments": {"query": "eigenvalues", "page_start": 3},
    }
}
EVIDENCE = (
    "Chapter 2, p.41. A procedure is a pattern for the local evolution of a "
    "computational process. "
) * 40

# Calibration cases fitted the renderer; held-out cases were written afterwards.
CASES = {
    "calibration": {
        "user_off": ([{"role": "user", "content": "Hello"}], None, False),
        "user_on": ([{"role": "user", "content": "Hello"}], None, True),
        "system_user": (
            [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "What is 2+2?"},
            ],
            None,
            False,
        ),
        "unicode": (
            [{"role": "user", "content": "数学 — ∫ x² dx ≈ ? 🚀\nline two"}],
            None,
            False,
        ),
        "tools_user": (
            [{"role": "user", "content": "Find eigenvalues."}],
            [SEARCH, PYTHON],
            False,
        ),
        "tools_system": (
            [
                {"role": "system", "content": "Use tools."},
                {"role": "user", "content": "Find eigenvalues."},
            ],
            [SEARCH],
            True,
        ),
        "tool_roundtrip": (
            [
                {"role": "user", "content": "Find eigenvalues."},
                {
                    "role": "assistant",
                    "content": "",
                    "thinking": "I should search.",
                    "tool_calls": [CALL],
                },
                {"role": "tool", "content": '{"status": "ok", "text": "p.3"}'},
            ],
            [SEARCH],
            True,
        ),
        "two_tools": (
            [
                {"role": "user", "content": "Go"},
                {
                    "role": "assistant",
                    "content": "Checking both.",
                    "tool_calls": [
                        CALL,
                        {
                            "function": {
                                "name": "execute_python",
                                "arguments": {"code": "print(1)\nprint(2)"},
                            }
                        },
                    ],
                },
                {"role": "tool", "content": "a"},
                {"role": "tool", "content": "b"},
            ],
            [SEARCH, PYTHON],
            False,
        ),
        "history": (
            [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1", "thinking": "old reasoning"},
                {"role": "user", "content": "Q2"},
            ],
            None,
            True,
        ),
    },
    "held_out": {
        "nested_schema_three_tools": (
            [
                {"role": "system", "content": "Policy line one.\n\nPolicy two."},
                {"role": "user", "content": "Is the claim on p.12 supported?"},
            ],
            [SEARCH, NESTED, PYTHON],
            False,
        ),
        "tool_history_then_user_think": (
            [
                {"role": "user", "content": "Integrate x^2."},
                {
                    "role": "assistant",
                    "content": "I will compute it.",
                    "thinking": "use sympy",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "execute_python",
                                "arguments": {"code": "result = sympy.integrate(x**2)"},
                            }
                        }
                    ],
                },
                {"role": "tool", "content": "x**3/3"},
                {"role": "assistant", "content": "The integral is x³/3 + C."},
                {"role": "user", "content": "And of x^3?"},
            ],
            [PYTHON],
            True,
        ),
        "long_evidence": (
            [
                {"role": "system", "content": "Answer only from evidence."},
                {
                    "role": "user",
                    "content": "Evidence:\n"
                    + EVIDENCE
                    + "\nQuestion: what is a procedure?",
                },
            ],
            None,
            False,
        ),
        "tool_result_after_thinking_off": (
            [
                {"role": "user", "content": "Search for 'fixed points' (σ, λ)."},
                {"role": "assistant", "content": "", "tool_calls": [CALL]},
                {"role": "tool", "content": "[]"},
            ],
            [SEARCH, NESTED],
            False,
        ),
    },
}

REQUIRED = (
    "identity",
    "chat_counts_match",
    "reasoning_cap_enforced",
    "tool_roundtrip",
    "fit",
    "cancel_unload",
)


def all_passed(checks: dict) -> bool:
    return all(checks.get(name, {}).get("passed") is True for name in REQUIRED)


def _identities() -> dict:
    with httpx.Client(trust_env=False, timeout=30) as client:
        version = client.get(config.OLLAMA_URL + "/api/version").json()["version"]
        tags = client.get(config.OLLAMA_URL + "/api/tags").json()["models"]
        show = client.post(
            config.OLLAMA_URL + "/api/show", json={"model": config.OLLAMA_MODEL}
        ).json()
    entry = next(m for m in tags if m["name"] == config.OLLAMA_MODEL)
    directives = [
        line
        for line in show.get("modelfile", "").splitlines()
        if line.startswith(("TEMPLATE", "RENDERER", "PARSER", "PARAMETER"))
    ]
    return {
        "ollama_version": version,
        "model": config.OLLAMA_MODEL,
        "digest": entry["digest"],
        "quantization": entry["details"].get("quantization_level"),
        "capabilities": show.get("capabilities"),
        "thinking": show.get("thinking"),
        "modelfile_directives": directives,
    }


def _vram() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
        return int(out.split()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


async def _sampled(coro):
    """Run a request while sampling device memory; returns (result, peak MiB)."""
    peak = [_vram()]
    done = asyncio.Event()

    async def sample():
        while not done.is_set():
            value = await asyncio.to_thread(_vram)
            if value is not None:
                peak.append(value)
            await asyncio.sleep(0.25)

    sampler = asyncio.create_task(sample())
    try:
        return await coro, max((p for p in peak if p is not None), default=None)
    finally:
        done.set()
        await sampler


async def _probe(report: dict) -> None:
    client = OllamaClient()
    counter = load_chat_counter(config.OLLAMA_MODEL)
    rows = report["rows"]

    async def turn(name, messages, tools, think, num_predict):
        stamp = time.monotonic()
        result = await client.achat(
            messages,
            model=config.OLLAMA_MODEL,
            tools=tools,
            think=think,
            num_predict=num_predict,
            timeout=config.AGENT_REQUEST_SECONDS,
        )
        row = {
            "name": name,
            "think": think,
            "num_predict": num_predict,
            "counted": counter.count(messages, tools, think=think),
            "served": result["prompt_eval_count"],
            "cached": result["prompt_eval_cached_count"],
            "eval_count": result["eval_count"],
            "done_reason": result["done_reason"],
            "load_seconds": (result["load_duration"] or 0) / 1e9,
            "seconds": time.monotonic() - stamp,
            "thinking_chars": len(result["message"]["thinking"]),
            "content_chars": len(result["message"]["content"]),
        }
        rows.append(row)
        return result, row

    async def gate_chat_counts_match():
        counts = {}
        for group, cases in CASES.items():
            for name, (messages, tools, think) in cases.items():
                _, row = await turn(f"{group}:{name}", messages, tools, think, 1)
                counts[row["name"]] = {
                    "served": row["served"],
                    "counted": row["counted"],
                    "cached": row["cached"],
                }
        report["checks"]["chat_counts_match"] = {
            "passed": all(
                v["served"] == v["counted"] and v["cached"] == 0
                for v in counts.values()
            ),
            "held_out_cases": sum(1 for k in counts if k.startswith("held_out:")),
            "counts": counts,
        }

    async def gate_reasoning_cap_enforced():
        bounds = {}
        question = (
            "Prove that the square root of 2 is irrational, with every step justified."
        )
        for cap in (
            config.AGENT_ROUTER_OUTPUT_TOKENS,
            config.AGENT_FINAL_OUTPUT_TOKENS,
        ):
            _, row = await turn(
                f"reasoning_cap:{cap}",
                [{"role": "user", "content": question}],
                None,
                True,
                cap,
            )
            bounds[cap] = {
                "eval_count": row["eval_count"],
                "done_reason": row["done_reason"],
                "thinking_chars": row["thinking_chars"],
                "content_chars": row["content_chars"],
            }
        report["checks"]["reasoning_cap_enforced"] = {
            "passed": all(
                b["eval_count"] is not None
                and b["eval_count"] <= cap
                and b["thinking_chars"] > 0
                and (b["done_reason"] == "length" or b["content_chars"] > 0)
                for cap, b in bounds.items()
            ),
            "bounds": bounds,
            "interpretation": "eval_count includes thinking tokens; num_predict caps reasoning+answer together",
        }

    async def gate_tool_roundtrip():
        messages = [
            {
                "role": "user",
                "content": "Use the add tool to add 17 and 25, then state the sum.",
            }
        ]
        first, _ = await turn("tool_roundtrip:request", messages, [ADD], False, 256)
        calls = first["message"]["tool_calls"]
        validated = None
        if len(calls) == 1 and calls[0]["function"]["name"] == "add":
            arguments = calls[0]["function"]["arguments"]
            jsonschema.Draft202012Validator(ADD["function"]["parameters"]).validate(
                arguments
            )
            validated = arguments
        roundtrip = {"tool_calls": calls, "validated_arguments": validated}
        if validated is not None:
            messages += [
                first["message"],
                {
                    "role": "tool",
                    "content": json.dumps({"sum": validated["a"] + validated["b"]}),
                    "tool_name": "add",
                },
            ]
            final, row = await turn("tool_roundtrip:final", messages, [ADD], False, 256)
            roundtrip.update(
                final_content=final["message"]["content"],
                final_complete=final["complete"],
                final_counted=row["counted"],
                final_served=row["served"],
            )
        report["checks"]["tool_roundtrip"] = {
            "passed": validated == {"a": 17, "b": 25}
            and roundtrip.get("final_complete") is True
            and "42" in roundtrip.get("final_content", "")
            and roundtrip.get("final_counted") == roundtrip.get("final_served"),
            **roundtrip,
        }

    async def gate_fit():
        budget = config.OLLAMA_CONTEXT_TOKENS - config.AGENT_FINAL_OUTPUT_TOKENS
        filler = "Evidence passage about recursion and iteration. " * 10
        body = ""
        while True:
            candidate = body + filler
            trial = [{"role": "user", "content": candidate + "Summarize in one line."}]
            if counter.count(trial, [SEARCH, PYTHON], think=False) > budget - 16:
                break
            body = candidate
        fit_messages = [{"role": "user", "content": body + "Summarize in one line."}]
        (_result, row), peak = await _sampled(
            turn(
                "fit:near_limit",
                fit_messages,
                [SEARCH, PYTHON],
                False,
                config.AGENT_FINAL_OUTPUT_TOKENS,
            )
        )
        device = (
            await asyncio.to_thread(
                subprocess.run,
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total,name",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        ).stdout.strip()
        report["checks"]["fit"] = {
            "passed": row["served"] == row["counted"]
            and row["served"] + config.AGENT_FINAL_OUTPUT_TOKENS
            <= config.OLLAMA_CONTEXT_TOKENS
            and row["done_reason"] in ("stop", "length"),
            "prompt_tokens": row["served"],
            "num_ctx": config.OLLAMA_CONTEXT_TOKENS,
            "output_reserve": config.AGENT_FINAL_OUTPUT_TOKENS,
            "eval_count": row["eval_count"],
            "peak_vram_mib": peak,
            "device": device,
        }

    async def gate_cancel_unload():
        started = time.monotonic()
        first_chunk = asyncio.Event()

        async def long_stream():
            async for _ in client.astream_chat(
                [
                    {
                        "role": "user",
                        "content": "Count slowly from 1 to 2000, one per line.",
                    }
                ],
                model=config.OLLAMA_MODEL,
                tools=None,
                think=False,
                num_predict=config.AGENT_FINAL_OUTPUT_TOKENS,
                timeout=config.AGENT_REQUEST_SECONDS,
            ):
                first_chunk.set()

        task = asyncio.create_task(long_stream())
        await asyncio.wait_for(first_chunk.wait(), config.AGENT_REQUEST_SECONDS)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        cancelled_at = time.monotonic() - started
        await asyncio.to_thread(client.unload_and_wait)
        with httpx.Client(trust_env=False, timeout=10) as http:
            loaded = http.get(config.OLLAMA_URL + "/api/ps").json().get("models", [])
        report["checks"]["cancel_unload"] = {
            "passed": task.cancelled() and loaded == [],
            "cancelled_after_seconds": cancelled_at,
            "unload_confirmed_seconds": time.monotonic() - started - cancelled_at,
            "loaded_after": loaded,
            "vram_after_mib": _vram(),
        }

    # Each gate is recorded independently; one failure never hides the others.
    for name, gate in (
        ("chat_counts_match", gate_chat_counts_match),
        ("reasoning_cap_enforced", gate_reasoning_cap_enforced),
        ("tool_roundtrip", gate_tool_roundtrip),
        ("fit", gate_fit),
        ("cancel_unload", gate_cancel_unload),
    ):
        try:
            await gate()
        except (
            TimeoutError,
            OSError,
            ValueError,
            RuntimeError,
            httpx.HTTPError,
            jsonschema.ValidationError,
        ) as exc:
            report["checks"][name] = {
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            # A failed or timed-out gate must still leave no model loaded.
            await asyncio.to_thread(client.unload_and_wait)


def probe(*, output: Path) -> dict:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve exclusively before any model call; prior evidence is never overwritten.
    with output.open("x") as stream:
        stream.write('{"status":"running"}\n')
        stream.flush()
        report = {
            "started_utc": datetime.now(UTC).isoformat(),
            "passed": False,
            "blocked_reason": None,
            "checks": {},
            "rows": [],
            "rendered_sample": render_chat(
                *CASES["held_out"]["nested_schema_three_tools"][:2], think=False
            ),
        }
        try:
            report["identities"] = ids = _identities()
            report["device_state"] = subprocess.run(
                [
                    "nvidia-smi",
                    (
                        "--query-gpu=name,pstate,clocks.sm,clocks.mem,power.draw,"
                        "enforced.power.limit,clocks_event_reasons.active"
                    ),
                    "--format=csv",
                ],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            report["checks"]["identity"] = {
                "passed": ids["digest"] == config.CHAT_TOKENIZER_MODEL_DIGEST
                and ids["ollama_version"] == config.CHAT_TOKENIZER_OLLAMA_VERSION
                and "RENDERER qwen3.5" in ids["modelfile_directives"],
            }
            if not report["checks"]["identity"]["passed"]:
                raise RuntimeError("Unvalidated model/runtime identity")
            asyncio.run(_probe(report))
        except jsonschema.ValidationError as exc:
            report["blocked_reason"] = f"Tool arguments failed schema: {exc.message}"
        except (
            OSError,
            RuntimeError,
            ValueError,
            TimeoutError,
            httpx.HTTPError,
        ) as exc:
            report["blocked_reason"] = f"{type(exc).__name__}: {exc}"
        report["passed"] = all_passed(report["checks"])
        if not report["passed"] and report["blocked_reason"] is None:
            report["blocked_reason"] = "Required runtime gate failed"
        report["finished_utc"] = datetime.now(UTC).isoformat()
        stream.seek(0)
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.truncate()
        stream.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = probe(output=args.output)
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "blocked_reason": result["blocked_reason"],
                "checks": {k: v.get("passed") for k, v in result["checks"].items()},
            }
        )
    )
    raise SystemExit(0 if result["passed"] else 1)
