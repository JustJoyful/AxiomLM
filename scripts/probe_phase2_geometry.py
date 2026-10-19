"""Task 1 only: frozen validation of experimental 300-token contextual chunks.

No Chroma connection, index writes, answer generation or production segmentation.
Run once; the report refuses overwrite and preserves all earlier evidence.
"""

from __future__ import annotations

import gc
import hashlib
import json
import re
import time
import weakref
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import itertools

from scripts import probe_phase2_late_chunking as old
from scripts.evaluate_phase2_baseline import (
    digest,
    load_labels,
    retrieval_metrics,
    tree_identity,
    verify_sources,
)
from src import config


def check_protocol(path, protocol):
    if digest(path) != protocol["label_sha256"]:
        raise ValueError("Validation labels changed after being frozen")


def geometry(pages, tokenizer, chunk_limit, block_limit):
    """Pack paragraphs across pages within sections; retain exact character ranges."""
    chunks = []
    chapter_titles = {
        "Building Abstractions with Procedures": "1",
        "Building Abstractions with Data": "2",
        "Modularity, Objects, and State": "3",
        "Metalinguistic Abstraction": "4",
        "Computing with Register Machines": "5",
    }
    section = "Unknown"
    text = ""
    sources = []

    def count(value):
        return len(tokenizer(value, add_special_tokens=False)["input_ids"])

    def flush():
        nonlocal text, sources
        if text:
            chunks.append(
                {
                    "text": text,
                    "section": section,
                    "chapter": section.split(".")[0],
                    "sources": sources,
                    "pages": sorted({s["page"] for s in sources}),
                }
            )
        text, sources = "", []

    for page, source in pages:
        # Keep whitespace in the source ranges, including paragraph delimiters.
        for match in re.finditer(r".+?(?:\n\s*\n|\Z)", source, re.DOTALL):
            start, stop = match.span()
            paragraph = match.group()
            heading = re.match(r"\s*#{1,6}\s+\**(\d+(?:\.\d+)+)\b", paragraph)
            title = re.match(r"\s*#{1,6}\s+\*\*(.*?)\*\*", paragraph)
            next_section = (
                heading[1]
                if heading
                else chapter_titles.get(title[1]) if title else None
            )
            if next_section and next_section != section:
                flush()
                section = next_section
            if text and count(text + paragraph) > chunk_limit:
                flush()
            while start < stop:
                remainder = source[start:stop]
                if count(text + remainder) <= chunk_limit:
                    text += remainder
                    sources.append({"page": page, "start": start, "stop": stop})
                    break
                encoded = tokenizer(
                    remainder, add_special_tokens=False, return_offsets_mapping=True
                )
                end = encoded["offset_mapping"][chunk_limit - 1][1]
                # Unicode byte-fallback tokens can share offsets; recheck exact slice.
                while end and count(remainder[:end]) > chunk_limit:
                    end -= 1
                if not end:
                    raise ValueError("Cannot split source within chunk token budget")
                text += remainder[:end]
                sources.append({"page": page, "start": start, "stop": start + end})
                start += end
                flush()
    flush()
    blocks = []
    group = []
    for i, chunk in enumerate(chunks):
        if group and chunk["section"] != chunks[group[-1][0]]["section"]:
            blocks.extend(old.make_blocks(group, tokenizer, block_limit))
            group = []
        group.append((i, chunk["text"]))
    if group:
        blocks.extend(old.make_blocks(group, tokenizer, block_limit))
    return chunks, blocks


def verify_serialization(chunks, blocks, tokenizer):
    for block in blocks:
        expected = (
            "Document: "
            + (tokenizer.eos_token + "\n\n").join(
                chunks[s["page"]]["text"] for s in block["spans"]
            )
            + tokenizer.eos_token
        )
        if (
            tokenizer.decode(block["input_ids"], clean_up_tokenization_spaces=False)
            != expected
        ):
            raise ValueError("Serialized block does not decode exactly")


def rank(vectors, queries, chunks, labels):
    from src.page_querying import extract_page_intent

    scores = queries @ vectors.T
    rows = []
    for i, case in enumerate(labels["cases"]):
        requested = extract_page_intent(case["question"])
        allowed = [
            j
            for j, c in enumerate(chunks)
            if not requested or set(c["pages"]) & set(requested)
        ]
        order = sorted(allowed, key=lambda j: (-float(scores[i, j]), j))[
            : config.TOP_K_RESULTS
        ]
        rows.append(
            {
                "case_id": case["case_id"],
                "retrieval_query": case["question"],
                "chunk_ids": order,
                "physical_pages": sorted(
                    {p for j in order for p in chunks[j]["pages"]}
                ),
                "scores": [float(scores[i, j]) for j in order],
            }
        )
    return rows


def finalize_report(report, output, protected, label_path, protocol_path):
    report["preserved"] = False
    try:
        after = {
            "index": tree_identity(config.VECTOR_STORE),
            "parsed": tree_identity(config.PARSED_MD),
            "baseline": digest(config.PHASE2_EVALUATION_PATH),
            "previous_probe": digest(config.PHASE2_PROBE_PATH),
            "config": digest(Path(config.__file__)),
            "labels": digest(label_path),
            "protocol": digest(protocol_path),
        }
        report["protected_after"] = after
        report["preserved"] = after == protected
        if not report["preserved"]:
            raise ValueError("Protected inputs changed")
    except (OSError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        report.update(
            status="failed",
            passed=False,
            preserved=False,
            integrity_error=error,
            blocked_reason=report.get("blocked_reason") or error,
        )
    finally:
        output.write_text(json.dumps(report, indent=2) + "\n")


def probe():
    import httpx
    import torch
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    from scripts.capture_runtime_manifest import capture_manifest
    from src.workloads import WorkloadCoordinator

    output = config.PHASE2_GEOMETRY_REPORT_PATH
    if output.exists():
        raise FileExistsError("Preserve previous geometry report before rerunning")
    label_path = config.PHASE2_VALIDATION_LABELS_PATH
    protocol_path = config.PHASE2_VALIDATION_PROTOCOL_PATH
    protocol = json.loads(protocol_path.read_text())
    check_protocol(label_path, protocol)
    if (
        protocol["chunk_content_token_ceiling"],
        protocol["block_token_ceiling"],
        protocol["k"],
        tuple(protocol["methods"]),
    ) != (
        config.PHASE2_PROBE_CHUNK_TOKENS,
        config.PHASE2_CONTEXT_TOKEN_LIMIT,
        config.TOP_K_RESULTS,
        old.METHODS,
    ):
        raise ValueError("Configured experiment differs from frozen protocol")
    labels = load_labels(label_path)
    verify_sources(labels, config.PARSED_MD)
    original = load_labels(old.LABELS_PATH)
    if set(labels["source_pages"]) & set(original["source_pages"]):
        raise ValueError("Validation source labels overlap original evaluation")
    snapshot = old.require_snapshot(
        snapshot_download(
            repo_id=config.EMBED_MODEL,
            revision=config.EMBED_REVISION,
            local_files_only=True,
        )
    )
    pages = [
        (int(p.stem.split("_")[1]) + 1, p.read_text())
        for p in sorted((config.CHECKPOINTS / labels["book_stem"]).glob("page_*.md"))
    ]
    protected = {
        "index": tree_identity(config.VECTOR_STORE),
        "parsed": tree_identity(config.PARSED_MD),
        "baseline": digest(config.PHASE2_EVALUATION_PATH),
        "previous_probe": digest(config.PHASE2_PROBE_PATH),
        "config": digest(Path(config.__file__)),
        "labels": digest(label_path),
        "protocol": digest(protocol_path),
    }
    report = {
        "status": "running",
        "passed": False,
        "model": config.EMBED_MODEL,
        "revision": config.EMBED_REVISION,
        "protocol": protocol,
        "protected_before": protected,
        "code_sha256": digest(Path(__file__)),
        "dependency_sha256": {
            str(p): digest(p)
            for p in [
                Path(old.__file__),
                config.ROOT / "scripts/evaluate_phase2_baseline.py",
            ]
        },
        "runtime": capture_manifest(),
        "limitations": "Agent-curated sufficient pages, single book, 12 answerable cases; unanswerable ranking is not abstention. No answers or human claim scores.",
    }
    coordinator = WorkloadCoordinator()
    admission = coordinator.try_acquire("ingest")
    if admission is None:
        raise RuntimeError("GPU workload admission unavailable")
    try:
        with httpx.Client(trust_env=False, timeout=3) as client:
            active = client.get(config.OLLAMA_URL + "/api/ps")
            active.raise_for_status()
            if active.json().get("models"):
                raise RuntimeError("Ollama model active; GPU experiment prohibited")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        warm = torch.ones((64, 64), device="cuda")
        product = warm @ warm
        del warm, product
        gc.collect()
        torch.cuda.empty_cache()
        idle = torch.cuda.memory_allocated()
        report["gpu_idle_runtime_allocated_bytes"] = idle
        if torch.cuda.mem_get_info()[0] < config.INGEST_MIN_FREE_VRAM_MB * 1024**2:
            raise RuntimeError("Insufficient free GPU memory")
        with old.owned_model(
            lambda: SentenceTransformer(
                snapshot, trust_remote_code=True, local_files_only=True, device="cpu"
            ),
            empty=torch.cuda.empty_cache,
        ) as get_model:
            model_ref = weakref.ref(get_model())
            refs = [weakref.ref(p) for p in get_model().parameters()]
            if (
                get_model().prompts.get("query") != "Query: "
                or get_model().prompts.get("document") != "Document: "
            ):
                raise ValueError("Unexpected effective prompts")
            report["prompts"] = get_model().prompts
            report["query_device"] = str(get_model().device)
            fixtures = json.loads(
                (config.ROOT / "tests/fixtures/phase2/planning_cases.json").read_text()
            )
            queries = (
                get_model()
                .encode(
                    [c["question"] for c in labels["cases"]],
                    prompt_name="query",
                    task="retrieval",
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
                .cpu()
            )
            control_queries = (
                get_model()
                .encode(
                    fixtures["control_queries"],
                    prompt_name="query",
                    task="retrieval",
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
                .cpu()
            )
            old.validate_vectors(queries)
            old.validate_vectors(control_queries)
            tokenizer = get_model()[0].tokenizer
            chunks, blocks = geometry(
                pages,
                tokenizer,
                config.PHASE2_PROBE_CHUNK_TOKENS,
                config.PHASE2_CONTEXT_TOKEN_LIMIT,
            )
            if geometry(
                pages,
                tokenizer,
                config.PHASE2_PROBE_CHUNK_TOKENS,
                config.PHASE2_CONTEXT_TOKEN_LIMIT,
            ) != (chunks, blocks):
                raise ValueError("Nondeterministic geometry")
            for page, source in pages:
                ranges = [r for c in chunks for r in c["sources"] if r["page"] == page]
                if "".join(
                    source[r["start"] : r["stop"]] for r in ranges
                ) != source or any(
                    a["stop"] != b["start"] for a, b in itertools.pairwise(ranges)
                ):
                    raise ValueError("Lost or duplicated source text")
            verify_serialization(chunks, blocks, tokenizer)
            chunk_record = json.dumps(chunks, sort_keys=True)
            report["geometry"] = {
                "chunks": len(chunks),
                "blocks": len(blocks),
                "pages": len(pages),
                "max_chunk_tokens": max(
                    len(tokenizer(c["text"], add_special_tokens=False)["input_ids"])
                    for c in chunks
                ),
                "max_block_tokens": max(len(b["input_ids"]) for b in blocks),
                "chapter_counts": {
                    ch: sum(c["chapter"] == ch for c in chunks)
                    for ch in sorted({c["chapter"] for c in chunks})
                },
                "chunks_sha256": hashlib.sha256(chunk_record.encode()).hexdigest(),
                "source_exactly_once": True,
                "deterministic": True,
                "exact_block_decode": True,
                "chunk_sources": [
                    {k: v for k, v in c.items() if k != "text"} for c in chunks
                ],
            }
            del tokenizer
            get_model().to("cuda")
            get_model().eval()
            report["dtype"] = str(next(get_model().parameters()).dtype)
            report["capacity"] = old.capacity_probe(get_model())
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            independent = (
                get_model()
                .encode(
                    [c["text"] for c in chunks],
                    prompt_name="document",
                    task="retrieval",
                    batch_size=config.EMBED_BATCH_SIZE,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
                .cpu()
            )
            old.validate_vectors(independent)
            report["independent_seconds"] = time.monotonic() - started
            started = time.monotonic()
            contextual = old.encode_blocks(get_model(), blocks)
            report["contextual_seconds"] = time.monotonic() - started
            controls = {name: [] for name in old.METHODS}
            for antecedent in fixtures["antecedents"]:
                cb = old.make_blocks(
                    [(1, antecedent), (2, fixtures["shared_span"])],
                    get_model()[0].tokenizer,
                    config.PHASE2_CONTEXT_TOKEN_LIMIT,
                )
                if len(cb) != 1:
                    raise ValueError("Control must share context block")
                states = old.encode_blocks(get_model(), cb)
                for name in old.METHODS:
                    controls[name].append(states[name][2])
            report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        gc.collect()
        torch.cuda.empty_cache()
        report["cleanup"] = {
            "model_released": model_ref() is None,
            "parameters_released": all(r() is None for r in refs),
            "allocated_after": torch.cuda.memory_allocated(),
            "initialized_baseline": idle,
        }
        clean = (
            report["cleanup"]["model_released"]
            and report["cleanup"]["parameters_released"]
            and report["cleanup"]["allocated_after"] <= idle
        )
        base_rows = rank(independent, queries, chunks, labels)
        base = retrieval_metrics(labels["cases"], base_rows)
        report["matched_independent"] = {"rows": base_rows, "metrics": base}
        report["methods"] = {}
        summaries = {}
        for name in old.METHODS:
            vectors = torch.stack([contextual[name][i] for i in range(len(chunks))])
            old.validate_vectors(vectors)
            c = torch.stack(controls[name])
            old.validate_vectors(c)
            scores = control_queries @ c.T
            rows = rank(vectors, queries, chunks, labels)
            metrics = retrieval_metrics(labels["cases"], rows)
            summaries[name] = {
                "controlled_top1": int(
                    (scores.argmax(1) == torch.arange(len(c))).sum()
                ),
                "controlled_cases": len(c),
                "precision": metrics["macro_page_precision"],
                "recall": metrics["macro_page_recall"],
            }
            report["methods"][name] = dict(
                summaries[name],
                metrics=metrics,
                rows=rows,
                controlled_scores=scores.tolist(),
                context_changed_cosine=float(c[0] @ c[1]),
            )
        fit = any(
            r["input_tokens"] == config.PHASE2_CONTEXT_TOKEN_LIMIT and r["passed"]
            for r in report["capacity"]
        )
        report["compatible_methods"] = old.compatible_methods(
            summaries,
            base["macro_page_recall"],
            base["macro_page_precision"],
            fit,
            clean,
        )
        report["passed"] = bool(report["compatible_methods"])
        report["blocked_reason"] = (
            None
            if report["passed"]
            else "No recipe passed the unchanged validation precision AND recall, control, capacity and cleanup gate."
        )
        report["status"] = "complete"
    except BaseException as exc:
        report.update(
            status="failed", passed=False, blocked_reason=f"{type(exc).__name__}: {exc}"
        )
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        coordinator.release(admission)
        finalize_report(report, output, protected, label_path, protocol_path)
    return report


if __name__ == "__main__":
    result = probe()
    print(
        json.dumps(
            {
                k: result[k]
                for k in ["status", "passed", "compatible_methods", "blocked_reason"]
            },
            indent=2,
        )
    )
