"""Local planning experiment: contextual page spans, not a production indexer.

All pages form a matched experimental corpus. Blocks preserve whole page texts,
with one Document prompt and explicit EOS markers between pages. No index writes.
"""

from __future__ import annotations

import gc
import json
import os
import time
import weakref
from contextlib import contextmanager
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from scripts.evaluate_phase2_baseline import (
    LABELS_PATH,
    digest,
    load_labels,
    retrieval_metrics,
    tree_identity,
    verify_sources,
)
from src import config

METHODS = ("mean", "last", "eos")


def ordered_queries(baseline, labels):
    rows = baseline.get("rows", [])
    expected = [row["case_id"] for row in labels["cases"]]
    by_id = {row["case_id"]: row for row in rows}
    if (
        baseline.get("status") != "complete"
        or len(rows) != len(expected)
        or set(by_id) != set(expected)
        or any(not by_id[cid].get("retrieval_query") for cid in expected)
    ):
        raise ValueError(
            "Baseline must contain exactly one completed result per frozen case"
        )
    return [by_id[cid]["retrieval_query"] for cid in expected]


def finalize_report(report, labels, before_store, page_hashes):
    report.setdefault("active_store_unchanged", False)
    report.setdefault("source_pages_unchanged", False)
    try:
        report["active_store_unchanged"] = (
            tree_identity(config.VECTOR_STORE) == before_store
        )
        report["source_pages_unchanged"] = all(
            digest(config.CHECKPOINTS / labels["book_stem"] / name) == sha
            for name, sha in page_hashes.items()
        )
        verify_sources(labels, config.PARSED_MD)
        if not report["active_store_unchanged"] or not report["source_pages_unchanged"]:
            raise ValueError("Source/index drift during probe")
    except (OSError, ValueError, KeyError) as exc:
        report.update(
            status="failed",
            passed=False,
            integrity_error=f"{type(exc).__name__}: {exc}",
        )
        if not report.get("blocked_reason"):
            report["blocked_reason"] = report["integrity_error"]
    path = config.PHASE2_PROBE_PATH
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def require_snapshot(path):
    if not Path(path).is_dir():
        raise FileNotFoundError(f"Pinned local snapshot missing: {path}")
    return str(path)


@contextmanager
def owned_model(factory, collect=gc.collect, empty=None):
    owners = []
    try:
        owners.append(factory())
        yield lambda: owners[0]
    finally:
        owners.clear()
        collect()
        if empty is not None:
            empty()


def make_blocks(pages, tokenizer, budget):
    blocks = []
    current = {"input_ids": [], "spans": []}
    for page, text in pages:
        # EOS separates spans while bidirectional attention still sees the block.
        prefix = "\n\n" if current["spans"] else "Document: "
        encoded = tokenizer(
            prefix + text,
            add_special_tokens=True,
            return_offsets_mapping=True,
            truncation=False,
        )
        ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        if len(current["input_ids"]) + len(ids) > budget and current["spans"]:
            blocks.append(current)
            current = {"input_ids": [], "spans": []}
            prefix = "Document: "
            encoded = tokenizer(
                prefix + text,
                add_special_tokens=True,
                return_offsets_mapping=True,
                truncation=False,
            )
            ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        if len(ids) > budget:
            raise ValueError(
                f"Page {page} exceeds block budget; production paragraph splitting required"
            )
        if ids[-1] != tokenizer.eos_token_id:
            raise ValueError(
                "Pinned tokenizer no longer appends the expected EOS token"
            )
        content = [
            i
            for i, (a, b) in enumerate(offsets)
            if b > len(prefix) and a < len(prefix + text) and b > a
        ]
        if not content:
            raise ValueError(f"Page {page} has no content tokens")
        base = len(current["input_ids"])
        current["spans"].append(
            {
                "page": page,
                "start": base + content[0],
                "stop": base + content[-1] + 1,
                "eos": base + len(ids) - 1,
            }
        )
        current["input_ids"].extend(ids)
    if current["spans"]:
        blocks.append(current)
    return blocks


def pool_span(hidden, span):
    return {
        "mean": F.normalize(hidden[span["start"] : span["stop"]].mean(0), dim=0)
        .detach()
        .cpu(),
        "last": F.normalize(hidden[span["stop"] - 1], dim=0).detach().cpu(),
        "eos": F.normalize(hidden[span["eos"]], dim=0).detach().cpu(),
    }


def validate_vectors(vectors):
    if (
        vectors.ndim != 2
        or vectors.shape[1] != config.EMBED_DIMENSION
        or not torch.isfinite(vectors).all()
    ):
        raise ValueError(
            "Invalid contextual vectors: require finite 768-dimensional rows"
        )
    if not torch.allclose(vectors.norm(dim=1), torch.ones(len(vectors)), atol=1e-4):
        raise ValueError("Contextual vectors must be normalized and nonzero")


def compatible_methods(results, baseline_recall, baseline_precision, fit, cleaned):
    return [
        name
        for name, row in results.items()
        if fit
        and cleaned
        and row["controlled_top1"] == row["controlled_cases"]
        and row["recall"] >= baseline_recall
        and row["precision"] >= baseline_precision
    ]


def token_states(model, ids):
    backbone = model[0].model
    backbone.set_adapter("retrieval")
    inputs = torch.tensor([ids], device="cuda")
    with torch.inference_mode():
        return backbone(
            input_ids=inputs, attention_mask=torch.ones_like(inputs)
        ).last_hidden_state[0]


def capacity_probe(model):
    records = []
    eos = model[0].tokenizer.eos_token_id
    seed = model[0].tokenizer(
        "Document: Git records source program versions.", add_special_tokens=False
    )["input_ids"]
    for size in config.PHASE2_PROBE_TOKEN_SIZES:
        ids = (seed * (size // len(seed) + 1))[: size - 1] + [eos]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        row = {"input_tokens": len(ids), "dtype": str(next(model.parameters()).dtype)}
        try:
            hidden = token_states(model, ids)
            torch.cuda.synchronize()
            row.update(
                passed=bool(torch.isfinite(hidden).all()), shape=list(hidden.shape)
            )
            del hidden
        except torch.cuda.OutOfMemoryError as exc:
            row.update(passed=False, error=str(exc))
        row.update(
            seconds=time.monotonic() - started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        records.append(row)
        torch.cuda.empty_cache()
        print(
            f'capacity {size}: {row["passed"]}, peak {row["peak_allocated_bytes"]//1024**2} MiB',
            flush=True,
        )
    return records


def encode_blocks(model, blocks):
    output = {name: {} for name in METHODS}
    for i, block in enumerate(blocks):
        hidden = token_states(model, block["input_ids"])
        for span in block["spans"]:
            for name, vector in pool_span(hidden, span).items():
                output[name][span["page"]] = vector
        del hidden
        if (i + 1) % 5 == 0 or i == len(blocks) - 1:
            print(f"context blocks {i+1}/{len(blocks)}", flush=True)
    return output


def ranked_rows(vectors, queries, pages, labels, traced_queries):
    from src.page_querying import extract_page_intent

    rows = []
    scores = queries @ vectors.T
    for i, case in enumerate(labels["cases"]):
        requested = extract_page_intent(case["question"])
        allowed = [j for j, p in enumerate(pages) if not requested or p in requested]
        order = sorted(allowed, key=lambda j: (-float(scores[i, j]), pages[j]))[
            : config.TOP_K_RESULTS
        ]
        rows.append(
            {
                "case_id": case["case_id"],
                "retrieval_query": traced_queries[i],
                "physical_pages": [pages[j] for j in order],
                "scores": [float(scores[i, j]) for j in order],
            }
        )
    return rows


def gpu_experiment(model, pages, controls, control_queries, report):
    model.to("cuda")
    model.eval()
    report["dtype"] = str(next(model.parameters()).dtype)
    report["attention_implementation"] = model[0].model.config._attn_implementation
    report["capacity"] = capacity_probe(model)
    passing_sizes = [r["input_tokens"] for r in report["capacity"] if r["passed"]]
    if not passing_sizes:
        raise RuntimeError("No tested context size fits the configured GPU")
    budget = max(passing_sizes)
    blocks = make_blocks(pages, model[0].tokenizer, budget)
    assert blocks == make_blocks(pages, model[0].tokenizer, budget)
    report["blocks"] = {
        "budget": budget,
        "count": len(blocks),
        "max_tokens": max(len(b["input_ids"]) for b in blocks),
        "source_pages_exactly_once": [s["page"] for b in blocks for s in b["spans"]]
        == [p for p, _ in pages],
        "scheme": "Whole pages, no overlap; EOS after each page and Document prefix once per block; paragraph/chapter segmentation is a future production test.",
    }
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    independent = model.encode(
        [text for _, text in pages],
        prompt_name="document",
        task="retrieval",
        batch_size=config.EMBED_BATCH_SIZE,
        convert_to_tensor=True,
        show_progress_bar=False,
    ).cpu()
    validate_vectors(independent)
    report["independent_seconds"] = time.monotonic() - started
    started = time.monotonic()
    contextual = encode_blocks(model, blocks)
    report["contextual_seconds"] = time.monotonic() - started
    report["corpus_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["corpus_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    ctrl = {name: [] for name in METHODS}
    for context in controls:
        control_blocks = make_blocks(
            [
                (1, context),
                (
                    2,
                    json.loads(
                        (
                            config.ROOT / "tests/fixtures/phase2/planning_cases.json"
                        ).read_text()
                    )["shared_span"],
                ),
            ],
            model[0].tokenizer,
            budget,
        )
        if len(control_blocks) != 1:
            raise ValueError("Controlled antecedent and target must share one block")
        states = encode_blocks(model, control_blocks)
        for name in METHODS:
            ctrl[name].append(states[name][2])
    decisions = {}
    for name in METHODS:
        vectors = torch.stack([contextual[name][p] for p, _ in pages])
        validate_vectors(vectors)
        c = torch.stack(ctrl[name])
        validate_vectors(c)
        s = control_queries @ c.T
        decisions[name] = {
            "controlled_top1": int((s.argmax(1) == torch.arange(len(controls))).sum()),
            "controlled_cases": len(controls),
            "controlled_scores": s.tolist(),
            "same_span_changed_context_cosine": float(c[0] @ c[1]),
            "vectors": vectors,
        }
    return independent, decisions


def probe():
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    from scripts.capture_runtime_manifest import capture_manifest
    from scripts.probe_phase2_planning import planning_checks
    from src.workloads import WorkloadCoordinator

    output = config.PHASE2_PROBE_PATH
    if output.exists():
        raise FileExistsError("Preserve the previous probe report before rerunning")
    labels = load_labels(LABELS_PATH)
    verify_sources(labels, config.PARSED_MD)
    snapshot = require_snapshot(
        snapshot_download(
            repo_id=config.EMBED_MODEL,
            revision=config.EMBED_REVISION,
            local_files_only=True,
        )
    )
    manifest = json.loads(
        (config.PARSED_MD / labels["book_stem"] / "page_manifest.json").read_text()
    )
    pages = []
    page_hashes = {}
    for row in manifest["pages"]:
        if (
            row["page_idx"] + 1 != row["physical_page"]
            or row["checkpoint_file"] != f"page_{row['page_idx']:04d}.md"
        ):
            raise ValueError("Inconsistent physical-page manifest")
        path = config.CHECKPOINTS / labels["book_stem"] / row["checkpoint_file"]
        text = path.read_text()
        page_hashes[row["checkpoint_file"]] = digest(path)
        if text.strip():
            pages.append((row["physical_page"], text))
    baseline = json.loads(config.PHASE2_EVALUATION_PATH.read_text())
    if baseline["dataset_sha256"] != digest(LABELS_PATH):
        raise ValueError("Frozen baseline labels changed")
    traced = ordered_queries(baseline, labels)
    fixture_path = config.ROOT / "tests/fixtures/phase2/planning_cases.json"
    fixtures = json.loads(fixture_path.read_text())
    controls = fixtures["antecedents"]
    raw_queries = fixtures["control_queries"]
    report = {
        "status": "running",
        "passed": False,
        "model": config.EMBED_MODEL,
        "revision": config.EMBED_REVISION,
        "dataset_sha256": digest(LABELS_PATH),
        "probe_code_sha256": digest(Path(__file__)),
        "experiment_code_sha256": {
            path: digest(config.ROOT / path)
            for path in (
                "scripts/probe_phase2_planning.py",
                "src/config.py",
                str(config.PHASE2_EVALUATION_PATH.relative_to(config.ROOT)),
            )
        },
        "planning_fixture_sha256": digest(fixture_path),
        "runtime": capture_manifest(),
        "page_hashes": page_hashes,
        "gate_rule": "Predeclared: finite normalized 768D vectors; both controlled antecedent rankings correct; macro gold-page recall AND precision no worse than matched independent-page baseline; 8192-token float32 fit; cleanup. No pooling hyperparameter tuning on these labels.",
        "controlled_fixture": {
            "antecedents": controls,
            "shared_target": json.loads(
                (config.ROOT / "tests/fixtures/phase2/planning_cases.json").read_text()
            )["shared_span"],
            "queries": raw_queries,
        },
        "comparison_note": "All 883 page spans are shared across matched methods. This changes geometry vs legacy 1162-chunk baseline; only the matched comparison isolates pooling/context. Ranking-only at K, page-intent filters; no calibrated confidence cutoff. Unanswerable ranking alone is not a refusal test.",
    }
    before_store = tree_identity(config.VECTOR_STORE)
    report["active_store_sha256_before"] = before_store
    coordinator = WorkloadCoordinator()
    token = coordinator.try_acquire("ingest")
    if token is None:
        raise RuntimeError("GPU workload admission unavailable")
    cuda_before = None
    model_ref = None
    try:
        import httpx

        with httpx.Client(trust_env=False, timeout=3) as client:
            active = client.get(config.OLLAMA_URL + "/api/ps")
            active.raise_for_status()
            if active.json().get("models"):
                raise RuntimeError(
                    "Ollama model active; finish generation before GPU probe"
                )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no CPU-ingest fallback in this probe")
        report["gpu_cold_allocated_bytes"] = torch.cuda.memory_allocated()
        # CUDA's cuBLAS workspace survives empty_cache without owning model tensors.
        warmup = torch.ones((64, 64), device="cuda")
        product = warmup @ warmup
        del warmup, product
        gc.collect()
        torch.cuda.empty_cache()
        cuda_before = torch.cuda.memory_allocated()
        report["gpu_idle_runtime_allocated_bytes"] = cuda_before
        report["gpu_free_bytes_before"] = torch.cuda.mem_get_info()[0]
        if report["gpu_free_bytes_before"] < config.INGEST_MIN_FREE_VRAM_MB * 1024**2:
            raise RuntimeError("Insufficient free GPU memory for probe")
        with owned_model(
            lambda: SentenceTransformer(
                snapshot, trust_remote_code=True, local_files_only=True, device="cpu"
            ),
            empty=torch.cuda.empty_cache,
        ) as get_model:
            model_ref = weakref.ref(get_model())
            parameter_refs = [weakref.ref(p) for p in get_model().parameters()]
            if (
                get_model().prompts.get("query") != "Query: "
                or get_model().prompts.get("document") != "Document: "
            ):
                raise ValueError("Unexpected internal prompts")
            report["internal_prompts"] = get_model().prompts
            report["query_device"] = str(get_model().device)
            queries = (
                get_model()
                .encode(
                    traced,
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
                    raw_queries,
                    prompt_name="query",
                    task="retrieval",
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
                .cpu()
            )
            validate_vectors(queries)
            independent, decisions = gpu_experiment(
                get_model(), pages, controls, control_queries, report
            )
        gc.collect()
        torch.cuda.empty_cache()
        report["gpu_allocated_bytes_after"] = torch.cuda.memory_allocated()
        report["model_released"] = model_ref() is None
        report["parameters_released"] = all(ref() is None for ref in parameter_refs)
        cleaned = (
            report["model_released"]
            and report["parameters_released"]
            and report["gpu_allocated_bytes_after"] <= cuda_before
        )
        page_ids = [p for p, _ in pages]
        rows = ranked_rows(independent, queries, page_ids, labels, traced)
        report["matched_independent"] = {
            "rows": rows,
            "metrics": retrieval_metrics(labels["cases"], rows),
        }
        summaries = {}
        report["methods"] = {}
        for name, result in decisions.items():
            vectors = result.pop("vectors")
            rows = ranked_rows(vectors, queries, page_ids, labels, traced)
            metrics = retrieval_metrics(labels["cases"], rows)
            report["methods"][name] = dict(result, rows=rows, metrics=metrics)
            summaries[name] = dict(
                result,
                recall=metrics["macro_page_recall"],
                precision=metrics["macro_page_precision"],
            )
        base = report["matched_independent"]["metrics"]
        fit = any(
            r["input_tokens"] == config.PHASE2_CONTEXT_TOKEN_LIMIT and r["passed"]
            for r in report["capacity"]
        )
        report["compatible_methods"] = compatible_methods(
            summaries,
            base["macro_page_recall"],
            base["macro_page_precision"],
            fit,
            cleaned,
        )
        report["passed"] = bool(report["compatible_methods"])
        report["blocked_reason"] = (
            None
            if report["passed"]
            else "No tested pooling recipe passed every predeclared compatibility, retrieval, context-fit and cleanup criterion."
        )
        if not cleaned:
            raise RuntimeError(
                "GPU model ownership/memory did not clear; skip Ollama checks"
            )
        coordinator.release(token)
        token = None
        # Heavy GPU embeddings are gone before any Ollama planning generation.
        report["planning"] = planning_checks(labels, pages, baseline, fixtures)
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
            report["final_gpu_allocated_bytes"] = torch.cuda.memory_allocated()
        if token is not None:
            coordinator.release(token)
        finalize_report(report, labels, before_store, page_hashes)
    return report


if __name__ == "__main__":
    result = probe()
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "compatible_methods": result.get("compatible_methods"),
                "report": str(config.PHASE2_PROBE_PATH),
            },
            indent=2,
        )
    )
