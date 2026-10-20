"""Deterministic contextual tokenization and the validated Jina mean recipe."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import chain

from src import config
from src.index_metadata import SourceSegment, normalize_metadata, stable_chunk_id


@dataclass(frozen=True)
class ContextBlock:
    text: str
    segments: tuple[SourceSegment, ...]
    token_spans: tuple[tuple[int, int], ...]
    chunk_spans: tuple[tuple[int, int], ...]
    token_ids: tuple[int, ...]
    chunk_eos_positions: tuple[int, ...]
    token_count: int
    chunk_texts: tuple[str, ...]
    chunk_segments: tuple[tuple[SourceSegment, ...], ...]
    chunk_ordinals: tuple[int, ...]


@dataclass(frozen=True)
class EmbeddedChunk:
    chunk_id: str
    text: str
    metadata: dict[str, str | int]
    embedding: list[float]


def split_context_blocks(
    segments: Sequence[SourceSegment],
    tokenizer,
    max_tokens: int = config.CONTEXT_BLOCK_MAX_TOKENS,
    chunk_tokens: int = config.RETRIEVAL_CHUNK_TOKENS,
) -> list[ContextBlock]:
    if (
        max_tokens < 1
        or max_tokens > config.CONTEXT_BLOCK_MAX_TOKENS
        or chunk_tokens < 1
    ):
        raise ValueError("Invalid contextual token budget")
    chunks = []
    current = []
    text = ""
    key = None

    def count(value):
        return len(tokenizer(value, add_special_tokens=False)["input_ids"])

    def flush():
        nonlocal text, current
        if text:
            chunks.append((text, tuple(current)))
        text, current = "", []

    for segment in segments:
        new_key = (segment.book_stem, segment.chapter, segment.section)
        if current and (new_key != key or count(text + segment.text) > chunk_tokens):
            flush()
        key = new_key
        start = 0
        while start < len(segment.text):
            remainder = segment.text[start:]
            if count(text + remainder) <= chunk_tokens:
                text += remainder
                current.append(
                    replace(
                        segment,
                        text=remainder,
                        source_start=segment.source_start + start,
                    )
                )
                break
            encoded = tokenizer(
                remainder, add_special_tokens=False, return_offsets_mapping=True
            )
            end = encoded["offset_mapping"][chunk_tokens - 1][1]
            while end and count(remainder[:end]) > chunk_tokens:
                end -= 1
            if not end:
                raise ValueError("Cannot split source within chunk budget")
            text += remainder[:end]
            current.append(
                replace(
                    segment,
                    text=remainder[:end],
                    source_start=segment.source_start + start,
                )
            )
            start += end
            flush()
    flush()
    blocks = []
    ids = []
    spans = []
    ends = []
    parts = []
    owners = []
    ordinals = []
    source_spans = []
    block_key = None

    def finish():
        nonlocal ids, spans, ends, parts, owners, ordinals, source_spans
        if ids:
            blocks.append(
                ContextBlock(
                    "".join(parts),
                    tuple(chain.from_iterable(owners)),
                    tuple(source_spans),
                    tuple(spans),
                    tuple(ids),
                    tuple(ends),
                    len(ids),
                    tuple(parts),
                    tuple(owners),
                    tuple(ordinals),
                )
            )
        ids, spans, ends, parts, owners, ordinals, source_spans = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

    for ordinal, (text, source_parts) in enumerate(chunks):
        first = source_parts[0]
        new_key = (first.book_stem, first.chapter, first.section)
        if ids and new_key != block_key:
            finish()
        prefix = "\n\n" if ids else "Document: "
        encoded = tokenizer(
            prefix + text,
            add_special_tokens=True,
            return_offsets_mapping=True,
            truncation=False,
        )
        if ids and len(ids) + len(encoded["input_ids"]) > max_tokens:
            finish()
            prefix = "Document: "
            encoded = tokenizer(
                prefix + text,
                add_special_tokens=True,
                return_offsets_mapping=True,
                truncation=False,
            )
        token_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        if len(token_ids) > max_tokens:
            raise ValueError("Chunk exceeds contextual block budget")
        if token_ids[-1] != tokenizer.eos_token_id:
            raise ValueError("Expected pinned EOS serialization")
        content = [
            i
            for i, (a, b) in enumerate(offsets)
            if b > len(prefix) and a < len(prefix + text) and b > a
        ]
        if not content:
            raise ValueError("Source has no content tokens")
        base = len(ids)
        spans.append((base + content[0], base + content[-1] + 1))
        ends.append(base + len(token_ids) - 1)
        offset = len(prefix)
        for source in source_parts:
            indices = [
                i
                for i, (a, b) in enumerate(offsets)
                if a < offset + len(source.text) and b > offset and b > a
            ]
            source_spans.append((base + indices[0], base + indices[-1] + 1))
            offset += len(source.text)
        ids.extend(token_ids)
        parts.append(text)
        owners.append(source_parts)
        ordinals.append(ordinal)
        block_key = new_key
    finish()
    return blocks


def token_states(model, ids):
    import torch

    backbone = model[0].model
    backbone.set_adapter("retrieval")
    inputs = torch.tensor([ids], device=model.device)
    with torch.inference_mode():
        return backbone(
            input_ids=inputs, attention_mask=torch.ones_like(inputs)
        ).last_hidden_state[0]


def embed_context_block(
    block: ContextBlock, model, pooling_method: str
) -> list[EmbeddedChunk]:
    import torch
    import torch.nn.functional as F

    if pooling_method != config.PHASE2_POOLING_METHOD or pooling_method != "mean":
        raise ValueError("Only validated mean pooling is supported")
    if (
        block.token_count != len(block.token_ids)
        or block.token_count > config.CONTEXT_BLOCK_MAX_TOKENS
    ):
        raise ValueError("Invalid contextual block size")
    hidden = token_states(model, block.token_ids)
    try:
        output = []
        for text, parts, (start, stop), ordinal in zip(
            block.chunk_texts,
            block.chunk_segments,
            block.chunk_spans,
            block.chunk_ordinals,
            strict=True,
        ):
            vector = F.normalize(hidden[start:stop].mean(0), dim=0).detach().cpu()
            if (
                vector.shape != (config.EMBED_DIMENSION,)
                or not torch.isfinite(vector).all()
                or not torch.allclose(vector.norm(), torch.tensor(1.0), atol=1e-4)
            ):
                raise ValueError("Require finite normalized 768D contextual vectors")
            pages = tuple(sorted({p for part in parts for p in part.source_pages}))
            first = parts[0]
            combined = replace(first, text=text, source_pages=pages)
            meta = normalize_metadata(combined)
            meta["source_paths"] = json.dumps(
                list(dict.fromkeys(p.source_path for p in parts))
            )
            cid = stable_chunk_id(
                first.book_stem,
                [p.source_path for p in parts],
                pages,
                first.chapter,
                first.section,
                ordinal,
                text,
            )
            output.append(EmbeddedChunk(cid, text, meta, vector.tolist()))
        return output
    finally:
        del hidden
