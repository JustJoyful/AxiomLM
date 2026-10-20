"""Count the served answer model's tokens without loading a second weight model."""

from __future__ import annotations

from functools import lru_cache

import httpx
from tokenizers import AddedToken, Regex, Tokenizer, pre_tokenizers
from tokenizers.models import BPE

from src import config

# Qwen's published tokenizer pattern; vocabulary/merges come from the local GGUF.
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/tokenization_qwen3_5.py
QWEN35_PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


class AnswerTokenCounter:
    def __init__(self, info):
        metadata = info["model_info"]
        if (
            info.get("template", "").strip() != "{{ .Prompt }}"
            or info.get("system")
            or metadata.get("tokenizer.ggml.add_bos_token", False)
        ):
            raise ValueError(
                "Unverified answer model template; token budget cannot be established"
            )
        if (
            metadata.get("tokenizer.ggml.pre") != "qwen35"
            or metadata.get("tokenizer.ggml.model") != "gpt2"
        ):
            raise ValueError(
                "Unverified answer tokenizer; use the configured validated model"
            )
        vocab = metadata["tokenizer.ggml.tokens"]
        merges = metadata["tokenizer.ggml.merges"]
        if len(set(vocab)) != len(vocab):
            raise ValueError("Duplicate answer vocabulary entries")
        self.tokenizer = Tokenizer(
            BPE(
                vocab={t: i for i, t in enumerate(vocab)},
                merges=[tuple(m.split(" ", 1)) for m in merges],
                unk_token=None,
                byte_fallback=False,
            )
        )
        self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(QWEN35_PATTERN), behavior="isolated"),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
        self.tokenizer.add_special_tokens(
            [
                AddedToken(t, special=True, normalized=False)
                for t, kind in zip(
                    vocab, metadata["tokenizer.ggml.token_type"], strict=True
                )
                if kind in {3, 4}
            ]
        )

    def __call__(self, prompt):
        return len(
            self.tokenizer.encode(
                config.ANSWER_PROMPT_PREFIX
                + prompt.strip()
                + config.ANSWER_PROMPT_SUFFIX,
                add_special_tokens=False,
            ).ids
        )


def load_answer_counter(model):
    with httpx.Client(
        trust_env=False, timeout=config.OLLAMA_REQUEST_TIMEOUT_SECONDS
    ) as client:
        response = client.get(config.OLLAMA_URL + "/api/tags")
        response.raise_for_status()
        entries = response.json().get("models", [])
    entry = next(
        (r for r in entries if r.get("name") == model or r.get("model") == model), None
    )
    if entry is None:
        raise ValueError("Selected answer model is not installed")
    return _counter_for(model, entry["digest"])


@lru_cache(maxsize=config.ANSWER_TOKENIZER_CACHE_SIZE)
def _counter_for(model, digest):
    if digest != config.ANSWER_TOKENIZER_MODEL_DIGEST:
        raise ValueError(
            "Answer tokenization is not validated for this model digest; select the configured Qwen model"
        )
    with httpx.Client(
        trust_env=False, timeout=config.OLLAMA_REQUEST_TIMEOUT_SECONDS
    ) as client:
        version = client.get(config.OLLAMA_URL + "/api/version")
        version.raise_for_status()
        if version.json().get("version") != config.ANSWER_TOKENIZER_OLLAMA_VERSION:
            raise ValueError(
                "Answer tokenization requires calibration for this Ollama version"
            )
        response = client.post(
            config.OLLAMA_URL + "/api/show", json={"model": model, "verbose": True}
        )
        response.raise_for_status()
        return AnswerTokenCounter(response.json())
