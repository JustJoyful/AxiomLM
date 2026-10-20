import pytest
from tokenizers.pre_tokenizers import ByteLevel

from src import answer_tokens as tokens


def info():
    return {
        "template": "{{ .Prompt }}",
        "model_info": {
            "tokenizer.ggml.pre": "qwen35",
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.tokens": sorted(ByteLevel.alphabet())
            + ["<|im_start|>", "<|im_end|>", "<think>", "</think>"],
            "tokenizer.ggml.token_type": [1] * 256 + [3] * 4,
            "tokenizer.ggml.merges": [],
            "tokenizer.ggml.add_bos_token": False,
        },
    }


def test_actual_model_byte_bpe_counts_complete_prompt_and_rejects_unknown_template():
    counter = tokens.AnswerTokenCounter(info())
    assert counter("abc\n123") == 32
    assert counter("θ") == 27
    with pytest.raises(ValueError, match="template"):
        tokens.AnswerTokenCounter(dict(info(), template="unverified chat template"))


def test_served_rendering_trims_prompt_and_preserves_combining_marks():
    counter = tokens.AnswerTokenCounter(info())
    assert counter(" abc\n123\n") == 32
    assert counter("e\u0301") == 28
