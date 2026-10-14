from src import config


def test_foundation_runtime_contract():
    assert config.OLLAMA_MODEL == "qwen3.5:4b-q4_K_M"
    assert getattr(config, "QUERY_EMBED_DEVICE", None) == "cpu"
    assert getattr(config, "INGEST_EMBED_DEVICE", None) == "cuda"
    assert config.EMBED_MODEL == "jinaai/jina-embeddings-v5-text-nano"
    assert len(getattr(config, "EMBED_REVISION", "")) == 40


def test_ocr_settings_preserved():
    assert config.MINERU_MIN_VRAM_GB == 4.5
    assert config.MINERU_BATCH_PAGES == 24
    assert config.MINERU_BACKEND_PRIMARY == "pipeline"
    assert config.AUTO_OCR_SAMPLE_PAGES == 8
    assert config.AUTO_OCR_TEXT_CHAR_THRESHOLD == 180
    assert config.AUTO_OCR_IMAGE_COVERAGE_THRESHOLD == 0.45
    assert config.AUTO_OCR_SCANNED_PAGE_RATIO_THRESHOLD == 0.50
