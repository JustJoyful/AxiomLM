"""
config.py — Single source of truth for all paths, model names, and tunable parameters.
Nothing is hardcoded in other scripts. Everything imports from here.
"""

from pathlib import Path

# ── Root paths ────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent.parent   # project root (axiom-lm/)
DATA         = ROOT / "data"
RAW_PDFS     = DATA / "raw_pdfs"
CLEAN_IMGS   = DATA / "clean_images"
PARSED_MD    = DATA / "parsed_md"
CHECKPOINTS  = PARSED_MD / "checkpoints"
VECTOR_STORE = DATA / "vector_store"

# ── Ollama / LLM ──────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "gemma:latest"

# ── Ollama sampling / response style ──────────────────────────────────────────
# Lower temperature favors structured, deterministic study answers.
OLLAMA_TEMPERATURE = 0.3
OLLAMA_TOP_P = 0.9
# Optional Mirostat controls. Set OLLAMA_MIROSTAT to 1 or 2 to enable.
OLLAMA_MIROSTAT = None
OLLAMA_MIROSTAT_TAU = 5.0
OLLAMA_MIROSTAT_ETA = 0.1

# ── Embeddings ────────────────────────────────────────────────────────────────
# jina-v5-text-nano: 239M params, 32K token context, CPU-only, MTEB 71.0
# IMPORTANT: Always use "Query: " / "Document: " prefixes for asymmetric retrieval
EMBED_MODEL   = "jinaai/jina-embeddings-v5-text-nano"

# ── Chunking & retrieval ──────────────────────────────────────────────────────
CHUNK_SIZE    = 800    # target tokens per chunk
CHUNK_OVERLAP = 100    # overlap tokens between adjacent chunks
TOP_K_RESULTS = 5      # top-k chunks retrieved per query

# ── MinerU VRAM guard ─────────────────────────────────────────────────────────
# Minimum free VRAM (GB) before MinerU mode is allowed to proceed.
# Below this threshold the script will pause and warn the user.
MINERU_MIN_VRAM_GB = 4.5

# ── OCR auto-routing heuristics ───────────────────────────────────────────────
# In auto mode, OCR samples pages and chooses MinerU for scan-heavy PDFs.
AUTO_OCR_SAMPLE_PAGES = 8
AUTO_OCR_TEXT_CHAR_THRESHOLD = 180
AUTO_OCR_IMAGE_COVERAGE_THRESHOLD = 0.45
AUTO_OCR_SCANNED_PAGE_RATIO_THRESHOLD = 0.50
