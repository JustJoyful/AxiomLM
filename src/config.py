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
