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
LOGS_DIR     = DATA / "logs"

# ── Ollama / LLM ──────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "qwen3.5:4b-q4_K_M"

# ── Ollama sampling / response style ──────────────────────────────────────────
# Lower temperature favors structured, deterministic study answers.
OLLAMA_TEMPERATURE = 0.3
OLLAMA_TOP_P = 0.9
# Optional Mirostat controls. Set OLLAMA_MIROSTAT to 1 or 2 to enable.
OLLAMA_MIROSTAT = None
OLLAMA_MIROSTAT_TAU = 5.0
OLLAMA_MIROSTAT_ETA = 0.1

# ── Embeddings ────────────────────────────────────────────────────────────────
# Prefixes and retrieval adapter must be identical across ingest/query.
EMBED_MODEL = "jinaai/jina-embeddings-v5-text-nano"
EMBED_REVISION = "8a7f00aac812071b69403df470f1038ec85f8925"
INGEST_EMBED_DEVICE = "cuda"
QUERY_EMBED_DEVICE = "cpu"
EMBED_DEVICE = QUERY_EMBED_DEVICE  # Legacy callers remain CPU-safe.
EMBED_BATCH_SIZE = 8

# Foundation budgets; measured by validate_foundation before phase acceptance.
OLLAMA_CONTEXT_TOKENS = 4096
OLLAMA_OUTPUT_TOKENS = 512
OLLAMA_REQUEST_TIMEOUT_SECONDS = 120.0
OLLAMA_READ_TIMEOUT_SECONDS = 5.0
OLLAMA_UNLOAD_TIMEOUT_SECONDS = 20.0
OLLAMA_POLL_INTERVAL_SECONDS = 0.2
OLLAMA_KEEP_ALIVE = 0
RUNTIME_MANIFEST_PATH = LOGS_DIR / "runtime_manifest.json"
FOUNDATION_REPORT_PATH = LOGS_DIR / "foundation_report.json"
TOOLS_REPORT_PATH = LOGS_DIR / "ollama_tools.json"
FOUNDATION_VECTOR_STORE = DATA / "foundation_vector_store"
INGEST_MIN_FREE_VRAM_MB = 2048
# ── Chunking & retrieval ──────────────────────────────────────────────────────
CHUNK_SIZE    = 800    # target tokens per chunk
CHUNK_OVERLAP = 100    # overlap tokens between adjacent chunks
TOP_K_RESULTS = 5      # top-k chunks retrieved per query (live-tunable in TUI)

# ChromaDB returns DISTANCE (lower = more similar), not similarity.
# Loosened to 1.5 because Jina-v5-nano results often fall between 0.8 and 1.2.
DISTANCE_THRESHOLD = 1.5

# ── MinerU VRAM guard ─────────────────────────────────────────────────────────
# Minimum free VRAM (GB) before MinerU mode is allowed to proceed.
# Below this threshold the script will pause and warn the user.
MINERU_MIN_VRAM_GB = 4.5
# Process scanned PDFs in page batches so long runs surface progress
# and checkpoint incrementally instead of waiting for one giant pass.
MINERU_BATCH_PAGES = 24
# MinerU backend to try first; falls back to MINERU_BACKEND_FALLBACK on failure.
# Options: pipeline | hybrid-auto-engine | vlm-auto-engine
MINERU_BACKEND_PRIMARY  = "pipeline"
MINERU_BACKEND_FALLBACK = "pipeline"   # kept identical; set to None to disable retry
# ISO 639-1 language hint passed to MinerU for better OCR accuracy.
# Use "ch" for Chinese, "en" for English, etc. None = let MinerU decide.
MINERU_LANG = "en"

# ── OCR/index timeout guards (seconds) ───────────────────────────────────────────────────
# Hard timeout to prevent hangs from freezing the TUI forever.
PYMUPDF4LLM_OCR_TIMEOUT_SECONDS = 300    # fast CPU path; 300s is generous even for 1000-page PDFs
MARKER_OCR_TIMEOUT_SECONDS = 3600
MINERU_OCR_TIMEOUT_SECONDS = 1800
INDEX_TIMEOUT_SECONDS = 1800
PROCESS_TERMINATE_GRACE_SECONDS = 8

# ── OCR auto-routing heuristics ───────────────────────────────────────────────
# In auto mode, OCR samples pages and chooses MinerU for scan-heavy PDFs.
AUTO_OCR_SAMPLE_PAGES = 8
AUTO_OCR_TEXT_CHAR_THRESHOLD = 180
AUTO_OCR_IMAGE_COVERAGE_THRESHOLD = 0.45
AUTO_OCR_SCANNED_PAGE_RATIO_THRESHOLD = 0.50


def ensure_directories():
    """Ensure all required data directories exist on disk."""
    for path in [DATA, RAW_PDFS, CLEAN_IMGS, PARSED_MD, CHECKPOINTS, VECTOR_STORE, LOGS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


# Auto-initialize on import so scripts don't crash on missing paths
ensure_directories()
