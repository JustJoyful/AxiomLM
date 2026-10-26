"""
config.py — Single source of truth for all paths, model names, and tunable parameters.
Nothing is hardcoded in other scripts. Everything imports from here.
"""

from pathlib import Path

# ── Root paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent  # project root (axiom-lm/)
DATA = ROOT / "data"
RAW_PDFS = DATA / "raw_pdfs"
CLEAN_IMGS = DATA / "clean_images"
PARSED_MD = DATA / "parsed_md"
CHECKPOINTS = PARSED_MD / "checkpoints"
VECTOR_STORE = DATA / "vector_store"
LOGS_DIR = DATA / "logs"

# ── Ollama / LLM ──────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434"
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
OLLAMA_OUTPUT_TOKENS = 768
OLLAMA_REQUEST_TIMEOUT_SECONDS = 120.0
OLLAMA_READ_TIMEOUT_SECONDS = 5.0
OLLAMA_UNLOAD_TIMEOUT_SECONDS = 20.0
OLLAMA_POLL_INTERVAL_SECONDS = 0.2
OLLAMA_KEEP_ALIVE = 0
RUNTIME_MANIFEST_PATH = LOGS_DIR / "runtime_manifest.json"
FOUNDATION_REPORT_PATH = LOGS_DIR / "foundation_report.json"
PHASE2_EVALUATION_PATH = LOGS_DIR / "phase2_context_evaluation.json"
PHASE2_PROBE_PATH = LOGS_DIR / "phase2_planning_probe.json"
EMBED_DIMENSION = 768
PHASE2_CONTEXT_TOKEN_LIMIT = 8192
PHASE2_BROAD_MAX_CHUNKS = 8
PHASE2_PROBE_TOKEN_SIZES = (2048, 4096, 8192)
QUERY_REWRITE_MAX_OUTPUT_TOKENS = 96
QUERY_REWRITE_TIMEOUT_SECONDS = 15
PHASE2_ANSWER_COUNT_CONTEXT_TOKENS = 8192
TOOLS_REPORT_PATH = LOGS_DIR / "ollama_tools.json"
FOUNDATION_VECTOR_STORE = DATA / "foundation_vector_store"
INGEST_MIN_FREE_VRAM_MB = 2048
# ── Chunking & retrieval ──────────────────────────────────────────────────────
CHUNK_SIZE = 800  # target tokens per chunk
CHUNK_OVERLAP = 100  # overlap tokens between adjacent chunks
TOP_K_RESULTS = 5  # top-k chunks retrieved per query (live-tunable in TUI)

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
MINERU_BACKEND_PRIMARY = "pipeline"
MINERU_BACKEND_FALLBACK = "pipeline"  # kept identical; set to None to disable retry
# ISO 639-1 language hint passed to MinerU for better OCR accuracy.
# Use "ch" for Chinese, "en" for English, etc. None = let MinerU decide.
MINERU_LANG = "en"

# ── OCR/index timeout guards (seconds) ───────────────────────────────────────────────────
# Hard timeout to prevent hangs from freezing the TUI forever.
PYMUPDF4LLM_OCR_TIMEOUT_SECONDS = (
    300  # fast CPU path; 300s is generous even for 1000-page PDFs
)
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
    for path in [
        DATA,
        RAW_PDFS,
        CLEAN_IMGS,
        PARSED_MD,
        CHECKPOINTS,
        VECTOR_STORE,
        LOGS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


# Auto-initialize on import so scripts don't crash on missing paths
ensure_directories()

# Terminal UI preferences and presentation budgets.
UI_CONFIG_PATH = Path.home() / ".config/axiomlm/config.toml"
UI_THEMES_DIR = Path.home() / ".config/axiomlm/themes"
UI_FRAME_INTERVAL = 1 / 30
UI_SCROLLBACK_BLOCKS = 2000
UI_PROMPT_MAX_LINES = 8
UI_OVERLAY_ROWS = 8
UI_CLIPBOARD_TIMEOUT = 1.0
UI_QUIT_CONFIRM_SECONDS = 1.0
UI_BACKGROUND_TIMEOUT = 0.1
UI_SPINNER_INTERVAL = 0.08
UI_SHIMMER_INTERVAL = 0.06
UI_SHIMMER_PAUSE = 0.6
UI_COUNTER_INTERVAL = 0.1
UI_BLINK_INTERVAL = 0.5
UI_COMPLETION_FLASH_SECONDS = 0.12
UI_COPY_CONFIRM_SECONDS = 2.0
UI_RETRIEVAL_REPORT_PATH = LOGS_DIR / "phase1_retrieval_comparison.json"
UI_CANCEL_TIMEOUT = 0.18
UI_LIVE_REPORT_PATH = LOGS_DIR / "phase1_live_cancellation.json"
UI_PROGRESS_WIDTH = 24
UI_PROGRESS_EASING = 0.25
UI_ACCEPTANCE_REPORT_PATH = LOGS_DIR / "phase1_tui_acceptance.json"
UI_STYLESHEET_PATH = Path(__file__).parent / "ui/axiom.tcss"

# Task 1 isolated pooling validation; no production chunk/index semantics change.
PHASE2_PROBE_CHUNK_TOKENS = 300
PHASE2_GEOMETRY_REPORT_PATH = LOGS_DIR / "phase2_geometry_validation.json"
PHASE2_VALIDATION_LABELS_PATH = (
    ROOT / "tests/fixtures/phase2/sicp_pooling_validation_labels.json"
)
PHASE2_VALIDATION_PROTOCOL_PATH = (
    ROOT / "tests/fixtures/phase2/pooling_validation_protocol.json"
)

# Phase 2 versioned contextual retrieval.
CONTEXT_BLOCK_MAX_TOKENS = PHASE2_CONTEXT_TOKEN_LIMIT
RETRIEVAL_CHUNK_TOKENS = PHASE2_PROBE_CHUNK_TOKENS
PHASE2_POOLING_METHOD = "mean"
PHASE2_INDEX_SCHEMA = 2
PHASE2_QUERY_POLICY = "named_retrieval_query"
PHASE2_INDEX_MANIFEST_PATH = VECTOR_STORE / "active_indexes.json"
PHASE2_BM25_DIR = VECTOR_STORE / "bm25"
HYBRID_CANDIDATE_LIMIT = 15
HYBRID_DENSE_FETCH = 30
HYBRID_BM25_FETCH = 30
HYBRID_FETCH_BATCH = 15
HYBRID_MAX_FETCH = 100
BROAD_CONTEXT_MIN_CHUNKS = 5
BOOK_OVERVIEW_RETRIEVAL_QUERY = (
    "book overview main topics purpose approach introduction preface"
)
BROAD_CONTEXT_MAX_CHUNKS = PHASE2_BROAD_MAX_CHUNKS
PHASE2_AFTER_EVALUATION_PATH = LOGS_DIR / "phase2_context_after_evaluation.json"
INDEX_WRITE_BATCH_SIZE = 100
ANSWER_TOKENIZER_CACHE_SIZE = 1
PHASE2_LABELS_DIR = ROOT / "tests/fixtures/phase2"
PHASE2_STAGING_REPORT_DIR = LOGS_DIR / "phase2_context_artifacts"
# Verified against the installed runtime's served prompt counts (think=False).
ANSWER_TOKENIZER_MODEL_DIGEST = (
    "d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04"
)
ANSWER_TOKENIZER_OLLAMA_VERSION = "0.40.4"
ANSWER_PROMPT_PREFIX = "<|im_start|>user\n"
ANSWER_PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

# Phase 3 native Python sandbox limits; host acceptance is independently required.
PYTHON_MEMORY_BYTES = 262144000
PYTHON_MAX_TASKS = 32
PYTHON_RESPONSE_SECONDS = 5.0
PYTHON_CLEANUP_SECONDS = 1.0
PYTHON_STARTUP_SECONDS = 15.0
PYTHON_MAX_CODE_BYTES = 8192
PYTHON_MAX_OUTPUT_BYTES = 8192
PYTHON_MAX_FRAME_BYTES = 65536
PYTHON_MAX_ID_CHARS = 128
PYTHON_SCRATCH_BYTES = 33554432
SYMPY_CACHE_SIZE = 128
# Native modules needed for the stdlib/SymPy worker; GUI/test/db extensions excluded.
PYTHON_EXTENSION_MODULES = frozenset(
    {
        "_asyncio",
        "_bisect",
        "_blake2",
        "_bz2",
        "_contextvars",
        "_csv",
        "_ctypes",
        "_datetime",
        "_decimal",
        "_hashlib",
        "_heapq",
        "_json",
        "_lzma",
        "_md5",
        "_multiprocessing",
        "_opcode",
        "_pickle",
        "_posixshmem",
        "_posixsubprocess",
        "_queue",
        "_random",
        "_sha1",
        "_sha256",
        "_sha3",
        "_sha512",
        "_socket",
        "_ssl",
        "_struct",
        "_typing",
        "_uuid",
        "array",
        "binascii",
        "cmath",
        "fcntl",
        "math",
        "mmap",
        "pyexpat",
        "select",
        "unicodedata",
        "zlib",
        "_statistics",
        "resource",
        "_elementtree",
        "termios",
    }
)

PYTHON_POLL_SECONDS = 0.01

# Phase 3 agent routing in the TUI; unvalidated models fall back to the Phase 2 path.
AGENT_ROUTING_ENABLED = True
# Phase 3 agent runtime caps: starting values to measure, not throughput guarantees.
AGENT_MAX_MODEL_REQUESTS = 6
AGENT_MAX_TOOL_CALLS = 6
AGENT_MAX_CALLS_PER_MESSAGE = 2
AGENT_REQUEST_SECONDS = 120.0
AGENT_ROUTER_OUTPUT_TOKENS = 256
AGENT_FINAL_OUTPUT_TOKENS = 768
# Native chat accounting is valid only for this digest/runtime pair after the probe.
CHAT_TOKENIZER_MODEL_DIGEST = ANSWER_TOKENIZER_MODEL_DIGEST
CHAT_TOKENIZER_OLLAMA_VERSION = ANSWER_TOKENIZER_OLLAMA_VERSION
PHASE3_RUNTIME_LOGS = LOGS_DIR / "phase3_runtime"
PHASE3_AGENT_LOGS = LOGS_DIR / "phase3_agent"

# Web evidence: off by default; /web on grants retrieval permission, not cloud inference.
WEB_ENABLED_DEFAULT = False
# Self-hosted SearXNG (JSON format enabled) bound to loopback; None disables search.
WEB_SEARCH_URL = "http://127.0.0.1:8888/search"
WEB_MAX_RESULTS = 5
WEB_REQUEST_SECONDS = 10.0
WEB_MAX_REDIRECTS = 3
WEB_MAX_BODY_BYTES = 1048576
WEB_MAX_TEXT_CHARS = 12000
WEB_ALLOWED_PORTS = frozenset({80, 443})
TOOL_MAX_RESULT_BYTES = 16384
TOOL_MAX_ARGUMENT_BYTES = 8192
TOOL_QUERY_MAX_CHARS = 2048

# Trusted symbolic-check parser limits (one real variable; unsupported input declines).
MATH_MAX_EXPRESSION_CHARS = 512
MATH_MAX_AST_NODES = 128
MATH_MAX_INTEGER_EXPONENT = 64
MATH_MAX_NUMERIC_DIGITS = 32
