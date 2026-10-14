<div align="center">

![AxiomLM logo](AxiomLM.png)

### Your Textbooks. Locally. Answered.

A terminal study assistant for STEM and computer science textbooks, powered by local Ollama inference and book retrieval.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/Runtime-Ollama-black?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Linux-orange?style=flat-square)

</div>

## Project status

AxiomLM is preparing a phased overhaul. The working baseline parses PDFs, indexes books and answers questions through a Textual TUI. The roadmap below describes **planned work**, not shipped features. OCR/scanned-PDF processing is frozen; effort moves to model lifecycle, UI, retrieval, tools and evaluation.

Core inference stays local with no silent cloud API fallback. Once models/dependencies are downloaded, the current book chat path can run offline. Planned web search/fetch are explicit evidence tools and require internet. A separate manually triggered Ask cloud button is optional, last, and outside core routing.

## Available today

- Textual terminal chat with local Ollama generation and model selection.
- Per-book ChromaDB collections and Jina asymmetric retrieval embeddings.
- Header/boundary-aware overlapping chunks with physical-page metadata.
- Follow-up context reuse, page-aware queries and source cues; citations are not yet checked by an entailment model.
- PDF engine selection, parse/index progress and resumable page checkpoints.
- Response/code copy actions via the current `pyperclip` helper.

The configured target is **Qwen3.5-4B Q4_K_M**, using the exact Ollama tag `qwen3.5:4b-q4_K_M`. Phase 0 acceptance passed on the RTX 3050 6GB development machine: Ollama 0.40.4 tool-calling, GPU generation at the 4096-token context limit, unload, CUDA Jina ingestion/release, and CPU live-query placement have recorded evidence. The baseline comparison uses a retrospective reconstruction with truncated before answers, so answer-quality comparisons are directional. Maintainer-local status and evidence are recorded in PHASES.md and MEMORY.md; those files are not part of the public README distribution.

## Screenshots

These show the current interface, before the planned overhaul.

![Current AxiomLM chat interface](image-2.png)
![Current AxiomLM study response](image-1.png)
![Current AxiomLM terminal layout](image.png)

## Setup and use

Linux, Python 3.11+, and [uv](https://docs.astral.sh/uv/getting-started/installation/) are required. Install [Ollama](https://ollama.com/) and start its local service. The primary overhaul target is a Linux/Niri/Wayland laptop with an RTX 3050 6GB GPU and 16GB RAM. Runtime defaults, paths and tuning parameters live in `src/config.py`.

Clone the project and run the installer. It syncs the pinned CUDA profile and development tools from `uv.lock`, then pulls the configured local Ollama model:

```bash
git clone https://github.com/JustJoyful/AxiomLM.git
cd AxiomLM
bash install.sh
source .venv/bin/activate
uv run --no-sync python -m src.tui
```

`install.sh` delegates to `scripts/setup_env.sh`. Use `--cpu` for the pinned CPU Torch profile, `--no-model` to skip both the pinned Jina snapshot download and Ollama model pull, and `--dry-run` to inspect all setup commands without changing the environment. Normal setup downloads the configured Jina revision and pulls the configured Ollama model. The sync preserves separately installed optional OCR engines. Use `uv run --no-sync axiomlm` to launch the installed entry point. `pyproject.toml` declares dependencies and `uv.lock` pins their resolved versions; the installer uses `uv`'s cache rather than staging multi-gigabyte CUDA wheels in `/tmp`.

The overhaul will require a tested pinned Textual 8.x release and GPU-capable ingest dependencies. Later phases add `wl-clipboard` for Wayland copying, Bubblewrap/cgroups v2/SymPy for execution, and Typst for PDFs. They are **planned prerequisites**, not capabilities enabled by this docs update.

### Parse and query a book

Enter an absolute PDF path in the TUI and select **Parse + Index**. Select the indexed book, then ask a question. Engine selection offers `auto`, `pymupdf4llm`, `mineru` and `marker`; heavy engines require their installed runtime and sufficient hardware. MinerU is installed separately when needed.

The CLI extraction/indexing path is:

```bash
uv run --no-sync python -m src.ocr --mode auto --pdf /absolute/path/to/book.pdf
uv run --no-sync python -m src.indexer book
```

Replace `book` with the PDF stem. Existing collection replacement is explicit:

```bash
uv run --no-sync python -m src.indexer book --reindex
```

Reindexing replaces that book's vector collection; retain the parsed page artifacts.

### Existing OCR behavior (frozen)

`src/ocr.py` is the unified entry point. PyMuPDF4LLM handles clean digital PDFs on CPU; MinerU handles scan-heavy PDFs with GPU batching/checkpoints; Marker remains a manual alternative for sufficiently capable hardware.

Auto-routing samples up to eight evenly spaced pages. It considers low extractable text and high image coverage, including image-heavy pages with an OCR text layer. MinerU is selected when at least half the samples are scan-like; thresholds and overrides remain in the existing code/configuration. Parse speed depends on the book, engine and hardware; old timing estimates are not guarantees.

Checkpoints are stored in `data/parsed_md/checkpoints/<book_stem>/page_NNNN.md`, merged Markdown in `data/parsed_md/<book_stem>/full.md`, and vectors in `data/vector_store`. Checkpoint indices begin at zero; cited physical PDF pages begin at one and may differ from printed book labels.

### Retrieval controls

The current TUI exposes K (retrieval count) and D (distance cutoff). The current configured defaults are K=5 and D=1.5; consult `src/config.py` for changes. Lower distance admits closer matches; raising K adds context and generation cost. Phase 1 moves these controls behind settings; Phase 2 adds hybrid retrieval, and Phase 4 introduces the calibrated entailment gate/rank.

## Overhaul roadmap

Build in order. Detailed acceptance criteria are in the maintainer's local `PRD.md`; the public scope summary is below.

| Phase | Planned outcome |
| --- | --- |
| **0 — Foundation** | Exact Qwen3.5-4B Q4_K_M tag/digest; verified Ollama tool round-trip; GPU ingest, explicit release, CPU queries; local inference boundary |
| **1 — Textual UI** | One accent and stylesheet, less chrome, settings-hidden K/D, native 8.x interaction, terminal ANSI themes, per-code/math `wl-copy` |
| **2 — Context** | Chapter/section metadata, 8192-token non-overlapping contextual blocks and late chunking, follow-up rewriting, BM25 + dense search, direct 5–8 chunk synthesis |
| **3 — Tools** | Local search, web search/fetch, sandboxed Python, PDF/citation tool interfaces; warm SymPy worker, verified solve flow and honest failure |
| **4 — Trust** | One sub-100M CPU entailment model for gate/rank and citation checks; labeled per-book evaluation |
| **5 — Study/documents** | Practice questions, coverage gaps, raw Typst → PDF with two retries maximum; optional flashcards |

Sandboxed Python uses Bubblewrap, network isolation and a cgroups v2 250M ceiling. The backend enforces a roughly 5s response deadline while preserving the warm worker; timeout cancellation and dead-worker recovery must be demonstrated before shipping.

Time-boxed floor: **Phases 0–2 plus the Phase 4 eval harness**. Begin labels/baseline before pipeline changes. Phase 3 is the most ambitious/time-consuming addition. Phase 1 may be explicitly deferred; Phase 1 and Phase 5 are the first permitted cuts, and flashcards are lowest priority. The floor does not require full Phase 3 or the Phase 4 trust scorer. No OCR overhaul, cloud routing, Pyodide or iterative map-reduce.

## Repository map

```text
src/
  config.py          Runtime paths, models and tunable parameters
  tui.py             Current Textual UI and query/ingest orchestration
  db.py              ChromaDB and Jina embedding integration
  indexer.py         Current chunking, metadata and indexing
  clipboard.py       Current response/code copy helper
  ocr.py             Frozen extraction entry point
  ocr_*.py           Frozen extraction engines
  checkpoint.py      Frozen per-page artifacts
  preprocessor.py    Frozen preprocessing
scripts/             Existing setup and page-validation utilities
data/                Local parsed artifacts, vector indexes and logs
PRD.md               Product scope, build order and acceptance gates
ARCHITECTURE.md      Current baseline and target implementation contracts
RULES.md             Mandatory development constraints
AGENTS.md            Coding-agent entry point and commands
```

## Contributing and verification

When working in the maintainer's local workspace, read `AGENTS.md` and `RULES.md` before making changes. Those local documents, `PRD.md` and `ARCHITECTURE.md` are intentionally excluded from GitHub. The public phase summary above records the scope; contributors should confirm the applicable phase before implementation. Keep scope/status documentation aligned with actual code and preserve the frozen extraction pipeline. Runtime configuration belongs in `src/config.py`.

```bash
uv run --no-sync --extra dev ruff check src/ scripts/
uv run --no-sync --extra dev black --check src/
uv run --no-sync python scripts/validate_page_metadata.py book
uv run --no-sync python scripts/validate_page_querying.py book "sample question"
```

The validation scripts require an indexed book. They check existing metadata/query behavior; they are not the planned labeled eval harness. Future pipeline changes require before/after retrieval precision/recall, citation accuracy and hallucination measurements. Use `type(scope): description` commit messages.

## License

MIT, as declared in `pyproject.toml`.
