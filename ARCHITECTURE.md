# AxiomLM — System Architecture (`ARCHITECTURE.md`)

> **Version:** 2.0.0  
> **Status:** ACTIVE  
> **Last Updated:** 2026-09-11  
> **Single Source of Truth (Config):** `src/config.py`

---

## 1. Overview

AxiomLM is a modular, local-first retrieval-augmented generation (RAG) study assistant designed for terminal environments. It is engineered specifically for STEM and engineering students processing dense, notation-heavy academic textbooks (calculus, probability, circuits, programming).

### Key Architectural Tenets
1. **100% Local Execution:** Zero external API calls, tracking, or cloud dependencies.
2. **GPU Budget Guardrail (6GB VRAM):** Heavy OCR (`MinerU`) and LLM generation (`qwen3:4b` via Ollama) are decoupled and mutually exclusive in VRAM.
3. **Adaptive Heuristic Routing:** High-speed direct extraction (`pymupdf4llm`) for clean digital PDFs (<10 seconds), reserving heavy vision-based OCR (`MinerU`) only for verified scanned/photographed books.
4. **Resilience via Checkpoints:** Page-level atomic markdown checkpoints (`page_NNNN.md`) ensure interrupted parsing jobs resume with zero lost compute.
5. **Faithful Math & Code RAG:** Grounded answering with `qwen3:4b`, asymmetric Jina v5 Nano embeddings, dynamic K/D retrieval sliders, and exact page/section citations.

---

## 2. System Diagram

```text
                                +---------------------------+
                                |  Input PDF                |
                                |  (data/raw_pdfs/*.pdf)    |
                                +-------------+-------------+
                                              |
                                              v
                              +-------------------------------+
                              |       src/ocr.py              |
                              |  Smart Heuristic Router       |
                              |  (8-page uniform sampling)    |
                              +---------------+---------------+
                                              |
                    +-------------------------+-------------------------+
                    | (scanned_ratio < 0.50)  | (scanned_ratio >= 0.50) | (manual override)
                    v                         v                         v
       +-------------------------+ +---------------------+ +--------------------+
       |  src/ocr_pymupdf4llm.py | |  src/ocr_mineru.py  | |  src/ocr_marker.py |
       |  Fast Direct Parse      | |  GPU OCR Pipeline   | |  Clean Math Parser |
       |  (CPU, <10s)            | |  (24-page batches)  | |  (Manual, 8GB+ RAM)|
       +------------+------------+ +----------+----------+ +---------+----------+
                    |                         |                      |
                    +-------------------------+----------------------+
                                              |
                                              v
                              +-------------------------------+
                              |      src/checkpoint.py        |
                              |  data/parsed_md/checkpoints/  |
                              |  page_0001.md ... page_NNNN.md|
                              +---------------+---------------+
                                              |
                                              v
                              +-------------------------------+
                              |  data/parsed_md/<book>/full.md|
                              +---------------+---------------+
                                              |
                                              v
                              +-------------------------------+
                              |      src/indexer.py           |
                              |  Header-Aware Text Chunking   |
                              +---------------+---------------+
                                              |
                                              v
                              +-------------------------------+
                              |         src/db.py             |
                              |  Prefix: "Document: "         |
                              |  Jina v5 Nano Embeddings      |
                              +---------------+---------------+
                                              |
                                              v
                              +-------------------------------+
                              |      data/vector_store        |
                              |  Persistent ChromaDB Client   |
                              +---------------+---------------+
                                              |
                           Retrieval (Top-K / Distance D)
                                              |
                                              v
                              +-------------------------------+
                              |         src/tui.py            |
                              |  Textual Terminal Interface   |
                              |  Prefix: "Query: "            |
                              +---------------+---------------+
                                              |
                               Stream Context + Query
                                              |
                                              v
                              +-------------------------------+
                              |  Ollama API (localhost:11434) |
                              |  Default Model: qwen3:4b      |
                              +-------------------------------+
                                              |
                    +-------------------------+-------------------------+
                    | (planned)                                         | (planned)
                    v                                                   v
       +-------------------------+                         +--------------------------+
       |   Anki Flashcard Export |                         |   Knowledge Gap Tracker  |
       |   Formula & Syntax Deck |                         |   Concept Mastery Audit  |
       +-------------------------+                         +--------------------------+
```

---

## 3. The Smart Routing Heuristic Engine

A critical architectural component in `src/ocr.py` is the **Heuristic Engine**, designed to avoid locking up user GPUs on false-positive scan detections.

### Heuristic Algorithm Specification
1. **Sampling:** When set to `auto` mode (the default), `_sample_indices` selects `AUTO_OCR_SAMPLE_PAGES = 8` pages spaced evenly throughout the PDF:
   $$\text{indices} = \left\{ \text{round}\left(i \cdot \frac{\text{page\_count} - 1}{\text{sample\_size} - 1}\right) \;\middle|\; i \in [0, \text{sample\_size}-1] \right\}$$
2. **Page Metrics Inspection (`_page_metrics`):**
   - Counts extractable text characters ($T$) via PyMuPDF.
   - Computes image bounding box area coverage ($C = \frac{\text{image\_area}}{\text{page\_area}}$).
3. **Scan-Like Classification:** A sampled page is marked as scan-like if:
   - $(T \le \text{AUTO\_OCR\_TEXT\_CHAR\_THRESHOLD} \land C \ge \text{AUTO\_OCR\_IMAGE\_COVERAGE\_THRESHOLD})$ (e.g., $T \le 180$ chars and $C \ge 0.45$), OR
   - $(T \le 60 \text{ chars} \land \text{has\_image\_block})$, OR
   - $(C \ge 0.90 \land \text{has\_image\_block})$ (catches scanned books containing an underlying poor-quality OCR text layer).
4. **Decision Boundary:**
   $$\text{scanned\_ratio} = \frac{\text{scanned\_votes}}{\text{sampled\_pages}}$$
   - If $\text{scanned\_ratio} \ge \text{AUTO\_OCR\_SCANNED\_PAGE\_RATIO\_THRESHOLD}$ (default: $0.50$): Route to `MinerU`.
   - Otherwise: Route to `pymupdf4llm`.

### Edge Case Handling (The "Page 4 Diagram" Problem)
If a clean 500-page digital programming textbook contains an isolated scanned diagram or high-resolution photo on page 4:
- In the 8 sampled pages, at most 1 page (or none) will exhibit high image coverage.
- The resulting scan ratio is $\le 12.5\%$ ($\le 0.125$), which falls far below the $50\%$ threshold.
- **Outcome:** The book is correctly routed to `pymupdf4llm`, completing in $\sim 5$ seconds instead of locking the GPU in MinerU for 40 minutes.

### Manual Overrides
Users retain deterministic control:
- **CLI Flag:** `python -m src.ocr --mode pymupdf4llm|mineru|marker path/to/book.pdf`
- **TUI Dropdown:** Select the preferred engine before clicking **Parse + Index**.

---

## 4. Subsystem Components

### 4.1 Configuration Layer (`src/config.py`)
- **Role:** Canonical single source of truth for runtime paths, thresholds, and model parameters.
- **Key Parameters:**
  - `OLLAMA_MODEL`: `"qwen3:4b"`
  - `EMBED_MODEL`: `"jinaai/jina-embeddings-v5-text-nano"`
  - `AUTO_OCR_SAMPLE_PAGES`: `8`
  - `AUTO_OCR_TEXT_CHAR_THRESHOLD`: `180`
  - `AUTO_OCR_IMAGE_COVERAGE_THRESHOLD`: `0.45`
  - `AUTO_OCR_SCANNED_PAGE_RATIO_THRESHOLD`: `0.50`
  - `MINERU_BATCH_PAGES`: `24`
  - `MINERU_MIN_VRAM_GB`: `4.0`

### 4.2 OCR Engines
1. **`src/ocr_pymupdf4llm.py`:** Directly converts digital PDF structure to Markdown using PyMuPDF4LLM. Runs entirely on CPU with negligible RAM.
2. **`src/ocr_mineru.py`:** Runs the MinerU pipeline for photographed or scanned pages. Employs 24-page batching, explicit VRAM garbage collection (`torch.cuda.empty_cache()`), and immediate checkpoint persistence.
3. **`src/ocr_marker.py`:** Manual alternative for math-heavy digital PDFs when extra GPU VRAM is available.

### 4.3 Checkpointing (`src/checkpoint.py`)
- Persists individual page Markdown as `page_NNNN.md`.
- Provides idempotent crash recovery: scans existing checkpoint files and resumes at the first uncompleted page index.
- Merges all checkpoints into `data/parsed_md/<book_stem>/full.md`.

### 4.4 Indexer & Vector Store (`src/indexer.py`, `src/db.py`)
- **Chunking:** Uses Markdown header-aware splitting (`#`, `##`, `###`) to preserve semantic section contexts, supplemented by character-level chunking (800 chars, 100 overlap).
- **Embedding:** Generates 512-dim vectors via `jinaai/jina-embeddings-v5-text-nano`. Embeds document chunks with prefix `Document: `.
- **ChromaDB Client:** Persistent collection created per book under `data/vector_store/`.

### 4.5 Terminal Interface & Inference (`src/tui.py`)
- Built with **Textual** for a native, fast terminal experience.
- Sidebar displays parsed books, chunk statistics, parse progress, and GPU embedding acceleration toggle.
- Chat loop prefixes user query with `Query: `, retrieves top-$K$ chunks within distance threshold $D$, formats context with citations, and streams tokens from Ollama `qwen3:4b`.

### 4.6 Planned Modules (Post-Voice Refocus)
- **`src/flashcards.py` (Anki Exporter):** Parses chat responses and textbook sections to generate Anki decks (`.apkg` / TSV) with math formulas and code blocks.
- **`src/gap_tracker.py` (Knowledge Gap Tracker):** Compares user query history against book table of contents to flag unmastered topics and missing prerequisites.

---

## 5. Integration Points

| Service / Dependency | Type | Purpose |
| :--- | :--- | :--- |
| **Ollama (`localhost:11434`)** | Local HTTP Service | Runs local `qwen3:4b` inference for RAG generation |
| **ChromaDB** | Embedded Python DB | Vector persistence and cosine similarity search |
| **PyMuPDF (`fitz`)** | C/Python Library | PDF text/image metric extraction for auto-routing |
| **PyMuPDF4LLM** | Python Library | Rapid digital PDF-to-Markdown extraction |
| **MinerU (`magic-pdf`)** | Python / PyTorch | Vision-based OCR for complex scanned textbooks |
| **SentenceTransformers** | Python / PyTorch | Runs Jina v5 Nano embedding model |
| **Textual** | Python TUI Framework | Renders reactive terminal user interface |

---

## 6. Development & Verification Conventions

- **Code Style:** Python 3.11+, typed hints, Ruff for linting and formatting.
- **Embedding Invariant:** Must always prefix indexed chunks with `Document: ` and retrieval queries with `Query: `.
- **VRAM Invariant:** Never initiate Ollama generation while MinerU OCR batch processing is active.
