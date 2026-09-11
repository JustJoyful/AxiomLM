# AGENTS.md — AxiomLM Developer & Agent Guide

> **Purpose:** Essential conventions, architectural invariants, and operational commands for coding agents working on AxiomLM.

---

## 1. Core Architecture & Invariants

* **Single Source of Truth:** `src/config.py` contains all paths, model identifiers, and tunable thresholds. Never hardcode model names, paths, or thresholds in other modules.
* **Default LLM:** `qwen3:4b` served locally via Ollama (`http://localhost:11434`).
* **Embedding Model:** `jinaai/jina-embeddings-v5-text-nano` running on CPU (or GPU when opted-in).
* **Asymmetric Embedding Schema (Mandatory):**
  * Stored document chunks MUST be prefixed with: `Document: `
  * Search/query strings MUST be prefixed with: `Query: `
* **Data Flow Pipeline:**
  $$\text{PDF} \xrightarrow{\text{src/ocr.py}} \text{data/parsed\_md/checkpoints/page\_NNNN.md} \to \text{full.md} \xrightarrow{\text{src/indexer.py}} \text{data/vector\_store} \xrightarrow{\text{src/tui.py}} \text{Ollama}$$
* **Page Checkpoints:** Must adhere strictly to `page_NNNN.md` (4-digit zero-padded integer).
* **VRAM Guardrail:** On consumer 6GB VRAM GPUs (e.g. RTX 3050), heavy OCR (`MinerU`) and LLM generation (`qwen3:4b`) must never execute concurrently.

---

## 2. OCR Engine Selection Rules

* **Entry Point:** Always `src/ocr.py`.
* **Engines:**
  * `pymupdf4llm` — Default for clean digital PDFs (CPU only, $<10$s for 200+ pages).
  * `mineru` — For complex or photographed scanned PDFs (GPU batching of 24 pages with VRAM clearing).
  * `marker` — Optional manual math parser for clean PDFs when $\ge 8$GB VRAM is available.
* **Auto-Routing Heuristic:** Samples 8 evenly spaced pages. Only routes to MinerU if $\ge 50\%$ of sampled pages exhibit scan-like metrics ($T \le 180$ chars and $C \ge 0.45$ image coverage).

---

## 3. Development Commands & Verification

* **Environment Setup:**
  ```bash
  bash scripts/setup_env.sh
  source .venv/bin/activate
  ```
* **Linting & Code Style:**
  ```bash
  .venv/bin/ruff check src/ scripts/
  .venv/bin/black --check src/
  ```
* **Database & Metadata Validation:**
  ```bash
  python scripts/validate_page_metadata.py <book_stem>
  python scripts/validate_page_querying.py <book_stem> "sample question"
  ```
* **Run Application:**
  ```bash
  python -m src.tui
  # or
  axiomlm
  ```
* **Commit Conventions:**
  `type(scope): description` (e.g., `feat(ocr): add heuristic preview`, `fix(tui): handle empty collection error`).