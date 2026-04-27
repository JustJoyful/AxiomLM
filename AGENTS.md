# AGENTS.md — AxiomLM Codebase Conventions

> **Goal:** A compact instruction file for autonomous agents to maintain context and follow idiosyncratic workflow patterns.
> **Directive:** Every line must answer: "Would an agent likely miss this without help?"

## 🚀 Core Development Workflow (GSD Protocol)
*   **Prerequisite:** No development or implementation (`feat`, `fix`) may begin until `PROJECT_RULES.md` is updated and `AGENTS.md` specifies `Status: FINALIZED` in `.gsd/SPEC.md`.
*   **Commit Granularity:** One single task = one commit. Never commit a wave of unrelated changes.
*   **Commit Format:** Must adhere to `type(scope): description` (e.g., `feat(tui): Added source cuing`).
    *   **Types:** `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.

## 🧐 Model & Context Management Quirks
*   **Context Decay:** The context window is highly volatile. State persistence *requires* updating `.gsd/STATE.md` after every major task or wave.
*   **Research First:** Before reading any file, use `grep` or `Select-String` to find specific lines or patterns. Reading entire files is context pollution.
*   **Source of Truth (Configuration):** All runtime paths, models, and tunables are hardcoded in `src/config.py`. Do not assume new paths or configurations are available elsewhere.
*   **Embedding Schema:** All embeddings must use specific prefixes:
    *   **Document Chunks:** Use the prefix `Document: `
    *   **Queries:** Use the prefix `Query: `

## ⚙️ Technical Architecture Boundaries
*   **Canonical Directory:** `src/config.py` is the single source of truth for the entire application.
*   **OCR Router:** The entry point for fetching PDF content is always `src/ocr.py`. This file selects the engine (`auto`, `marker`, or `mineru`).
*   **Data Flow (Mandatory Order):** `PDF` $\to$ `src/ocr.py` $\to$ `data/parsed_md/*.md` $\to$ `src/indexer.py` $\to$ `data/vector_store` $\to$ `src/tui.py`.
*   **Extraction Logic:** OCR processes must save page-level markdown checkpoints using the fixed format: `page_NNNN.md` (Zero-padded).

## 🔨 Mandatory Development Commands & Checks
*   **Verification Chain:** For production readiness, the sequence is non-negotiable: `npm run lint` $\to$ `npm run typecheck` $\to$ `npm test` (or equivalent platform commands).
*   **Code Search (ripgrep):** Use `rg "pattern" --type "*.ts"` for robust code artifact searching.
*   **Testing Quirks:** No dedicated unit test suite exists (`tests/` is absent). Verification relies on `src/validate-*.sh|ps1` scripts.
*   **Debugging/Recovery:**
    *   **Pattern:** If debugging fails 3 times, STOP. Update `STATE.md` with the attempt log and use a fresh session.
    *   **State Recovery:** Always check `.gsd/STATE.md` for the next action before continuing any run.

## ♻️ State Management & Cleanup
*   **Wave Protocol:** Grouping work into labeled "Waves" is required for atomic committing. Must update `ROADMAP.md` and commit proof.
*   **Context Compression:** When moving between waves, summarize/reference previous work in `STATE.md` rather than reloading vast chunks of code/logs.
*   **Environment:** Assume `OLLAMA_HOST=http://localhost:11434` is the inference endpoint.

***Source Documentation:** This file synthesizes operational details from `PROJECT_RULES.md`, `ARCHITECTURE.md`, `GSD-STYLE.md`, and `docs/runbook.md` to enforce adherence to unique and critical workflow patterns.*