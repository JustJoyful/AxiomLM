# Phase 6 quality work

Phase 6 is in progress. The quality tools below are implemented; release and portability acceptance remain pending. The development runtime keeps its existing context/output limits, local Ollama model, serial GPU scheduling and default-off learned checks.

The original seven-case Phase 2 macro regression remains open. Its exact 1,387-chunk geometry was reconstructed from the hash-verified archived evaluator and sources. RRF scored macro precision/recall **0.2513605442/0.5000000000**, below the frozen baseline **0.4571428571/0.5952380952**. The active indexes use repaired text; their micro measurements remain a separate comparison. Human factual/support/citation inventories and Phase 5 practice/PDF fidelity reviews remain pending. CPU-only generation and an additional backend have not been accepted. No release or portability claim follows from unit-test counts.

The quality validator writes evidence exclusively and refuses to overwrite prior files. It rejects mismatched labels, source hashes, runtime/model identities, indexes and metric cutoffs. Missing, failed, partial and empty answers remain in completion and coverage denominators. Review sheets contain blank human fields: valid pages, scorer votes and retrieved passages are never treated as proof of factual correctness or claim support.

Run from the repository after the documented environment setup:

```bash
# Diagnose both rank policies using frozen calibration queries and a temporary
# copy of the active store. Requires local cached embedding dependencies.
uv run --no-sync python scripts/validate_phase6_quality.py diagnose sicp \
  --output data/logs/phase6/new-run/sicp-diagnosis.json

# Export a blank review inventory for an immutable ordinary-Q&A report.
uv run --no-sync python scripts/validate_phase6_quality.py review-sheet \
  --report data/logs/phase4_eval/run/sicp-test.json \
  --output data/logs/phase6/new-run/sicp-review.json

# Compare matched reports and independently completed human sidecars.
uv run --no-sync python scripts/validate_phase6_quality.py compare \
  --before data/logs/phase6/before/sicp-test.json \
  --after data/logs/phase6/after/sicp-test.json \
  --review-before data/logs/phase6/reviews/before-sicp-test.json \
  --review-after data/logs/phase6/reviews/after-sicp-test.json \
  --output data/logs/phase6/new-run/sicp-quality.json
```

Use actual existing report paths and a fresh output path. `compare` exits nonzero while acceptance is failed or pending. It reports retrieval, completion, store preservation, latency and human judgments separately. Supply `--original-before` and `--original-after` to include the separate original macro comparison. It verifies the frozen baseline bytes (including relocated exact copies), original labels/pages, runtime and reconstructed geometry; a lower replacement baseline is rejected. Without those inputs the original gate stays pending. The measured reconstructed candidate fails that gate. This tool compares one book/split at a time; full-set release aggregation and final release approval remain outstanding.

`interleave` remains the shipping fusion policy. The opt-in `rrf` experiment uses reciprocal ranks with K=60, preserves whole candidates and page provenance, and records selection from both books' calibration results before held-out testing. Calibration gains alone do not authorize promotion. Selection binds the complete centralized configuration hash, ranking/retrieval/rewrite/tool implementation hashes and frozen query inputs; unrelated trace instrumentation may differ. Evaluation reports also retain final stop reasons and prompt/output-budget traces; a length stop remains incomplete.

No OCR changes, active reindexing, model installation or backend replacement is included in this quality deliverable. Further measured retrieval correction and a maintainer quality disposition are required before dependent runtime/product advancement.

The completed held-out experiment improved SICP precision/recall from 0.190/0.462 to 0.234/0.577, but reduced Think Python from 0.308/1.000 to 0.217/0.813. RRF therefore remains unpromoted. Both modern before/after samples completed all 38 answers; candidate traces matched served token counts and fit within 4096 tokens. Human factual/support/premise judgments remain unmeasured. Final unit verification: 833 passed, 17 skipped and 19 snapshots; skipped host/terminal checks do not establish release acceptance.
