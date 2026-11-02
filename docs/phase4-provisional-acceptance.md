# Phase 4 provisional acceptance / Phase 5 Task 0

Decision date: 2026-11-01. Decision maker: project maintainer, by explicit instruction to record a provisional pass and commit/merge the Phase 4 branch.

## Accepted

- The per-book evaluation harness, hash-pinned source/label/report workflow, isolated index evaluation and saved-answer rescoring.
- The `verify_citation` tool and shared claim-check implementation as advisory/experimental infrastructure, including answer-preservation, timeout/error reporting and source provenance.
- Advancement past Phase 5 planning Task 0 on this provisional basis. This is not a claim that all Phase 4 quality targets passed, nor authorization to silently enable experimental warnings.

## Not accepted

The retrieval gate failed selection: only **4/10** positive question pairs passed, below the required **8/10**. `ENTAILMENT_GATE_ENABLED = False` stays in place. No retrieval-gate quality pass is claimed.

## Shipping limitation

The claim scorer marks many correct claims unsupported. **`CITATION_CHECK_ENABLED = False` remains the default.** Ordinary answers show no scorer warning block and do no NLI checking. Explicit advisory checks and offline reports preserve the original answer and all page references; they do not strip citations or rewrite prose. Live warning quality is not accepted. Re-enabling the signal requires separate evidence and maintainer approval.

## Reviewed sample

Ten complete saved test answers were selected without scorer scores or flags: five per book, one per direct/paraphrase/followup/contradiction/broad kind, using the first case ID within each kind.

The maintainer authorized a targeted finalization: retain the lower-risk correct first-pass judgments and check only the seven incorrect and two unclear rows against their passages. Those nine rows were passage-checked; the remaining 56 correct judgments were retained, not independently re-audited. The claim inventories were consolidated and their scoped review flags recorded. This is an explicitly provisional review, not an independent human passage audit of every claim.

| Book | Correct judgments retained | Incorrect, passage-checked | Unclear, passage-checked | Non-factual | Factual-error rate |
| --- | --- | --- | --- | --- | --- |
| SICP | 23 | 3 | 1 | 0 | 3/26 = 11.5% |
| Think Python | 33 | 4 | 1 | 2 | 4/37 = 10.8% |
| Combined | 56 | 7 | 2 | 2 | **7/63 = 11.1%** |

Denominator: determinate factual claims (`correct + incorrect`), excluding the two unclear and two non-factual rows from 67 inventory rows. The estimate applies to these ten kind-stratified answers; it is not a population-wide or causal before/after estimate. The earlier scorer-selected warning table measures flag precision, not answer hallucination rate. New schema-2 report fields distinguish scorer vote rates from factual-error measurements.

## Known issue and evidence limits

- False-premise rejection is unreliable: **1/2** selected contradiction answers reject the premise. This answer-level failure is recorded separately from atomic claim counts and remains open for Phase 6.
- The 38-case live experimental capture completed with unchanged stores and no check timeouts. Those reports preceded final parser/window repairs and default-off policy; regression tests cover the repairs. No fresh end-to-end generation run is implied by this decision.
- Model/scorer identity: local `qwen3.5:4b-q4_K_M`; CPU `cross-encoder/nli-deberta-v3-xsmall` revision `a150876415327c80daeff35ca6f68f5ed8cf5c24`, 70,831,107 parameters. Existing selected thresholds remain experimental values, not a new acceptance guarantee.
- Local review records, raw book text, personal labels and evaluation JSON remain private/ignored. The detailed nine-row audit and report hashes are retained in the maintainer workspace.

## Conditions for Phase 5

Keep both gate/check defaults off and preserve the advisory-only contract. Keep the known error rate, two unclear rows, selective-audit scope and 1/2 premise rejection visible in subsequent acceptance reports. Phase 5 implementation still follows its reviewed plans; this record clears the advancement prerequisite, not the remaining design review or Phase 5 acceptance gates.

## Verification before integration

`uv run --no-sync pytest -q`: **732 passed, 10 skipped**, 17 snapshots passed (61.67 s). Scoped production Ruff/Black and `git diff --check` passed. Inventory arithmetic, the nine-row audit count, saved-report hashes, review-scope flags, relative links and both disabled defaults were checked. Skipped host/model cases are not newly accepted by this record.
