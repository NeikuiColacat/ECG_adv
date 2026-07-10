# Task 4 report: downgrade invalid evidence and close K500 lineage

## Status

DONE

Baseline: `b20ea7eae22c600ac14ef5297fee1a48e505115c`

## Result

- Closed the current-K500 -> target-adapted-init -> evaluation-exclusion lineage at launch, evaluator, and registry-audit boundaries.
- Explicitly allows source-only initialization; target-adapted initialization must match the current center, K, record-id hash, and recorded seed when present.
- Preserved absolute metrics, but deprecated the protocol-invalid EfficientNet claim/method/bundle and ECGFounder support entry with `paper_use: prohibited_protocol_invalid`.
- Renamed matched comparative deltas to `observed_unmatched_not_for_claim`; no union K>500 protocol and no new status enum were introduced.
- Updated the 2026-07-09 ECGFounder report with the observed K500 identities and leakage counts: Ningbo 488, Chapman-Shaoxing 456, CPSC 2018 445, Georgia 466; total 1855/2000.

## TDD evidence

RED covered the missing ECGFounder exclusion CLI, actual-exclusion mismatch, target-adapted-init mismatch, launch preflight mismatch, and registry artifact-identity mismatch. The source-only control already passed and remained supported.

GREEN/final verification:

- `python -m pytest -q` over the five Task 4 test files plus config-loader and latest-mainline golden tests: `300 passed, 13 warnings`.
- `scripts/agent/build_latest_mainline_golden.py --check`: passed; SHA256 `b59903cf95b6a6907075363e0ab034bad1617bce197f7c5bf9ef88b98fede67d`.
- CPU-only registry audit in `/dev/shm/task4_registry_audit_final`: `passed: true`, `error_count: 0`, `warning_count: 14`.

The warnings are existing historical run-integrity, deprecated-bundle, guarded external-model, and dirty-worktree warnings; none is a new Task 4 correctness failure.

## Ponytail review

Production-code diff was reduced from `+306/-6` to `+281/-6` lines while retaining the fail-closed checks.

The two public metadata APIs are both necessary production seams:

- `validate_target_init_k500_identity(...)` is shared by launch preflight and the ECGFounder evaluator, so both fail before GPU/model loading without duplicating identity semantics.
- `evaluation_k500_identities(...)` is consumed by the evidence-registry audit to extract the actual recorded evaluation exclusion identity independently of current YAML/ref metadata.

No run-record schema expansion, Task 2/3 topology change, model-handle edit, union exclusion, or speculative abstraction was added.

## Files changed

- K500 identity helpers and clean-metadata preservation under `ecg_adv_gen/evaluation/` and the PN2021-C runner.
- Launch, ECGFounder adapter/evaluator, and registry fail-closed gates.
- Active evidence registry, latest-mainline golden contract, and the archived ECGFounder evidence report.
- Focused regression tests in the five Task 4 test files.

## Concerns

No blocking concern. Pre-existing dirty `model/*` handles and unrelated untracked planning documents were left untouched and must remain unstaged.
