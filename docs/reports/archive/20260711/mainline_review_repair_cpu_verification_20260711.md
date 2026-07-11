# Mainline Review Repair And CPU Verification

Date: 2026-07-11
Repository: `/home/linbinhao/ECG_rebuild_review_20260711`
Branch: `codex/mainline-review-repair-20260711`
Implementation HEAD before this report: `e4fe280`
Mapping: `v7_super5_sjr_rgq_review_20260528 / 555ec85d5b51`

## Outcome

Tasks 1-7 are mechanically implemented and CPU-verified. The branch is ready
for a seed-20260601 one-center smoke and Task 8 config preflight, but it is not
yet mechanically ready for the final three-seed launch and does not contain new
experimental evidence for F-001, F-002, F-004, or F-005.

The paper-safe status is:

| Finding | Code | CPU tests | Experimental evidence |
|---|---|---|---|
| F-001 matched primary comparison | implemented | whole-branch verified | pending |
| F-002 A0/A2/A3/A4/A5 matrix | implemented | whole-branch verified | pending three paired seeds |
| F-003 corruption scope | resolved for narrowed claim | whole-branch verified | claim scope narrowed; no unseen-family claim |
| F-004 measurable target objective | implemented | whole-branch verified | pending rho runs and floors |
| F-005 anchor-collapse repair | provisional implementation | whole-branch verified | pending real geometry/attack gates |

No old Direct-K500, locked-rawfirst, selector, or single-seed artifact was
promoted into the new matched evidence scope. The active evidence registry was
left unchanged.

## Implementation Series

| Task | Terminal commit | Mechanical result |
|---|---|---|
| Managed protocol fields | `46327cc` | source lineage, selection, hull/attack knobs, resume identity |
| Matched contract and compression | `666815f` | A0/A5 source/split/budget/source-floor contract with reduced duplication |
| F-004 objective | `21cbe00` | explicit target clean/adversarial objective and realized accounting |
| F-005 geometry | `89b4ecc` | balanced provisional recipe and strict initial/final diagnostics |
| ECGFounder resume | `8d1bebb` | atomic unit progress, identity/digest guards, deterministic reconstruction |
| Five-arm runtime | `d71d2a4` | canonical A0/A2/A3/A4/A5 executable component semantics |
| Managed matrix/claim | `e4fe280` | 20-command stages, narrowed claim, paths, reporting, golden |

## Final Managed Contract

The current seed-20260601 EfficientNet train, PN2021 clean, PN2021-C S5, and
PN2021-C depth-2/3 stages each resolve to exactly 20 deterministic commands:

```text
4 target centers x [A0, A2, A3, A4, A5]
```

- A0: matched Direct-K500 baseline; no VAE, raw chains, view BCE, or JSD.
- A2: raw AugMix clean-third control; view BCE and JSD; no VAE/PGD.
- A3: VAE-LHAT only; no raw-chain route, view BCE, or JSD.
- A4: VAE-LHAT plus raw three-chain and view BCE; JSD disabled.
- A5: full VAE-LHAT plus raw three-chain, view BCE, and JSD.

All arms share the verified PTB-XL source checkpoint, deterministic K500
split/selection, source floor, optimizer budget, preprocessing, F-005 defaults,
and ref-exclusion policy. Every clean/S5/depth consumer points to its exact
producer directory and `best_model.pt`.

This tracked/golden surface hard-locks `seed/subset_seed=20260601`, and managed
CLI overrides intentionally reject paper-protocol/K-shot seed changes. Task 8
must add tracked per-seed configs or an atomic seed-by-center-by-arm mechanism
that keeps train and all three consumer paths aligned before launching
`20260531` and `20260611`.

The claim resolver permits train/eval overlap only under
`known_family_corruption_robustness`. Official depth-2/3 inputs are described as
compositions of the same five represented corruption families. Atomic order,
composite family membership, global/child trace fields, and fail-closed empty,
unknown, duplicate, reordered, and wrong-scope cases are tested.

The generic train postprocessor is disabled for the center-by-case layout. The
dedicated clean center-by-arm stage is the sole paper-table producer. A real
20-artifact sentinel passes through metrics export and paper-table aggregation
and proves five arm rows with four centers each.

## Verification Evidence

All temporary test and audit outputs were placed under `/dev/shm`.

| Gate | Result |
|---|---|
| `util/tests` | 602 passed, 13 dependency warnings, 0 failed |
| `methods/augmix/tests` | 35 passed, 0 failed |
| Total CPU pytest | 637 passed, 0 failed |
| Task 6 focused suite | 277 passed, 13 dependency warnings |
| Independent Task 6 re-review | APPROVED, 0 Critical / 0 Important / 0 Minor |
| Managed config audit | 10/10 passed; protocol/command audits passed |
| Latest-mainline golden | check passed |
| Golden SHA-256 | `da4a4a4dc8af7f331ea6f078137c1a8f891ae94a2f7cedd165214b4b5e3bc5a9` |
| Golden/generator size | 1,715 / 200 lines |
| Config parsing | 31 YAML and 5 JSON files parsed |
| Python compilation | `ecg_adv_gen` and `scripts` compileall passed with pycache in `/dev/shm` |
| External models | passed, error_count=0, two non-blocking warnings |
| Full workspace audit | passed; active scripts and registry passed |
| Diff whitespace | passed |

The Matplotlib/PyParsing warnings are existing dependency deprecations.

The full workspace audit reports eight non-blocking historical warnings:

- five old managed-run indexes reference deliberately absent
  `last_model.pt`/`env.json` artifacts and are not replay-ready;
- the old Direct-versus-threechain comparison bundle is deprecated;
- DeepECG and advdiff targets exist but are not Git checkouts.

These warnings are preserved as evidence debt. They must not be hidden by
editing old run indexes. Task 8 should create fresh matched run indexes and a
new comparison bundle.

The audit was run before committing this ledger/report and therefore correctly
reported `handoff_readiness.status=attention_required`,
`ready_for_commit=false`, and `ready_for_handoff=false` for the untracked
findings document. This is a pre-commit state, not a clean-handoff assertion;
the audit must be rerun after the documentation commit.

## Findings Ledger Import

The user-owned findings file in the dirty main worktree was not edited. It was
read only after confirming SHA-256:

```text
80a8f9f66f41b7373aa12245ea9be6e2bdeb93a6407a02651555314e4e991590
```

A copy was imported into the isolated branch, then updated to separate Code,
Tests, and Evidence states. Historical diagnoses and acceptance gates were
retained. Two real non-blocking implementation debts were also retained:

- `include_clean_anchor_in_batch`, `partner_policy`,
  `forbid_norm_abnormal_mix_main`, and `rare_class_anchor_quota` remain partly
  declarative metadata;
- the ASR high bound reaches config/audit/resume but has no independent
  training-control action.

## One-Day Review Surface

The complete active first-party Python closure is too large for a credible
one-day line review:

| Surface | Files | Review SLOC |
|---|---:|---:|
| Scientific production | 79 | 20,018 |
| Full active first-party closure | 109 | 29,868 |

The reviewable surface is the bounded Tasks 1-6 delta:

| Delta layer | Files | Added | Deleted |
|---|---:|---:|---:|
| Scientific production Python | 14 | 2,113 | 368 |
| Launch/config production Python | 7 | 571 | 52 |
| Human-authored config/evidence | 11 | 207 | 100 |

The generated golden JSON is not a manual review target; review its 200-line
generator and run `--check`.

Recommended one-day order:

1. active comparison boundary, five-arm table, F-005 defaults, claim resolver,
   and run naming;
2. target objective and A0/A2/A3/A4/A5 runtime semantics;
3. latent-hull geometry and strict diagnostics;
4. ECGFounder atomic resume block;
5. matrix adapters, producer-consumer paths, reporting identity, and golden;
6. focused tests, audits, and findings ledger.

The largest safe review-surface improvement is to separate
`latest_mainline.primary` matched EfficientNet from historical Direct and
optional ECGFounder sections. This would remove about 5,127 physical scientific
lines from the default review closure without deleting reproduction code.

Further true low-risk compression is modest, roughly 120-220 lines:

- share the 39 overlapping explicit Latent-Hull/AugMix CLI parameters;
- consolidate pure float/signal/crop/z-score helpers;
- reuse exact-argv adapter append helpers.

Do not unify the EfficientNet and ECGFounder training loops or generalize the
atomic resume engine before reproduction evidence passes; those changes carry
more semantic risk than review benefit.

## Remaining Task 8 Gates

1. Preflight the frozen seed-20260601 managed surfaces without changing the
   five-arm contract.
2. Run a one-center smoke and require finite F-004/F-005 mechanism diagnostics.
3. Add and audit tracked per-seed configs or one atomic
   seed-by-center-by-arm expansion for `20260531`, `20260601`, and `20260611`.
   Train, clean, S5, and depth-2/3 must resolve matching producer-consumer paths
   and must remain fail-closed against partial or cross-seed plans.
4. Add and preflight a separate full-topology rho 0/0.25/0.5 sweep. Only rho
   may change; do not use A0/A2 versus A3/A4/A5 as the F-004 causal comparison
   and do not cross this sweep with the five-arm matrix.
5. Add and preflight a separate F-005 balanced-versus-neighbor-only
   (`hull_lambda=1.0`) matched control before claiming F-005 evidence closure.
6. Run A0/A2/A3/A4/A5 for paired seeds `20260531`, `20260601`, and `20260611`.
7. Evaluate ref-excluded PN2021 clean, official S5, and depth-2/3 for both
   all-zero-kept and drop-all-zero views.
8. Verify PTB-XL source and target-clean floors, real post-projection effective
   anchor share, `atk_init`, `atk_anchor`, loss gain, ASR, decode validity, and
   per-arm optimizer budgets.
9. Build a new matched comparison bundle with paired deltas, variability, and
   uncertainty against A0.
10. Keep ECGFounder A0/A3/A5 optional unless a backbone-independent claim is
   explicitly pursued with its own matched contract.

No GPU was used during Tasks 6-7 verification, and no branch was pushed.
