# ECG_rebuild Refactor Goal Completion Report

Date: 2026-07-10
Branch: `refactor/data-module-20260620`
Goal baseline: `fc5766299bdeaeb0e53bf3e9a9f255d7bb7ce557`
Implementation snapshot: `6e4e4cf13810f1ce986b5d4475a1bd7fe432019a`
Upstream base after explicit fetch: `afec2b883106b0e05339f62cb62476b967301e45`
Push status: **not pushed**

This report closes the evidence integration, P0 repair, behavior-preserving
compression, one-day review-pack, run-integrity, and verification work started
from `fc57662`. Resolve the commit containing this file without a
self-referential hash using:

```bash
git log -1 --format=%H -- docs/refactor_cleanup/refactor_goal_completion_20260710.md
```

## Outcome

- ECGFounder method commit `b57ee16` is in branch history.
- Evidence source commit `02f7cef` is present exactly once as patch-equivalent
  local commit `b799e13`; that commit changes only the registry and archived
  evidence report.
- ECGFounder depth2+3 remains a single-seed provisional result:
  `0.8161 / 0.5163`, with `+4.48 / +7.53 pp` over matched direct fullFT. It is
  not presented as a multi-seed paper conclusion.
- Startup P0 failures are closed, including the copies/JSD contract, stale
  source-of-truth paths, and legacy-directory assumptions.
- Ten low/medium-risk compression commits remove a raw net 821 LOC without
  changing the locked mapping, K500, preprocessing, checkpoint, or metric-view
  contracts.
- The algorithm review surface is organized into 15 files and 5,885 reviewed
  LOC: 3,008 LOC deep review plus 2,877 LOC guided invariant review.
- High-risk public API, data-preparation, active-index, and PN2021-C schema
  changes are deferred behind an explicit RFC and deletion gates.
- Managed-run and active-evidence integrity are fail-closed and independently
  reviewed. Only the complete direct run is replay-ready.
- No GPU workload was launched and no commit was pushed.

## Goal commits

The 21 implementation/evidence commits after the baseline are:

| Commit | Role |
|---|---|
| `b799e13` | Integrate the exact ECGFounder evidence patch. |
| `4e14ada` | Clear ECGFounder startup blockers. |
| `7ba7ab8` | Add shared RAM/SSD guardrails to `AGENTS.md`. |
| `ffb0e97` | Complete the P0 startup-blocker boundary. |
| `98a5f7d` | Deduplicate EfficientNet postprocess config. |
| `a0bc283` | Collapse config-audit shims. |
| `769de43` | Remove the unused target-split helper. |
| `c13a195` | Inline ECGFounder pass-through helpers. |
| `ae8c95f` | Remove synth-checkpoint pass-throughs. |
| `7bd501b` | Remove the unused PTB-XL helper API. |
| `e1aad61` | Reuse shared process streaming. |
| `1afacc9` | Lock the 10-stage latest-mainline golden contract. |
| `3b8a127` | Reuse canonical data helpers. |
| `8a6237c` | Lock preprocessing behavior and reduce duplicate YAML. |
| `328562f` | Lock the ECGFounder adapter-to-runner preflight contract. |
| `88d4f41` | Remove unused adversarial entrypoints. |
| `0a2fa09` | Trim unused AugMix surface. |
| `a4757d6` | Lock PN2021-C metadata compatibility. |
| `500bf34` | Keep all-zero reporting views distinct. |
| `a883315` | Add schema-v2 managed-run integrity snapshots. |
| `6e4e4cf` | Enforce active-evidence Registry integrity. |

The documentation closeout commit is the commit containing this report and the
one-day pack/RFC/handoff updates. Use the `git log` command above for its exact
hash.

## Compression accounting

### Compression-only changes

| Commits | Raw net LOC |
|---|---:|
| `98a5f7d`, `a0bc283`, `769de43`, `c13a195`, `ae8c95f` | -331 |
| `7bd501b`, `e1aad61`, `3b8a127` | -94 |
| `88d4f41`, `0a2fa09` | -396 |
| **Total** | **-821** |

The same 15 algorithm files selected for the review pack changed from 11,609
whole-file LOC at `fc57662` to 11,410 LOC, a net reduction of 199. The smaller
number is the honest core-surface result because P0 fixes added necessary
runtime checks inside some selected files.

### Whole-Goal changes

At implementation snapshot `6e4e4cf`, the entire Goal range changes 47 files
with 5,757 insertions and 1,282 deletions, net +4,475 lines. The increase is
primarily golden data, behavioral tests, Registry/run-record integrity code,
and evidence documentation. It must not be described as a repository-wide net
reduction. The final documentation-only delta is reported below.

## One-day review surface

The authoritative review order and exact ranges are in
`docs/refactor_cleanup/one_day_core_review_20260710.md`.

| Scope | Files/LOC | Meaning |
|---|---:|---|
| Selected core files | 15 files / 11,410 whole LOC | Complete implementation files in the call graph. |
| Tier 1 | 3,008 LOC | Line-by-line mechanism review. |
| Tier 2 | 2,877 LOC | Guided review against golden/adapters/invariant tests. |
| One-day reviewed total | 5,885 LOC | In the requested 4.5k-6k window; not all deep review. |

The locked call graph starts at `configs/active_scripts.yaml:latest_mainline`,
passes through `scripts/run_experiment.py` and typed adapters, and reaches the
EfficientNet/ECGFounder training and PN2021/PN2021-C evaluators. Evidence and
reporting tooling are intentionally outside the algorithm LOC.

## Evidence and reproducibility state

### ECGFounder provisional evidence

- Artifact-integrity manifest SHA-256:
  `f4a102770b673d1074540688400f9f2630e3b863908b78a19ea7df66dbe4e9ce`.
- Integrity scope: 33 selected artifacts totaling 1,129,751,679 bytes.
- Provenance-gap SHA-256:
  `a28b084a5a6e5eb8b89caa7e5c0df54750988998240f8041f526360fd7c1d3f4`.

The gap record preserves the missing exact launch command, artifact-proven Git
SHA/dirty patch, managed run card/index, and multi-seed replication. Integrity
coverage does not turn this one-off run into a replay.

### Managed records

| Surface | Critical files | Missing | Replay state |
|---|---:|---:|---|
| EfficientNet direct K500 | 45 | 0 | verified, replay-ready |
| EfficientNet three-chain | 53 | 4 | historical evaluation only |
| ECGFounder source fullFT | 13 | 2 | artifact missing |
| ECGFounder K500 fullFT | 32 | 5 | artifact missing |
| ECGFounder VAE three-chain | 35 | 8 | artifact missing |
| ECGFounder PN2021-C S5 | 31 | 5 | historical evaluation only |

The five incomplete historical records retain surviving evaluation evidence
but cannot be represented as byte-identical replay. Regenerating missing
checkpoints would require new GPU work and was outside this Goal.

## Verification evidence

All commands were CPU-only; transient output and pytest state were placed under
`/dev/shm/ecg-rebuild-goal-20260710/`.

| Gate | Result |
|---|---|
| Complete project pytest (`util/tests`, `methods/augmix/tests`) | 414 passed, 13 third-party deprecation warnings, 44.50 s |
| Registry + run-record contract suite | 114 passed |
| Independent Registry robustness review | PASS; truncated/non-UTF-8 artifacts return structured issues |
| Handoff/index focused tests | 13 passed |
| Latest-mainline golden check | PASS, SHA-256 `df5d6597167e3eb27b63054ed0e62e823b43c5a48f037c048e8bb6c307b404e1` |
| Active managed-config audit | 10/10 passed |
| Live evidence/workspace audit | passed; Registry `error_count=0`, managed 6/6 |
| YAML parse | 6 managed records, 1 active claim |
| `git diff --check` / cached check | passed |

The pre-closeout workspace audit had 13 expected warnings: five historical
non-ready records, five guarded dirty external-model handles, two external
targets that are directories rather than Git checkouts, and one dirty
source-of-truth warning while closeout documentation was uncommitted.

## Deferred high-risk work

`docs/refactor_cleanup/deferred_high_risk_refactor_rfc_20260710.md` inventories
five data modules totaling 1,282 LOC and 39 exported names. Nine exports have a
current production consumer. A future 850-1,100 net-line reduction is plausible
only after managed raw1000 creation, typed reference-set boundaries, external
consumer/deprecation evidence, normalized active-index migration, and
versioned PN2021-C metadata/spec tests. No such high-risk deletion was silently
performed here.

## Preserved dirty state and artifact boundary

The Goal does not own the following local state and never staged it:

```text
model/DeepECG
model/ECGTwin
model/advdiff
model/ecg_ptbxl_benchmarking
model/ecgfounder
docs/superpowers/
```

No dataset, run directory, checkpoint, weight, cache, generated sample, or
external-model handle is part of a Goal commit. Exact-path staging and cached
artifact checks were run before each local commit.

## Remote and future-push boundary

The upstream ref was refreshed immediately before closeout:

```text
origin/refactor/data-module-20260620 = afec2b883106b0e05339f62cb62476b967301e45
```

At implementation snapshot `6e4e4cf`, the relation was ahead 63 / behind 0.
The complete future-push expression is:

```bash
git log --oneline afec2b883106b0e05339f62cb62476b967301e45..HEAD
```

This range intentionally includes 42 commits already local before the Goal as
well as the Goal commits. No part of the range has been pushed by this work.

## Final post-documentation verification

This section is updated after committing the review pack, RFC, handoff,
evidence amendment, and initial report. It records the final committed-state
SHA, ahead/behind count, whole-Goal diff, audit warnings, and repeated CPU
gates.
