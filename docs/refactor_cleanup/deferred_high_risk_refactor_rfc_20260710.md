# RFC: Deferred High-Risk Refactor and Deletion Gates

Date: 2026-07-10
Status: proposed migration contract; no runtime behavior change
Scope: public ECG data helpers, duplicated active-index metadata, and PN2021-C
metadata/CLI compatibility surfaces

## Decision

Do not delete these surfaces in the current cleanup. Use a staged deprecation
and proof process, then delete only when the replacement is replayable and the
locked K500, preprocessing, mapping, and evaluation contracts remain identical.

This RFC records the high-risk work intentionally excluded from the current
low/medium-risk compression. It does not authorize a public API break, change
PN2021-C evaluation semantics, or promote historical artifacts to trusted
evidence.

## Why this is deferred

The five candidate data modules expose 39 names through `ecg_adv_gen.data` and
contain 1,282 lines. Only nine exports have a current repository production
consumer, but zero in-repository references are not enough to prove that a
public API is safe to remove. Direct tests for these helpers were also removed
historically: 31 tests and 791 test lines no longer protect their behavior.

Two nominally unused implementations remain active source-of-truth entries,
and the orphan raw1000 materializer is the only remaining implementation that
can create inputs frozen into the current golden mainline contract. PN2021-C
metadata and CLI fields are bound to four active evaluation stages, real
registry artifacts, and the version-1 golden argv snapshot.

## Alternatives considered

1. Delete every zero-consumer export now. This maximizes immediate line
   reduction but can silently remove external API and artifact-creation
   capability. It is rejected.
2. Freeze all surfaces indefinitely. This avoids compatibility risk but leaves
   approximately 850–1,100 net removable lines and two competing sources of
   truth. It is rejected as a long-term architecture.
3. Deprecate, migrate, prove, then delete. This keeps the current evidence
   readable while turning implicit behavior into tested contracts. This is the
   selected approach.

## Candidate inventory

| Surface | Current LOC | Current fact | Estimated eventual gross reduction |
|---|---:|---|---:|
| `ecg_adv_gen/data/kshot.py` | 374 | Five live ref-ID/K500 consumers; about 12 public names have no production consumer | 250–265 |
| `ecg_adv_gen/data/kshot_artifacts.py` | 323 | Path construction is active; ref-meta/relabel branch has no consumer | 200–210 |
| `ecg_adv_gen/data/pn2021_raw_kshot.py` | 159 | No consumer, but only remaining creator for golden `.raw1000.npz` inputs | 165–170 |
| `ecg_adv_gen/data/pn2021_records.py` | 131 | No consumer, still indexed as active source of truth | 140–150 |
| `ecg_adv_gen/data/real_anchors.py` | 295 | Explicit record-ID loader is live; implicit K selection is not | 110–115 |
| `configs/active_scripts.yaml:managed_experiments` | 101 YAML lines | Duplicates the 10 latest-mainline stages but is read by audit/tests | 50–75 net |
| PN2021-C metadata and public CLI | model-specific runners/adapters | Four active stages and registered artifacts depend on current shapes/flags | 65–330 net, depending on scope |

After required replacements and tests, realistic aggregate net reduction is
approximately 850–1,100 lines. This estimate is not a completion metric and
must not justify skipping migration evidence.

## Target architecture

### 1. One narrow K500 reference contract

Create a package-owned typed ref-set boundary that owns:

- exact ref-meta parsing and center validation;
- ordered record IDs, count, and SHA-256;
- canonical K/seed/center path construction;
- include/exclude semantics used by clean and PN2021-C evaluation.

Migrate the five live consumers from `kshot.py` to this boundary. Keep artifact
selection, relabeling, and proportional sampling out of the runtime API.

### 2. One managed target-anchor preparation stage

Before deleting the raw1000/relabel code, provide a CPU-managed preparation
stage that accepts versioned ref metadata and emits:

- raw1000 signals in PTB-XL lead order with no premature z-score;
- labels, ordered record IDs, and ref-ID digest;
- mapping version/hash and preprocessing contract ID;
- run card, v2 file index, command, environment, and integrity hashes.

The current golden inputs must be reproducible through that stage or explicitly
remain historical artifacts with a provenance record.

### 3. Explicit-only real-anchor loading

Make `selected_ids` mandatory for every active runner and manifest. Migrate
path discovery to the canonical artifact-group object. Deprecate implicit
proportional selection and fallback-root APIs only after array alignment,
missing-ID, center, ordering, and K500 hash tests exist.

### 4. One normalized active experiment index

Make `latest_mainline.stages` canonical. Derive launcher, runner entrypoint,
adapter, mapping, and active status from each resolved configuration. Retain
only genuinely non-derivable metadata such as role, evidence scope, and a short
method-family label. During migration, load both old and new shapes and assert
their normalized views are identical; remove `managed_experiments` only after
10/10 audit and golden equivalence.

### 5. Versioned PN2021-C metadata before compatibility removal

Define a typed, versioned metadata structure containing mapping, class order,
preprocessing mode/order, crop/target length, center, cache version, ref count
and hash, corruption profile/order, and z-score timing. Readers may migrate
legacy registered artifacts; writers emit only the canonical version. Remove
legacy aliases only after redacted real-artifact fixtures prove migration and
all registered artifacts remain readable.

### 6. Typed evaluation spec before CLI deletion

Keep EfficientNet raw-first and ECGFounder bottleneck5000 evaluator behavior
separate. First extract a typed spec for the 20 truly common fields and prove
legacy argv-to-spec equality for all four active PN2021-C stages. Public flag
deletion or evaluator-loop unification is a later migration that requires a
golden-contract version bump and preserved historical commands.

## Required tests before deletion

- Ref-meta list/object shapes, duplicates, wrong center, wrong K, wrong seed,
  ordering, count, and SHA-256.
- Exact include-then-ref-exclude behavior for all four target centers.
- Raw1000 shape, sample rate, lead order, label order, no premature z-score,
  and deterministic record ordering.
- Explicit real-anchor missing-ID, center isolation, latent/label/signal
  alignment, and exact selected-ID hash.
- Real legacy clean/corrupt PN2021-C metadata fixtures migrated to one canonical
  version, including negative ref-count/hash cases.
- PN2021 and PN2021-C all-zero-kept/drop-all-zero views remain distinct.
- All 10 latest-mainline configurations validate, render, and pass managed
  audit with unchanged normalized golden output unless a reviewed version bump
  is intentional.

## Migration and deletion gates

No candidate deletion may merge until all applicable statements are proven:

1. Active source-of-truth entries point to the replacement.
2. Required golden inputs have a managed creator and integrity record.
3. Live consumers use the canonical boundary and produce identical IDs,
   hashes, waveform contracts, and commands.
4. Removed top-level exports have completed a documented deprecation interval
   or an external/user-script inventory proves no consumers.
5. Registered artifacts remain readable or have an explicit offline migrator.
6. Focused tests, the full CPU suite, golden check, registry/workspace audit,
   artifact guard, and `git diff --check` pass.
7. No `model/*`, data, run, cache, or checkpoint artifact is staged.

## Rollback rule

Each migration phase must be a separate commit. If normalized commands,
record-ID hashes, waveform behavior, metadata views, or registered artifact
readability differ unexpectedly, revert only that migration phase and keep the
legacy surface. Do not compensate by changing the locked protocol or expected
metrics.

## Proposed implementation order

1. Restore the smallest active-contract fixtures from the deleted helper tests.
2. Add the managed raw1000/target-anchor preparation record.
3. Introduce and migrate to the narrow ref-set/artifact-group APIs.
4. Deprecate zero-consumer top-level data exports.
5. Normalize `latest_mainline` and remove the duplicate experiment table.
6. Version and migrate PN2021-C metadata.
7. Introduce the typed PN2021-C evaluation spec.
8. Delete compatibility branches only after every deletion gate is green.
