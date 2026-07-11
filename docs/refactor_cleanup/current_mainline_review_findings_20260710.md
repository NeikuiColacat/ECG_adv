# Current Mainline Review Findings

Date: 2026-07-10

## Purpose

Track verified design, protocol, and implementation problems found while
walking through the current managed mainline. This is a review ledger, not an
experiment-result source. Each finding records Code, Tests, and Evidence
separately. Mechanical implementation or CPU tests do not constitute
experimental closure; paper-facing resolution requires the finding-specific
managed run records and ref-excluded artifacts.

Each finding tracks three independent states:

- `Code`: `OPEN`, `PARTIAL`, `MECHANICALLY_IMPLEMENTED`, or
  `MECHANICALLY_RESOLVED_FOR_NARROWED_CLAIM`.
- `Tests`: `NOT_RUN`, `TARGETED_CPU_VERIFIED`, or
  `WHOLE_BRANCH_CPU_VERIFIED`.
- `Evidence`: `EVIDENCE_PENDING`, `FAILED_GATE`, `EVIDENCE_VERIFIED`, or
  `CLAIM_SCOPE_NARROWED` for a deliberately narrowed non-experimental claim.

`BLOCKING` is a paper-evidence gate. It does not mean that the corresponding
code is still missing.

## Current repair queue

Imported from the user-owned main-worktree copy with source SHA-256
`80a8f9f66f41b7373aa12245ea9be6e2bdeb93a6407a02651555314e4e991590`.
The source file was not edited in place.

- Overall status: `READY_FOR_ONE_SEED_SMOKE_AND_TASK8_CONFIG_PREFLIGHT`
- Overall mechanical status: `TASKS_1_TO_7_MECHANICALLY_VERIFIED`
- Experimental status: `EVIDENCE_PENDING`
- Paper claim status: `BLOCKED_PENDING_TASK_8_PAIRED_ARTIFACTS`

All findings and protocol decisions in this document are retained as the
authoritative repair queue. Do not use the current EfficientNet VAE-LHAT run as
paper-facing main-method evidence until the blocking items below are resolved.

| Priority | Item | Mechanical status | Remaining gate |
|---|---|---|---|
| P0 | F-001 | implemented | fresh matched A0/A5 artifacts and deltas |
| P0 | Module-1 adapter audit | paper-critical fields implemented; residual declarative metadata noted below | consumed by F-001/F-004/F-005 evidence gates |
| P0 | F-005 | implemented provisionally | real post-projection anchor share and attack diagnostics |
| P0 | F-004 | implemented | matched rho 0/0.25/0.5 artifacts and source/clean floors |
| P1 | F-002 | implemented as A0/A2/A3/A4/A5 | three paired seeds and complete comparison bundle |
| P1 | F-003 | narrowed mechanically | keep all wording within known-family scope |

Locked decisions now implemented mechanically:

- PTB-XL source-checkpoint initialization;
- VAE-LHAT adversarial waveform as the third chain;
- K500-internal validation for checkpoint ranking, with PTB-XL used only as a
  hard source-floor rejection check;
- simultaneous all-zero-kept and drop-all-zero reporting;
- balanced Latent-Hull provisional managed candidate: `M=20`, no anchor
  inside the candidate set, uniform coefficient initialization,
  `hull_lambda=0.60`, five inner steps, and `hull_lr=0.25`;
- neighbor-only `hull_lambda=1.0` remains an unpromoted ablation until
  experimentally validated.

No item above is considered fixed merely because it appears in this ledger.
Resolution requires the acceptance criteria under the corresponding finding.

## F-001: Matched EfficientNet primary comparison contract

Status:

- Code: `MECHANICALLY_IMPLEMENTED`
- Tests: `WHOLE_BRANCH_CPU_VERIFIED`
- Evidence: `EVIDENCE_PENDING`, `BLOCKING`

### 2026-07-11 mechanical update

Commits `eb9bb55..666815f`, followed by the canonical arm/runtime work in
`d71d2a4` and managed matrix/consumer contract in `e4fe280`, now bind matched
arms to the same PTB-XL source lineage, deterministic K500 internal split,
optimizer budget, target selection metric, source floor, preprocessing, and
arm-named output paths. Historical Direct-K500 remains explicitly unmatched
provenance and cannot be relabeled as A0. Fresh run evidence is still absent.

Affected surfaces:

- `configs/experiments/effnet_direct_k500_v7_sjr_rgq.yaml`
- `configs/experiments/effnet_vae_lhat_augmix_threechain_locked_k500.yaml`
- `configs/defaults/effnet_matched_f005_locked.yaml`
- `ecg_adv_gen/config/adapters/direct_finetune.py`
- `ecg_adv_gen/config/adapters/effnet_vae_lhat.py`
- `ecg_adv_gen/runner/effnet_direct_finetune.py`
- `docs/pipelines/pn2021c_vae_lhat_augmix_locked_protocol_20260618.md`

### Historical mismatch that opened this finding

The historical Direct K500 baseline started from the PTB-XL EfficientNet source checkpoint,
trained for 30 epochs on an 80/20 K500 train/validation split, and selected
`best_model.pt` by K500 validation AUPRC.

The pre-repair VAE-LHAT plus three-chain AugMix configuration started from that
already target-adapted Direct K500 `best_model.pt`, trains for another 30
epochs, uses all K500 records for training, and evaluates `last_model.pt`.

The comparison therefore changes more than the proposed method. It also changes:

- initialization;
- total optimization steps and K500 exposure;
- K500 train/validation usage;
- checkpoint-selection policy.

Any observed delta can contain gains from additional target fine-tuning and
selection differences, so it cannot be attributed solely to VAE-LHAT or the
three-chain AugMix topology.

This also conflicts with the locked EfficientNet protocol, which states that
the Direct K500 and VAE-LHAT/AugMix variants use the same PTB-XL Super5 source
initialization and aligned full-fine-tuning semantics.

### Required primary comparison

Run both arms from the same PTB-XL source checkpoint:

```text
PTB-XL source checkpoint
├── Direct K500 full fine-tuning
└── K500 + VAE-LHAT + locked three-chain AugMix full fine-tuning
```

Keep the following identical between arms:

- backbone and source checkpoint;
- K500 record IDs and seed;
- epochs, optimizer steps, batch size, learning rate, scheduler, and trainable scope;
- preprocessing and target-real normalization;
- checkpoint-selection rule;
- PN2021/PN2021-C ref exclusion and reporting views.

The only intended difference is the VAE-LHAT plus three-chain augmentation and
its directly associated loss terms.

### Optional staged-method ablation

If Direct K500 initialization is retained as a separate staged method, compare
two matched continuation arms:

```text
Direct K500 checkpoint
├── clean-only K500 continuation for the same additional budget
└── VAE-LHAT + three-chain continuation for the same additional budget
```

Label this as a staged continuation ablation, not as the primary matched
baseline comparison.

### Acceptance criteria

Criteria 1-2 below are mechanically implemented. Criteria 3-4 remain the
experimental gate before this finding can become `EVIDENCE_VERIFIED`:

1. Managed YAMLs encode matched initialization, K500 identity, training budget,
   and selection rules for the primary Direct and method arms.
2. Dry-run manifests prove those fields match except for the intended method
   components.
3. Both arms produce complete run records and ref-excluded PN2021/PN2021-C
   evaluation artifacts under the same mapping version/hash.
4. Reported deltas are computed only between the matched arms.

### Existing evidence status

`configs/active_evidence_registry.yaml` already prohibits the historical
EfficientNet comparison from paper use because of an unmatched K500-exclusion
protocol. The initialization and training-budget mismatch recorded here is an
additional independent reason not to use the historical delta as a main-method
claim.

## F-002: Paper-complete matched baseline and ablation matrix

Status:

- Code: `MECHANICALLY_IMPLEMENTED`
- Tests: `WHOLE_BRANCH_CPU_VERIFIED`
- Evidence: `EVIDENCE_PENDING`, `BLOCKING`

### 2026-07-11 mechanical update

Commits `d71d2a4` and `e4fe280` define one canonical A0/A2/A3/A4/A5 component
table and an exact 4-center by 5-arm managed matrix for train, PN2021 clean,
official S5, and depth-2/3 evaluation. All 20 consumers resolve to their
producer directory and `best_model.pt`. Per-arm reporting is protected by an
end-to-end 20-artifact sentinel. No paired multi-seed experiment evidence has
yet been produced, and the currently tracked surface is locked to seed
`20260601`; Task 8 must add an auditable multi-seed surface before the final
three-seed launch.

Affected surfaces:

- `configs/active_scripts.yaml`
- `configs/experiments/`
- `configs/defaults/effnet_matched_f005_locked.yaml`
- `configs/defaults/vae_lhat_defaults.yaml`
- `ecg_adv_gen/config/adapters/effnet_vae_lhat.py`
- `ecg_adv_gen/runner/synth_online_at_super5.py`
- `docs/pipelines/pn2021c_vae_lhat_augmix_locked_protocol_20260618.md`

### Historical gap that opened this finding

The pre-repair EfficientNet mainline exposed a Direct K500 baseline and the full
VAE-LHAT plus locked three-chain AugMix method, but it did not expose a complete
set of matched managed experiments that isolates the contribution of each main
component.

The minimum paper-facing matrix is:

| ID | K500 supervision | VAE-LHAT | Raw corruption chains | JSD consistency | Purpose |
|---|---:|---:|---:|---:|---|
| A0 | yes | no | no | no | Matched Direct K500 baseline |
| A2 | yes | no | yes | yes | Raw AugMix / clean-third-chain control |
| A3 | yes | yes | no | no | VAE-LHAT-only contribution |
| A4 | yes | yes | yes | no | Three-chain method with view BCE, without JSD |
| A5 | yes | yes | yes | yes | Full method |

A1 is intentionally absent because the active primary design is not a staged
Direct-K500 continuation. PTB-XL source-only remains a reference row, not a
sixth K500 training arm.

The active comparison surface includes this context row, which does not use
K500 training:

- PTB-XL source-only EfficientNet evaluation.

If the paper claims backbone-independent effectiveness, ECGFounder must at least
repeat the matched Direct K500, VAE-LHAT-only, and full-method rows with the same
K500 identities, corruption protocol, trainable scope, and selection rule.

### Required comparison contract

For every paired row, keep the following fixed:

- source initialization and backbone;
- K500 record IDs and subset seed;
- epochs, optimizer steps, batch size, learning rate, scheduler, and trainable scope;
- preprocessing and normalization;
- hyperparameter-selection budget and checkpoint-selection rule;
- PN2021/PN2021-C ref exclusion, mapping version/hash, and evaluation code.

The final main rows require at least three paired seeds. Report mean and standard
deviation, identify the variability source, and report a confidence interval for
the paired method-minus-baseline delta when the saved predictions support it.

Required result views are:

- PTB-XL source floor;
- PN2021 clean all-zero-kept and drop-all-zero, ref-excluded;
- PN2021-C corrupted absolute AUROC/AUPRC and clean-to-corrupted drop;
- per-center, per-class, and per-corruption results;
- attack diagnostics including `atk_init`, `atk_anchor`, `loss_gain`, ASR,
  decoded-invalid rate, and hull-weight entropy.

### Implemented component order

1. A0 is the matched Direct-K500 primary baseline.
2. A2 is the raw AugMix clean-third control and bypasses VAE/PGD.
3. A3 isolates VAE-LHAT and bypasses raw corruption chains.
4. A4 keeps VAE-LHAT, raw chains, and augmented-view BCE while disabling JSD.
5. A5 enables the full VAE-LHAT plus raw-chain plus JSD topology.
6. The remaining action is the final paired-seed matrix, not more arm wiring.

### Acceptance criteria

Criteria 1-2 are mechanically implemented. Criteria 3-4 remain required for
`EVIDENCE_VERIFIED`:

1. Every required row has a tracked managed YAML and appears in the active
   experiment index with an explicit baseline or ablation role.
2. Dry-run manifests prove that paired rows differ only in the intended
   components.
3. Main rows have paired multi-seed run records and complete required metrics.
4. The comparison bundle computes deltas only against the matching baseline
   scope and records confidence/variability metadata.

## F-003: Training and PN2021-C evaluation use the same corruption families

Status:

- Code: `MECHANICALLY_RESOLVED_FOR_NARROWED_CLAIM`
- Tests: `WHOLE_BRANCH_CPU_VERIFIED`
- Evidence: `CLAIM_SCOPE_NARROWED`; unseen-family claims remain prohibited

### 2026-07-11 mechanical update

Commit `e4fe280` adds a fail-closed operator resolver, validates global and
per-child train/eval/overlap sets, and permits overlap only for the exact scope
`known_family_corruption_robustness`. Atomic S5 order is exact; official
depth-2/3 composite tokens reduce to the same declared five-family set. The
active index, generated golden, manifest trace, and locked pipeline all use
known-family wording. The evidence registry was not changed.

Affected surfaces:

- `configs/experiments/effnet_vae_lhat_augmix_threechain_locked_k500.yaml`
- `configs/defaults/pn2021c_official_s5_locked_protocol.yaml`
- `configs/defaults/pn2021c_official_s5_depth23_composite_protocol.yaml`

### Verified overlap

The full method trains its raw AugMix chains with:

- `powerline_noise`;
- `emg_noise`;
- `baseline_wander`;
- `baseline_shift`;
- `random_leads_masking`.

The primary PN2021-C evaluation uses the same five corruption families. The
current experiment can therefore support a claim about robustness to known
corruption families, but it cannot by itself establish robustness to unseen
corruption types.

### Chosen correction

The active protocol chose option 1 below. Options 2-3 remain future work:

1. Narrow claim: explicitly describe PN2021-C as evaluation on corruption
   families represented during training.
2. Generalization claim: add leave-one-corruption-family-out experiments, where
   each held-out operator is absent from training and used only for evaluation.
3. Strongest claim: also add an external or newly defined corruption family that
   is not used by any training augmentation.

Do not present ordinary same-operator severity changes as evidence of unseen
corruption-family generalization.

### Acceptance criteria

The narrowed-claim branch is mechanically complete because the first condition
is enforced by config, manifest, golden, tests, and pipeline wording. It does
not create unseen-family evidence:

- the paper claim and all table captions are narrowed to known-family
  corruption robustness; or
- tracked leave-one-family-out/unseen-family configs and paired evaluations are
  complete, with train/test operator separation recorded in each run manifest.

## F-004: Measurable target-domain adversarial contribution

Status:

- Code: `MECHANICALLY_IMPLEMENTED`
- Tests: `WHOLE_BRANCH_CPU_VERIFIED`
- Evidence: `EVIDENCE_PENDING`, `BLOCKING` for a learned hard-sample claim

### 2026-07-11 mechanical update

Commits `92acabe` and `21cbe00` implement the explicit target objective
`(1-rho) * L_clean + rho * L_adv`, keep source loss separate, make rho
independent of source-row count, skip the adversarial model forward at rho 0,
and record realized group counts/contributions. Runtime/adapter semantics for
rho 0, 0.25, and 0.5 are validated and resume-checked; the tracked canonical
matrix currently executes only rho 0/0.5, so the separate 0.25 sweep surface
still belongs to Task 8. CPU tests establish semantics only; real contribution,
source floor, and attack quality still require Task 8 runs.

Affected surfaces:

- `configs/defaults/vae_lhat_defaults.yaml`
- `configs/defaults/effnet_matched_f005_locked.yaml`
- `configs/experiments/effnet_vae_lhat_augmix_threechain_locked_k500.yaml`
- `ecg_adv_gen/training/losses.py`
- `ecg_adv_gen/training/online_buffer.py`
- `ecg_adv_gen/runner/synth_online_at_super5.py`
- `ecg_adv_gen/runner/effnet_vae_lhat.py`

### Historical gap that opened this finding

The pre-repair `adv_weight=0.3` was a sampler weight, not an adversarial-loss
coefficient. With 17,418 PTB-XL source records at weight 1, 500 target records
at weight 80, and a 2,048-record adversarial buffer, its main-sampler
adversarial probability has an upper bound of approximately:

```text
(2048 * 0.3) / (17418 * 1 + 500 * 80 + 2048 * 0.3) = 1.06%
```

Quality-aware buffer weights reduced the realized probability further. The
separate latent-AugMix consistency stage previously contributed only three
optimizer batches per epoch, so the nominal YAML values do not establish a
literature-comparable adversarial-training contribution.

### Locked repair target

The paper-facing target is **at least 50% effective adversarial contribution
inside the target-domain adaptation objective**, preferably implemented as at
least one adversarial counterpart for every clean target sample. This follows
the clean/adversarial 1:1 or 50/50 structure used by canonical mixed
adversarial-training objectives and ECG adversarial distillation training.

This target does not require adversarial samples to exceed 50% of the combined
PTB-XL source plus target workload. Report these separately:

- clean/adversarial contribution within the target adaptation objective;
- realized global clean/source/target/adversarial sampler proportions;
- realized per-stream loss contributions after all coefficients and warmup.

Do not satisfy this finding by increasing `adv_weight` alone. Before increasing
adversarial exposure, the generated samples must be non-degenerate: log
`atk_init`, reduce the near-one-hot anchor initialization, and verify meaningful
`loss_gain` and hull-weight utilization. Keep a PTB-XL source floor because the
current target-heavy continuation already reduces source performance.

### Required comparison

Use the same initialization and training budget for a minimal paired ablation:

| Target-domain adversarial contribution | Role |
|---:|---|
| 0% | matched clean continuation |
| 25% | low-pressure control |
| 50% | minimum paper-facing target |
| >50% | optional high-pressure stress ablation |

This must be a separate full-topology rho sweep in which VAE-LHAT, raw chains,
view BCE, JSD, source initialization, split, and budget are identical and only
rho changes. A0/A2 versus A3/A4/A5 is not an F-004 causal comparison because
multiple method components change at once. The rho sweep must replace, not
cross-product with, the canonical five-arm matrix.

### Acceptance criteria

The objective and logging surfaces in criteria 1-2 are implemented. Their real
values, together with criteria 3-4, remain required for `EVIDENCE_VERIFIED`:

1. The managed full-method objective gives adversarial examples at least 50%
   effective contribution within target-domain adaptation after warmup.
2. Run artifacts record realized sampler proportions and realized loss
   contributions rather than only nominal weights.
3. `atk_init`, `atk_anchor`, `loss_gain`, hull entropy/top-1 weight, and decoded
   validity show that the adversarial branch is non-degenerate.
4. The matched 0%/25%/50% comparison and PTB-XL source-floor evaluation are
   complete under the same K500 identities and ref-excluded protocol.

## Protocol decisions confirmed on 2026-07-10

The following choices are now encoded in the locked EfficientNet experiment
contract:

1. The primary VAE-LHAT plus three-chain method starts from the PTB-XL source
   checkpoint, not from a Direct K500 checkpoint.
2. The third chain uses the VAE-LHAT adversarial waveform. A clean third chain
   remains an ablation only.
3. Checkpoint ranking uses a split internal to the fixed K500 subset. PTB-XL
   source performance may be used only as a hard rejection floor, not as an
   additional positive ranking signal. Held-out PN2021 labels remain forbidden.
4. PN2021 and PN2021-C report both all-zero-kept and drop-all-zero views.

These decisions are mechanically implemented and covered by managed-command,
resume, golden, and producer-consumer tests. Commits `46327cc..666815f`,
`d71d2a4`, and `e4fe280` replace the old Direct-K500 initialization and
last-checkpoint behavior. Fresh experiment evidence remains pending.

## Module-1 mechanical audit

Audit scope:

```text
YAML inheritance
-> resolved config
-> typed adapter
-> managed wrapper argv
-> training/evaluation child commands
-> dry-run manifest and expected artifacts
```

### Checks that passed

- Defaults, experiment YAML, local host config, CLI overrides, and interpolation
  have an explicit deterministic merge order.
- Scientific commands are built by typed adapters; raw `runner.argv` is not an
  accepted managed launch surface.
- The matched train, clean, S5, and depth-2/3 stages each expand to exactly
  four centers x A0/A2/A3/A4/A5 = 20 deterministic commands with distinct
  run-id-scoped child directories.
- K500 seed/path identity, v7 mapping version/hash, target-reference exclusion,
  and the two all-zero reporting views are present in the managed dry-run.
- The source-of-truth audit found no missing latest-mainline stage config and no
  staged or tracked artifact-policy violation.
- Fresh whole-branch CPU verification at `e4fe280` plus this ledger update
  passed:

  ```text
  util/tests:          602 passed, 13 dependency deprecation warnings
  methods/augmix:       35 passed
  total:               637 passed, 0 failed
  managed configs:      10/10 passed
  golden:               SHA-256 da4a4a4dc8af7f331ea6f078137c1a8f891ae94a2f7cedd165214b4b5e3bc5a9
  YAML/JSON parse:       31/5 passed
  package/script compileall: passed
  ```

The warnings are from Matplotlib/PyParsing and do not affect the protocol
audit.

### Historical failures and current disposition

| Pre-repair failure | Current mechanical disposition |
|---|---|
| Direct-K500 checkpoint was constructed in the adapter | verified PTB-XL source lineage is passed through the managed contract |
| K500 selection existed only as metadata and consumers used `last_model.pt` | K500-internal selection plus PTB-XL source floor selects canonical `best_model.pt` |
| `include_anchor` was forced true | true/false behavior is adapter-tested; F-005 uses false |
| consistency enablement was ignored | canonical arm table controls view BCE and JSD independently |
| `init_logit_gap=4` was hidden in Python | YAML owns the value; F-005 locks 0 |
| PGD radius and attack thresholds were runner defaults | epsilon and both bounds reach child/audit/resume; the low threshold drives abort, while the high bound has no separate training-control action |
| anchor source/clean-anchor/rare-quota policy booleans were treated as executable | `include_clean_anchor_in_batch`, `partner_policy`, `forbid_norm_abnormal_mix_main`, and `rare_class_anchor_quota` remain declarative metadata debt |
| `atk_init` was null | three-logit diagnostics and strict-finite summaries are required for VAE arms |

Module-1 is `MECHANICALLY_IMPLEMENTED` for the paper-critical fields. The
residual declarative policy metadata and ASR high-bound control semantics stay
open as non-blocking engineering debt. Its tests prove the executable config
contract, not the real experiment acceptance gates in F-001/F-004/F-005.

## F-005: Latent-hull anchor-collapse repair

Status:

- Code: `MECHANICALLY_IMPLEMENTED` for the provisional balanced recipe and
  diagnostics
- Tests: `WHOLE_BRANCH_CPU_VERIFIED`
- Evidence: `EVIDENCE_PENDING`, `BLOCKING` for a meaningful hard-sample claim

### 2026-07-11 mechanical update

Commit `89b4ecc` locks the provisional M20/no-anchor/gap0/lambda0.6/five-step/
epsilon2 recipe and records initial/final coefficient geometry, identity-aware
anchor share, projection scale, effective lambda, strict-finite summaries, and
real `atk_init` semantics. Commit `e4fe280` shares the same F-005 defaults
across A0/A2/A3/A4/A5 while requiring hull diagnostics only for VAE arms.
Nominal 40% anchor share is not experimental proof of post-projection behavior.

Affected surfaces:

- `configs/defaults/vae_lhat_defaults.yaml`
- `configs/defaults/effnet_matched_f005_locked.yaml`
- `ecg_adv_gen/config/adapters/effnet_vae_lhat.py`
- `ecg_adv_gen/adaptation/latent_hull_torch.py`
- `adversarial/latent_hull_pgd.py`
- `ecg_adv_gen/runner/synth_online_at_super5.py`

### Historical collapsed behavior that opened this finding

The pre-repair method used:

```text
M = 20
include_anchor = true
hull_lambda = 0.05
init_logit_gap = 4.0  # hidden EfficientNet default
hull_steps = 3
hull_lr = 0.25
```

With the anchor at logit `+4` and the other 19 candidates at `-4`, the exact
initial candidate weights are:

```text
candidate anchor weight = 0.993666578
each other candidate    = 0.000333338
weight entropy          = 0.057020939  # maximum for M=20 is log(20)=2.995732
```

The outer blend then applies:

```text
z_adv = 0.95 * z0 + 0.05 * z_mix
```

Therefore the effective original-anchor coefficient at initialization is:

```text
0.95 + 0.05 * 0.993666578 = 0.999683329
```

The completed Ningbo run confirms that three Adam steps do not escape this
initialization: epoch-level `hull_weight_top1_mean` remains about `0.99`,
entropy is about `0.07-0.09`, `atk_init` is absent, and BCE `loss_gain` is only
about `1.4e-4` to `2.1e-4`. The current output is consequently almost a
reconstruction of the original anchor rather than a meaningful M-sample
adversarial interpolation.

### What M, learning rate, steps, and epochs actually mean

- `M=20` creates 20 mixture logits per attacked ECG. They are per-sample inner
  optimization variables, not persistent model parameters.
- `hull_lr=0.25` is the Adam learning rate for those logits only.
- `hull_steps=3` means each selected anchor receives only three coefficient
  updates before decoding.
- The coefficient logits are reinitialized for every attack batch. They are not
  carried across the 30 classifier-training epochs, so there is no separate
  meaningful "coefficient epoch" count.
- With `K_anchor=300` and `pgd_batch=32`, the historical setting performed roughly
  ten attack batches and 30 inner forward/backward updates per outer epoch. A
  five-step setting would perform about 50 per outer epoch.

### Implemented provisional repaired recipe

The mechanically implemented provisional recipe is:

```text
M: 20
include_anchor: false
init_logit_gap: 0.0       # uniform 1/20 initialization
hull_lambda: 0.60         # nominal original-anchor share 40%
hull_steps: 5
hull_lr: 0.25
pgd_eps: 2.0
neighbor_mode: local_random
neighbor_distance_space: standardized
```

This retains the original ECG as a 40% outer reference but removes it from the
20 optimized candidates. All 20 neighbor coefficients start equally at 5% and
are optimized together. Five steps are a reasonable first training setting;
use a fixed 10-20-step stronger audit to test whether the five-step search is
materially under-optimized.

The L2 projection can shrink the convex-hull move and raise the effective
anchor contribution above the nominal 40%. The runner now exposes `pgd_eps`
and records pre/post-projection scale, effective hull lambda, candidate-anchor
share, and effective original-anchor share with strict finite/row-alignment
checks. Promotion still requires real median effective anchor share below 50%;
nominal YAML values and CPU formulas are insufficient.

### No-anchor alternative

The strict alternative is:

```text
include_anchor: false
hull_lambda: 1.0
init_logit_gap: 0.0
```

This optimizes a convex combination of neighbors only. Keep it as an ablation
until decoded validity, label consistency, source floor, and K500-internal
validation show that removing the anchor does not create manifold intrusion.
For this arm, use exact multi-hot label partners first; the current broad
`compatible` pool can mix unrelated abnormal classes and is too permissive
when the anchor no longer dominates.

The recommendation follows the ordinary Mixup/Manifold Mixup principle that
training examples may be convex combinations rather than near-copies of one
privileged endpoint, while retaining Local Mixup's warning that distant or
label-incompatible interpolation can create contradictory virtual samples:

- https://arxiv.org/abs/1710.09412
- https://arxiv.org/abs/1806.05236
- https://arxiv.org/abs/2201.04368
- https://arxiv.org/abs/2003.02484

### Minimal comparison before promotion

Keep initialization, K500 split, training budget, and target-domain
adversarial contribution matched across:

| Hull arm | Effective anchor target | Role |
|---|---:|---|
| historical near-one-hot | approximately 100% | diagnosed failure/control |
| balanced outer anchor | below 50% | recommended candidate |
| neighbor-only | 0% | stress ablation |

Do not sweep M, learning rate, steps, lambda, and anchor policy simultaneously
beyond these three predeclared arms. First verify coefficient entropy/top-1,
`atk_init`, `atk_anchor`, `loss_gain`, projection activity, decoded validity,
K500-internal validation, and PTB-XL source floor.

### Acceptance criteria

Criteria 1-2 are mechanically implemented. Criteria 3-4 require real artifacts
before promotion:

1. `include_anchor`, `init_logit_gap`, `pgd_eps`, and selection policy are
   operational managed-config fields with adapter tests proving both true and
   false/non-default behavior.
2. Final mixture diagnostics report initial and final coefficients, effective
   anchor share after projection, entropy/top-1, and `atk_init`.
3. The promoted recipe has non-degenerate coefficient utilization and median
   effective anchor share below 50%.
4. The balanced and neighbor-only arms use the same PTB-XL initialization,
   K500-internal selection protocol, outer training budget, and ref-excluded
   evaluation views.

## Supporting non-finding repairs completed

- `8d1bebb` makes ECGFounder PN2021-C evaluation interruption-safe with atomic
  progress sidecars, identity/digest/checkpoint guards, deterministic resume,
  and byte-identical final public JSON reconstruction.
- `d71d2a4` defines the canonical matched EfficientNet runtime semantics for
  A0/A2/A3/A4/A5, including A4 augmented-view BCE without JSD and A2's
  VAE/PGD bypass.
- `e4fe280` wires the exact 20-command managed train/clean/S5/depth matrix,
  fail-closed known-family claim trace, canonical producer-consumer paths,
  per-arm paper-table identity, and generated golden contract.

## Verification and evidence boundary

The Task 7 CPU gate passed 637 tests with zero failures, managed-config audit
passed 10/10, workspace/latest-mainline audit passed, golden regeneration check
passed, all tracked YAML/JSON parsed, and package/script compilation passed.
The full workspace audit passes with eight non-blocking historical warnings:
five old managed-run indexes still reference intentionally absent
`last_model.pt`/`env.json` artifacts, the old comparison bundle is deprecated,
and DeepECG/advdiff targets are not Git checkouts. ECGTwin, ECGFounder, the
registry, and the active managed config/command paths pass their checks. These
warnings are not repaired by relabeling old artifacts; the new reruns must
produce fresh run indexes and a new matched comparison bundle.

These results establish executable and reporting contracts only. They do not
upgrade F-001/F-002/F-004/F-005 to experimental evidence. Task 8 must still
produce the predeclared paired-seed artifacts, real F-004/F-005 mechanism
diagnostics, source/clean floors, ref-excluded clean/S5/depth metrics, and
comparison-bundle uncertainty statistics. F-004 additionally requires a
separate matched full-topology rho 0/0.25/0.5 sweep; F-005 requires balanced
versus neighbor-only (`hull_lambda=1.0`) matched artifacts if the provisional
recipe is promoted. The current managed/golden train and three consumer stages
are hard-locked to seed `20260601`, and paper-protocol seed override is
forbidden. Task 8 must first add tracked per-seed configs or one atomic
seed-by-center-by-arm expansion whose producer/consumer paths remain identical
for `20260531`, `20260601`, and `20260611`. The active evidence registry remains
unchanged until those artifacts exist.
