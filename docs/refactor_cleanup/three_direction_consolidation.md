# Three-direction consolidation — 2026-09-12

## Confirmed scope

One repository and three long-lived development worktrees:

| Direction | Existing directory under `/home/linbinhao/` | Target branch |
|---|---|---|
| ECG LLM | `ECG_manual_refactor_paper_kernel_v2` | `direction/ecg-llm` |
| Traditional SimCLR | `ECG_manual_refactor_clean` | `direction/traditional-simclr` |
| Traditional JSD | `ECG_manual_refactor_jsd_width_20260911` | `direction/traditional-jsd` |

SimCLR keeps its locked two-stage recipe. JSD keeps the selected single-stage
R18 recipe: JSD coefficient 1.5, supervised loss masses clean 0.05, rotating
corruptions 0.25, two-chain AugMix 0.50, contracted LHAT 0.20. No parameter
search or retraining is part of consolidation. The two-stage JSD-12 width
experiment is retained as auxiliary development evidence.

Shared code is synchronized through Git, never another worktree's imports.
Each tree has one index owner. The common Git directory remains in
`ECG_manual_refactor_clean/.git`; no fourth permanent management checkout is
needed. Direction branches can share a code tree while having distinct focus.

## Recovery checkpoint

Before retained code edits, a private snapshot was created at:

`/home/linbinhao/ECG_adv_data/archives/three_worktree_consolidation_20260912_initial`

- All 19 worktrees: 8,358 regular files / symlinks, 98,621,428 source bytes.
- Every archived regular file SHA256 and symlink target verified by reading
  the archive back. External link targets were not copied or modified.
- Git metadata, per-tree index, staged/unstaged binary patches, complete
  untracked status and an all-refs Git bundle are included.
- Cache exclusions are explicit in `snapshot.json`: Python bytecode and
  pytest/ruff/mypy caches only. Necessary ignored probes and local artifacts
  remain in the private backup, not in the public source commit.
- A bare clone of the bundle succeeded; the original online trainer and
  isolated width coordinator were independently read back and hash-checked.
- Baseline CPU suite: 523 passed, 14 skipped in 87.67 seconds.

Validation output root:
`/home/linbinhao/ECG_adv_data/analytics/three_direction_consolidation_20260912`

Recovery is explicit, not an automatic broad restore: choose the recorded
worktree and files, extract to a fresh user-owned directory, verify against
`snapshot.json`, then review differences before restoring any live file/index.
The metadata archive is not to be untarred over a running shared Git directory.

## First integration slice

- Imported 24 unique JSD-width files and reviewed seven shared-file deltas;
  the original isolated worktree was not overwritten during this integration.
- Fixed result-summary labels by schema; historical manifests remain unchanged.
- Added GPU reservations to the old queue's CPU-loading interval.
- Fixed the ineffective forbidden-source-token test with negative examples.
- Reduced the recipe contract's 19-argument forwarding; all 74 selector
  descriptions and recipe hashes remain equal to the pre-change snapshot.
- Aligned preprocessing mapping closure with the owning YAML directory,
  preserving all existing YAML bytes and checking bundle escape boundaries.
- Added a small non-executable direction catalog and concise collaboration
  guide. Historical configs and replay identities remain available; replay
  requiring weights is subject to the missing-artifact caveat below.
- Made the width-queue closure test use its temporary output directory, so
  completed real experiments cannot make a read-only unit test fail.
- Reused the existing cache JSON reader in split loading: identical function
  body and error behavior, 12 fewer production lines, five new cases.
- Removed stale Git-ignore exceptions for local model handles and legacy
  scripts. No actual model handle, payload or legacy file was removed.

Post-integration CPU suite: **558 passed, 14 skipped**, 91.56 seconds, one
existing PyTorch TypedStorage warning. Evidence: `validation/final_cpu.xml`
below the external validation root. Independent reviews covered recipe AST
equivalence, queue reservation lifecycle, mapping closure, direction links,
and Git artifact safety. All 74 recipe descriptions and hashes are unchanged.
The JSD checkout independently passed the same **558 tests, 14 skipped** in
90.81 seconds (`validation/jsd_checkout_cpu.xml`). Representative LLM, SimCLR
and JSD dry-runs resolved their local bundles without loading data/model/GPU.
All three correctly reported existing historical output directories as a
collision; new execution still requires a fresh run directory. Their outputs
are saved in `validation/direction_dry_runs.json`.
CPU-only checks do not establish GPU or author-model equivalence. No full
benchmark was launched for this code and navigation consolidation.

## Historical artifact caveat

Four sampled historical `last.pt` paths were missing at their registered
locations, while their parent checkpoint directories and manifests exist:
SimCLR main / no-VAE / A1 (EffNet Ningbo) and R19 seed0 baseline (Founder
Ningbo). This is not proof of deletion. Locate cold archives by their recorded
hashes before claiming checkpoint recovery; a newly retrained model must never
be labelled with an old checkpoint hash. Source PTB-XL checkpoints were present
during the planning audit.

## Source checkpoint and checkout status

Local source commit: `e8b92427663f7b2e07e3419af2709393b875aafd`.
It captures the previously uncommitted retained baseline plus this turn's
reviewed integration; its 1,200 changed files are not 1,200 newly implemented
files. The reviewed index contained 1,395 source/evidence files, 4,053,115 bytes.
Nothing was pushed. Data, private drafts, local model handles, intermediate
artifacts, and the two unchanged tracked historical evidence files were not
newly staged. The original index state remains recoverable from the snapshot.

| Direction | Branch | Checkout state |
|---|---|---|
| LLM | `direction/ecg-llm` | Checked out in `paper_kernel_v2`; clean source index |
| JSD | `direction/traditional-jsd` | Checked out in `jsd_width_20260911`; same source tree |
| SimCLR | `direction/traditional-simclr` | Branch prepared; old `clean` checkout not switched yet |

JSD synchronization changed 17 files and added four, with no deletions.
Its old files were checked against the initial snapshot; its resulting index
and worktree were proven equal to the committed source before a normal Git
switch. No reset, force checkout or force staging was used. Direction branches
share code; a branch's focus does not silently switch the experiment recipe.

The full initial staged whitespace check reported 19 historical YAMLs: three
trailing spaces and 16 EOF blank lines. All 19 were byte-identical across
initial snapshot, actual archive member, staged blob, and working file.
They remain unchanged to preserve frozen configuration hashes. A second check
excluding exactly those 19 literal paths passed for all other 1,181 changed
files; no Git setting or attribute weakened the check. Later code/doc diffs
must pass the ordinary check.

## Retirement candidates — NOT approved for removal

The following 16 registered worktrees are historical candidates. Prefixes
below are relative to `/home/linbinhao/`. Their commits remain in Git refs and
the recovery bundle, and local unique files remain in the snapshot.

| Exact directory | Retained purpose |
|---|---|
| `ECG_manual_refactor_augmix_four_ablation_20260818` | AugMix / contrastive controls |
| `ECG_manual_refactor_corruption_ft_test` | A0/A1 baseline provenance |
| `ECG_manual_refactor_latent_augmix_20260818` | Latent mixing experiment |
| `ECG_manual_refactor_lhat_extreme_min_20260815` | LHAT minimum/anchor search |
| `ECG_manual_refactor_lhat_hybrid_rescue_20260819` | LHAT rescue variants |
| `ECG_manual_refactor_lhat_lambda060_teacher_on_20260816` | Teacher / hull ablation |
| `ECG_manual_refactor_lhat_min_ablation_20260814` | LHAT geometry search |
| `ECG_manual_refactor_lhat_raw_alpha050_20260816` | Raw LHAT alpha experiment |
| `ECG_manual_refactor_lhat_step10_lambda060_no_teacher_20260816` | Attack / hull control |
| `ECG_manual_refactor_lhat_step10_rescue_20260816` | Attack step experiment |
| `ECG_manual_refactor_lhat_teacher_off_20260815` | Teacher-off experiment |
| `ECG_manual_refactor_margin_hardview_20260822` | Margin-hard SimCLR |
| `ECG_manual_refactor_stage1_ablation_20260814` | Stage1 controls |
| `ECG_manual_refactor_teacher_ablation_20260814` | Stage2 teacher controls |
| `workspaces/ecg_external_bench_20260811/repo` | MERL external comparison |
| `workspaces/ecg_locked_backbone_port_20260828` | Competitor/parameter-budget experiments |

Before each removal: recheck HEAD, full dirty/ignored identity, source and
artifact references, and this user's process cwd/open files. Verify recoverable
unique contents and obtain the user's explicit batch approval. Use Git
worktree operations, never ordinary `mv`, broad `git clean`, or force removal.
Unchanged historical references keep their original source identity; live
consumers must migrate before an old path is retired. Data roots and report
services are outside this removal scope.

## Live retirement gates

Read-only audit: `retirement_readonly_audit_20260912_v2.json` under the
external validation root. On 2026-09-12, all 16 candidate HEADs,
indexes, patches and 5,114 files/links matched the initial snapshot.
Three retained local model links do not point into the candidate trees.

Fifteen candidates had no observed current-user process references. The
competitor worktree `workspaces/ecg_locked_backbone_port_20260828` still had
nine bash working directories: PIDs 439337, 552921, 634103, 637030, 637041,
775881, 775884, 4138030, 4174925. Those terminals must first move elsewhere;
do not terminate them automatically. The audit covered 63 owned processes and
664 file-descriptor links; some surfaces of sd-pam, a zombie bash, and sshd
were unreadable. Recheck live state before any approved retirement.

Updating the old `ECG_manual_refactor_clean` checkout from `a1892b2` to the
consolidated source would remove the following **47 previously tracked files**.
All are backed up and absent from the new source tree. This checkout is held
at its old commit until the user approves this exact removal batch; the
prepared SimCLR branch already contains the validated new source.

```text
boot_scripts/refit_pn2021_direct.py
boot_scripts/select_pn2021_direct.py
boot_scripts/tune_pn2021_direct.py
configs/experiments/manual_refactor_pn2021_ecgfounder_direct_select.yaml
configs/experiments/manual_refactor_pn2021_ecgfounder_direct_tune_chapman_shaoxing.yaml
configs/experiments/manual_refactor_pn2021_ecgfounder_direct_tune_cpsc_2018.yaml
configs/experiments/manual_refactor_pn2021_ecgfounder_direct_tune_georgia.yaml
configs/experiments/manual_refactor_pn2021_ecgfounder_direct_tune_ningbo.yaml
configs/experiments/manual_refactor_pn2021_ecgfounder_fixed20_refit_chapman_shaoxing.yaml
configs/experiments/manual_refactor_pn2021_ecgfounder_fixed20_refit_cpsc_2018.yaml
configs/experiments/manual_refactor_pn2021_ecgfounder_fixed20_refit_georgia.yaml
configs/experiments/manual_refactor_pn2021_ecgfounder_fixed20_refit_ningbo.yaml
configs/experiments/manual_refactor_pn2021_effnet_direct_select.yaml
configs/experiments/manual_refactor_pn2021_effnet_direct_tune_chapman_shaoxing.yaml
configs/experiments/manual_refactor_pn2021_effnet_direct_tune_cpsc_2018.yaml
configs/experiments/manual_refactor_pn2021_effnet_direct_tune_georgia.yaml
configs/experiments/manual_refactor_pn2021_effnet_direct_tune_ningbo.yaml
configs/experiments/manual_refactor_pn2021_effnet_fixed20_refit_chapman_shaoxing.yaml
configs/experiments/manual_refactor_pn2021_effnet_fixed20_refit_cpsc_2018.yaml
configs/experiments/manual_refactor_pn2021_effnet_fixed20_refit_georgia.yaml
configs/experiments/manual_refactor_pn2021_effnet_fixed20_refit_ningbo.yaml
configs/experiments/manual_refactor_pn2021_effnet_matched_lhat_aux_e2_ningbo.yaml
configs/experiments/manual_refactor_pn2021_effnet_matched_raw_aux_e2_ningbo.yaml
configs/experiments/manual_refactor_pn2021_effnet_matched_vae_reconstruction_aux_e2_ningbo.yaml
configs/train/PN2021_direct_tune.yaml
configs/train/PN2021_matched_lhat_aux_tune.yaml
configs/train/PN2021_matched_raw_aux_tune.yaml
configs/train/PN2021_matched_vae_reconstruction_aux_tune.yaml
configs/train/methods/direct_depth23_fixed20_lhat_aux.yaml
configs/train/methods/direct_depth23_fixed20_raw_aux.yaml
configs/train/methods/direct_depth23_fixed20_vae_reconstruction_aux.yaml
configs/train/methods/exp_augmix_guided_latent_simplex_v1.yaml
configs/train/methods/exp_lhat_as_sixth_branch_v1.yaml
configs/train/methods/exp_lhat_replay_pool_v1.yaml
configs/train/methods/exp_paired_augmix_latent_bridge_v1.yaml
core/methods/executor.py
core/methods/nodes/__init__.py
core/methods/nodes/buffers.py
core/methods/nodes/codecs.py
core/methods/nodes/latent_ops.py
core/methods/nodes/mixers.py
core/methods/nodes/selectors.py
core/methods/nodes/waveform_ops.py
core/pn2021_tuning.py
util/evaluation/direct_baseline_selection.py
util/tensorboard_logging.py
util/visualize_ecg.py
```

## Acceptance

- [x] User confirmed all three directions and single-stage JSD parameters.
- [x] Verified recoverable pre-change snapshot and CPU baseline.
- [x] Integrated unique width implementation and added focused fixes/tests.
- [x] Full combined regression and independent code review.
- [x] Reviewed local source commit and three direction branches created.
- [ ] All three direction branches checked out (SimCLR awaits removal approval).
- [ ] Final retirement audit and user approval of the exact old-tree batch.
- [ ] Exactly three registered worktrees; old refs/artifacts still recoverable.

No worktree removal is authorized by this document alone.
