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
| SimCLR | `direction/traditional-simclr` | Checked out in `clean` after explicit cleanup approval; same source tree |

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

## Retirement candidates — completed in two separately authorized batches

The user explicitly replied “同意清理” to the 47-file and 15-idle-worktree
batch on 2026-09-12. The first 15 directories below were removed by ordinary
`git worktree remove`, without force; the final occupied directory was excluded
from that first approval and retired only after the later authorization below.
Prefixes are relative to
`/home/linbinhao/`. Their commits remain in Git refs and the recovery bundle;
local unique files remain in the verified snapshot.

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

## Pre-retirement audit and approved clean-file batch

Read-only audit: `retirement_readonly_audit_20260912_v2.json` under the
external validation root. On 2026-09-12, all 16 candidate HEADs,
indexes, patches and 5,114 files/links matched the initial snapshot.
Three retained local model links do not point into the candidate trees.

Fifteen candidates had no observed current-user process references. The
competitor worktree `workspaces/ecg_locked_backbone_port_20260828` still had
nine bash working directories: PIDs 439337, 552921, 634103, 637030, 637041,
775881, 775884, 4138030, 4174925. At this first gate those terminals required
release; termination had not been authorized. The audit covered 63 owned processes and
664 file-descriptor links; some surfaces of sd-pam, a zombie bash, and sshd
were unreadable. Recheck live state before any approved retirement.

The approved switch of `ECG_manual_refactor_clean` from `a1892b2` to
`direction/traditional-simclr` at `0e8d305` removed the following **47 previously
tracked files**. All are backed up and verified absent after the switch.
This was a normal Git checkout of the already validated source, not a force
checkout, fresh code deletion, experiment rerun, or change to historical refs.

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

## First approved execution — final tree still protected at this point

Transaction artifacts under the external validation root:
`approved_cleanup_20260912_v1/{plan.json,journal.jsonl,result.json}`.
The immutable per-file plan SHA256 is
`6602b90134c782503521c959f559b629afe5f228a3159d2ac473b58309e3ae48`.

Before each action, the transaction revalidated the exact target, archive
hash, file identities/content/ownership and current-user process references.
The 15 retired trees contained 4,834 backed-up files/links, excluding the
explicitly disposable cache categories. Their directory entries and Git
worktree registrations are gone. All Git refs were unchanged, protected
source and the occupied tree matched the pre-action inventory, and all three
local model links retained their targets. No process was stopped; data,
weights, report services and recovery archives were outside the deletion scope.

Post-cleanup full CPU regression from the SimCLR checkout: **558 passed,
14 skipped**, 95.24 seconds, with the existing TypedStorage warning only
(`post_cleanup_cpu.xml`). All three checkout-local dry-runs passed using
fresh non-created output paths; none loaded data, a model or a GPU
(`post_cleanup_dry_runs.json`). `git fsck --connectivity-only --no-dangling`
passed and `git worktree prune --dry-run --verbose` found no stale records.

The first batch left **three active direction worktrees plus one occupied
historical worktree**. It did not establish the final three-tree goal or
authorize removal of `workspaces/ecg_locked_backbone_port_20260828`.

## Final authorized retirement — exactly three worktrees

The user then explicitly requested closing the identified tmux windows and
continuing the consolidation goal. The final transaction lives under
`final_cleanup_20260912_v1/` in the external validation root.

- Revalidated all 280 old-tree files/links against the original SHA-verified
  snapshot; saved a new immutable per-file/pane plan with SHA256
  `fb306af993c83d42163c30c2743b2addaa218f5487b725eb3964bb2818e5c14c`.
- Closed only `cmp_locked_r0` panes `%39,%40,%41,%42,%44,%47,%49,%50,%51`:
  eight idle bash windows (nine shell processes including one nested shell)
  and one already-dead window. No training or Codex child was present.
  The other 17 panes, including Codex and report services, were preserved.
  Scrollback is private external recovery material, never staged in Git.
- Preserved six modified source files and nine untracked YAMLs byte-for-byte
  in archive-only commit `0aba16bfb60c9f9efc543ee0350c1b292eeef2c6`, branch
  `archive/locked-competitor-local-20260912`. The original competitor branch
  remains at `db26ed5ad80949b38dcc005e79f1ff3d2e95d9fe`. Three focused CPU
  tests passed in 7.15 seconds; these drafts are not promoted into the active
  directions or revalidated as scientific results.
- Moved only the absolute `model/ECGTwin` symlink into
  `final_cleanup_20260912_v1/local_handles/ECGTwin`. Its external payload was
  untouched. The original link also remains recorded in the initial snapshot.
- After another source/process check and verified all-refs bundle, ordinary
  `git worktree remove` removed the now-clean final old checkout (279 source
  files after moving the link). No force, reset, broad clean or system change.
- Verified exactly the three directories in the scope table remain registered;
  all preexisting refs and all three active checkout contents were preserved.
  `result.json`, `pane_journal.jsonl`, `panes_closed.json`, and
  `before_final_remove.bundle` retain the audit and recovery evidence.

To recover the final drafts, create a separate clone from the verified bundle
and select the archive commit, or extract chosen original files from the
initial tar snapshot. Do not overwrite the live common Git metadata. The
historical missing-checkpoint caveat above still applies; this cleanup did not
reconstruct missing weights, change recipes, or launch training.

Final post-retirement CPU suite: **558 passed, 14 skipped**, 95.44 seconds,
with the same TypedStorage warning (`final_cleanup_20260912_v1/final_cpu.xml`).
Three checkout-local launcher dry-runs passed with fresh non-created output
paths and no data/model/GPU loading (`final_dry_runs.json`). The final local
source commit, equal checkout heads, Git connectivity, all-ref bundle and
bare-clone readback of all 15 archived drafts are recorded in `handoff.json`.
No commits were pushed; no GPU parity or new performance claim is implied.

## Acceptance

- [x] User confirmed all three directions and single-stage JSD parameters.
- [x] Verified recoverable pre-change snapshot and CPU baseline.
- [x] Integrated unique width implementation and added focused fixes/tests.
- [x] Full combined regression and independent code review.
- [x] Reviewed local source commit and three direction branches created.
- [x] All three direction branches checked out.
- [x] User-approved 47-file and 15-worktree cleanup executed and verified.
- [x] Remaining occupied worktree released, re-audited and separately approved.
- [x] Exactly three registered worktrees; old source refs and local drafts recoverable.

No further worktree removal is authorized by this document alone.

## 2026-09-13 reviewed publication and remote archive policy

The user authorized review, commit/push of the three directions, and archival
of the old remote branches. This is publication of the current validated
slices, not completion of every planned architecture simplification.

- Preserve the remote default `main` at `ffe513a3472bc118bdeb04a7a08a28ae65dcb141`.
- Publish `direction/ecg-llm`, `direction/traditional-simclr` and
  `direction/traditional-jsd`; configure each local branch's matching upstream.
- First push annotated tags `archive/20260913/<old-branch>` and verify their
  peeled commit hashes. Only then remove the 12 original branch refs below,
  with exact expected-old hashes and an atomic transaction. Abort if refs move.
- Preserve all old commits, local history, datasets, checkpoints and run snapshots.
  No default-branch change, forced branch update, GPU run or file cleanup is included.

| Old remote branch | Commit protected by the archive tag |
|---|---|
| `archive/thesis-repro-cleanup` | `93c89067fdd2dec6c50a053c742c32aff110250d` |
| `chore/streamlit-uv-migration` | `2f1f4df0ba844889fbbf093ed6fc9f44b591fd66` |
| `codex/mainline-review-repair-20260711` | `df68b6f9eda9586c902e1b2bc8892f3237c782f0` |
| `graduate-project` | `108c6eec68b13c2fe11ddf917e1ccfcaa560eff7` |
| `handoff/k500-data-interface-20260731` | `314559b77bfae3dc950809c9f0c8fdcd2520f690` |
| `mainline/simplified-augmix-lhat-20260805` | `3a39a516420b52c219a782a7b7440f82746f4b90` |
| `paper/vae500-online-at-findings-20260603` | `f6838b1668b703082c32902f6b29a1edfff1fb6a` |
| `refactor/data-module-20260620` | `6e7988ef26e4d30f802bc3f275c12b63dd03b735` |
| `refactor/manual-20260715` | `fb51790dd1bd6ecb986e6bbd5fa341f989efddec` |
| `refactor/manual-clean-v1-20260806` | `a1892b2bd1eebceeae8d89591a58379fe3476e0e` |
| `refactor/paper-kernel-v2-20260811` | `5b117985089e78a503b47006c03a1a33b0aa3864` |
| `refactor/pn2021c-vae-lhat-agent-cleanup-20260630` | `02f7cef37ba9e5603be5810dcc105a69b7d9da91` |

Pre-publication CPU checks: LLM 630 passed / 14 skipped; SimCLR 610 / 14;
JSD 574 / 14. Each had the existing TypedStorage warning. All 27 project-skill
validators and three representative managed launcher dry-runs passed; dry-runs
created no output directories. These are CPU/config checks, not new GPU or
scientific-performance validation.

The reviewed JSD repeat-study config bundle remains development evidence.
Its prior run/config/source snapshots must not be rewritten to the publication
commit. The shared guidance and local code slices are committed separately
from that experiment bundle.

The immutable before-inventory and final execution receipt are retained under
`/home/linbinhao/ECG_adv_data/analytics/remote_publication_20260913_PMm0gT/`.
Consult the receipt or live remote refs for the actual publication outcome;
this section records the reviewed scope and recovery map.

Recovery example (choose an unused local branch name; no force push):

```bash
git fetch origin tag archive/20260913/refactor/paper-kernel-v2-20260811
git switch -c recovered-paper-kernel archive/20260913/refactor/paper-kernel-v2-20260811
```
