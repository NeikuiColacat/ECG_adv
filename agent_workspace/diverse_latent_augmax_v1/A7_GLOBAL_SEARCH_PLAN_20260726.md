# A7 VAE-LHAT + AugMix global search

Status: `heldout_tuned_development_only`

## Boundary

- Adaptation data are exactly the center-specific K500 records.
- Ref-excluded PN2021 Clean and PN2021-C are metric-only development feedback.
- Outside-K500 waveforms, labels, class counts, batch-normalization statistics,
  latent pools, pseudo-labels, teacher inputs, and source-replay inputs are
  forbidden.
- Candidate selection uses only the four-center equal-weighted macro AUROC and
  macro AUPRC. It does not use per-center or per-class weights or manual choices.
- Class order and mapping stay
  `CD,HYP,MI,NORM,STTC` and
  `v7_super5_sjr_rgq_review_20260528 / 555ec85d5b51`.
- Results are development evidence, not blind final-test evidence.

## Current narrowed paper-facing comparison

Every promoted candidate must be compared with Direct+fixed20 using the same:

- PTB-XL source checkpoint;
- center-specific K500 population;
- epoch and cosine-scheduler horizon;
- outer learning rate, weight decay, batch size, seed policy, and trainable scope;
- final-checkpoint selection and ref-excluded evaluation contract.

The original A7 stack below is retained only as historical search provenance.
It is superseded for the paper-facing method because raw auxiliary views,
VAE-random, source replay, EMA teachers, residual heads, JSD/SupCon/PCGrad
stacks, class-specific weights and checkpoint blends make causal attribution
unnecessarily difficult.

The current method has only two claimed components:

1. K500-only two-chain AugMix contrastive pre-training;
2. one short exact-label, non-self VAE-LHAT post-training refinement.

Direct+fixed20 is the shared supervised baseline rather than a claimed method
component. SimCLR/VICReg specifies the fixed Stage-1 representation objective;
it is not exposed as another paper contribution or independent ablation axis.
The five classes remain equally treated. No outside-K500 target waveform,
label, class distribution or batch statistic may update the model.

The causal table is exactly:

| cell | Stage 1 | Stage 2 |
|---|---|---|
| A Direct | identity-view contrastive control | identity polish control |
| B AugMix-only | two-chain AugMix contrastive pre-training | identity polish control |
| C VAE-only | identity-view contrastive control | VAE-LHAT polish |
| D AugMix-to-VAE | two-chain AugMix contrastive pre-training | VAE-LHAT polish |

Only a scalar internal to AugMix or VAE-LHAT may change during development.
Changing such a scalar does not add a component, but the final paper ablation
must use one frozen global setting rather than list the whole tuning history.

## Staged search

1. Close and document the completed teacher-soft, hard-label, chain-three and
   consistency VAE coupling tests; do not add another auxiliary objective.
2. Optimize only the simple AugMix Stage-1 budget/strength against its
   identity-view control.
3. Reopen one four-cell factorial only when the new Stage-1 pair advances the
   absolute Clean/PN2021-C frontier.
4. Promote a recipe only when the VAE effect is non-null, Clean and PN2021-C
   both satisfy the predeclared gate, and gains are not concentrated in one
   center.
5. Freeze one global EfficientNet recipe, replay it with at least two seeds,
   then transfer the frozen two-component recipe to ECGFounder. Only backbone
   optimizer LR/batch size may differ.

The requested target is EfficientNet macro AUROC at least +4 pp over its matched
Direct+fixed20 control on both Clean and PN2021-C, with positive AUPRC, while
ECGFounder also improves and remains roughly 2 pp or more above EfficientNet in
absolute AUROC. The controller reports achieved evidence honestly; it never
converts the target into a metric-selection oracle.

## Resource policy

- Use only GPUs verified free immediately before launch.
- Current four-center waves use at most four GPUs concurrently even if six are
  authorized; additional GPUs are used only for independent controls or seeds.
- Keep at least 48 GiB host RAM available, 24 GiB `/dev/shm` free, and 150 GiB
  NVMe free.
- Controller stdout/stderr live under `/dev/shm/a7_global_search_20260726`.
- Persistent managed run records and final checkpoints live under
  `/home/linbinhao/ECG_adv_data/runs/agent_workspace/a7_global_search_20260726`.
- Each training config uses `checkpoint_write_policy: final_only`.

## Completed strict matched screens

All deltas below are four-center macro percentage points versus the same
P00 Direct+fixed20 reference (`LR=1.5e-4`, batch 64, E40/T40).

| candidate | change from A7 base | Clean AUROC/AUPRC delta | PN2021-C AUROC/AUPRC delta |
|---|---|---:|---:|
| P00 | SimCLR 128, auxiliary alpha 1.0 | +0.759 / +0.249 | +0.879 / +0.584 |
| P01 | P00 + auxiliary alpha 1.5 | +0.837 / +0.461 | +0.995 / +0.816 |
| P02 | P00 + SimCLR 256 | +0.790 / +0.227 | +1.195 / +1.310 |
| P03 | SimCLR 256 + auxiliary alpha 1.5 | +0.864 / +0.428 | +1.300 / +1.522 |
| P06 | P03 + Dirichlet/Beta 1.0/1.0 | +0.804 / +0.188 | +1.241 / +1.371 |
| P08 | SimCLR 256; reallocate BCE mass from raw to VAE random/hard | +0.814 / +0.267 | +1.153 / +1.177 |
| P09 | P08; move VAE mass from random to LHAT hard | +0.794 / +0.264 | +1.144 / +1.169 |
| P13 | Direct+fixed20 + AugMix-SimCLR/source-logit-anchor only | +0.516 / -0.521 | +0.864 / +0.584 |
| P12 | P13 + strong VAE-LHAT as actual third AugMix chain | +0.635 / -0.136 | +0.968 / +0.850 |
| P07 | P03 with VICReg replacing SimCLR | +1.106 / +0.662 | +0.992 / +0.792 |
| P11 | P03 with SimCLR warm-up extended from 256 to 512 steps | +0.755 / +0.265 | +1.550 / +2.196 |
| P10 | P03 with Barlow Twins replacing SimCLR | +1.033 / +0.511 | +1.021 / +1.060 |
| P14 | Strong VAE-LHAT third chain + SimCLR 512 | +0.598 / -0.208 | +1.278 / +1.631 |
| P15 | P11 with SSL source-logit anchor raised from 5 to 10 | +0.886 / +0.238 | +1.425 / +1.727 |
| P16 | P11 with VICReg mixed into SimCLR at weight 0.03 | +0.774 / +0.382 | +1.642 / +2.457 |
| P17 | P16 with VICReg mix weight raised from 0.03 to 0.06 | +0.754 / +0.379 | +1.652 / +2.561 |
| P18 | Invalid outer-only D19 LHAT override; audited, not evidence | INVALID | INVALID |
| P19 | P14 strong LHAT third chain + SimCLR512/VICReg0.03 | +0.596 / -0.019 | +1.363 / +1.948 |
| P20 | P16 with K500-only SSL steps raised from 512 to 1024 | +0.748 / +0.570 | +2.018 / +3.093 |
| P21 | P20 with K500-only SSL steps raised from 1024 to 2048 | +0.819 / +0.254 | +2.320 / +3.362 |
| P22 | P21 with K500-only SSL steps raised from 2048 to 4096 | +0.928 / +0.439 | +2.590 / +3.827 |
| P23 | Direct supervised-BCE budget filler for 4096 steps, then the same E40 stage | -2.724 / -2.904 | -2.271 / -1.928 |
| P24 | ECGFounder transfer of P20: SSL1024, SimCLR/VICReg0.03, auxiliary alpha 1.5 | -1.022 / -3.091 | +2.906 / +2.390 |
| P26 | P24 with auxiliary alpha reduced from 1.5 to 1.0 | -1.139 / -3.196 | +2.759 / +2.308 |
| P27 | ECGFounder Direct/P24 weight soup, P24 weight 0.25 | +0.163 / -0.489 | +2.115 / +2.117 |
| P28 | ECGFounder Direct/P24 weight soup, P24 weight 0.50 | -0.283 / -1.511 | +2.824 / +2.670 |
| P29 | ECGFounder Direct/P24 weight soup, P24 weight 0.75 | -0.640 / -2.316 | +3.038 / +2.749 |

P08 is the mechanism-isolating comparison for VAE branch weighting.  Relative
to P02, it changes only the auxiliary BCE allocation from `0.4/0.3/0.3`
(raw/random/hard) to `0.2/0.4/0.4`, preserving total auxiliary BCE mass.
P08 minus P02 is `+0.024/+0.041` pp on Clean but `-0.042/-0.133` pp on
PN2021-C.  Therefore this reallocation is not positive evidence for an
independent VAE contribution.

P09 keeps raw BCE mass at `0.2` and total VAE BCE mass at `0.8`, but changes
the VAE random/hard split from `0.4/0.4` to `0.2/0.6`.  P09 minus P08 is
`-0.020/-0.003` pp on Clean and `-0.010/-0.007` pp on PN2021-C.  This is a
near-exact null result: the current low-radius D19 LHAT-hard branch does not
outperform its paired VAE-random branch.

P03 remains the current frontier.  Its P01/P00 and P02/P00 factorial contrasts
show a clearer benefit from longer AugMix-SimCLR warm-up than from globally
raising the combined auxiliary branch.  These results still do not isolate
AugMix from the complete Target-SSL warm-up or VAE from the other auxiliary
terms.

P06 changes only the Target-SSL mixing distributions from
`Dirichlet(0.5)/Beta(0.5)` to the classic AugMix `1.0/1.0`.  P06 minus P03 is
`-0.060/-0.239` pp on Clean and `-0.059/-0.151` pp on PN2021-C.  The standard
image-domain setting is therefore not promoted and the current `0.5/0.5`
setting remains the local frontier.

## Mechanism escalation after the D19 weighting null

The next paired comparison reuses existing runtimes and changes the coupling
mechanism rather than adding another scalar grid:

1. P13: Direct+fixed20 plus K500-only AugMix-SimCLR/source-logit-anchor.
2. P12: the same SSL256 warm-up and Direct+fixed20 base, plus the existing
   stronger LHAT endpoint (`hull_lambda=0.60`, five hard steps) as the third
   branch of one actual three-chain AugMix composition.

P13 minus P00 measures the complete AugMix-SSL adaptation stage.  P12 minus
P13 measures the additional VAE-LHAT chain contribution under a shared
optimizer/data/epoch contract.  This comparison is simpler and more
attributable than the D19 separate raw/random/hard objective.

P13 confirms that the AugMix-SimCLR/source-logit-anchor stage is useful for
four-center AUROC and PN2021-C AUPRC, but not sufficient by itself: Clean AUPRC
drops by `0.521` pp, and CPSC Clean AUROC/AUPRC drop by
`1.974/1.724` pp.  It is therefore an attribution control, not a promoted
standalone method.

P12 minus P13 is positive on all four pooled metrics:
`+0.119/+0.386` pp on Clean and `+0.104/+0.266` pp on PN2021-C.
Every center also improves on all four metrics in this paired contrast.  The
strong VAE-LHAT chain therefore has consistent independent value in this
coupling, unlike the low-radius D19 hard/random reweighting.  The effect remains
small, however, and P12 still trails Direct by `0.136` pp on Clean AUPRC.
P12 is mechanism-positive evidence, not a target-reaching recipe.

P07 minus P03 is `+0.242/+0.235` pp on Clean but
`-0.308/-0.730` pp on PN2021-C.  VICReg therefore shifts the trade-off toward
Clean rather than improving the combined frontier.  CPSC Clean also remains
below Direct (`-1.340/-0.942` pp), so P07 is not promoted.

P11 minus P03 is `-0.109/-0.162` pp on Clean but
`+0.250/+0.674` pp on PN2021-C.  Longer K500-only AugMix-SimCLR warm-up is the
clearest scalable robustness lever found so far, particularly for robust
AUPRC.  It does not solve CPSC Clean degradation and remains well below the
requested `+4` AUROC target.

P10 minus P03 is `+0.169/+0.084` pp on Clean but
`-0.279/-0.461` pp on PN2021-C.  Like VICReg, Barlow Twins shifts the
trade-off toward Clean without advancing the combined frontier.  SimCLR
therefore remains the selected SSL objective for the next mechanism pair.

P14 minus P12 is `-0.037/-0.073` pp on Clean but
`+0.310/+0.781` pp on PN2021-C, so the SSL-length robustness signal transfers
to the strong three-chain method.  P14 nevertheless trails P11 on all four
metrics (`-0.157/-0.474` Clean and `-0.272/-0.565` PN2021-C).  The strong
VAE-LHAT third-chain coupling remains more attributable but is not the best
performing coupling.

P15 minus P11 is `+0.131/-0.027` pp on Clean and
`-0.125/-0.469` pp on PN2021-C.  The stronger SSL source-logit anchor mildly
repairs CPSC Clean (`+0.206/+0.304` pp relative to P11) but leaves it below
Direct and reduces robustness.  Further anchor-strength gridding is stopped.

P16 minus P11 is positive on all four pooled metrics:
`+0.019/+0.116` pp on Clean and `+0.092/+0.261` pp on PN2021-C.
Mixing a small VICReg regularizer into SimCLR is therefore preferable to
replacing SimCLR with VICReg.  P16 is the new combined frontier, although CPSC
Clean remains below Direct and the AUROC target is still unmet.

P17 minus P16 is `-0.020/-0.002` pp on Clean and
`+0.010/+0.104` pp on PN2021-C.  The doubled VICReg mixture mostly trades a
small amount of Clean for robust AUPRC; AUROC is flat.  P16 remains the balanced
point, P17 the robust-AUPRC point, and the VICReg-mixture axis is stopped.

P18 is invalid and must not enter any scientific comparison.  Its generated
outer method changed `resources.lhat_config`, but D19 constructs its actual
candidate runtime from `contracts.pcgrad_candidate_profile`; that referenced
profile retained the original low-radius compatible runtime.  Four-center P18
model tensors were therefore exactly identical to P16 (571/571 tensors per
center, maximum absolute difference 0).  The controller now rejects this
candidate explicitly.

P19 is the corrected mechanism test.  It does not invent a new D19 runtime.
Instead, it directly reuses the already exercised strong VAE-LHAT third-chain
method from P12/P14 (`hull_lambda=0.60`, five hard steps, epsilon 8) and combines
it with the P16 Target-SSL recipe (`SimCLR512 + 0.03 VICReg`).  Runtime resource
capture must confirm `train/lhat_h060_s5_eps8.yaml` before P19 is launched.

P19 runtime capture confirmed the strong LHAT config and its four-center
diagnostics showed real attack pressure (`loss_gain` about `0.09-0.14`,
`atk_anchor_l2` about `8`, decoded invalid rate `0`).  Relative to P14, P19 is
`-0.003/+0.189` pp on Clean and `+0.085/+0.317` pp on PN2021-C.  The light
VICReg mixture therefore helps strong-LHAT AUPRC and robustness.  P19 still
trails P16 by `0.178/0.401` pp on Clean and `0.279/0.510` pp on PN2021-C.
The strong third-chain branch is retained as attribution evidence but is not
the performance frontier.

P20 scales the current P16 frontier from 512 to 1024 K500-only
AugMix-SimCLR/VICReg steps while keeping the supervised method, loss allocation,
learning rate, epoch budget and all class weights fixed.  This is the next
single-axis test because SSL length has been the clearest robust-performance
lever so far.

P20 minus P16 is `-0.026/+0.189` pp on Clean and
`+0.376/+0.636` pp on PN2021-C.  Relative to Direct it reaches
`+0.748/+0.570` pp on Clean and `+2.018/+3.093` pp on PN2021-C.  All four
centers improve PN2021-C AUROC; Ningbo improves robust AUPRC by `+4.328` pp.
The SSL-length axis is therefore not saturated at 1024 steps.

P21 continues only this axis from 1024 to 2048 steps.  Two thousand forty-eight
steps correspond to about 262 batch-64 exposures over a 500-record pool, which
is within ordinary long-horizon contrastive-learning budgets.  All supervised
training and evaluation contracts remain unchanged.

P21 minus P20 is `+0.071/-0.317` pp on Clean and
`+0.303/+0.269` pp on PN2021-C.  Robustness continues to improve, but the gain
is diminishing and Clean AUPRC now pays a visible cost.  Relative to Direct,
P21 reaches `+0.819/+0.254` pp on Clean and `+2.320/+3.362` pp on PN2021-C.

P22 is the final SSL-length boundary point at 4096 steps.  It keeps every other
scalar fixed.  Stop this axis after P22 unless robust AUROC makes a materially
larger jump without further Clean collapse; otherwise freeze P20 as the balanced
point and P21 as the robust point.

The agent-workspace runtime validation ceiling for
`pretrain_steps_per_epoch` was expanded from 2048 to 4096 only for this boundary
test.  The pretraining loop and cosine scheduler were already total-step based;
no whitelist file, data boundary, checkpoint format or allocation policy was
changed.

P22 minus P21 is `+0.109/+0.186` pp on Clean and
`+0.269/+0.465` pp on PN2021-C.  Relative to Direct it reaches
`+0.928/+0.439` pp on Clean and `+2.590/+3.827` pp on PN2021-C.  The robust
AUPRC target is nearly reached, but robust AUROC remains `1.410` pp short of
the requested `+4` target.  The SSL-length axis stops here: the marginal
robust-AUROC gain from 2048 to 4096 is only `0.269` pp, so another doubling
would be difficult to justify as a mechanism improvement.

Freeze P20 as the balanced EfficientNet point and P22 as the robust EfficientNet
point.  The next stage transfers the same global, equal-class recipe to
ECGFounder under its own matched Direct baseline; no per-center or per-class
hyperparameters may be introduced.

Before transfer, P23 closes the optimizer-step fairness gap.  P00 Direct has
about `40 * ceil(500/64) = 320` supervised steps, whereas P22 adds 4096
contrastive steps before the same E40 training.  P23 therefore reuses the
existing `supervised_bce_budget_filler` runtime for 4096 K500-only supervised
steps at the Direct learning rate, then runs the identical Direct+fixed20 E40
stage.  P22 versus P23 is the budget-matched mechanism comparison; P22 versus
P00 remains only the practical total-recipe comparison.

P23 is substantially worse than P00 on every pooled metric:
`-2.724/-2.904` pp on Clean and `-2.271/-1.928` pp on PN2021-C.  Every center
also degrades on all four metrics.  Therefore the P22 gain cannot be explained
by the claim that any additional 4096 K500 optimizer steps would improve the
model.  P22 minus P23 is `+3.652/+3.343` pp on Clean and
`+4.860/+5.755` pp on PN2021-C, with all four centers positive.

This is a useful training-budget counterexample, not a complete causal
ablation.  P23 uses supervised BCE and the Direct learning rate, while P22 uses
contrastive AugMix-SimCLR/VICReg with a different pretraining optimizer
contract.  Consequently:

- cite P22 versus P00 as the practical total-recipe development result;
- cite P22 versus P23 only as evidence that extra step count alone is
  insufficient;
- do not call the full P22-P23 gap a pure VAE or AugMix effect;
- keep P12-P13 as the paired evidence for the independent strong VAE-LHAT
  third-chain contribution.

## ECGFounder transfer

P24 transfers the complete P20 recipe without per-center or per-class changes:
the locked ECGFounder source checkpoint, `2e-5` full fine-tuning, batch 64,
1024 K500-only AugMix-SimCLR/VICReg steps, equal-five logit anchors, the same
D19 VAE-random/VAE-hard auxiliary views, and E40 supervised training.

Relative to its own newly rerun Direct+fixed20 control, P24 improves PN2021-C
by `+2.906/+2.390` pp AUROC/AUPRC, and all four centers improve both robust
metrics.  Its robust absolute score is `0.867855/0.550925`, which is
`+1.510/+3.149` pp above EfficientNet P22.  The cross-backbone robustness
signal therefore transfers.

P24 is not promotable as a balanced recipe because Clean falls by
`-1.022/-3.091` pp.  Do not escalate directly to the pre-generated 4096-step
P25: longer SSL is likely to amplify the same trade-off.  P26 changes one
scalar only, reducing `pcgrad_auxiliary_alpha` from `1.5` to `1.0` while
keeping SSL1024, source anchors, optimizer, epoch budget, class weights and
all data identities fixed.  Its purpose is to test whether ECGFounder needs a
lighter VAE/raw/JSD auxiliary gradient to retain Clean performance.

P26 does not repair the trade-off.  Relative to P24 it changes
Clean by `-0.117/-0.105` pp and PN2021-C by `-0.147/-0.081` pp.  The auxiliary
weight axis is therefore stopped and P24 remains the ECGFounder robustness
point.  This negative result also indicates that the Clean loss is not caused
simply by setting the VAE/raw/JSD auxiliary multiplier to `1.5`.

P27-P29 interpolate only floating-point model tensors between the per-center
ECGFounder Direct and P24 checkpoints.  Optimizer state is excluded and all
non-floating buffers come from P24.  The same one-dimensional global alpha is
used for every center and class.

The interpolation forms a smooth Clean/robustness Pareto curve.  P27 is the
balanced point: it keeps Clean AUROC slightly above Direct and limits the Clean
AUPRC cost to `0.489` pp while retaining about `+2.12/+2.12` pp PN2021-C.
P29 is the robustness point: `+3.038/+2.749` pp PN2021-C with
`-0.640/-2.316` pp Clean.  P29 dominates the original P24 endpoint on all four
metrics (`+0.383/+0.775` pp Clean and `+0.132/+0.359` pp PN2021-C), but it
still does not satisfy the requested simultaneous Clean improvement or
`+4` robust-AUROC target.

These alpha results remain `heldout_tuned_development_only`: the outside-K500
records never enter training or blending, but their global macro metrics were
used to inspect the alpha curve.  A paper claim would need to freeze one global
alpha and validate it on an independent protocol or seeds.

P25 is the final ECGFounder SSL-length boundary test.  A generated-config diff
against P24 confirmed that, apart from candidate names and output paths, the
only scientific change was `pretrain_steps_per_epoch: 1024 -> 4096`.  The
source checkpoint, equal-five class policy, K500 identities, optimizer
settings, auxiliary branches and supervised E40 budget were unchanged.

The longer warm-up fails decisively.  Relative to the matched ECGFounder
Direct+fixed20 control, P25 changes Clean by `-3.783/-10.522` pp and PN2021-C
by `-0.183/-4.407` pp AUROC/AUPRC.  Relative to P24 itself, it changes Clean by
`-2.761/-7.431` pp and PN2021-C by `-3.089/-6.797` pp.  PN2021-C AUPRC falls
in all four centers; robust AUROC is positive only in Chapman and Ningbo and is
negative in CPSC and Georgia.

All four training and evaluation jobs exited successfully, and the four-center
summary retained `outside_k500_model_access: false`, equal class weights, the
v7 mapping and ref exclusion.  This is therefore treated as K500 contrastive
over-training / representation drift under the fixed `1e-3` SSL learning rate,
not a resource failure or configuration mismatch.  Stop the ECGFounder
SSL-length axis at P25, retain P24 as the original robust endpoint and P29 as
the current robust checkpoint-interpolation point, and do not blend or promote
P25.

P30 fills the missing ECGFounder midpoint at 2048 SSL steps with the same
`1e-3` SSL learning rate.  A P24/P30 config diff again confirmed that only the
step count changed.  Relative to Direct, P30 changes Clean by
`-1.977/-6.680` pp and PN2021-C by `+1.686/-0.774` pp.  Relative to P24, it
changes Clean by `-0.955/-3.589` pp and PN2021-C by `-1.220/-3.164` pp.
Robust AUPRC is below Direct in every center.

The 1024/2048/4096 sequence therefore shows a clear ECGFounder over-training
curve at fixed SSL learning rate rather than a hidden optimum between P24 and
P25.  P24 remains the selected endpoint on this axis.  Any further duration
test must lower the SSL learning rate so that the cumulative update scale is
matched; simply increasing steps at `1e-3` is stopped.

P31 performs the single allowed cumulative-update control: 2048 SSL steps at
`5e-4`, so `steps * learning_rate` approximately matches P24's 1024 steps at
`1e-3`.  Relative to P30, lowering the learning rate recovers
`+0.074/+0.606` pp Clean and `+0.246/+0.634` pp PN2021-C.  This confirms that
part of P30's loss was excessive representation movement.

The recovery is not enough.  Relative to Direct, P31 changes Clean by
`-1.903/-6.074` pp and PN2021-C by `+1.932/-0.140` pp.  Relative to P24, it
remains lower by `-0.881/-2.983` pp Clean and `-0.974/-2.530` pp PN2021-C.
Thus extra AugMix/VAE view coverage at a matched cumulative update scale does
not improve the endpoint.  Freeze ECGFounder pretraining at P24's 1024 steps
and stop the duration/learning-rate axis; P29 remains the best robustness
point after the already documented global checkpoint interpolation.

## 2026-07-26 stopping point

P32 is the valid moderate VAE-LHAT geometry test on top of the balanced P20
recipe.  It changes only the compatible D19 hard-search geometry from the
default effective `hull_lambda=0.05`, three steps to requested
`hull_lambda=0.10`, five steps; the runtime reports an actual effective lambda
near `0.09` and a stable hard-search `loss_gain` around `0.010-0.012`.

Despite the stronger and non-collapsed attack signal, P32 is effectively tied
with P20:

- P32 absolute Clean: `0.860326 / 0.539315` AUROC/AUPRC.
- P32 absolute PN2021-C: `0.847022 / 0.512061`.
- P32 versus Direct: Clean `+0.749 / +0.572` pp; PN2021-C
  `+2.017 / +3.090` pp.
- P32 minus P20: Clean `+0.001 / +0.002` pp; PN2021-C
  `-0.001 / -0.004` pp.

All four centers trained for the fixed E40 budget, evaluation excluded each
center's K500 references, the mapping remained
`v7_super5_sjr_rgq_review_20260528 / 555ec85d5b51`, and no per-center or
per-class selection was used.  This negative result stops the radius axis:
P33 (`hull_lambda=0.15`) is not launched because the old Ningbo-only pilot
already showed `0.10` and `0.15` nearly tied and P32 proves that increasing
attack pressure alone does not move the four-center endpoint.

The next planned experiment is P34, a sequential attribution design:

1. initialize both arms from the per-center P22 checkpoint;
2. disable repeated Target-SSL pretraining;
3. run eight matched continuation epochs at learning rate `3e-5`;
4. control arm: Direct+fixed20 continuation;
5. candidate arm: the same continuation plus VAE-random, VAE-LHAT-hard and JSD.

This design would measure the independent VAE refinement gain after the
already established AugMix-SSL stage.  Generated configs and their per-center
initializer SHA256 identities are prepared and all eight train dry-runs pass.
The first real control launch stopped before training because the managed
trainer correctly rejected P22 as differing from the locked PTB-XL source
registry.  The four failed run directories were removed with the controller's
owned `clean-failed` command; no incomplete checkpoint remains.

To resume P34, generate a candidate-local per-center source registry that
records the exact P22 initializer path and SHA256, point each online config's
`references.source_baseline_registry` to that snapshot, rerun all dry-runs,
then launch the Direct control before the VAE candidate.  Do not weaken or
bypass `validate_locked_source_checkpoint`, and do not launch P33 first.

## 2026-07-26 corrected sequential 2x2 design

P34 is retired before training. It cannot isolate a sequential AugMix-to-VAE
effect because P22's E40 supervised stage already used raw, VAE-random,
VAE-LHAT-hard and JSD auxiliaries. The corrected experiment is a true
Stage-1-by-Stage-2 factorial:

| endpoint | Stage 1 | Stage 2 |
|---|---|---|
| A Direct | clean-clean SimCLR/VICReg control, then Direct+fixed20 E40 | Direct+fixed20 E8 |
| B AugMix-only | clean-AugMix SimCLR/VICReg, then Direct+fixed20 E40 | Direct+fixed20 E8 |
| C VAE-only | same initializer as A | Direct+fixed20 E8 plus VAE-LHAT |
| D AugMix->VAE | same initializer as B | Direct+fixed20 E8 plus VAE-LHAT |

Stage 1 candidates:

- `s00_effnet_ssl4096_identity_e40`
- `s01_effnet_ssl4096_augmix_e40`

Stage 2 paired continuations:

- `s02_effnet_identity_vae_refine8`
- `s03_effnet_augmix_vae_refine8`

The two Stage-1 arms share source checkpoint, K500 order, 4096 AdamW/cosine
updates, SimCLR/VICReg objective, projection-head initialization,
source-logit anchor, full-FT E40 schedule and equal-five class anchor weights.
Their only intended mechanism difference is `pretrain_view_mode`:
`clean_identity` versus `twochain_augmix`. PTB-XL source replay is disabled in
both arms; no VAE checkpoint is passed.

The Stage-2 VAE package is intentionally narrow: exact-label nonself
VAE-LHAT, direct LHAT BCE, clean/LHAT JSD, PCGrad alpha `0.25`, repaired
`hull_lambda=0.60`, five attack steps and explicit epsilon projection. It
contains no AugMix node, raw auxiliary view, VAE-random view, source replay,
checkpoint blend or class-specific weight.

Each post-refine pair shares initializer SHA, batch/order, optimizer restart,
LR `3e-5`, E8 cosine horizon, full-FT scope, one optimizer step per fixed20
base batch and `comparison_rng_identity=sequential_effnet_stage2_v1`.
VAE attack compute is additional generation compute and must be reported
separately; the comparison is optimizer-budget matched, not wall-clock
matched.

Before Stage 2 can be prepared, the controller now requires the parent
checkpoint, parent manifest, run manifest, train result, method hash, exact
K500 split/hash identity, non-empty ordered-batch digest, zero source-replay
samples and explicit proof that the Stage-1 run had no VAE resource or
`--vae-checkpoint`.

Current validation status:

- both Stage-1 config closures generated;
- all eight four-center managed train dry-runs pass;
- generated method profiles compile with one shared RNG identity;
- Stage-1 method diff is limited to arm labels and `pretrain_view_mode`;
- VAE post profile compiles and passes its frozen allowlist;
- targeted PCGrad positive/conflicting-gradient tests pass;
- BTVR regression tests pass.

All results remain single-seed, heldout-tuned development evidence. Outside-K500
target records may be read only for global macro reporting and may not update
model state, select per-center epochs or set class weights.

## 2026-07-26 Stage-2 coupling closure

Two narrower E4 continuation designs were run from the completed Stage-1
initializers with full K500 exposure, 32 optimizer steps per center, LR
`3e-5`, a cosine E4 horizon, equal-five class treatment and ref-excluded
Clean/PN2021-C evaluation.

H2 tested direct clean plus VAE-LHAT supervision:

- effective objective:
  `0.50 clean BCE + 0.25 LHAT BCE + 0.25 clean/LHAT JSD`;
- no fixed20, AugMix, PCGrad, VAE-random, raw auxiliary or source replay;
- epsilon 8 attacks were active, with positive loss gain, approximately
  18-30% any-flip ASR, zero invalid decodes and zero quality rejection;
- after AugMix Stage 1, VAE minus its matched clean-polish control was
  `-0.094/-0.089` pp Clean and `-0.124/-0.117` pp PN2021-C
  AUROC/AUPRC.

J3 removed direct LHAT BCE and used the same hard waveform only as AugMix
chain three and in three-view JSD. Its matched control replaced chain three
with a second identity view while preserving the three model forwards,
objective weights, raw-chain AugMix RNG, initializer, record order and
optimizer budget.

- effective objective:
  `0.50 clean BCE + 0.25 clean-preserve BCE + 0.125 AugMix BCE +`
  `0.125 three-view JSD`;
- epsilon 8 attacks again remained active: `atk_anchor_l2` approximately 8,
  positive loss gain approximately `0.075-0.105`, any-flip ASR approximately
  17-30%, zero invalid decodes and zero quality rejection;
- J3 minus its matched identity-chain3 control was
  `-0.008/-0.014` pp Clean and `-0.009/-0.016` pp PN2021-C
  AUROC/AUPRC;
- J3 absolute metrics were `0.856208/0.526300` Clean and
  `0.846566/0.510626` PN2021-C, below P22 by approximately
  `0.590/1.169` and `0.619/0.881` pp respectively.

These results isolate a coupling failure rather than a null attack or resource
failure. Direct hard-label LHAT supervision is mildly harmful, while LHAT as
an unsupervised chain-three direction is effectively neutral under this
frozen E4 recipe. Do not spend an ECGFounder transfer or an identity-side
factorial on H2/J3. Retain the Stage-1 AugMix result as the positive component
and require a new VAE training signal, not another epsilon-only adjustment,
before reopening the post-training branch.

Execution used only physical GPUs 4 and 5. The controller's existing
`--jobs-per-gpu 2` mode safely ran four independent center cells at once:
candidate training used about 6 GiB for the final remaining cell, and paired
evaluation used about 3 GiB total per GPU while two cells were co-located.
Each cell retained a unique run directory and log.

## 2026-07-26 J4 frozen-teacher chain3 closure

J4 tested whether a frozen Stage-2 initializer teacher could make the J3
chain-three signal safer without changing its geometry or optimizer budget.
The experiment used candidate
`j4a_effnet_augmix_vae_chain3_teacher04_polish4` and a matched
clean-identity-chain3 control. Both arms started from the same per-center S01
checkpoint, ran four epochs and 32 optimizer steps, and used the same K500
order, exact-label eligibility mask, AugMix random identity, objective weights
and evaluation records.

The only candidate-side mechanism difference was the VAE-LHAT waveform in
chain three. In both arms, only the AugMix BCE target was changed to
`0.60 * hard_label + 0.40 * frozen_initializer_probability`; both clean BCE
terms remained hard-label trained and the three-view JSD was unchanged. The
teacher was snapshotted before the first method generation, and each center
recorded `32 generate = 32 objective = 32 soft-BCE` calls with a teacher
checkpoint SHA matching its S01 initializer.

Four-center all-zero-kept, reference-excluded means were:

| arm | Clean AUROC | Clean AUPRC | PN2021-C AUROC | PN2021-C AUPRC |
|---|---:|---:|---:|---:|
| J4 VAE-LHAT chain3 | 0.856353 | 0.526329 | 0.846694 | 0.510731 |
| matched identity chain3 | 0.856269 | 0.526311 | 0.846694 | 0.510777 |
| candidate minus control, pp | +0.0084 | +0.0019 | -0.0000 | -0.0046 |

The drop-all-zero view was also null. J4 improved J3 by only approximately
`+0.015/+0.003` pp Clean and `+0.013/+0.011` pp PN2021-C AUROC/AUPRC, while
remaining below P22 by `-0.576/-1.166` pp Clean and `-0.606/-0.871` pp
PN2021-C.

The null result is not explained by a disabled runtime or failed waveform QC:

- exact-label eligibility was 95.8%, 95.0%, 92.6% and 97.2% for Ningbo,
  Chapman-Shaoxing, CPSC and Georgia;
- every decoded waveform was finite and accepted, with zero QC rejection;
- LHAT loss gain was positive, approximately `0.077-0.105`;
- any-flip ASR was approximately 17-28%;
- `atk_anchor_l2` mean, median and p90 all reached the epsilon-8 boundary.

The likely failure is signal dilution rather than attack absence. The
teacher-soft AugMix BCE contributes only about 12% of effective objective mass,
the LHAT branch receives only about one third of the AugMix mixture, and the
weighted candidate-control JSD difference is approximately `0.0011-0.0014`.
The attack is geometrically saturated but has weak decision-flip efficiency.

J4 therefore closes the frozen-teacher rescue of the current chain3 recipe.
Do not spend a PTB-XL source-floor run, second seed or ECGFounder transfer on
this candidate. Retain P22 as the stronger EfficientNet endpoint and require a
materially different VAE coupling objective before reopening post-training.
This evidence remains single-seed `heldout_tuned_development_only` and must not
enter the active evidence registry or a paper main table.

The full J4 run used only physical GPUs 4 and 5 with two independent center
processes per GPU. Training occupied about 12.2 GiB per GPU with approximately
11.8 GiB free; paired evaluation occupied about 3 GiB per GPU. All eight train
and eight evaluation cells completed with separate run records and no failed
process.

### Post-run BN audit correction

The numerical J4a result above is retained as development provenance, but its
claim of a strictly matched waveform exposure is superseded by a post-run
audit. The trainer's BatchNorm plan forwards the complete batch for each
objective view even where `valid_mask` is false. J4a control masked the
ineligible AugMix samples out of the loss but left their corrupted waveform in
the full-batch forward, whereas the candidate runtime kept those positions as
clean fallback waveforms. This affects 2.8-7.4% of records depending on the
center.

J4a therefore cannot close the causal comparison by itself. Preserve all J4a
artifacts, fix the control's ineligible AugMix waveform to the same clean
fallback semantics, and rerun both arms under a new J4b run identity. Do not
overwrite or reinterpret the J4a run.

## 2026-07-26 J4b BN-matched replay and final decision

J4b, `j4b_effnet_augmix_vae_chain3_teacher04_bnmatched_polish4`, reran both
arms without overwriting J4a. Its complete candidate configuration is
single-source derived from J4a: initializer, optimizer, four-epoch/32-step
budget, epsilon, teacher mix and comparison RNG identity are unchanged.

The matched-control runtime now applies the exact-label eligibility mask and
also replaces every ineligible AugMix waveform with the corresponding clean
waveform before the full-batch BatchNorm forward. Eligible positions retain
the ordinary identity-chain3 AugMix result. The original AugMix tensor is not
modified in place, and labels, order, valid masks, metadata and stochastic
traces are preserved.

Each training run also persists byte-for-byte copies of eight critical runtime
sources, including the objective and teacher adapters plus the dirty whitelist
trainer/runtime files. The snapshot manifest records original paths, SHA256 and
sizes; a post-run recheck fails closed on source drift. All eight J4b training
runs recorded stable sources and included the snapshot manifest in the managed
run file index.

Four-center all-zero-kept, reference-excluded means were:

| arm | Clean AUROC | Clean AUPRC | PN2021-C AUROC | PN2021-C AUPRC |
|---|---:|---:|---:|---:|
| J4b VAE-LHAT chain3 | 0.856358 | 0.526320 | 0.846694 | 0.510743 |
| BN-matched identity chain3 | 0.856253 | 0.526346 | 0.846689 | 0.510771 |
| candidate minus control, pp | +0.0105 | -0.0026 | +0.0005 | -0.0028 |

The drop-all-zero candidate-minus-control result was also null:
`+0.0086/-0.0188/-0.0041/-0.0161` pp in the same metric order. J4b reproduced
J4a within approximately 0.001 pp and remained below P22 by
`-0.575/-1.167` pp Clean and `-0.606/-0.869` pp PN2021-C AUROC/AUPRC.

All split, mapping, reference-exclusion, initializer, teacher, optimizer-step,
objective-valid-count, loss-mass and evaluation-view checks passed. The
mechanism remained active, finite and QC-clean. The strict BN-matched replay
therefore confirms that frozen-clean-teacher label softening does not create a
measurable VAE-chain3 benefit under this recipe.

Mechanism diagnostics explain the null without implying a runtime failure:
the attack saturated the epsilon-8 boundary but reached only 22.7% any-flip
ASR; after AugMix interpolation the LHAT waveform contributed approximately
16.4% of the final waveform; and the realized JSD loss magnitude was only about
0.7% of total logged loss. The hard direction was therefore geometrically
active but weakly coupled to the classifier update.

J4b fails the predeclared robust-AUPRC improvement gate and must not be
transferred to ECGFounder. Do not spend a second seed or PTB-XL source-floor
evaluation on this closed candidate. Keep it as a reproducible single-seed
development null result; P22 remains the stronger EfficientNet endpoint.

## 2026-07-26 K1 pure frozen-teacher LHAT closure

K1 tested whether the negative H2/J3/J4 results came from mixing hard labels
into the VAE-LHAT outer objective. It started from the same completed S01
AugMix-SimCLR/VICReg plus Direct+fixed20 checkpoint and ran four epochs,
32 optimizer steps per center, LR `3e-5`, epsilon 8 and five LHAT steps.

The matched pair used:

- candidate: effective `0.75 clean hard BCE + 0.25 pure frozen-initializer
  soft BCE` on the direct LHAT endpoint;
- control: the same objective and second model forward, but the soft BCE was
  applied to an identity waveform;
- no hard-label LHAT BCE, JSD, post-stage AugMix, fixed20, VAE-random or
  source replay in this polish stage;
- identical initializer, K500 order, eligibility, teacher snapshot, RNG,
  optimizer budget and BatchNorm exposure weights.

Four-center candidate minus control was:

| metric | delta, pp |
|---|---:|
| Clean AUROC | -0.0272 |
| Clean AUPRC | -0.0373 |
| PN2021-C AUROC | -0.0410 |
| PN2021-C AUPRC | -0.0654 |

The mechanism was active: all attacks reached the epsilon-8 boundary, loss
gain was `0.071-0.098`, any-flip ASR was `17.8-28.9%`, and no waveform failed
finite/QC checks. Candidate and control clean BCE trajectories were nearly
identical, while the LHAT soft-BCE remained persistently harder. The negative
result therefore is not an implementation failure and does not justify an
ECGFounder transfer. A future low-dose 0.10 auxiliary test is allowed only
after the higher-information Stage-1 source-replay factorial below.

## 2026-07-26 next experiment: Stage-1 view by source replay 2x2

P22 and S01 share the same target SSL and supervised budgets, but P22 also
uses labeled PTB-XL train semantic replay during Stage 1. Existing S00/S01
already show the key tradeoff: AugMix views sacrifice clean performance while
substantially improving PN2021-C. The next experiment isolates whether source
replay restores clean performance without erasing that robustness gain.

The frozen cells are:

1. S00: clean-identity SimCLR/VICReg, source replay off;
2. S01: two-chain AugMix SimCLR/VICReg, source replay off;
3. R00: clean-identity SimCLR/VICReg, source replay weight `0.30`;
4. R01: two-chain AugMix SimCLR/VICReg, source replay weight `0.30`.

All cells use the same PTB-XL source checkpoint, per-center full K500 target
pool, 4096 target SSL steps, batch 64, LR `1e-3`, SimCLR+VICReg mix `0.03`,
equal-five logit anchors `5/2`, then 40 epochs / 320 optimizer steps of
Direct+fixed20 at LR `1.5e-4`, with fixed last-epoch selection. Replay-on
means one labeled PTB-XL train batch per target SSL step, not a 30% sampling
probability: each center consumes 262,144 source exposures at loss weight
`0.30`.

This remains target-center K500-only adaptation but is not source-data-free.
The accurate statement is: no target-center records outside K500 are used for
model updates; replay-on cells additionally consume labeled PTB-XL train.
Ref-excluded Clean/PN2021-C metrics are development feedback only.

The runtime now records both ordered target K500 and ordered PTB-XL source
batch SHA256 values. The factorial summary must verify identical target order
across all four cells, identical source order across R00/R01, identical
fixed20 supervised exposure, and report both main effects plus the
view-by-replay interaction. If R01 is promising, freeze that Stage-1 recipe
before reopening exactly one matched VAE post-training comparison.

## 2026-07-26 K2/K3 minimal two-component closure

The paper-facing design is now restricted to two claimed components:

1. target-K500 AugMix-SimCLR/VICReg pre-training;
2. a short VAE-LHAT post-training refinement.

Direct+fixed20 is the shared supervised baseline, not a third claimed
component. Source replay, EMA teacher training, JSD, SupCon, PCGrad,
VAE-random, raw auxiliary views, residual heads and class-specific weighting
must not be added to the main comparison. The required causal table is the
four-cell Direct, AugMix-only, VAE-only and AugMix-to-VAE factorial.

K2 tested a low-dose frozen-initializer teacher target in a four-epoch,
32-optimizer-step refinement. The nominal objective was 90% clean hard-label
BCE plus 10% auxiliary soft BCE. Its matched control used the same frozen
teacher and exact-label eligibility mask on an identity waveform.

| K2 arm | Clean AUROC | Clean AUPRC | PN2021-C AUROC | PN2021-C AUPRC |
|---|---:|---:|---:|---:|
| identity control | 0.863813 | 0.539799 | 0.819971 | 0.462314 |
| identity then VAE-LHAT | 0.863638 | 0.539461 | 0.819432 | 0.461570 |
| AugMix control | 0.856268 | 0.526349 | 0.846649 | 0.510789 |
| AugMix then VAE-LHAT | 0.856176 | 0.526279 | 0.846525 | 0.510619 |

The VAE effect was `-0.054/-0.074` pp PN2021-C AUROC/AUPRC after the
identity Stage1 and `-0.012/-0.017` pp after the AugMix Stage1.

K3 removed teacher distillation and tested the academically simpler
true-label hard-BCE coupling. Its effective objective was 90% clean hard-label
BCE plus 10% exact-label VAE-LHAT hard BCE. The matched control used the same
second forward and exact-label eligibility mask on an identity waveform.

| K3 arm | Clean AUROC | Clean AUPRC | PN2021-C AUROC | PN2021-C AUPRC |
|---|---:|---:|---:|---:|
| identity control | 0.863689 | 0.539626 | 0.819801 | 0.462173 |
| identity then VAE-LHAT | 0.863474 | 0.539527 | 0.819297 | 0.461731 |
| AugMix control | 0.856117 | 0.526318 | 0.846492 | 0.510677 |
| AugMix then VAE-LHAT | 0.856027 | 0.526149 | 0.846278 | 0.510461 |

The VAE effect was `-0.050/-0.044` pp PN2021-C AUROC/AUPRC after the
identity Stage1 and `-0.021/-0.022` pp after the AugMix Stage1. K2 and K3
therefore agree that the current short VAE-LHAT coupling is near-null rather
than hidden by the choice of teacher-soft versus true-label targets.

The positive component is Stage1 AugMix: in the K3 matched controls it changed
PN2021-C by `+2.669/+4.850` pp AUROC/AUPRC, while changing Clean by
`-0.757/-1.331` pp. The combined K3 endpoint remains below P22 by
`-0.648/-0.898` pp on PN2021-C and `-0.609/-1.184` pp on Clean.

Decision:

- do not transfer K2 or K3 to ECGFounder;
- do not rescue them by stacking more losses or auxiliary branches;
- keep one simple hard-label VAE-LHAT implementation as the sole VAE arm;
- next test whether shortening the supervised stage after the fixed 4096-step
  AugMix pre-training preserves more of the learned invariance;
- reopen the same four-cell factorial only if the shorter AugMix initializer
  improves the absolute Clean/PN2021-C trade-off.

All K2/K3 evidence is single-seed, K500-only adaptation with ref-excluded
development evaluation. It is not eligible for the final paper table until a
recipe is frozen, the PTB-XL source floor is checked and the required seed
replication is complete.

## 2026-07-26 E20 AugMix preservation check

S10/S11 changed only the supervised Direct+fixed20 horizon after the fixed
4096-step target SSL stage:

- S10: clean-identity SimCLR/VICReg control, then E20 supervised training;
- S11: two-chain AugMix SimCLR/VICReg, then E20 supervised training.

The pair shared source checkpoint, K500 membership and order, target SSL
budget, projection head, optimizer, learning-rate schedule, batch size,
fixed20 exposure and 160 supervised optimizer steps per center. All four
centers passed the paired order, split and exposure checks.

| endpoint | Clean AUROC | Clean AUPRC | PN2021-C AUROC | PN2021-C AUPRC |
|---|---:|---:|---:|---:|
| S10 E20 identity | 0.857813 | 0.524697 | 0.820308 | 0.456490 |
| S11 E20 AugMix | 0.854735 | 0.521377 | 0.845661 | 0.504798 |
| S00 E40 identity | 0.859572 | 0.529771 | 0.826412 | 0.469837 |
| S01 E40 AugMix | 0.853737 | 0.518349 | 0.845809 | 0.505579 |
| P22 development reference | 0.862112 | 0.537990 | 0.852754 | 0.519437 |

At E20, AugMix minus identity was `-0.308/-0.332` pp Clean and
`+2.535/+4.831` pp PN2021-C AUROC/AUPRC. Relative to E40 AugMix, E20 changed
Clean by `+0.100/+0.303` pp and PN2021-C by `-0.015/-0.078` pp. The mean of
Clean and PN2021-C AUPRC therefore increased from `0.511964` to `0.513088`.

E20 improves the clean-robust trade-off but does not close the P22 gap:
S11 remains below P22 by `-0.738/-1.661` pp Clean and `-0.709/-1.464` pp
PN2021-C. This is sufficient to reopen exactly one minimal K3-style
hard-label VAE-LHAT factorial from S10/S11. It does not justify another
supervised-horizon grid, another loss, or ECGFounder transfer.

Machine-readable evidence:
`s11_effnet_ssl4096_augmix_e20/stage1_e20_comparison.json`.

## 2026-07-26 K4 E20 two-component factorial

K4 reused the exact K3 hard-label VAE-LHAT implementation from the E20
initializers. No new loss, teacher, replay stream or auxiliary view was added.
All four Stage2 cells ran four epochs, 32 optimizer steps per center and the
same K500 record order:

| cell | Stage1 | Stage2 |
|---|---|---|
| A Direct | identity SSL E20 | 90% clean + 10% identity hard-BCE control |
| B AugMix-only | AugMix SSL E20 | same identity hard-BCE control |
| C VAE-only | identity SSL E20 | 90% clean + 10% VAE-LHAT hard BCE |
| D AugMix-to-VAE | AugMix SSL E20 | same VAE-LHAT hard BCE |

Four-center all-zero-kept, ref-excluded means:

| cell | Clean AUROC | Clean AUPRC | PN2021-C AUROC | PN2021-C AUPRC |
|---|---:|---:|---:|---:|
| A Direct | 0.861824 | 0.534113 | 0.813751 | 0.448951 |
| B AugMix-only | 0.857280 | 0.528380 | 0.847881 | 0.510944 |
| C VAE-only | 0.861559 | 0.533929 | 0.813218 | 0.448374 |
| D AugMix-to-VAE | 0.857052 | 0.528251 | 0.847674 | 0.510779 |

Factorial effects in pp:

- AugMix without VAE: `-0.454/-0.573` Clean and
  `+3.413/+6.199` PN2021-C AUROC/AUPRC;
- AugMix with VAE: `-0.451/-0.568` Clean and
  `+3.446/+6.240` PN2021-C;
- VAE after identity: `-0.027/-0.018` Clean and
  `-0.053/-0.058` PN2021-C;
- VAE after AugMix: `-0.023/-0.013` Clean and
  `-0.021/-0.017` PN2021-C;
- interaction: `+0.004/+0.006` Clean and
  `+0.033/+0.041` PN2021-C.

The VAE runtime was active rather than silently bypassed. Across all centers
and epochs the attack reached approximately L2 epsilon 8, produced positive
loss gain, had non-zero any-flip ASR and produced zero invalid decodes or QC
rejections. The null effect is therefore a training-signal result.

D remains below P22 by `-0.506/-0.974` pp Clean and
`-0.508/-0.866` pp PN2021-C. K4 fails the predeclared VAE transfer gate:
robust AUPRC is not at least +0.15 pp, clean AUPRC is negative, and the
per-center robust effect is not non-negative in at least three centers.
Do not transfer K4 to ECGFounder and do not run a PTB-XL source floor or
second seed.

Machine-readable evidence:
`k4a_effnet_e20_augmix_vae_hard_bce_lowdose4/factorial_summary.json`.

## 2026-07-26 K5 E20 consistency-only factorial

K5 kept the same E20 initializers and four-epoch, 32-step Stage2 budget as K4,
but replaced the hard label on the LHAT endpoint with one symmetric Bernoulli
consistency term. This was the last predeclared simple coupling test; it did
not introduce a teacher, replay stream, extra corruption branch, class weight
or checkpoint blend.

The effective candidate objective was:

```text
0.50 clean hard BCE
+ 0.40 duplicate clean-preservation hard BCE
+ 0.10 JSD(clean prediction, VAE-LHAT prediction)
```

The matched control used the same two clean forwards, exact-label eligibility,
batch-normalization exposure and optimizer budget, but replaced the LHAT
waveform with an identity auxiliary waveform.

Four-center all-zero-kept, ref-excluded means:

| cell | Stage1 | Stage2 | Clean AUROC | Clean AUPRC | PN2021-C AUROC | PN2021-C AUPRC |
|---|---|---|---:|---:|---:|---:|
| A Direct | identity SSL E20 | identity consistency control | 0.861821 | 0.534078 | 0.813746 | 0.448851 |
| B AugMix-only | AugMix SSL E20 | identity consistency control | 0.857291 | 0.528332 | 0.847895 | 0.510939 |
| C VAE-only | identity SSL E20 | VAE-LHAT consistency | 0.861762 | 0.534011 | 0.813421 | 0.448403 |
| D AugMix-to-VAE | AugMix SSL E20 | VAE-LHAT consistency | 0.857351 | 0.528400 | 0.847920 | 0.510921 |

Factorial effects in pp:

- AugMix without VAE: `-0.453/-0.575` Clean and
  `+3.415/+6.209` PN2021-C AUROC/AUPRC;
- AugMix with VAE: `-0.441/-0.561` Clean and
  `+3.450/+6.252` PN2021-C;
- VAE after identity: `-0.006/-0.007` Clean and
  `-0.033/-0.045` PN2021-C;
- VAE after AugMix: `+0.006/+0.007` Clean and
  `+0.002/-0.002` PN2021-C;
- interaction: `+0.012/+0.013` Clean and
  `+0.035/+0.043` PN2021-C.

The VAE runtime again reached approximately L2 epsilon 8, produced positive
loss gain and non-zero any-flip ASR, with zero invalid decodes and zero quality
rejections. Per-center VAE-after-AugMix PN2021-C AUPRC effects were
`+0.005`, `-0.013`, `+0.004` and `-0.002` pp for Chapman, CPSC, Georgia and
Ningbo respectively. The aggregate effect is therefore a genuine null result,
not a silently disabled attack.

D remains below P22 by `-0.476/-0.959` pp Clean and `-0.483/-0.852` pp
PN2021-C. K5 fails the same transfer gate as K4. Combined with the K2
teacher-soft and K3/K4 hard-label results, three simple LHAT endpoint couplings
all agree that the current post-training VAE component contributes no stable
gain.

Decision:

- close the teacher-soft, hard-label and consistency coupling grid;
- do not add more auxiliary losses or branches to rescue the VAE result;
- do not transfer K5 to ECGFounder or run its PTB-XL source floor/second seed;
- retain AugMix as the demonstrated positive component;
- any further VAE experiment must change the VAE component's own mechanism,
  remain a single post-training component and use the same four-cell factorial.

Machine-readable evidence:
`k5a_effnet_e20_augmix_vae_consistency_lowdose4/factorial_summary.json`.

## 2026-07-27 S12/S13 predeclared simple AugMix scaling check

K2 through K5 show that adding or reweighting another VAE endpoint loss is not
the next useful axis. The next experiment therefore changes no method
component and tests only whether the positive AugMix representation stage has
finished scaling.

The matched pair is:

- S12: clean-identity SimCLR/VICReg control, 8192 SSL updates, then E20;
- S13: two-chain AugMix SimCLR/VICReg, 8192 SSL updates, then E20.

Relative to S10/S11, only the total pre-training update budget changes from
4096 to 8192 (`1x4096` to `2x4096`) and a new shared comparison RNG identity
is used. The pair keeps batch 64, Stage-1
LR `1e-3`, SimCLR/VICReg mix `0.03`, Dirichlet/Beta `0.5/0.5`, no source
replay, no VAE resource, E20/T20 supervised training, fixed final checkpoint,
equal-five class treatment, K500-only adaptation and ref-excluded evaluation.

Promotion gates:

1. S13 must improve the mean of Clean and PN2021-C AUPRC over S11 by at least
   `0.15` pp;
2. S13 must not reduce Clean AUPRC by more than `0.25` pp versus S11;
3. the AugMix-minus-identity robust AUPRC effect must remain positive in at
   least three of four centers;
4. if S13 still trails P22 on both Clean and PN2021-C AUPRC, stop the SSL-step
   axis rather than doubling again.

Only if these gates advance the absolute frontier may one already-implemented,
hard-label VAE-LHAT four-cell factorial be reopened from S12/S13. No new loss,
teacher, replay stream, residual head or class weight may be added.

The first S12 launch used `1x8192`, which the frozen runtime correctly rejected
before the first optimizer step because its per-epoch safety bound is 4096.
The executable representation is therefore `2x4096`; total optimizer updates
and cosine horizon are unchanged. Four task-owned failed run directories are
kept out of the comparison. S14 provides a fresh shared Direct run under the
same final source snapshot and RNG identity used by the successful S12/S13
pair, so the earlier completed Direct artifact is not reused for a paper-style
delta.

### S12/S13 result and decision

All four S14 Direct, S12 identity-SSL and S13 AugMix-SSL runs completed under
the frozen source snapshot. Each S12/S13 center records 8192 target SSL steps,
524288 K500 waveform exposures, no target labels in Stage 1, zero PTB-XL
replay samples, the same K500 hash and the same target-batch-order hash.
Supervised training is E20 with 500 clean plus 10000 canonical fixed20
corrupted exposures per epoch and family-balanced 0.5/0.5 loss.

Four-center all-zero-kept, ref-excluded means:

| cell | Clean AUROC | Clean AUPRC | PN2021-C AUROC | PN2021-C AUPRC |
|---|---:|---:|---:|---:|
| S14 shared Direct | 0.854347 | 0.519512 | 0.822135 | 0.462734 |
| S12 identity SSL 8192 | 0.856977 | 0.524036 | 0.820057 | 0.457559 |
| S13 AugMix SSL 8192 | 0.854446 | 0.519848 | 0.847330 | 0.507096 |
| S11 AugMix SSL 4096 | 0.854735 | 0.521377 | 0.845661 | 0.504798 |
| P22 development reference | 0.862112 | 0.537990 | 0.852754 | 0.519437 |

S13 minus S12 is `-0.253/-0.419` pp on Clean and `+2.727/+4.954` pp on
PN2021-C AUROC/AUPRC. Robust AUPRC is positive in all four centers, so the
AugMix view remains a clear causal component. However, S13 minus S11 is only
`+0.167/+0.230` pp robust and `-0.029/-0.153` pp Clean; the balanced
Clean/robust AUPRC score improves by only `+0.038` pp, below the predeclared
`+0.15` pp gate. S13 also remains below P22 by `-0.767/-1.814` pp Clean and
`-0.542/-1.234` pp robust.

Decision:

- stop the SSL-step axis; do not run 16384 updates;
- retain 4096-step S11 as the compute-efficient AugMix reference;
- do not reopen the VAE-LHAT factorial from S12/S13;
- do not transfer this non-frontier recipe to ECGFounder.

Machine-readable evidence:
`s13_effnet_ssl8192_augmix_e20/stage1_ssl_scaling_comparison.json`.

## 2026-07-27 S15 predeclared classic AugMix alpha check

The next experiment changes only AugMix's tied mixing concentration from
Dirichlet/Beta `0.5/0.5` to the classic neutral `1.0/1.0`. It returns to the
compute-efficient S11 budget: 4096 target SSL updates followed by E20
supervised training. Source checkpoint, K500 membership/order, batch 64,
Stage-1 LR `1e-3`, SimCLR/VICReg mix `0.03`, fixed20 supervision, equal-five
class treatment, zero source replay, zero VAE resources and fixed final
checkpoint are unchanged.

This is one AugMix parameter, not a new method component. Alpha `0.5` is
U-shaped and often samples near-clean or near-fully-corrupted mixtures;
alpha `1.0` samples the interpolation simplex uniformly and tests whether
more medium-strength mixtures improve the clean/robust trade-off.

Promotion gates:

1. S15 balanced Clean/PN2021-C AUPRC must improve over S11 by at least
   `0.15` pp;
2. S15 Clean AUPRC must not drop by more than `0.25` pp versus S11;
3. S15 PN2021-C AUPRC must be non-negative versus S11 in at least three of
   four centers;
4. if S15 remains behind S11 on the balanced score or behind P22 on both
   Clean and PN2021-C AUPRC, close the alpha axis rather than adding a grid.

### S15 result and decision

All four S15 Direct and alpha-1 AugMix runs completed with the same K500
membership, target-batch order, 4096 SSL updates, E20 supervised budget and
ref-excluded evaluator used by S11.

Four-center all-zero-kept means:

| cell | Clean AUROC | Clean AUPRC | PN2021-C AUROC | PN2021-C AUPRC |
|---|---:|---:|---:|---:|
| S15 matched Direct | 0.854498 | 0.519524 | 0.822107 | 0.462694 |
| S15 AugMix alpha 1.0 | 0.854681 | 0.521921 | 0.844713 | 0.503771 |
| S11 AugMix alpha 0.5 | 0.854735 | 0.521377 | 0.845661 | 0.504798 |
| P22 development reference | 0.862112 | 0.537990 | 0.852754 | 0.519437 |

S15 minus its Direct control is `+0.018/+0.240` pp on Clean and
`+2.261/+4.108` pp on PN2021-C AUROC/AUPRC, so AugMix remains useful.
Against S11, however, alpha 1.0 changes Clean by `-0.005/+0.054` pp and
PN2021-C by `-0.095/-0.103` pp. Its balanced Clean/robust AUPRC is
`-0.024` pp below S11, and robust AUPRC improves in only one of four centers.

Decision:

- retain alpha `0.5/0.5` and S11 as the compact AugMix reference;
- close the alpha axis without a grid;
- the next experiment must alter the VAE-LHAT mechanism itself, not add
  another AugMix hyperparameter or auxiliary training component.

Machine-readable evidence:
`s15_effnet_ssl4096_augmix_alpha100_e20/summary.json`.

## 2026-07-27 L6 predeclared VAE-LHAT latent-path factorial

K2--K5 show that direct endpoint distillation, hard BCE and prediction
consistency do not provide a stable VAE effect. L6 changes the VAE mechanism
itself while retaining the simple sequential story:

```text
S10 identity SSL or S11 AugMix SSL
-> E20 Direct+fixed20 adaptation
-> matched E8 short polish
   control: clean identity
   candidate: VAE-LHAT direction followed by latent-path sampling
```

The four budget-matched cells are:

- A: S10 initializer + identity-path control;
- B: S11 initializer + identity-path control;
- C: S10 initializer + VAE-LHAT path;
- D: S11 initializer + VAE-LHAT path.

Every Stage-2 cell uses LR `3e-5`, batch 64, cosine E8/T8, one optimizer step
per clean batch and equal-five class treatment. Candidate and control both
execute the same exact-label/nonself LHAT search (`epsilon=8`, five steps,
`hull_lambda=0.60`), VAE path decode and RNG schedule. The candidate trains on
one uniformly sampled valid path point per eligible record from
`t={0.4,0.7,1.0}`; the control substitutes clean identity only at the
supervised auxiliary view. The effective objective is:

```text
0.75 clean hard-label BCE + 0.25 sampled latent-path hard-label BCE
```

Path geometry is the standardized clean-to-LHAT ray. Decoded points use linear
clean/hard endpoint residual correction. Selection among path points uses only
the frozen finite/flatline/amplitude QC gate and never model score, class
distribution, held-out labels or outside-K500 records. Fixed20, post-stage
AugMix, JSD, teacher, source replay, PCGrad, raw auxiliary, VAE-random,
residual heads and class weights are absent.

Promotion gates:

1. the matched VAE main effect must improve PN2021-C AUPRC by at least
   `0.15` pp;
2. D must improve the balanced Clean/PN2021-C AUPRC over B;
3. D must not reduce Clean AUPRC by more than `0.25` pp versus B;
4. the D-minus-B robust AUPRC effect must be non-negative in at least three
   centers;
5. ECGFounder transfer is allowed only if D reaches the current EfficientNet
   frontier rather than merely improving a weak control.

The first runtime attempt required all three frozen path points to pass QC.
Ningbo and Georgia correctly failed closed when an intermediate point crossed
the amplitude gate, while Chapman and CPSC completed. Those mixed-code results
were archived and are excluded. The final frozen rule samples uniformly among
valid path points; the already accepted `t=1` LHAT endpoint guarantees at
least one valid choice. All four centers are rerun from scratch under this
single runtime identity.

### L6 result and decision

All four cells completed for all four logical centers under the final
uniform-valid-path runtime. The evaluation is PN2021 Clean and PN2021-C,
ref-excluded, with the v7 mapping
`v7_super5_sjr_rgq_review_20260528 / 555ec85d5b51`.

Four-center all-zero-kept means:

| cell | Clean AUROC | Clean AUPRC | PN2021-C AUROC | PN2021-C AUPRC |
|---|---:|---:|---:|---:|
| A: identity SSL + matched identity polish | 0.863137 | 0.537343 | 0.814165 | 0.451766 |
| B: AugMix SSL + matched identity polish | 0.857441 | 0.530895 | 0.848050 | 0.512879 |
| C: identity SSL + VAE-LHAT path polish | 0.862587 | 0.536962 | 0.812892 | 0.450302 |
| D: AugMix SSL + VAE-LHAT path polish | 0.857083 | 0.530656 | 0.847672 | 0.512498 |
| P22 development reference | 0.862112 | 0.537990 | 0.852754 | 0.519437 |

Main effects in percentage points:

| effect | Clean AUROC | Clean AUPRC | PN2021-C AUROC | PN2021-C AUPRC |
|---|---:|---:|---:|---:|
| AugMix without VAE, B-A | -0.570 | -0.645 | +3.388 | +6.111 |
| AugMix with VAE, D-C | -0.550 | -0.631 | +3.478 | +6.220 |
| VAE after identity, C-A | -0.055 | -0.038 | -0.127 | -0.146 |
| VAE after AugMix, D-B | -0.036 | -0.024 | -0.038 | -0.038 |
| mean VAE main effect | -0.045 | -0.031 | -0.083 | -0.092 |

The D-minus-B PN2021-C AUPRC effects are approximately `-0.016`,
`-0.047`, `+0.006` and `-0.095` pp for Chapman-Shaoxing, combined
CPSC 2018+Extra, Georgia and Ningbo respectively. Only one of four centers is
non-negative. D is also `-0.503/-0.733` pp below P22 on Clean AUROC/AUPRC and
`-0.508/-0.694` pp below P22 on PN2021-C AUROC/AUPRC.

This is not a dead-attack result. Across centers and epochs, accepted LHAT
endpoints reach `atk_anchor_l2 ~= 8`, loss gain remains positive,
sample-any-flip ASR is roughly 16--30%, selected-path validity is 100%, and
decoded invalid rate plus quality rejection count are both zero.

Decision:

- retain S11 as the compact causal AugMix result;
- reject L6 latent-path polish because it fails gates 1, 2, 4 and 5;
- do not transfer L6 to ECGFounder;
- close endpoint/path hard-label VAE-LHAT post-training under the current
  frozen simple two-stage recipe instead of adding another VAE loss or
  hyperparameter grid;
- do not freeze a source-preservation claim because the rejected L6 cells were
  not sent through a separate PTB-XL source-floor evaluation.

Machine-readable evidence:
`l6a_effnet_e20_augmix_vae_lhat_path8/factorial_summary.json`.

## 2026-07-27 L7 predeclared VAE-LHAT feature-invariance factorial

L6 closes direct hard-label supervision of both the LHAT endpoint and sampled
points on its latent ray.  L7 changes the Stage-2 learning signal rather than
adding another attack-strength or loss-weight grid.  It tests whether the
hard view is more useful as a representation-invariance constraint:

```text
S10 identity SSL or S11 AugMix SSL
-> E20 supervised adaptation
-> matched E8 short polish
   control: clean versus clean penultimate feature alignment
   candidate: clean versus VAE-LHAT penultimate feature alignment
```

The four cells remain:

- A: S10 initializer + matched identity feature polish;
- B: S11 initializer + matched identity feature polish;
- C: S10 initializer + VAE-LHAT feature polish;
- D: S11 initializer + VAE-LHAT feature polish.

Every Stage-2 cell uses LR `3e-5`, batch 64, cosine E8/T8, one optimizer step
per clean batch, equal-five class treatment and the fixed final checkpoint.
Candidate and control both execute the same exact-label/non-self LHAT search
(`epsilon=8`, five steps, `hull_lambda=0.60`), decoder, eligibility mask,
two classifier forwards and RNG schedule.  The control replaces the auxiliary
waveform with clean identity only after matched LHAT generation.

The effective Stage-2 objective is:

```text
0.75 clean hard-label BCE
+ 0.25 mean(1 - cosine(clean_feature, auxiliary_feature))
```

Features are the input to the managed linear five-class classifier head.
There is no projection head, teacher, pseudo-label, direct LHAT BCE, JSD,
AugMix in Stage 2, fixed20, source replay, VAE-random, raw auxiliary,
residual head, per-class weight or held-out selection.  The auxiliary feature
term is evaluated only over the exact matched eligibility intersection.

The sandbox method graph uses a typed two-view objective slot so the generic
trainer performs one aligned clean/LHAT forward.  The allowlisted
`objective_scale_adapter.py` replacement must fail closed unless the method
id, two-view topology, effective weights, feature-capture head, validity mask,
and explicit `penultimate_cosine_distance_v1` contract all match.  Raw and
weighted feature-loss means must be persisted in training history.

Promotion gates are frozen before execution:

1. the matched VAE main effect must improve PN2021-C AUPRC by at least
   `0.15` pp;
2. D must improve balanced Clean/PN2021-C AUPRC over B;
3. D must not reduce Clean AUPRC by more than `0.25` pp versus B;
4. D-minus-B PN2021-C AUPRC must be non-negative in at least three centers;
5. ECGFounder transfer requires D to exceed P22 on both Clean and PN2021-C
   AUPRC, not merely improve its matched B control.

If L7 fails these gates, do not tune its weight, epsilon or epoch count from
held-out feedback.  Close feature-invariance post-training and move to a
different VAE coupling mechanism.

### L7 result and decision

All four cells completed for all four logical centers with the same
ref-excluded v7 evaluation contract used by L6:
`v7_super5_sjr_rgq_review_20260528 / 555ec85d5b51`.

Four-center all-zero-kept means:

| cell | Clean AUROC | Clean AUPRC | PN2021-C AUROC | PN2021-C AUPRC |
|---|---:|---:|---:|---:|
| A: identity SSL + matched identity feature polish | 0.863290 | 0.537421 | 0.814177 | 0.451365 |
| B: AugMix SSL + matched identity feature polish | 0.857918 | 0.531320 | 0.848520 | 0.513261 |
| C: identity SSL + VAE-LHAT feature polish | 0.863126 | 0.537685 | 0.813541 | 0.450714 |
| D: AugMix SSL + VAE-LHAT feature polish | 0.858121 | 0.531378 | 0.848627 | 0.513304 |
| P22 development reference | 0.862112 | 0.537990 | 0.852754 | 0.519437 |

Main effects in percentage points:

| effect | Clean AUROC | Clean AUPRC | PN2021-C AUROC | PN2021-C AUPRC |
|---|---:|---:|---:|---:|
| AugMix without VAE, B-A | -0.537 | -0.610 | +3.434 | +6.190 |
| AugMix with VAE, D-C | -0.500 | -0.631 | +3.509 | +6.259 |
| VAE after identity, C-A | -0.016 | +0.026 | -0.064 | -0.065 |
| VAE after AugMix, D-B | +0.020 | +0.006 | +0.011 | +0.004 |
| mean VAE main effect | +0.002 | +0.016 | -0.026 | -0.030 |

The D-minus-B PN2021-C AUPRC effects are approximately `+0.018`,
`-0.072`, `-0.015` and `+0.086` pp for Chapman-Shaoxing, combined
CPSC 2018+Extra, Georgia and Ningbo respectively.  Only two of four centers
are non-negative.  D is `-0.399/-0.661` pp below P22 on Clean AUROC/AUPRC and
`-0.413/-0.613` pp below P22 on PN2021-C AUROC/AUPRC.

The VAE branch was active rather than dead: the real-GPU smoke reached
`atk_anchor_l2 ~= 8`, positive LHAT loss gain, nonzero attack success and
zero decoded-invalid or quality-rejected samples.  The failure is therefore a
learning-signal result: pairwise penultimate cosine invariance contributes
essentially no matched robustness benefit.

Decision:

- retain S11 as the compact causal AugMix result;
- reject L7 because it fails promotion gates 1, 4 and 5 and its balanced
  D-minus-B gain is negligible;
- do not transfer L7 to ECGFounder;
- do not tune the L7 feature weight, epsilon or epoch count from held-out
  feedback;
- audit the historically supported local, anchor-dominant latent hull and
  multi-label-preserving target mechanism before predeclaring the next
  factorial.

Machine-readable evidence:
`l7a_effnet_e20_augmix_vae_feature8/factorial_summary.json`.

## 2026-07-27 L8 predeclared local anchor-soft VAE-LHAT factorial

L6 and L7 used a deliberately far exact-label endpoint
(`hull_lambda=0.60`, standardized L2 cap 8).  The locked project mechanism and
the earlier positive VAE family instead use a local, anchor-dominant hull.
L8 restores that mechanism without adding another training component:

```text
S10 identity SSL or S11 AugMix SSL
-> E20 Direct+fixed20 adaptation
-> matched E8 short polish
   control: local LHAT is generated, but the supervised auxiliary waveform is clean
   candidate: the same local LHAT waveform is supervised
```

The four cells remain:

- A: S10 initializer + matched local-search identity control;
- B: S11 initializer + matched local-search identity control;
- C: S10 initializer + local anchor-soft VAE-LHAT;
- D: S11 initializer + local anchor-soft VAE-LHAT.

Every Stage-2 cell uses LR `3e-5`, batch 64, cosine E8/T8, one optimizer step
per clean batch, equal-five class treatment and a fixed last checkpoint.  The
candidate and control both execute the same exact-positive-set/non-self latent
pool lookup, optimized hull search, VAE decode, quality gate, RNG schedule and
two classifier forwards.  Their only difference is whether the accepted
auxiliary waveform is the decoded LHAT view or clean identity.

The hard waveform QC policy is also frozen: non-finite, flatline, or
`abs(x)>20 mV` decoded views are clean-only for that record and receive no
auxiliary BCE.  Both arms execute LHAT before this mask is applied; the
identity control replaces accepted LHAT waveforms with clean only after
preserving the same runtime accounting.  A run is invalid if more than 10% of
K500 is rejected in any epoch or if any other rejection reason appears.

The frozen local geometry is:

```text
M = 20 distinct non-self neighbors
anchor = explicit outer interpolation origin, not duplicated in the M20 simplex
candidate policy = exact positive set, non-self
local pool size = 80
hull_lambda = 0.05
PGD L2 cap = 2.0 in standardized latent coordinates
steps = 3
learning rate = 0.25
```

The repaired whitelist LHAT formulation is
`anchor + 0.05 * (neighbor_mix - anchor)`, so the decoded latent retains at
least 95% effective anchor mass.  The anchor is deliberately excluded from
the M20 neighbor softmax to avoid counting it twice.  This is a local
manifold search, not the far endpoint used by L6/L7.

The effective Stage-2 objective is:

```text
0.75 clean hard-label BCE
+ 0.25 auxiliary anchor-soft BCE
```

The auxiliary target is `0.95` for the anchor's positive classes and `0.0`
otherwise.  Exact-positive-set candidates cannot admit a new class, and the
same soft target is used in the matched clean-waveform control.  Therefore the
candidate-minus-control contrast isolates the VAE waveform rather than label
smoothing.

There is no Stage-2 AugMix, fixed20, JSD, teacher, source replay, VAE-random,
raw auxiliary, PCGrad, residual head, per-class weight or held-out
checkpoint selection.

Promotion gates are frozen before execution:

1. mean matched VAE main effect on PN2021-C AUPRC must be at least `+0.15` pp;
2. D must improve balanced Clean/PN2021-C AUPRC over B;
3. D must not reduce Clean AUPRC by more than `0.25` pp versus B;
4. D-minus-B PN2021-C AUPRC must be non-negative in at least three centers;
5. ECGFounder transfer requires D to exceed P22 on both Clean and PN2021-C
   AUPRC.

If L8 fails, do not tune hull lambda, soft-target value, auxiliary weight or
epoch count from held-out feedback.  Close local anchor-soft post-training
under this sequential recipe and retain S11 as the compact positive AugMix
result.

Execution IDs are `l8ri_effnet_e20_identity_vae_local_anchor_soft8` and
`l8ra_effnet_e20_augmix_vae_local_anchor_soft8`.  The `r` records an
implementation-only repair made before the complete factorial: the first
pilot IDs exposed the repaired-core `include_anchor=false` requirement, a
reserved teacher-policy field collision, and an overly strict whole-run
failure on hard-QC fallback.  No scientific hyperparameter, target metric, or
promotion gate changed.  Partial pilot outputs are retained as invalid
engineering evidence and are not eligible for the factorial summary.

## 2026-07-27 L8 result: local anchor-soft post-training rejected

The repaired L8 factorial completed all four centers, both initializers and
both matched Stage-2 arms. Every arm used 8 epochs, 64 optimizer steps,
K500-only adaptation, fixed-last selection and ref-excluded evaluation.
The managed summary is:

```text
/home/linbinhao/ECG_adv_data/runs/agent_workspace/
  a7_global_search_20260726/
  l8ra_effnet_e20_augmix_vae_local_anchor_soft8/factorial_summary.json
```

| Cell | Clean AUROC | Clean AUPRC | PN2021-C AUROC | PN2021-C AUPRC |
|---|---:|---:|---:|---:|
| A identity SSL + control | 0.863015 | 0.537392 | 0.814153 | 0.451869 |
| B AugMix SSL + control | 0.857299 | 0.530720 | 0.847951 | 0.512770 |
| C identity SSL + VAE | 0.862889 | 0.537366 | 0.813199 | 0.450228 |
| D AugMix SSL + VAE | 0.857214 | 0.530390 | 0.847754 | 0.512427 |
| P22 development reference | 0.862112 | 0.537990 | 0.852754 | 0.519437 |

The matched VAE effect is negative:

- C minus A: `-0.013/-0.003` pp Clean and `-0.095/-0.164` pp PN2021-C;
- D minus B: `-0.008/-0.033` pp Clean and `-0.020/-0.034` pp PN2021-C;
- mean PN2021-C AUPRC main effect across the two initializers is
  `-0.099 pp`;
- D-minus-B PN2021-C AUPRC is negative in all four centers
  (`-0.095`, `-0.021`, `-0.004`, `-0.017` pp);
- D remains below P22 by `-0.490/-0.760` pp Clean and
  `-0.500/-0.701` pp PN2021-C.

The failure is not a QC artifact. Maximum per-epoch hard-QC rejection was
`3/500 = 0.6%`, decoded-invalid rate among accepted views was zero and
effective anchor share was 0.95. However, the attack itself was
underpowered: mean sample-any-flip ASR was only `1.49%` after AugMix
initialization (`1.81%` after identity initialization), maximum ASR was
`4.60%`, and mean BCE loss gain was only about `0.0035`.

Therefore L8 fails every VAE promotion gate. It is not transferred to
ECGFounder, and no L8 scalar is tuned from the ref-excluded target metrics.
The clean causal conclusion is:

> S11's benefit comes from AugMix pretraining; the tested mild VAE-LHAT
> post-training adds no measurable robustness.

## L9 predeclaration: one precedent-backed attack calibration

One final VAE post-training check is allowed because the frozen internal
attack diagnostic is far outside the project ASR target, not because of a
favorable held-out center or class. L9 keeps the L8 architecture, target,
outer loss, optimizer, E8 budget and K500-only protocol unchanged and replaces
only the attack geometry with the already documented 2026-07-22 Gate-2
calibration:

```text
hull_lambda = 0.20
steps = 10
learning_rate = 0.40
standardized L2 epsilon = 4.0
```

This is a single historical setting, not a new held-out grid. It previously
raised an internal Ningbo attack screen from about 5% to 16.8% ASR while
keeping decoded-invalid rate zero. L9 first runs only the matched
AugMix-initialized B/D pair across all four centers. The identity A/C pair is
run only if D-minus-B PN2021-C AUPRC is at least `+0.15 pp`, Clean AUPRC loses
no more than `0.25 pp`, and at least three centers are non-negative. P22 and
ECGFounder gates remain unchanged. If this one calibrated geometry fails,
close VAE-LHAT post-training rather than expanding another scalar sweep.

L9 is encoded as the frozen preset `historical_gate2_calibrated_v1`; arbitrary
geometry values are rejected by the controller. Its execution IDs are
`l9ri_effnet_e20_identity_vae_calibrated_anchor_soft8` and
`l9ra_effnet_e20_augmix_vae_calibrated_anchor_soft8`. The distinct method IDs
bind the stronger geometry without changing or invalidating the archived L8
profile. Before launch, the controller suite passes 81 tests; the runtime
adapter suite passes 59 tests with only the pre-existing legacy v8 fixture
failure caused by its missing `validation_predictions.execution` mapping.

## 2026-07-27 L9 result: calibrated VAE post-training closed

The predeclared AugMix-initialized B/D pair completed all four centers with
matched E8 optimizer budgets, complete K500 exposure, the same input order,
the same local search/decode workload and ref-excluded Clean/PN2021-C
evaluation. The machine-readable result, including per-epoch paired attack and
hard-QC evidence, is:

```text
/home/linbinhao/ECG_adv_data/runs/agent_workspace/
  a7_global_search_20260726/
  l9ra_effnet_e20_augmix_vae_calibrated_anchor_soft8/summary.json
```

| Arm | Clean AUROC | Clean AUPRC | PN2021-C AUROC | PN2021-C AUPRC |
|---|---:|---:|---:|---:|
| B AugMix initializer + matched control | 0.857315 | 0.530720 | 0.847942 | 0.512763 |
| D AugMix initializer + calibrated VAE-LHAT | 0.856874 | 0.530322 | 0.847520 | 0.512223 |
| D minus B (pp) | -0.044 | -0.040 | -0.042 | -0.054 |

The PN2021-C AUPRC deltas are negative in all four centers:
Ningbo `-0.120 pp`, Chapman-Shaoxing `-0.065 pp`, combined CPSC
`-0.025 pp`, and Georgia `-0.007 pp`. Clean AUPRC also falls by
`-0.040 pp` on average. D remains below P22 by `-0.524/-0.767 pp`
Clean AUROC/AUPRC and `-0.523/-0.721 pp` PN2021-C AUROC/AUPRC.

The stronger search did change the attack rather than silently reproducing L8:
mean any-flip ASR increased from about `1.49%` to `10.98%`, mean BCE loss gain
increased to `0.0419`, mean standardized anchor distance reached `3.98`, and
mean effective anchor share was `0.833`. Maximum epoch ASR was `14.62%`.
Accepted decoded-invalid rate remained zero; the maximum hard-QC fallback was
`3/500 = 0.6%` and was matched in the control.

L9 therefore fails the predeclared `+0.15 pp` PN2021-C AUPRC gate and the
three-of-four non-negative-center gate. The identity A/C pair and ECGFounder
transfer are not run. No additional hull-lambda, epsilon, step-count,
soft-target, loss-weight or epoch grid is permitted from these held-out
metrics. Under the tested sequential recipe, VAE-LHAT short post-training is
closed; S11 remains the compact positive AugMix result.

## L10 predeclaration: bounded AugMax-M20 coupling

L8/L9 close direct endpoint supervision, not every possible coupling between
VAE-LHAT and AugMix. The one remaining mechanism-level candidate is the
already implemented `pure_m20_lhat_augmax_bounded` runtime. It changes how the
hard waveform enters training rather than adding another loss or tuning the
attack against ref-excluded target metrics.

The prior K500-internal Ningbo 400/100 run is used only to freeze one existing
recipe. At epoch 6, the `hull_lambda=0.35`, 10-step, epsilon-10 bounded AugMax
run exceeded ordinary three-chain AugMix by approximately `+0.15 pp` clean
AUPRC and `+0.58 pp` robust AUPRC on the 100-record internal validation split.
Its mean internal attack-success rate was `29.5%`, decoded-invalid rate was
zero, and the bounded coefficient search found an admissible harder mixture
for approximately `74.1%` of views. This is justification for one global
four-center test, not permission for another scalar or center-specific grid.

L10 freezes the following Stage-2 recipe:

```text
initializer = S10 identity-SSL E20 or S11 AugMix-SSL E20
epochs = 6
batch size = 128
learning rate = 5e-5
cosine horizon = 40
LHAT = exact-positive-set, non-self M20
hull_lambda = 0.35
steps = 10
learning rate inside LHAT = 0.25
standardized L2 epsilon = 10
outer objective =
  0.50 clean BCE
  + 0.50 mean over 20 bounded AugMax BCE views
  + 0.30 classic Bernoulli JSD
```

Each of the 20 views fixes chain one to one canonical depth-2/3 composition,
draws an independent raw chain two, and reuses one VAE-LHAT hard waveform as
chain three. For each record/view, the runtime evaluates the ordinary AugMix
draw plus five fixed convex candidates and selects the highest BCE gain in
`[0, 0.15]`; otherwise it falls back to the ordinary draw. No fixed20
endpoint is directly supervised.

The matched control executes the same latent-pool lookup, LHAT optimization,
VAE decode, hard QC, raw-chain generation, six-way coefficient search,
classifier forwards, RNG schedule, 20-view objective, JSD, BatchNorm exposure,
optimizer and K500 order. Only after LHAT generation and QC, its chain-three
waveform is replaced by the corresponding clean waveform. Therefore
candidate-minus-control isolates the decoded VAE-LHAT contribution to bounded
AugMax rather than extra compute, an extra view, or classifier-guided search.

The two factors remain:

- Stage 1: clean-identity SimCLR/VICReg versus two-chain AugMix
  SimCLR/VICReg;
- Stage 2: clean-chain3 bounded AugMax control versus VAE-LHAT-chain3 bounded
  AugMax.

No teacher, source replay, residual head, PCGrad, VAE-random branch, SupCon,
class weighting, per-center rule, per-class rule, checkpoint blending or
held-out checkpoint selection is allowed. All adaptation records remain the
center's K500; evaluation remains reference-excluded with mapping
`v7_super5_sjr_rgq_review_20260528 / 555ec85d5b51`.

Execution first runs the AugMix-initialized B/D pair under IDs
`l10ri_effnet_e20_identity_vae_bounded_augmax6` and
`l10ra_effnet_e20_augmix_vae_bounded_augmax6`. The identity A/C pair is opened
only if D-minus-B PN2021-C AUPRC is at least `+0.15 pp`, clean AUPRC loses no
more than `0.25 pp`, and at least three centers have non-negative robust AUPRC
change. ECGFounder transfer additionally requires D to exceed P22 on both
clean and PN2021-C AUPRC. Failure closes this bounded AugMax coupling; no
threshold, candidate-mixture, attack-geometry, epoch, or loss-weight grid may
be selected from ref-excluded feedback.

## 2026-07-27 L10 result: bounded AugMax coupling closed

The AugMix-initialized B/D pair completed all four centers on physical GPUs
4-7. Both arms used the same S11 E20 initializer, complete K500 order, E6
schedule, 24 optimizer steps, 20 online depth-2/3 AugMax views per base batch,
six-way coefficient-search compute, full VAE-LHAT search/decode/QC workload
and ref-excluded evaluation. The control replaced only the accepted chain-three
waveform with clean after LHAT/QC. Machine-readable metrics and paired
per-epoch evidence are stored at:

```text
/home/linbinhao/ECG_adv_data/runs/agent_workspace/
  a7_global_search_20260726/
  l10ra_effnet_e20_augmix_vae_bounded_augmax6/summary.json
```

| Arm | Clean AUROC | Clean AUPRC | PN2021-C AUROC | PN2021-C AUPRC |
|---|---:|---:|---:|---:|
| B AugMix initializer + full-compute clean-chain3 control | 0.856635 | 0.530997 | 0.847450 | 0.513317 |
| D AugMix initializer + VAE-LHAT bounded AugMax | 0.856086 | 0.530488 | 0.846916 | 0.512635 |
| D minus B (pp) | -0.055 | -0.051 | -0.053 | -0.068 |

Per-center PN2021-C AUPRC deltas were Ningbo `-0.150 pp`,
Chapman-Shaoxing `+0.006 pp`, combined CPSC `-0.096 pp`, and Georgia
`-0.033 pp`. Candidate decoded-invalid and hard-QC rejection rates were zero
in every epoch. In the final epoch, candidate mean selected BCE gains were
approximately `0.030-0.044`, mean VAE-chain weights `0.404-0.440`, and LHAT
any-flip ASR `20.9-33.5%`; therefore this is an active but ineffective
hardness intervention, not a collapsed attack.

L10 fails the predeclared robust-AUPRC and three-of-four-center gates. The
identity A/C pair and ECGFounder transfer are not run. No ref-excluded
threshold, coefficient candidate, attack scalar, epoch or loss-weight
retuning is permitted. Together with L8/L9, this closes both direct
post-training and classifier-guided bounded AugMax; the next admissible mechanism
must return to the historically positive mild coupling where VAE-LHAT is only
an ordinary AugMix third chain, without an outer hardness maximizer.

## L11 predeclaration: integrated mild third-chain factorial

P12 versus P13 is the only completed comparison in which the VAE component was
positive on every pooled metric and in every center: adding a moderate
exact-label VAE-LHAT endpoint as an ordinary AugMix third chain changed Clean
by `+0.119/+0.386 pp` and PN2021-C by `+0.104/+0.266 pp` AUROC/AUPRC.
Unlike L8-L10, it did not directly supervise the hard endpoint or maximize the
outer AugMix mixture against the current classifier.

L11 replays this single historically positive coupling inside the simpler
S10/S11 E20 representation design. It is a full four-cell factorial from the
same locked PTB-XL source checkpoint; no completed checkpoint is used as a
continuation initializer:

| cell | K500-only Stage 1 | E20 supervised Stage 2 |
|---|---|---|
| A Direct | clean-identity SimCLR/VICReg control | Direct+fixed20 |
| B AugMix-only | two-chain AugMix SimCLR/VICReg | Direct+fixed20 |
| C VAE-only | clean-identity SimCLR/VICReg control | fixed20 + mild VAE-LHAT third-chain auxiliary |
| D AugMix-to-VAE | two-chain AugMix SimCLR/VICReg | fixed20 + mild VAE-LHAT third-chain auxiliary |

Every cell freezes:

```text
target data = one logical center's complete K500 only
Stage-1 updates = 4096
Stage-1 batch/LR = 64 / 1e-3
SSL objective = SimCLR + 0.03 VICReg
source replay = off
supervised epochs/steps = E20 / 160 per center
supervised batch/LR/WD = 64 / 1.5e-4 / 1e-4
cosine horizon = 20
class treatment = equal five
checkpoint selection = fixed final epoch
```

The only C/D addition is the frozen P12 third-chain mechanism:

```text
partner policy = exact-positive-set, non-self M20
hull_lambda = 0.60
LHAT steps = 5
standardized L2 epsilon = 8
chain mixing = ordinary three-chain AugMix, Dirichlet/Beta 0.5/0.5
auxiliary terms = AugMix BCE + clean/LHAT/AugMix Bernoulli JSD
auxiliary gradient scale = 0.25
fixed20 base gradient = preserved; conflicts projected by PCGrad
```

PCGrad is a fixed gradient-combination rule, not another searched prediction
module. There is no teacher, source replay, VAE-random view, raw auxiliary
branch, residual head, SupCon, class weight, checkpoint blend, coefficient
search, per-center rule or per-class rule. Candidate and control must record
the same ordered K500 Stage-1 batches, E20 learning-rate schedule, fixed20
exposure counts and canonical corruption RNG traces.

The run identities are
`l11c_effnet_ssl4096_identity_vae_mild_chain3_e20` and
`l11d_effnet_ssl4096_augmix_vae_mild_chain3_e20`; each owns both its Direct
control and VAE candidate arm. Evaluation remains reference-excluded under
`v7_super5_sjr_rgq_review_20260528 / 555ec85d5b51`.

The predeclared VAE gate is:

1. D minus B PN2021-C AUPRC at least `+0.15 pp`;
2. D minus B Clean AUPRC no worse than `-0.25 pp`;
3. D minus B PN2021-C AUPRC non-negative in at least three centers.

ECGFounder transfer additionally requires D to exceed P22 on both Clean and
PN2021-C AUPRC. Failure closes the ordinary mild-chain coupling; no LHAT
radius, step, alpha, epoch or AugMix scalar may be selected from the resulting
ref-excluded metrics.

## 2026-07-27 L11 result: mechanism passes, transfer gate fails

All four L11 cells completed from the locked PTB-XL source checkpoint with
fixed final-epoch selection. The machine-readable causal summary is:

```text
/home/linbinhao/ECG_adv_data/runs/agent_workspace/
  a7_global_search_20260726/
  l11d_effnet_ssl4096_augmix_vae_mild_chain3_e20/
  factorial_summary.json
```

| Cell | Clean AUROC | Clean AUPRC | PN2021-C AUROC | PN2021-C AUPRC |
|---|---:|---:|---:|---:|
| A identity SSL + Direct | 0.857813 | 0.524697 | 0.820308 | 0.456490 |
| B AugMix SSL + Direct | 0.854735 | 0.521377 | 0.845661 | 0.504798 |
| C identity SSL + mild VAE chain3 | 0.858894 | 0.527910 | 0.821592 | 0.459645 |
| D AugMix SSL + mild VAE chain3 | 0.855793 | 0.524466 | 0.846447 | 0.507095 |

AugMix pretraining is the dominant robustness factor: B minus A is
`-0.308/-0.332 pp` Clean and `+2.535/+4.831 pp` PN2021-C AUROC/AUPRC.
The mild VAE third chain remains positive after AugMix: D minus B is
`+0.106/+0.309 pp` Clean and `+0.079/+0.230 pp` PN2021-C AUROC/AUPRC.
PN2021-C AUPRC is positive in all four centers. The preregistered mechanism
gate therefore passes.

D remains below historical P22 by `-0.632/-1.352 pp` Clean and
`-0.631/-1.234 pp` PN2021-C AUROC/AUPRC. The ECGFounder transfer gate fails,
so L11 is not transferred.

## L12 predeclaration: source-preserving correction only

The P22 gap is not explained by its longer E40 supervised horizon. Historical
S01 E40 versus L11 B E20 changes PN2021-C AUPRC by only about `+0.08 pp` and
reduces Clean AUPRC by about `0.30 pp`. P22 also contains two differences:

1. Stage-1 PTB-XL source replay at weight `0.30`;
2. the compound D19 objective: raw corruption, VAE-random, VAE-hard and JSD
   with auxiliary scale `1.5`.

At 512 Stage-1 updates, historical P16 D19 exceeds P19 mild chain3 by about
`0.51 pp` PN2021-C AUPRC, but restoring D19 would obscure the intended
AugMix-pretrain then VAE-LHAT story. L12 therefore tests only the simpler,
standard anti-forgetting correction: supervised PTB-XL source replay during
Stage 1. No target-center record outside K500 is consumed.

The single managed candidate
`l12d_effnet_ssl4096_augmix_replay030_vae_mild_chain3_e20` owns two matched
arms:

| arm | Stage 1 | Stage 2 |
|---|---|---|
| B12 control | two-chain AugMix SimCLR/VICReg + PTB-XL replay 0.30 | Direct+fixed20 |
| D12 candidate | identical Stage 1 | fixed20 + mild VAE-LHAT third chain |

Every other L11D field is frozen: 4096 Stage-1 updates, batch 64, LR `1e-3`,
SimCLR plus `0.03` VICReg, E20 supervised training, LR `1.5e-4`, equal five
classes, final-epoch checkpoint, exact-label non-self M20 partners,
`hull_lambda=0.60`, five LHAT steps, standardized L2 epsilon 8, ordinary
three-chain AugMix and PCGrad auxiliary scale `0.25`.

The gates are frozen before execution:

1. D12 minus B12 PN2021-C AUPRC at least `+0.15 pp`;
2. D12 minus B12 Clean AUPRC no worse than `-0.25 pp`;
3. D12 minus B12 PN2021-C AUPRC non-negative in at least three centers;
4. D12 minus L11D PN2021-C AUPRC at least `+0.50 pp`, with Clean AUPRC no
   worse than `-0.25 pp`;
5. ECGFounder transfer only if D12 exceeds P22 on both Clean and PN2021-C
   AUPRC.

Failure closes this correction. No D19 restoration, replay-weight grid,
auxiliary-alpha grid, epoch extension or held-out checkpoint selection follows
from L12.

## 2026-07-27 L12 result: correction works, P22 transfer gate still fails

Both matched L12 arms completed all four centers from the locked PTB-XL source
checkpoint. They used the same K500-only target records, 4096 Stage-1 updates,
E20 supervised budget, final-epoch selection and reference-excluded evaluation.
The only Stage-2 difference was the predeclared mild VAE-LHAT third chain.
Machine-readable results are stored at:

```text
/home/linbinhao/ECG_adv_data/runs/agent_workspace/
  a7_global_search_20260726/
  l12d_effnet_ssl4096_augmix_replay030_vae_mild_chain3_e20/
  summary.json
```

| Arm | Clean AUROC | Clean AUPRC | PN2021-C AUROC | PN2021-C AUPRC |
|---|---:|---:|---:|---:|
| B12 AugMix SSL + source replay + Direct | 0.861552 | 0.533739 | 0.850973 | 0.513037 |
| D12 same Stage 1 + mild VAE chain3 | 0.862951 | 0.536866 | 0.851983 | 0.515472 |
| D12 minus B12 (pp) | +0.140 | +0.313 | +0.101 | +0.243 |

The D12-minus-B12 PN2021-C AUPRC changes are positive in every center:
Chapman-Shaoxing `+0.338 pp`, combined CPSC `+0.247 pp`, Georgia
`+0.140 pp`, and Ningbo `+0.249 pp`. Gates 1-3 therefore pass.

Relative to L11D, D12 changes Clean by `+0.716/+1.240 pp` and PN2021-C
by `+0.554/+0.838 pp` AUROC/AUPRC. Gate 4 passes. This isolates PTB-XL
source replay during AugMix SSL as a substantial preservation correction;
the VAE third chain remains a smaller, consistent refinement.

Relative to historical P22, D12 changes Clean by `+0.084/-0.112 pp` and
PN2021-C by `-0.077/-0.397 pp` AUROC/AUPRC. It therefore fails both AUPRC
requirements in gate 5. The frozen recipe is not transferred to ECGFounder,
and no post-hoc D19, replay-weight, auxiliary-alpha, epoch or checkpoint grid
is opened. L12 remains single-seed development evidence; PTB-XL source-floor
evaluation is unnecessary because the recipe is not eligible for freezing.

## L13 predeclaration: gradient-budget-matched VAE coupling

L12 closes the replay correction but also provides an internal, target-label
independent reason for one new coupling experiment. Its VAE-LHAT search is
active and QC-clean, while the projected VAE auxiliary-gradient norm added to
the fixed20 gradient is only `0.345-0.360` of the base-gradient norm across
the four centers. The mean alpha required to make the two norms equal is
`0.713`. This diagnosis comes only from training-time K500 gradients, not
reference-excluded PN2021 labels.

L13 keeps the complete L12 data and method graph and changes only how the
already computed VAE auxiliary gradient is coupled:

```text
Stage 1 = two-chain AugMix SimCLR/VICReg, 4096 updates
          + labeled PTB-XL source replay weight 0.30
Stage 2 base = E20 family-balanced Direct+fixed20
VAE view = the same exact-label non-self LHAT third AugMix chain
requested PCGrad auxiliary alpha = 0.75
post-projection auxiliary norm cap = 1.0 * current fixed20 base-gradient norm
```

The alpha is the single rounded value implied by the observed base/auxiliary
norm ratio; it is not a held-out scalar grid. The cap prevents the auxiliary
branch from dominating late or low-base-gradient steps. Conflict projection,
combined-gradient clipping, K500 order, optimizer, learning rate, attack
geometry, AugMix distributions, labels, source replay, final-E20 selection
and evaluation remain unchanged. No raw D19 branch, VAE-random branch,
teacher, class weight, GroupDRO, SupCon, residual head, checkpoint blend or
outside-K500 target record is added.

The candidate id is
`l13d_effnet_ssl4096_augmix_replay030_vae_gradbalanced_chain3_e20`. Its matched
Direct reference is the completed L12 Direct arm because all Direct-visible
training fields and stochastic identities are identical.

Promotion gates are frozen before implementation:

1. L13 minus L12 Direct PN2021-C AUPRC must be at least `+0.15 pp`;
2. L13 minus L12 Direct Clean AUPRC must be no worse than `-0.25 pp`;
3. PN2021-C AUPRC must be non-negative in at least three centers;
4. L13 must exceed P22 on both Clean and PN2021-C AUPRC;
5. cap diagnostics must be finite, the final requested addition must never
   exceed the base-gradient norm, decoded-invalid rate must remain zero, and
   attack success must not collapse below the L12 regime;
6. only after gates 1-5 pass may L13 receive matched PTB-XL source-floor and
   second-seed replication; ECGFounder transfer requires those checks too.

Failure closes gradient-strength calibration. No second alpha, cap, attack,
epoch or held-out checkpoint candidate is selected from this result.

## 2026-07-27 L13 result: stronger VAE coupling helps, P22 gate narrowly fails

L13 completed the frozen E20/final-checkpoint protocol for all four centers
and was evaluated on the reference-excluded Clean and PN2021-C populations.
Its matched Direct reference is the already completed L12 Direct arm. The
machine-readable result is:

```text
/home/linbinhao/ECG_adv_data/runs/agent_workspace/
  a7_global_search_20260726/
  l13d_effnet_ssl4096_augmix_replay030_vae_gradbalanced_chain3_e20/
  summary.json
```

| Arm | Clean AUROC | Clean AUPRC | PN2021-C AUROC | PN2021-C AUPRC |
|---|---:|---:|---:|---:|
| L12 matched Direct | 0.861552 | 0.533739 | 0.850973 | 0.513037 |
| L13 gradient-balanced VAE chain3 | 0.864408 | 0.540221 | 0.853038 | 0.517936 |
| L13 minus Direct (pp) | +0.286 | +0.648 | +0.207 | +0.490 |

PN2021-C AUPRC improves in every center relative to the matched Direct arm:
Chapman-Shaoxing `+0.702 pp`, combined CPSC `+0.580 pp`, Georgia
`+0.280 pp`, and Ningbo `+0.397 pp`. Gates 1-3 pass. Relative to the
otherwise identical L12 mild-coupling candidate, L13 changes Clean by
`+0.146/+0.336 pp` and PN2021-C by `+0.106/+0.246 pp` AUROC/AUPRC. This
supports the training-gradient diagnosis: L12's VAE branch was useful but
under-coupled.

The gradient budget behaved as designed. Each center executed 160 optimizer
steps with alpha `0.75`. Mean auxiliary-addition norms remained below mean
fixed20 base-gradient norms. The cap activated on `16.25%-23.75%` of steps,
conflict projection activated on `30.63%-51.88%`, no pending auxiliary
gradient remained at exit, sample-anyflip ASR was `21.48%-23.18%`, decoded
invalid rate was zero, and no waveform failed hard QC. Runtime source
identities were stable and all new runs snapshot the complete 17-file sandbox
runtime surface.

Relative to historical P22, L13 changes Clean by `+0.230/+0.223 pp` and
PN2021-C by `+0.028/-0.150 pp` AUROC/AUPRC. The robust-AUPRC deficit is
concentrated in Chapman-Shaoxing (`-0.904 pp` versus P22); CPSC, Georgia and
Ningbo differ by `+0.184`, `-0.028` and `+0.148 pp`, respectively. Gate 4
therefore fails despite the strong matched-control result.

Per the frozen decision rule, L13 remains single-seed development evidence.
No PTB-XL source-floor run, second seed or ECGFounder transfer is launched,
and the alpha/cap/epoch/checkpoint axis is closed. The run also exposed a
reporting-only legacy defect: `pcgrad_diagnostics.json` describes unused D19
term-scaling metadata even though the generated method profile, training
history and gradients correctly execute the strong LHAT third-chain
objective. This does not change the measurements above but must be repaired
before any future replicated run.

## L14 predeclaration: diverse VAE post-training views

L13 closes the gradient-strength axis but leaves one mechanism difference from
P22 unresolved. The E40-versus-E20 evidence shows that the longer supervised
horizon explains only about `+0.08 pp` PN2021-C AUPRC. At the matched
512-step Stage-1 budget, the P16 separate-view objective exceeds the P19
single strong-LHAT third-chain objective by `+0.510 pp` PN2021-C AUPRC. The
remaining interpretable hypothesis is therefore view diversity, not another
alpha, cap, attack-radius, epoch or checkpoint setting.

L14 keeps the complete L12 Stage-1 correction and E20 supervised budget:

```text
Stage 1 = two-chain AugMix SimCLR/VICReg, 4096 updates
          + labeled PTB-XL source replay weight 0.30
Stage 2 base = E20 family-balanced Direct+fixed20
Stage 2 auxiliary views =
  one online raw depth2/3 corruption reference
  + one local compatible VAE-random view
  + one matched VAE-LHAT-hard view
  + Bernoulli JSD across clean/raw/random/hard
PCGrad auxiliary alpha = frozen historical D19 value 1.5
```

This is one post-training VAE-diversity component. GroupDRO, SupCon,
PairRank, class weighting, residual heads, EMA online teacher, checkpoint
blend, candidate replay and outside-K500 target records remain absent or at
their exact disabled values. The raw reference is retained because historical
P08/P09 reallocations away from it were neutral-to-negative; no new term
weight is selected.

The candidate id is
`l14d_effnet_ssl4096_augmix_replay030_vae_diverse_d19_e20`. Its matched Direct
reference is the completed L12 Direct arm: source checkpoint, K500 order,
Stage-1 target/source batches, Stage-2 optimizer, E20 schedule, fixed20
exposures and final-checkpoint rule are identical. The existing full-K500
authorization remains development-only; a promotable replay must use a
frozen, non-oracle promotion record.

Promotion gates are frozen before implementation:

1. L14 minus L12 Direct PN2021-C AUPRC must be at least `+0.50 pp`;
2. L14 minus L12 Direct Clean AUPRC must be at least `+0.25 pp`;
3. PN2021-C AUPRC must be non-negative in all four centers;
4. L14 must exceed P22 on both Clean and PN2021-C AUPRC;
5. all diagnostics must be finite, decoded-invalid rate must remain zero,
   hard-search loss gain must be non-negative, and only D19-active diagnostic
   metadata may be emitted;
6. only after gates 1-5 pass may the recipe receive a PTB-XL source-floor
   evaluation and an independently seeded frozen replay;
7. ECGFounder transfer requires gates 1-6 and uses the same global method
   graph without per-center or per-class settings.

Failure closes the diverse-view restoration. No second D19 term allocation,
auxiliary alpha, attack geometry, supervised epoch, held-out checkpoint or
center-specific variant is selected from the result.

## 2026-07-27 L14 result: diverse VAE views clear the frozen P22 gate

L14 completed the frozen E20/final-checkpoint protocol for all four centers
and was evaluated on the reference-excluded Clean and PN2021-C populations.
Its matched Direct reference is the completed L12 Direct arm. The
machine-readable result is:

```text
/home/linbinhao/ECG_adv_data/runs/agent_workspace/
  a7_global_search_20260726/
  l14d_effnet_ssl4096_augmix_replay030_vae_diverse_d19_e20/
  summary.json
```

| Arm | Clean AUROC | Clean AUPRC | PN2021-C AUROC | PN2021-C AUPRC |
|---|---:|---:|---:|---:|
| L12 matched Direct | 0.861552 | 0.533739 | 0.850973 | 0.513037 |
| L14 diverse VAE views | 0.865053 | 0.541556 | 0.854184 | 0.519961 |
| L14 minus Direct (pp) | +0.350 | +0.782 | +0.321 | +0.692 |

PN2021-C AUPRC improves in every center relative to the matched Direct arm:
Chapman-Shaoxing `+1.295 pp`, combined CPSC `+0.485 pp`, Georgia
`+0.415 pp`, and Ningbo `+0.575 pp`. Gates 1-3 pass.

Relative to historical P22, L14 changes Clean by `+0.294/+0.357 pp` and
PN2021-C by `+0.143/+0.052 pp` AUROC/AUPRC. Gate 4 therefore passes, but the
PN2021-C AUPRC margin is narrow and is not treated as a stable superiority
claim before an independent frozen replay.

All four centers completed 20 supervised epochs and 160 optimizer steps.
Stage 1 used source replay weight `0.30` and consumed `262144` labeled PTB-XL
source samples per center. Hard-search loss gain remained positive, decoded
invalid rate was zero, runtime source identities remained stable, and only
D19-active diagnostic metadata was emitted. Ningbo and Georgia recorded
`120` and `40` hard-QC rejections respectively; the runtime used the
predeclared fallback behavior and completed without non-finite decoded
waveforms. Gate 5 passes.

L14 remains `heldout_tuned_development_only` and single-seed evidence. Per
gates 6-7, the next operations are limited to:

1. evaluate all four frozen L14 checkpoints on PTB-XL fold10 as a source
   retention audit;
2. if source retention is acceptable, replay the exact L14-versus-Direct pair
   with one independent global seed;
3. freeze the resulting global recipe before any ECGFounder transfer.

No L14 hyperparameter, center-specific setting, per-class weight, epoch or
checkpoint is changed from the result above.

The source audit is frozen before reading its metrics. Both the four L14
checkpoints and the four matched L12 Direct checkpoints receive the identical
PTB-XL fold10 evaluation:

- an absolute strict-retention pass requires mean AUROC and AUPRC to be no
  more than `1.0 pp` below the locked source checkpoint, with no individual
  checkpoint more than `1.5 pp` below the locked source AUPRC;
- a target-specialist comparative pass requires L14 mean source AUPRC to be no
  more than `0.25 pp` below matched Direct and no individual center-paired
  L14 checkpoint more than `1.0 pp` below its Direct counterpart.

Failure of the absolute rule forbids a source-preservation claim. Failure of
the comparative rule stops the independent-seed replay and ECGFounder
transfer. Passing the comparative rule but failing the absolute rule may
advance only as explicitly labeled target-specialist adaptation evidence.

## 2026-07-27 L14 source-floor result: target gain costs excess forgetting

All eight frozen checkpoints received the same PTB-XL fold10 evaluation after
the source rule above was written. The results are:

| Arm | PTB-XL AUROC | PTB-XL AUPRC | Delta from locked source (pp) |
|---|---:|---:|---:|
| Locked EffNet source | 0.901281 | 0.767951 | — |
| L12 matched Direct mean | 0.882475 | 0.736034 | -1.881 / -3.192 |
| L14 diverse VAE mean | 0.875043 | 0.719734 | -2.624 / -4.822 |
| L14 minus Direct (pp) | -0.743 | -1.630 | — |

Neither arm passes the absolute strict-retention rule. More importantly, L14
also fails the comparative rule: its source AUPRC is `1.630 pp` below matched
Direct. Every center-paired L14 checkpoint is worse than Direct on source
AUPRC: Chapman-Shaoxing `-2.234 pp`, combined CPSC `-1.130 pp`, Georgia
`-1.516 pp`, and Ningbo `-1.640 pp`.

The interpretation is narrow but decisive. Diverse VAE post-training produces
a real target-domain gain, yet the current Stage-2 coupling obtains part of
that gain by moving farther away from the source solution. L14 remains useful
mechanism evidence, but it is not a frozen final recipe and cannot be described
as source preserving.

Per the predeclared rule, the L14 second-seed replay and ECGFounder transfer
are stopped. Any next candidate must address Stage-2 source retention as a
single explicit stability correction while keeping the AugMix pretraining and
VAE-random/VAE-LHAT/JSD mechanism fixed. It must restart from the locked source
checkpoint and receive a fresh matched Direct comparison; it may not continue
from L14 weights.

## L15 predeclaration: matched Stage-2 source replay

L15 changes one stability control only. It restarts from the same locked
EfficientNet source checkpoint and retains the complete L14 target mechanism:

```text
Stage 1 = two-chain AugMix SimCLR/VICReg, 4096 updates
          + labeled PTB-XL source replay weight 0.30
Stage 2 base = E20 family-balanced Direct+fixed20
Stage 2 auxiliary = raw depth2/3 + VAE-random + VAE-LHAT-hard + JSD
PCGrad auxiliary alpha = 1.5
```

During Stage 2, both the candidate and its newly trained Direct control receive
the same deterministic PTB-XL train replay batch on every clean base-batch
objective. Its total loss weight is the already implemented historical value
`0.10`. Source forwards do not update BatchNorm running statistics and restore
the process RNG state, so only the source gradient enters the shared optimizer
step. This is a stability control and is not presented as method novelty.

The candidate id is
`l15d_effnet_ssl4096_augmix_replay030_stage2replay010_vae_diverse_d19_e20`.
Relative to L14, all target data, K500 identity, target RNG identity, AugMix
views, VAE geometry, D19 terms, optimizer, epoch, checkpoint and evaluation
fields remain fixed. There is no continuation from L14 weights.

Promotion gates are frozen before execution:

1. L15 minus its new replay-matched Direct must improve PN2021-C AUPRC by at
   least `+0.50 pp`;
2. Clean AUPRC must improve by at least `+0.25 pp`;
3. PN2021-C AUPRC must be non-negative in all four centers;
4. L15 must exceed P22 on both Clean and PN2021-C AUPRC;
5. source replay must execute the same number and order of PTB-XL batches in
   candidate and Direct for each center, with finite BCE and stable runtime
   identities;
6. L15 source AUPRC may be no more than `0.25 pp` below its replay-matched
   Direct mean, and no center pair may be worse by more than `1.0 pp`;
7. only after gates 1-6 pass may the exact L15-versus-Direct pair receive an
   independent-seed replay, followed by a globally frozen ECGFounder transfer.

No second replay weight, D19 term allocation, auxiliary alpha, epoch,
checkpoint, per-center setting or per-class setting will be selected from this
result.

## 2026-07-27 L15 result: target gain survives, replay does not fix forgetting

L15 and its newly trained Direct arm completed the exact E20/final-checkpoint
protocol for all four centers. Each arm consumed 160 deterministic Stage-2
source batches and 10,240 source records per center. Candidate and Direct
source-order hashes match exactly within every center.

Target evaluation:

| Arm | Clean AUROC | Clean AUPRC | PN2021-C AUROC | PN2021-C AUPRC |
|---|---:|---:|---:|---:|
| L15 replay-matched Direct | 0.861159 | 0.533206 | 0.850539 | 0.512627 |
| L15 diverse VAE + replay | 0.864688 | 0.541318 | 0.853825 | 0.519613 |
| L15 minus Direct (pp) | +0.353 | +0.811 | +0.329 | +0.699 |

PN2021-C AUPRC improves in all four centers: Chapman-Shaoxing
`+1.326 pp`, combined CPSC `+0.480 pp`, Georgia `+0.433 pp`, and Ningbo
`+0.554 pp`. L15 exceeds P22 by `+0.333 pp` Clean AUPRC and only
`+0.018 pp` PN2021-C AUPRC. Target gates 1-5 pass, but the P22 robust margin
remains too narrow to treat as stable superiority.

PTB-XL fold10 source audit:

| Arm | PTB-XL AUROC | PTB-XL AUPRC | Delta from locked source (pp) |
|---|---:|---:|---:|
| Locked EffNet source | 0.901281 | 0.767951 | — |
| L15 replay-matched Direct mean | 0.884533 | 0.739913 | -1.675 / -2.804 |
| L15 diverse VAE + replay mean | 0.877327 | 0.724085 | -2.395 / -4.387 |
| L15 minus Direct (pp) | -0.721 | -1.583 | — |

The `0.10` Stage-2 replay improves L14 absolute source AUPRC by about
`+0.435 pp`, but improves the candidate-versus-Direct source gap by only
about `+0.047 pp`. Every center pair remains worse than Direct:
Chapman-Shaoxing `-2.129 pp`, combined CPSC `-1.058 pp`, Georgia
`-1.523 pp`, and Ningbo `-1.621 pp`. Gate 6 fails.

The conclusion is that low-weight supervised replay is not the missing source
retention mechanism. L15 is not promoted, and its second seed and ECGFounder
transfer are stopped. Per the predeclaration, the replay-weight axis is closed;
no larger replay weight is selected from this result.

## L16 predeclaration: frozen independent target-specialist replay

The active research objective is a per-center K500-only target-adaptation
method, not a source-preserving continual-learning method. L11 has already
completed the requested optimizer-, data-, seed- and budget-matched four-cell
factorial:

```text
A Direct
B AugMix-SimCLR/VICReg only
C VAE-LHAT only
D AugMix-SimCLR/VICReg -> VAE-LHAT
```

L14/L15 additionally show that the diverse VAE post-training branch produces
positive Clean and PN2021-C deltas over a fresh matched Direct arm in all four
centers, but their PTB-XL source audits prohibit any source-preservation claim.
That diagnostic does not by itself answer whether the K500-only target gain is
replicable. L16 therefore changes the claim boundary before execution:
PTB-XL forgetting remains reported as a limitation and is not used as an
ECGFounder transfer gate; no source-retention claim will be made.

The single candidate
`l16d_effnet_ssl4096_augmix_replay030_vae_diverse_d19_e20_seed1` is an exact
frozen replay of L14. The only changed field is:

```text
random_seed.replicate_id: 0 -> 1
```

Everything else remains fixed: locked PTB-XL source checkpoint, per-center
K500 identities, two-chain AugMix SimCLR plus `0.03` VICReg, 4096 Stage-1
updates, source replay weight `0.30`, E20 family-balanced Direct+fixed20,
raw/VAE-random/VAE-LHAT-hard/JSD auxiliary bundle, PCGrad scale `1.5`, equal
five classes, final checkpoint, v7 mapping and reference-excluded Clean plus
PN2021-C evaluation. Its Direct control is rerun under the same replicate-1
identity; seed-0 checkpoints are not continued.

The independent replay passes only if:

1. L16 minus its seed-matched Direct improves PN2021-C AUPRC by at least
   `+0.50 pp`;
2. Clean AUPRC improves by at least `+0.25 pp`;
3. PN2021-C AUPRC is non-negative in all four centers;
4. L16 itself exceeds P22 on both Clean and PN2021-C AUPRC;
5. both L14 seed 0 and L16 seed 1 satisfy gates 1-4 independently, rather than
   only their two-seed mean;
6. runtime records prove `replicate_id=1`, candidate/Direct K500 identity and
   matched optimizer/exposure budgets.

Only after these gates pass is the recipe frozen for ECGFounder transfer. No
loss weight, replay weight, epoch, checkpoint, class, center or attack
hyperparameter may be changed from the independent result.

## 2026-07-27 L16 audit: partial rather than full independent RNG replay

L16 completed and its target metrics passed gates 1-5. However, the
post-run stochastic audit found that the Stage-1 target-batch and PTB-XL
source-replay hashes were identical to seed 0. The online YAML carried
`replicate_id: 1`, and the outer supervised runtime consumed it, but the
target SSL adapter built its own target loader, source loader, projection-head
seed and AugMix generator without including `replicate_id`.

Therefore L16 is retained as useful repeated-execution evidence but is not
counted as a fully independent second seed. Its metrics must not be used in a
two-independent-seed claim.

The adapter is corrected in place so that `replicate_id` enters:

1. the Stage-1 target-loader namespace;
2. the Stage-1 PTB-XL replay-loader namespace;
3. projection-head initialization;
4. the Stage-1 AugMix generator;
5. residual-head initialization when that optional path is active.

The correction also retains source-loader identity until diagnostics are
serialized. Candidate and matched Direct still share the same replicate
identity; only different replicates receive different streams.

## L17 predeclaration: full-RNG independent replay

`l17d_effnet_ssl4096_augmix_replay030_vae_diverse_d19_e20_seed1_fullrng`
is an exact L14 recipe replay with `replicate_id: 1` under the corrected
Stage-1 seed propagation. It uses a new run directory and does not overwrite
or continue L16.

All L16 metric gates remain unchanged. In addition, promotion now requires:

1. L17 candidate and Direct have identical target/source Stage-1 order hashes
   within each center;
2. L17 Stage-1 target and source hashes differ from seed-0 L14/L12 hashes;
3. L17 diagnostics contain the complete PTB-XL source-loader cache, split and
   seed identities;
4. candidate and Direct still have identical 160-step E20 fixed20 traces and
   final-checkpoint selection.

The Stage-1 seed repair changes no loss, data population, method component,
optimizer, epoch, class weight, center setting, attack setting or evaluation
rule.

## L18 predeclaration: exact L14 transfer to ECGFounder

Candidate:
`l18d_ecgfounder_ssl4096_augmix_replay030_vae_diverse_d19_e20`.

This run answers one narrow question: does the frozen L14 target-center
adaptation graph transfer from EfficientNet1DV2 to ECGFounder? It changes only
the backbone-specific fields:

- model: `efficientnet1dv2` -> `ecgfounder`;
- locked PTB-XL source checkpoint: the managed ECGFounder source checkpoint;
- supervised full-finetuning learning rate: `1.5e-4` -> `2e-5`;
- evaluation batch size follows the existing ECGFounder memory contract.

Everything that defines L14 remains frozen: per-center K500-only adaptation,
two-chain AugMix SimCLR/VICReg Stage 1, 4096 Stage-1 updates at `1e-3`,
PTB-XL replay weight `0.30`, equal-five source-logit anchors, E20/T20
family-balanced clean+fixed20 supervision, and the D19 raw/VAE-random/
VAE-LHAT-hard/JSD auxiliary views with coefficient `1.5`.

Four logical centers are trained independently from the same locked ECGFounder
source checkpoint. The first wave runs the candidate on four GPUs to obtain
Clean and PN2021-C metrics quickly. A generated matched Direct arm is retained
for a later budget-matched causal comparison; historical locked Direct and P29
are context only and are not substituted for that control.

This is an exact-transfer stress test, not a tuned ECGFounder recipe. Historical
P25 evidence warns that 4096 Stage-1 updates at `1e-3` may overtrain
ECGFounder, so a negative result must not be repaired by silently shortening
Stage 1 after seeing held-out metrics.

### L18 result

The four-center candidate wave completed successfully on 2026-07-27. All four
managed runs completed 4096 Stage-1 updates, E20/160 Stage-2 optimizer updates,
strict checkpoint loading, and ref-excluded Clean plus PN2021-C evaluation.

| Center | Clean AUROC | Clean AUPRC | PN2021-C AUROC | PN2021-C AUPRC |
|---|---:|---:|---:|---:|
| Ningbo | 0.884863 | 0.498191 | 0.870693 | 0.478692 |
| Chapman-Shaoxing | 0.894046 | 0.501301 | 0.877632 | 0.485885 |
| CPSC 2018 + Extra | 0.843810 | 0.480375 | 0.819296 | 0.455715 |
| Georgia | 0.809756 | 0.477646 | 0.788145 | 0.469734 |
| Four-center mean | 0.858119 | 0.489378 | 0.838942 | 0.472507 |

The exact L14 transfer is rejected. Relative to the previously locked
ECGFounder Direct+fixed20 result (`0.889740/0.582508` Clean and
`0.830200/0.496628` PN2021-C), the non-matched historical comparison is:

- Clean: `-3.16 pp AUROC / -9.31 pp AUPRC`;
- PN2021-C: `+0.87 pp AUROC / -2.41 pp AUPRC`.

The result is also below P29 on both slices. It is consistent with the
predeclared P25 warning: 4096 Stage-1 updates at `1e-3` over-adapt ECGFounder.
The generated L18 Direct arm was intentionally not run in the first
time-critical wave, so this result establishes absolute transfer failure but
is not presented as a budget-matched causal delta.

## 2026-07-29 next phase: make VAE-LHAT independently stable

R4/R8 now isolate a positive but small current VAE effect:
`+0.016/+0.167 pp` PN2021-C AUROC/AUPRC, with AUPRC positive in all four
centers.  The same VAE branch costs `-0.778/-1.775 pp` PTB-XL AUROC/AUPRC
relative to the no-VAE R8 arm.  This is mechanism evidence, not a satisfactory
final trade-off.

The next phase must not add another loss stack. It reuses the existing
controller and already implemented L12/L13 strong-LHAT third-chain profiles:

1. audit the completed seed-0 L13 source floor;
2. replay the exact L13 versus its matched Direct control with one full-RNG
   independent seed;
3. only if the VAE effect remains positive on Clean and PN2021-C AUPRC and is
   non-negative in at least three centers, run a third seed;
4. then remove one mechanism at a time: optimized hard search versus a
   compute-matched non-adversarial latent control, and gradient conflict
   projection versus norm cap;
5. freeze one global setting before ECGFounder transfer.

The promotion target is not a guaranteed favorable number. It is:

- positive mean Clean and PN2021-C AUPRC in every completed seed;
- PN2021-C AUPRC non-negative in at least three centers per seed;
- finite search diagnostics, zero decoded-invalid rate, and useful but not
  saturated attack pressure;
- smaller PTB-XL source penalty than the current R4 branch;
- fixed K500-only adaptation, final-checkpoint selection, equal-five classes,
  and no per-center/per-class tuning.

Historical seed-0 L13 is the starting point because it already isolates a
larger VAE effect over a matched Direct arm:
`+0.286/+0.648 pp` Clean and `+0.207/+0.490 pp` PN2021-C AUROC/AUPRC, with
PN2021-C AUPRC positive in all four centers. Its independent replay, source
floor, and mechanism controls remain outstanding.

## 2026-07-29 L13/L19 VAE-LHAT independent replication

Before replay, the controller audit found a metadata-only defect: the
historical L13 candidate object still declared standardized epsilon `2`,
whereas its generated method, copied runtime config and completed diagnostics
all used `train/lhat_h060_s5_eps8.yaml` and reached `atk_anchor_l2=8`. No
historical waveform or result was generated with epsilon 2. The controller now
fails closed when candidate metadata and the referenced LHAT YAML disagree;
L13 and its exact replays explicitly record epsilon 8.

The seed-0 L13 source floor is:

| Arm | PTB-XL AUROC | PTB-XL AUPRC |
|---|---:|---:|
| matched L12 Direct | 0.882475 | 0.736034 |
| L13 VAE-LHAT | 0.876298 | 0.722040 |
| L13 minus Direct (pp) | -0.618 | -1.399 |

The full-RNG independent replay
`l19d_effnet_ssl4096_augmix_replay030_vae_gradbalanced_chain3_e20_seed1`
then completed all four target centers from the locked source checkpoint.
Candidate and Direct used the same K500 identities and the same Stage-1 target
and PTB-XL replay order within each center, while both order hashes differed
from seed 0. Both arms completed E20 and 160 outer optimizer updates and used
the final checkpoint.

| Arm | Clean AUROC | Clean AUPRC | PN2021-C AUROC | PN2021-C AUPRC |
|---|---:|---:|---:|---:|
| seed-1 matched Direct | 0.861024 | 0.533005 | 0.850246 | 0.513058 |
| seed-1 VAE-LHAT | 0.864102 | 0.540529 | 0.852344 | 0.518530 |
| VAE-LHAT minus Direct (pp) | +0.308 | +0.752 | +0.210 | +0.547 |

PN2021-C AUPRC is positive in every center: Chapman-Shaoxing `+0.855
pp`, combined CPSC `+0.501 pp`, Georgia `+0.340 pp`, and Ningbo `+0.493
pp`. Across seeds 0 and 1, the mean VAE-LHAT delta is `+0.297/+0.700 pp`
Clean AUROC/AUPRC and `+0.208/+0.519 pp` PN2021-C AUROC/AUPRC. Both seeds
independently have positive Clean and PN2021-C AUPRC and non-negative
PN2021-C AUPRC in all four centers.

Seed-1 attack diagnostics are active but not degenerate: mean any-flip ASR is
approximately `19.7-24.9%` across centers, mean BCE gain is
`0.056-0.074`, standardized anchor distance reaches the declared epsilon 8,
and decoded-invalid plus quality-rejected counts are both zero. Source
retention remains a limitation, but the seed-1 candidate-versus-Direct PTB-XL
gap (`-0.633/-1.412 pp` AUROC/AUPRC) is smaller than the R4-versus-R8 VAE
penalty (`-0.778/-1.775 pp`).

These results pass the predeclared two-seed target and source-penalty gates.
They establish a small, repeatable positive VAE-LHAT contribution; they do not
show that VAE-LHAT is the dominant source of the full pipeline gain.

## L20 predeclaration: third frozen VAE-LHAT seed

`l20d_effnet_ssl4096_augmix_replay030_vae_gradbalanced_chain3_e20_seed2`
is an exact L13 replay with only `replicate_id: 0 -> 2`. It changes no data,
method, loss, attack, optimizer, epoch, class, center, checkpoint or evaluation
field. Its matched Direct arm shares the same replicate-2 Stage-1 and Stage-2
random identities.

The third seed passes only if Clean and PN2021-C AUPRC are both positive
against its matched Direct, PN2021-C AUPRC is non-negative in at least three
centers, attack diagnostics remain finite with zero invalid decodes, and its
PTB-XL source penalty is no worse than the R4 VAE branch. No hyperparameter is
changed in response to this seed.

## L21 predeclaration: compute-matched non-adversarial latent control

L21 is run only after L20 completes. It asks whether L13's repeatable gain
requires classifier-guided LHAT search or merely any VAE latent-neighbor view.
It reuses the complete L20 replicate-2 graph and Direct owner and changes one
mechanism:

```text
L20: five Adam steps update the M20 softmax weights to maximize BCE
L21: the same five decode/classify/backward passes execute, but the
     float32 softmax logits remain numerically uniform
```

The control keeps exact-label non-self nearest M20 partners, hull lambda
`0.60`, epsilon `8`, the same VAE decode, QC, third-chain AugMix draw,
auxiliary losses, alpha `0.75`, conflict projection, norm cap, optimizer,
E20 budget, K500 order, seed, checkpoint and evaluation. Its generated LHAT
config records `weight_mode=uniform_compute_matched` and learning rate
`1e-30`; diagnostics must confirm coefficient top-1 approximately `0.05`,
entropy approximately `log(20)`, and gain versus the initial uniform hull
approximately zero.

The adversarial-search mechanism is supported only if L20 exceeds L21 on mean
PN2021-C AUPRC, does not lose more than `0.25 pp` Clean AUPRC, and is
non-negative in at least three centers. If not, the honest conclusion is that
VAE latent diversity is useful but the five-step adversarial weight search is
not independently established.

## L20 execution audit and L22 frozen replay

The first L20 execution completed all four candidate and four matched-Direct
training arms through epoch 20 and optimizer step 160, but every managed run
was correctly marked failed because `a7_global_search.py` changed while the
delegates were active. A runtime-snapshot comparison found no change in the
model, trainer, data, augmentation or LHAT runtime files; nevertheless, these
checkpoints are retained only as failed provenance and are not admitted as
formal replicate evidence.

`l22d_effnet_ssl4096_augmix_replay030_vae_gradbalanced_chain3_e20_seed2_replay`
therefore replays L20 in a new output namespace. It is byte-for-byte identical
at the scientific-candidate level except for the candidate identifier and keeps
`replicate_id=2`. No method or hyperparameter is changed in response to the
failed attempt. All tracked runtime/controller sources remain frozen from
prepare through completion.

## L21 result and direct-supervision follow-up

The formal L22 seed-2 replay passed and extended the frozen L13 result to three
independent seeds. Mean candidate-minus-matched-Direct improvement is
`+0.299/+0.701 pp` on Clean AUROC/AUPRC and `+0.212/+0.520 pp` on PN2021-C
AUROC/AUPRC. All three seeds improve both Clean and PN2021-C AUPRC and all four
centers improve PN2021-C AUPRC in seed 2. The target-domain contribution is
therefore repeatable, but the candidate is not source preserving: seed-2
candidate-minus-Direct PTB-XL is approximately `-0.637/-1.472 pp`.

L21 confirms that its uniform control is genuinely non-adversarial:
`gain_vs_initial` is approximately zero while the optimized L22 search produces
substantial positive BCE gain. Nevertheless, L22-minus-L21 is only
`+0.003/+0.017 pp` PN2021-C AUROC/AUPRC and is positive in only two of four
centers. The predeclared adversarial-search gate therefore fails. The current
evidence supports the VAE latent-neighborhood branch, not an independent claim
for the five-step coefficient search.

The next frozen mechanism screen moves the LHAT waveform out of stochastic
three-chain mixing and supervises it directly:

- L23 is the compute-matched uniform control for the existing R4 direct-LHAT
  path at hull lambda `0.60`, five steps and epsilon `2`.
- L24 uses the same R4 path with a local low-dose geometry: hull lambda `0.10`,
  three search steps and epsilon `2`.
- L25 is L24's compute-matched uniform control.

All three preserve the R4 Stage-1 two-chain AugMix SimCLR, source replay,
anchors, E40 Direct+fixed20 base, direct hard BCE, clean/hard JSD, K500-only
population, optimizer and RNG identity. The direct-supervision adversarial
mechanism is supported only if hard exceeds its geometry-matched uniform
control on PN2021-C AUPRC, loses no more than `0.25 pp` Clean AUPRC, and is
non-negative in at least three centers.

## 2026-07-29 L26/L27 objective screen and pruned ECGFounder transfer

L26 replaced the local-hard multilabel BCE search objective with a per-record
equal-positive/equal-negative BCE objective. Relative to its geometry- and
compute-matched L25 local-uniform control, it changed Clean by
`+0.006/+0.037 pp` AUROC/AUPRC and PN2021-C by `+0.008/+0.031 pp`.
PN2021-C AUPRC was positive in three of four centers, but the frozen
`+0.10 pp` mean-AUPRC gate failed.

L27 replaced only the attack objective with Bernoulli KL divergence from the
decoded anchor prediction. The KL objective increased throughout training and
decoded-invalid rate remained zero, but relative to L25 it changed Clean by
`-0.013/-0.042 pp` and PN2021-C by `-0.006/-0.018 pp`; PN2021-C AUPRC was
negative in all four centers. The attack-objective axis is closed. These
results do not support an independent gain from balanced-BCE or VAT-style
coefficient search.

The remaining current-mainline task is a backbone transfer of the already
pruned R4 graph, not another loss stack. Two candidates are frozen:

```text
L28: ECGFounder, SSL1024 two-chain AugMix-SimCLR
     + PTB-XL replay 0.30 + equal-five logit anchors
     -> E40 family-balanced Direct+fixed20

L29: identical L28 graph and RNG identity
     + one exact-label VAE-LHAT hard view
     + direct hard BCE and clean/hard Bernoulli JSD
```

Candidate ids:

```text
l28d_ecgfounder_ssl1024_augmix_replay030_no_vae_e40
l29d_ecgfounder_ssl1024_augmix_replay030_vae_hard_e40
```

They change no K500 identity, target population, class weight, optimizer,
epoch, checkpoint-selection, source checkpoint, Stage-1 view, replay, anchor,
or evaluation field. Both use final E40 and the same comparison RNG identity.
No outside-K500 target record enters model adaptation.

The VAE branch is promoted only if L29 minus L28:

1. improves mean PN2021-C AUPRC by at least `+0.15 pp`;
2. is non-negative on PN2021-C AUPRC in at least three centers;
3. loses no more than `0.50 pp` mean Clean AUPRC;
4. has finite attack diagnostics and zero decoded-invalid rate.

Outside-K500 labels are used only for the final global development comparison,
so the result remains `heldout_tuned_development_only` until an independent
frozen-protocol replication.

### L28/L29 result

Both candidates completed four independently adapted logical centers through
the frozen final E40 checkpoint and reference-excluded Clean/PN2021-C
evaluation.

| Arm | Clean AUROC | Clean AUPRC | PN2021-C AUROC | PN2021-C AUPRC |
|---|---:|---:|---:|---:|
| L28 no VAE | 0.885928 | 0.565970 | 0.867065 | 0.543365 |
| L29 VAE-LHAT | 0.888339 | 0.576046 | 0.868788 | 0.547580 |
| L29 minus L28 (pp) | +0.241 | +1.008 | +0.172 | +0.422 |

PN2021-C AUPRC improves in all four centers: Chapman-Shaoxing `+0.213 pp`,
combined CPSC `+0.856 pp`, Georgia `+0.175 pp`, and Ningbo `+0.442 pp`.
All frozen target-domain promotion gates pass. At E40, mean LHAT BCE gain is
positive in every center (`0.0072-0.0088`), sample-anyflip ASR is
`2.3%-4.0%`, and decoded-invalid rate is zero.

For practical context only, L29 versus the historical locked ECGFounder
Direct+fixed20 result changes Clean by approximately `-0.140/-0.646 pp` and
PN2021-C by `+3.859/+5.095 pp`. This is a whole-recipe comparison, not the
causal VAE delta; the causal matched VAE delta is the L29-minus-L28 row above.

The PTB-XL source-floor result does not pass:

| Arm | PTB-XL AUROC | PTB-XL AUPRC |
|---|---:|---:|
| locked ECGFounder source | 0.929407 | 0.824164 |
| L28 no VAE | 0.915229 | 0.794021 |
| L29 VAE-LHAT | 0.907931 | 0.779246 |
| L29 minus L28 (pp) | -0.730 | -1.478 |

The current minimal narrative is therefore supported on the target domain:
two-chain AugMix-SimCLR supplies the main robust adaptation, and one online
VAE-LHAT hard branch adds a smaller but positive refinement on both Clean and
PN2021-C. The remaining limitation is source retention. Do not present L29 as
a no-trade-off global improvement, and do not attribute the full practical
gain to VAE-LHAT.

## L24/L25 result and effective-objective repair

The local direct-supervision comparison completed on all four logical centers.
L24 minus its compute-matched L25 uniform-neighbor control is:

| Slice | AUROC delta | AUPRC delta |
|---|---:|---:|
| Clean | +0.007 pp | +0.052 pp |
| PN2021-C depth2+3 | +0.003 pp | +0.029 pp |

PN2021-C AUPRC is positive for Chapman-Shaoxing (`+0.042 pp`), Georgia
(`+0.009 pp`) and Ningbo (`+0.076 pp`), but slightly negative for combined
CPSC (`-0.010 pp`). The local VAE-neighbor branch itself is stable:
L25 minus the no-VAE R8 arm is `+0.130 pp` PN2021-C AUPRC and is positive in
all four centers. The current three-step BCE coefficient search therefore
passes the sign gate only narrowly and is too small to support a strong
independent LHAT-search claim.

The audit also found that `hull_attack.objective` was recorded in every LHAT
YAML but ignored by `core/lhat.py`; generation always maximized mean
five-label BCE. Earlier `targeted_lhat_runtime.py` experiments are not valid
objective evidence because that adapter patched a symbol that the active
generator did not call. The whitelist implementation now dispatches the YAML
objective directly and logs initial, final and gained attack objective for
every generated view.

## L26/L27 predeclaration: simpler effective LHAT objectives

Both candidates retain the exact L24 data, Stage-1 two-chain AugMix SimCLR,
source replay, equal-five logit anchors, Direct+fixed20 outer training,
direct LHAT BCE, clean/LHAT JSD, local exact-label M20 pool, hull lambda
`0.10`, three Adam search steps, epsilon `2`, E40 budget, K500-only population,
optimizer and random identity. They change only the inner attack objective:

- L26 gives each record's positive-label family and negative-label family
  equal `0.5/0.5` search mass. It is per-record and class-symmetric; it does
  not use held-out prevalence or a HYP-specific weight.
- L27 maximizes independent-Bernoulli KL divergence from the decoded anchor
  prediction, a label-free VAT/TRADES-style local boundary objective.

L26 is tested first because the completed standard-BCE attack gain was
concentrated in NORM while positive labels received only their ordinary
one-of-five BCE mass. L27 is run only if L26 does not materially improve the
hard-minus-uniform result. Promotion requires:

- mean PN2021-C AUPRC above L25 by at least `0.10 pp`;
- PN2021-C AUPRC non-negative in at least three centers;
- Clean AUPRC no worse than L25 by more than `0.25 pp`;
- finite objective gain, zero decoded-invalid rate and no class-specific
  held-out tuning.

Any promoted objective must then pass independent-seed replay and a PTB-XL
source-floor check before ECGFounder transfer.

## 2026-07-29 L30/L31 fixed20 dependency ablation

The L28/L29 result established a positive VAE-LHAT delta when the supervised
base was family-balanced clean+fixed20. It did not establish whether fixed20
was required for the VAE branch itself. The following four-cell ECGFounder
ablation therefore kept the same K500 population, source checkpoint, Stage-1
two-chain AugMix-SimCLR, PTB-XL replay weight `0.30`, equal-five logit anchors,
E40/T40 optimizer schedule, batch size `64`, learning rate `2e-5`, random
identity, last-checkpoint rule and reference-excluded evaluation:

```text
L30: Stage 1 -> clean-only supervised adaptation
L31: Stage 1 -> clean + 1.5 * (0.5 LHAT BCE + 0.5 clean/LHAT JSD)
L28: Stage 1 -> family-balanced clean+fixed20
L29: Stage 1 -> family-balanced clean+fixed20
                  + 1.5 * (0.5 LHAT BCE + 0.5 clean/LHAT JSD)
```

The L31 implementation uses the ordinary graph objective directly:
`1.0 clean BCE + 0.75 exact-label LHAT BCE + 0.75 clean/LHAT Bernoulli JSD`.
It does not use objective rescaling, conflict projection, supervised-stage
AugMix, raw corruption auxiliaries, VAE-random, or a materialized corruption
cache. All target adaptation remains per-center K500-only; records outside
K500 are used only for the final global development comparison.

### Four-cell result

The canonical metric view keeps all-zero Super5 records and excludes the
center-specific K500 references.

| Arm | Clean AUROC | Clean AUPRC | PN2021-C AUROC | PN2021-C AUPRC |
|---|---:|---:|---:|---:|
| L30 clean-only | 0.889877 | 0.585698 | 0.869890 | 0.553588 |
| L31 clean + VAE-LHAT | 0.887274 | 0.588326 | 0.866931 | 0.554488 |
| L28 clean+fixed20 | 0.885928 | 0.565970 | 0.867065 | 0.543365 |
| L29 clean+fixed20 + VAE-LHAT | 0.888339 | 0.576046 | 0.868788 | 0.547580 |

The matched effects in percentage points are:

| Contrast | Clean AUROC | Clean AUPRC | PN2021-C AUROC | PN2021-C AUPRC |
|---|---:|---:|---:|---:|
| VAE without fixed20: L31-L30 | -0.260 | +0.263 | -0.296 | +0.090 |
| VAE with fixed20: L29-L28 | +0.241 | +1.008 | +0.172 | +0.422 |
| fixed20 without VAE: L28-L30 | -0.395 | -1.973 | -0.283 | -1.022 |
| fixed20 with VAE: L29-L31 | +0.107 | -1.228 | +0.186 | -0.691 |
| factorial interaction | +0.501 | +0.745 | +0.468 | +0.332 |

L31-L30 PN2021-C AUPRC is positive for Chapman-Shaoxing (`+0.149 pp`),
combined CPSC (`+0.248 pp`) and Georgia (`+0.382 pp`), but negative for
Ningbo (`-0.420 pp`). The four-center mean is only `+0.090 pp`, below the
predeclared `+0.15 pp` promotion threshold, while PN2021-C AUROC drops
`0.296 pp`. Therefore the evidence does not support a stable, fixed20-free
VAE-LHAT benefit under this frozen strength.

The VAE mechanism was active in all four centers: final attack-objective gain
was `0.0066-0.0077`, decoded-invalid rate was zero, and quality acceptance was
`92.6%-97.0%`. The weak matched gain is not explained by a dead VAE branch.

### Source floor

| Arm | PTB-XL AUROC | PTB-XL AUPRC |
|---|---:|---:|
| L30 clean-only | 0.886623 | 0.740103 |
| L31 clean + VAE-LHAT | 0.880683 | 0.725698 |
| L31 minus L30 (pp) | -0.594 | -1.440 |

Both arms fail the absolute source-retention rule, and the VAE branch worsens
that trade-off.

### Decision

Clean supervised adaptation remains necessary and was present in every cell;
this ablation does not authorize a no-clean method. Fixed20 is not necessary
for the highest absolute target AUPRC in this four-cell experiment: both
fixed20-free arms exceed their fixed20 counterparts on Clean and PN2021-C
AUPRC. However, fixed20 acts as an enabling scaffold for a clearer VAE
contribution: it changes the VAE AUROC effect from negative to positive and
increases the PN2021-C AUPRC delta from `+0.090` to `+0.422 pp`.

Accordingly:

1. do not claim fixed20-free VAE-LHAT as a stable promoted component;
2. do not claim fixed20 itself improves absolute AUPRC;
3. retain clean supervision in every mainline;
4. if the paper prioritizes a clean VAE attribution, keep fixed20 as an
   explicit robustness scaffold until a frozen fixed20-free VAE strength
   passes the same gate;
5. the next mechanism screen should reduce the fixed20-free VAE dose and
   improve source retention, not add more components.

All results remain `heldout_tuned_development_only`, single-seed, mapping
`v7_super5_sjr_rgq_review_20260528` / `555ec85d5b51`.

## 2026-07-29 L32/L33 fixed20-free VAE-LHAT dose screen

This screen follows directly from the L30/L31 decision and adds no component.
It keeps L30 as the single clean-only reference and freezes the source
checkpoint, four logical K500 populations, Stage-1 initialization, E40/T40
optimizer schedule, batch size, learning rate, VAE-LHAT search geometry,
random identity, final-checkpoint rule and reference-excluded metric view.
Only the supervised VAE auxiliary multiplier changes:

```text
L30: clean BCE
L32: clean BCE + 0.5 * (0.5 LHAT BCE + 0.5 clean/LHAT JSD)
L33: clean BCE + 1.0 * (0.5 LHAT BCE + 0.5 clean/LHAT JSD)
L31: clean BCE + 1.5 * (0.5 LHAT BCE + 0.5 clean/LHAT JSD), already complete
```

The ordinary graph objective is used directly for every dose. No objective
global-scale adapter, fixed20, supervised-stage AugMix, conflict projection,
raw auxiliary, VAE-random or source replay enters Stage 2. Promotion gates are
frozen before reading L32/L33 results:

- PN2021-C AUPRC versus L30 at least `+0.15 pp`;
- PN2021-C AUPRC non-negative in at least three of four logical centers;
- PN2021-C AUROC no worse than L30 by more than `0.10 pp`;
- Clean AUPRC no worse than L30 by more than `0.50 pp`;
- PTB-XL source AUPRC no worse than L30 by more than `0.50 pp`;
- finite positive attack-objective gain and zero decoded-invalid rate.

If no dose passes all gates, fixed20-free VAE-LHAT remains unpromoted rather
than triggering another component search. These are single-seed development
experiments and cannot become final paper evidence without a frozen-dose
independent-seed replay.

### Dose-screen result

Both new doses completed all four independently adapted logical centers,
reference-excluded Clean/PN2021-C evaluation and PTB-XL fold-10 source-floor
evaluation. L30 is the common zero-dose reference; L31 is the previously
completed `1.5` dose.

| VAE multiplier | Clean AUROC | Clean AUPRC | PN2021-C AUROC | PN2021-C AUPRC | PTB-XL AUROC | PTB-XL AUPRC |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0, L30 | 0.889877 | 0.585698 | 0.869890 | 0.553588 | 0.886623 | 0.740103 |
| 0.5, L32 | 0.888245 | 0.587705 | 0.867416 | 0.555767 | 0.882037 | 0.730970 |
| 1.0, L33 | 0.888052 | 0.588303 | 0.867313 | 0.554850 | 0.883262 | 0.729571 |
| 1.5, L31 | 0.887274 | 0.588326 | 0.866931 | 0.554488 | 0.880683 | 0.725698 |

Relative to L30, the matched percentage-point effects are:

| VAE multiplier | Clean AUROC | Clean AUPRC | PN2021-C AUROC | PN2021-C AUPRC | PTB-XL AUROC | PTB-XL AUPRC |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5, L32 | -0.163 | +0.201 | -0.247 | +0.218 | -0.459 | -0.913 |
| 1.0, L33 | -0.182 | +0.260 | -0.258 | +0.126 | -0.336 | -1.053 |
| 1.5, L31 | -0.260 | +0.263 | -0.296 | +0.090 | -0.594 | -1.440 |

L32 is the best fixed20-free VAE dose for PN2021-C AUPRC. Its per-center
AUPRC deltas are positive in all four centers: Chapman-Shaoxing `+0.025 pp`,
combined CPSC `+0.360 pp`, Georgia `+0.175 pp`, and Ningbo `+0.311 pp`.
Nevertheless, L32 fails the frozen PN2021-C AUROC gate (`-0.247 pp`) and
source-AUPRC gate (`-0.913 pp`). L33 fails the mean robust-AUPRC threshold,
the three-center sign threshold, the robust-AUROC gate and the source-AUPRC
gate. L31 remains weaker on robust AUPRC and source retention.

The mechanism was active rather than silently bypassed:

- L32 final attack-objective gain is `0.00730-0.00913`, any-flip ASR is
  `2.40%-4.08%`, and decoded-invalid rate is zero in every center;
- L33 final attack-objective gain is `0.00660-0.00823`, any-flip ASR is
  `1.93%-3.93%`, and decoded-invalid rate is zero in every center;
- all 16 new managed train/evaluation records pass manifest, result,
  checkpoint and file-index SHA checks.

### Dose-screen decision

No fixed20-free dose passes all predeclared gates, so none is promoted. The
experiment supports a narrow statement: a low VAE dose can improve
PN2021-C AUPRC consistently, but under the current E40 full-fine-tuning
recipe that gain trades away ranking AUROC and PTB-XL source retention.

This closes auxiliary-dose-only tuning. Do not add another loss component or
select L32 by AUPRC alone. Under the currently tested recipe, family-balanced
clean+fixed20 remains the enabling scaffold for the clearest joint
AUROC/AUPRC VAE contribution (L29 minus L28), even though fixed20 itself does
not maximize absolute target AUPRC. Clean supervision remains mandatory in
every arm.

## 2026-07-29 L34/L35 rotating-four corruption scaffold

This matched experiment tests the simplest version of the proposed pipeline:

```text
K500-only two-chain AugMix-SimCLR, 1024 updates
  + PTB-XL replay weight 0.30
  + equal-five source-logit anchor weight 5.0
-> clean + four deterministic rotating corruption views
  + one online exact-label VAE-LHAT hard view
-> direct multilabel BCE + clean/hard Bernoulli JSD
```

The fixed20 supervised scaffold is replaced, not supplemented. For every base
record and epoch, the runtime executes clean plus two depth-2 and two depth-3
views. A stable per-record hash offset and epoch stride of two cover all twenty
canonical depth2+3 compositions every five epochs. Clean and corruption
families retain equal `0.5/0.5` mass, and the five outer views accumulate into
one optimizer update per base batch.

The two arms are:

- L34: the shared AugMix-SimCLR stage followed by clean + rotating-four
  supervision;
- L35: L34 plus one online VAE-LHAT hard view for each eligible record,
  weighted by `1.5 * (0.5 hard BCE + 0.5 clean/hard JSD)`.

The VAE-LHAT search uses an exact-positive-set, non-self neighborhood, local
pool size `80`, `M=20` candidates, standardized latent coordinates,
`hull_lambda=0.60`, five Adam search steps, search learning rate `0.25`, and
standardized L2 radius `2.0`. The decoder is frozen. Decoded 1024-point ECGs
are linearly bridged to the classifier domain and normalized only after
decoding.

### Pair audit

Both arms start from the same locked PTB-XL ECGFounder checkpoint and use the
same K500 records, batch size `64`, learning rate `2e-5`, weight decay `1e-4`,
E40/T40 cosine schedule, last-checkpoint rule, global seed identity and
reference-excluded evaluation. The completed artifacts prove, for all four
centers:

- the same 1024-step target SSL batch order;
- the same PTB-XL source-replay batch order;
- the same 320 supervised optimizer updates;
- identical learning rate, exposure metadata, input identity and all three
  corruption composition/depth/operator-mask hashes in each of 40 epochs.

All eight train/evaluation records pass result, checkpoint, run-manifest and
file-index SHA checks. Adaptation is per-center K500-only; records outside K500
are not accessed by the model and are used only for the global development
comparison.

### Target-domain result

The canonical metric view keeps all-zero Super5 records and excludes each
center's K500 references.

| Arm | Clean AUROC | Clean AUPRC | PN2021-C AUROC | PN2021-C AUPRC |
|---|---:|---:|---:|---:|
| L34 rotating-four, no VAE | 0.886861 | 0.567558 | 0.868133 | 0.542863 |
| L35 rotating-four + VAE-LHAT | 0.887536 | 0.577748 | 0.867777 | 0.547598 |
| L35 minus L34 (pp) | +0.068 | +1.019 | -0.036 | +0.474 |

Per-center matched deltas are:

| Center | Clean AUROC | Clean AUPRC | PN2021-C AUROC | PN2021-C AUPRC |
|---|---:|---:|---:|---:|
| Ningbo | -0.013 | +0.843 | -0.313 | +0.232 |
| Chapman-Shaoxing | +0.180 | +1.698 | -0.011 | +0.577 |
| CPSC 2018 + Extra | +0.230 | +0.734 | +0.114 | +0.581 |
| Georgia | -0.127 | +0.801 | +0.067 | +0.504 |

Thus Clean and PN2021-C AUPRC improve in all four centers. Mean robust AUPRC
exceeds the `+0.15 pp` mechanism threshold, Clean AUPRC does not regress, and
mean robust AUROC is effectively flat. The target-domain mechanism gate
passes.

For context, L35 is nearly equivalent to the heavier fixed20 L29 arm:

| Contrast: L35 minus L29 | Clean AUROC | Clean AUPRC | PN2021-C AUROC | PN2021-C AUPRC |
|---|---:|---:|---:|---:|
| rotating-four minus fixed20 (pp) | -0.080 | +0.170 | -0.101 | +0.002 |

The theoretical supervised view count falls from 21 to 5 per base batch
(`-76.2%`). Mean managed training wall time per center falls from `818` to
`394` seconds without VAE (`-51.8%`) and from `1131` to `720` seconds with VAE
(`-36.3%`). These wall-clock comparisons are same-host development
measurements rather than formal throughput benchmarks.

The VAE branch is active in every center: final attack-objective gain is
`0.00734-0.00854`, ordinary BCE gain is `0.00460-0.00704`, and decoded-invalid
rate is zero. Final any-flip ASR is only `0.8%-4.7%`, so this is a mild local
hard-sample pressure rather than a high-success adversarial attack.

### Source floor and decision

| Arm | PTB-XL AUROC | PTB-XL AUPRC |
|---|---:|---:|
| locked ECGFounder source | 0.929407 | 0.824164 |
| L34 rotating-four, no VAE | 0.915372 | 0.795770 |
| L35 rotating-four + VAE-LHAT | 0.908378 | 0.780632 |
| L35 minus L34 (pp) | -0.699 | -1.514 |

The absolute source-retention rule fails for both arms, and VAE-LHAT adds a
material source-domain cost. Therefore L35 is **mechanism-positive and
compute-efficient on the target domain, but not promoted as a final
no-trade-off paper method**.

The supported conclusion is narrower:

1. fixed20 is not required to retain the L29 target AUPRC frontier;
2. deterministic rotating-four supervision provides essentially the same
   target AUPRC with much less supervised computation;
3. one VAE-LHAT hard view supplies a consistent four-center AUPRC refinement;
4. source retention, low attack-success rate and single-seed
   `heldout_tuned_development_only` status remain unresolved.

Do not tune another target-domain component from these held-out scores. The
next admissible step is either an independent-seed replay of the frozen L34/L35
pair or one separately declared source-retention control; it must not change
the target K500-only adaptation population or select center-specific settings.

## 2026-07-29 L36/L37 EfficientNet rotating-four transfer

This experiment transfers the frozen L34/L35 mechanism to EfficientNet1DV2
without changing its locked Direct+fixed20 Stage-2 optimizer recipe:

- source checkpoint: locked PTB-XL EfficientNet1DV2;
- Stage 1: 1024 K500-only two-chain AugMix-SimCLR updates with PTB-XL semantic
  replay weight `0.30`;
- Stage 2: learning rate `5e-5`, batch size `128`, weight decay `1e-4`,
  E23 with a T30 cosine horizon, and exactly `92` optimizer updates;
- supervised views: clean plus deterministic rotating-four depth2+3
  corruption views;
- L37 only: one exact-label online VAE-LHAT hard view with
  `1.5 * (0.5 hard BCE + 0.5 clean/hard Bernoulli JSD)`.

L36 and L37 use the same K500 records, source checkpoint, target and source
Stage-1 batch order, Stage-1 losses, epoch learning rates, Stage-2 input
identity and corruption composition/depth/operator-mask traces in every center.
The only method-family difference is the VAE-LHAT branch. All 16 managed
train/evaluation records pass the controller's result, checkpoint, manifest
and file-index checks. The sandbox controller regression suite passes
`126/126` tests.

### Target-domain result

The canonical metric view keeps all-zero Super5 records, excludes the adapted
center's K500 references, and combines CPSC 2018 with CPSC Extra as one logical
center.

| Arm | Clean AUROC | Clean AUPRC | PN2021-C AUROC | PN2021-C AUPRC |
|---|---:|---:|---:|---:|
| Locked Direct+fixed20 | 0.850009 | 0.505380 | 0.809939 | 0.438257 |
| L36 AugMix-SimCLR + rotating-four, no VAE | 0.855507 | 0.518518 | 0.839711 | 0.491366 |
| L37 L36 + online VAE-LHAT | 0.858001 | 0.523483 | 0.841685 | 0.494762 |
| L36 minus locked Direct (pp) | +0.550 | +1.314 | +2.977 | +5.311 |
| L37 minus locked Direct (pp) | +0.799 | +1.810 | +3.175 | +5.651 |
| L37 minus L36 (pp) | +0.249 | +0.496 | +0.197 | +0.340 |

The matched VAE-only increments are positive in every center:

| Center | Clean AUROC | Clean AUPRC | PN2021-C AUROC | PN2021-C AUPRC |
|---|---:|---:|---:|---:|
| Ningbo | +0.163 | +0.393 | +0.026 | +0.219 |
| Chapman-Shaoxing | +0.133 | +0.454 | +0.122 | +0.356 |
| CPSC 2018 + Extra | +0.439 | +0.833 | +0.355 | +0.577 |
| Georgia | +0.263 | +0.305 | +0.287 | +0.206 |

Thus the full recipe improves both Clean and PN2021-C four-center means over
the locked Direct+fixed20 baseline. L36 alone supplies most of the gain. L37
adds a smaller but sign-consistent VAE-LHAT refinement.

This is not a whole-pipeline compute-matched causal contrast against
Direct+fixed20: L36/L37 include 1024 extra SSL updates and source replay, and
rotate four corruption views per epoch instead of evaluating all twenty views
inside every base batch. The `L37 - L36` contrast is the controlled
VAE-LHAT attribution; `L36/L37 - Direct` describes the complete recipe.

### VAE mechanism diagnostics

Across centers and epochs, L37 has mean attack-objective gain `0.01212`, mean
BCE loss gain `0.01773`, mean effective latent displacement `0.0916`, and zero
decoded-invalid rate. Mean any-flip ASR is only `4.22%`, so the current search
is active but mild. It should be described as local hard-sample
regularization, not as a high-success adversarial attack.

### PTB-XL source floor

| Arm | PTB-XL AUROC | PTB-XL AUPRC |
|---|---:|---:|
| Locked source checkpoint | 0.901281 | 0.767951 |
| Locked Direct+fixed20 supplementary replay | 0.876042 | 0.714541 |
| L36 rotating-four, no VAE | 0.885400 | 0.743584 |
| L37 rotating-four + VAE-LHAT | 0.880577 | 0.732554 |
| L36 minus locked Direct (pp) | +0.936 | +2.904 |
| L37 minus locked Direct (pp) | +0.453 | +1.801 |
| L37 minus L36 (pp) | -0.482 | -1.103 |

The Direct+fixed20 row above is a same-evaluator current-run supplementary
probe of its four locked checkpoints; unlike the L36/L37 source-floor JSON
files, it is not yet a managed registry artifact. Both new arms retain PTB-XL
better than Direct+fixed20, but both still fail the predeclared absolute source
retention rule relative to the original source checkpoint. VAE-LHAT adds a
measurable source-domain cost over L36.

### Decision

The EfficientNet transfer is positive development evidence:

1. the compact rotating-four pipeline beats the locked Direct+fixed20 target
   baseline by `+2.98 / +5.31 pp` on PN2021-C;
2. the matched online VAE-LHAT branch adds `+0.20 / +0.34 pp` and is positive
   in all four centers;
3. the full L37 pipeline reaches `+3.17 / +5.65 pp` over Direct+fixed20 while
   preserving PTB-XL better than that baseline.

It remains `heldout_tuned_development_only`, single-seed evidence. Do not call
L37 a locked paper method until the exact frozen L36/L37 pair is replayed at an
independent seed and the VAE source-retention trade-off is either accepted as
an explicit limitation or reduced by a separately declared source-retention
control.

Primary generated artifacts:

- L37 versus L36 summary:
  `/home/linbinhao/ECG_adv_data/runs/agent_workspace/a7_global_search_20260726/l37d_effnet_ssl1024_augmix_replay030_rot4_vae_hard_h30_e23/summary.json`;
- L36 source floor:
  `/home/linbinhao/ECG_adv_data/runs/agent_workspace/a7_global_search_20260726/l36d_effnet_ssl1024_augmix_replay030_rot4_no_vae_h30_e23/source_floor/candidate/summary.json`;
- L37 source floor:
  `/home/linbinhao/ECG_adv_data/runs/agent_workspace/a7_global_search_20260726/l37d_effnet_ssl1024_augmix_replay030_rot4_vae_hard_h30_e23/source_floor/candidate/summary.json`.
