# Literature And Repository Priors

Use these as priors for ECGTwin VAE latent-hull online AT. They are guidance,
not permission to drift away from the PTB-XL/PN2021 mainline.

## Adversarial Training

- Madry et al., *Towards Deep Learning Models Resistant to Adversarial
  Attacks*, ICLR 2018. PGD as first-order robust optimization.
  https://openreview.net/forum?id=rJzIBfZAb
- TRADES, Zhang et al., ICML 2019. Clean loss plus boundary/robust
  regularization, with `beta` controlling accuracy-robustness tradeoff.
  https://github.com/yaodongyu/TRADES
- MART, Wang et al., ICLR 2020. Separate behavior for clean-correct and
  misclassified samples; useful for clean-correct conditioned ASR.
  https://openreview.net/forum?id=rklOg6EFwS
- AWP, Wu et al., NeurIPS 2020. Weight perturbation can flatten robust loss
  landscape, but should be late-stage after stable PGD settings.
  https://github.com/csdongxian/AWP
- Robust overfitting, Rice et al., ICML 2020. Early stopping and robust
  validation matter.
  https://github.com/locuslab/robust_overfitting
- AutoAttack, Croce and Hein, ICML 2020. Fixed strong attack suite to reduce
  attack-protocol tuning.
  https://github.com/fra31/auto-attack
- RobustBench. Standardized clean and robust reporting.
  https://github.com/RobustBench/robustbench
- MadryLab robustness library. Separates training and stronger evaluation
  attacks.
  https://github.com/MadryLab/robustness
- ART adversarial trainer. Clean/adv ratio is a first-class setting.
  https://github.com/Trusted-AI/adversarial-robustness-toolbox
- torchattacks. PGD/MultiAttack implementation patterns and explicit random
  start/restart controls.
  https://github.com/Harry24k/adversarial-attacks-pytorch

## Mixup And Latent Interpolation

- Chapelle et al., *Vicinal Risk Minimization*, NeurIPS 2000.
  https://bottou.org/papers/chapelle-2000
- Zhang et al., *mixup: Beyond Empirical Risk Minimization*, ICLR 2018.
  https://openreview.net/forum?id=r1Ddp1-Rb
- Verma et al., *Manifold Mixup*, ICML 2019.
  https://arxiv.org/abs/1806.05236
- Interpolation Consistency Training, Verma et al., IJCAI 2019.
  https://arxiv.org/abs/1903.03825
- VAE, Kingma and Welling, *Auto-Encoding Variational Bayes*.
  https://arxiv.org/abs/1312.6114
- AdaMixup / manifold intrusion, Guo et al. Mixing can create contradictory or
  out-of-manifold samples.
  https://arxiv.org/abs/1809.02499
- Local Mixup, Baena et al. Locality reduces harmful far-pair mixing.
  https://arxiv.org/abs/2201.04368
- Latent Embedded Graphs. Generative-model latent spaces may be non-Euclidean;
  linear interpolation is not automatically valid.
  https://www.sciencedirect.com/science/article/pii/S0010448521001020

## Engineering Repositories

- facebookresearch mixup-cifar10.
  https://github.com/facebookresearch/mixup-cifar10
- Manifold Mixup reference code.
  https://github.com/vikasverma1077/manifold_mixup
- OpenMixup collection.
  https://github.com/Westlake-AI/openmixup
- timm Mixup/CutMix docs.
  https://timm.fast.ai/augmentation
- fastai mixup docs: Beta alpha behavior and implementation conventions.
  https://docs.fast.ai/callback.mixup.html
- MosaicML Composer mixup method card.
  https://docs.mosaicml.com/projects/composer/en/latest/method_cards/mixup.html

## Reviewer-Safe External Shortlist (verified 2026-08-10)

No published score below is directly comparable with the locked
PTB-XL-Super5 -> per-center K500 -> ref-excluded PN2021/PN2021-C contract.
Only locally reproduced results may enter quantitative tables.

Qualification scope: `declared-data-clean candidate` means only that the
paper/repository's public data-source statement does not name a locked target
dataset. No released checkpoint has yet passed download, hash, metadata, or
record-level manifest auditing here. It is therefore a candidate status—not a
leakage-free finding—and remains pending checkpoint-level provenance audit.

### Matched Whole-System Candidates

Scope note: [ECGFounder](https://github.com/PKUDigitalHealth/ECGFounder) is
already a project backbone and reference control, not a newly added external
competitor. Keep its matched Direct/ablation rows as anchors when comparing the
systems below; this role does not itself certify checkpoint provenance.

- **MERL** — ICML 2024
  ([paper](https://proceedings.mlr.press/v235/liu24bg.html),
  [code/weights](https://github.com/cheliu-computation/MERL-ICML2024)). MIT
  code and ResNet/ViT pretrained weights are released. The official repository
  states that pretraining uses MIMIC-IV-ECG; PTB-XL, CPSC2018, and
  Chapman-Shaoxing-Ningbo are downstream datasets, so use only a base
  pretrained checkpoint and rebuild the locked Super5/K500 head. Role:
  high-priority declared-data-clean multimodal candidate, pending
  checkpoint-level provenance audit; medium reproduction cost.
- **HeartLang** — ICLR 2025
  ([paper](https://openreview.net/forum?id=6Hz1Ko087B),
  [code](https://github.com/PKUDigitalHealth/HeartLang)). Code and base/VQ-HBR
  weights are released under MIT. Pretraining uses MIMIC-IV-ECG, with no
  declared PTB-XL, Ningbo, Chapman-Shaoxing, CPSC, or Georgia overlap. Role:
  declared-data-clean 12-lead ECG-system candidate, pending checkpoint-level
  provenance audit and a local Super5/K500 adapter; medium-high reproduction
  cost.
- **D-BETA** — ICML 2025
  ([paper](https://proceedings.mlr.press/v267/pham-hung25a.html),
  [code](https://github.com/manhph2211/D-BETA),
  [weights](https://huggingface.co/Manhph2211/D-BETA)). Code/checkpoint are
  CC BY-NC 4.0 and weight access is gated by terms. Pretraining uses paired
  MIMIC-IV-ECG reports; use only the base checkpoint, not target-dataset
  fine-tunes. Role: declared-data-clean ECG-language candidate, pending
  checkpoint-level provenance audit; medium cost.
- **ECGFlow** — NEJM AI 2025
  ([paper](https://doi.org/10.1056/AIoa2500164),
  [code/weights](https://github.com/4dm-labs/ecgflow)). Apache-2.0 code and an
  SSL checkpoint are released. SSL pretraining is MIMIC-IV-ECG only; PTB-XL
  task heads must not be reused. Role: declared-data-clean MAE/1dViT candidate,
  pending checkpoint-level provenance audit; medium cost.
- **ECG-CPC** — ICLR 2026
  ([paper](https://openreview.net/forum?id=xXRqWpt3Xr),
  [code/weights](https://github.com/AI4HealthUOL/ecg-fm-benchmarking)). Code and
  weights exist, but the repository declares no license. Its HEEDB count
  matches the MGH-only release described by the
  [official HEEDB record](https://bdsp.io/content/heedb/5.0/), so target
  overlap is not indicated; freeze this as conditional until a checkpoint
  record manifest and usage permission are obtained. Role:
  declared-data-clean compact-FM candidate, pending checkpoint-level
  provenance audit and licensing clarification; medium-high cost.

### Matched Mechanism Ports

- **TransPL** — ICML 2025
  ([paper](https://proceedings.mlr.press/v267/kim25w.html),
  [code](https://github.com/eai-lab/TransPL)). Code is released without weights
  or a declared license. It has no ECG pretraining overlap when trained from
  scratch. Role: recent pseudo-label/VQ-transition UDA-K500 port; medium-high
  cost, with an independent implementation if licensing remains unresolved.
- **SSSS-TSA** — TMLR 2025
  ([paper](https://openreview.net/forum?id=8C8LJIqF4y),
  [code](https://github.com/nerdslab/SSSS_TSA)). Code is released without
  weights or a declared license; its original non-ECG wearable datasets do not
  overlap the locked ECG sets. Role: channel-selective/Sinkhorn UDA port,
  especially relevant to lead/domain shift; medium cost.
- **ECGMatch** — IEEE TPAMI 2024
  ([paper](https://doi.org/10.1109/TPAMI.2023.3342828),
  [code](https://github.com/KAZABANA/ECGMatch)). Code is released without
  weights or a declared license. Its published experiments use PTB-XL,
  Chapman, Ningbo, and Georgia, so published results are protocol-mismatched;
  a from-scratch locked-split port is clean. Role: closest ECG-specific
  SSDA-K500 mechanism; medium cost.

### Contamination Or Related-Only

- **ECG-LFM** — Nature Communications 2026
  ([paper](https://doi.org/10.1038/s41467-026-72436-2),
  [code](https://github.com/biomed-AI/ECG-LFM),
  [checkpoint](https://doi.org/10.5281/zenodo.20388950)). Code carries MIT,
  but the 7.5-GB checkpoint record does not state a license. Pretraining uses
  MIMIC-IV-ECG plus an HEEDB count matching MGH-only data; use no downstream
  PTB-XL/Chapman/CPSC task head. Role: declared-data-clean 500-Hz candidate
  pending checkpoint-level provenance and weight-license audits; very-high-cost
  appendix system, not a minimal-main-table baseline.
- **DeepECG-SSL** — European Heart Journal 2026
  ([paper](https://academic.oup.com/eurheartj/article/47/18/2174/8436833),
  [code](https://github.com/HeartWise-AI/DeepECG_Docker),
  [weights](https://huggingface.co/collections/heartwise/deepecg-models)). Base
  pretraining uses MHI, CODE-15, and MIMIC-IV rather than the locked datasets,
  but neither repository nor base model card states a reuse license. Role:
  declared-data-clean 250-Hz candidate pending checkpoint-level provenance and
  license audits; high-cost appendix adapter.
- **CSFM** — Nature Machine Intelligence 2026
  ([paper](https://doi.org/10.1038/s42256-026-01180-5),
  [code/access](https://github.com/guxiao0822/Cardiac-Sensing-FM)). Pretraining
  uses MIMIC-III-WDB, MIMIC-IV-ECG, and CODE-Full, with no declared locked-set
  overlap. Weights require a signed Academic Access Agreement and no repository
  license is stated. Role: declared-data-clean multimodal/lead-agnostic
  candidate pending checkpoint-level provenance, access, and license audits;
  high-cost appendix system.
- **ST-MEM** — ICLR 2024
  ([paper](https://openreview.net/forum?id=WcOohbsF4H),
  [code/weights](https://github.com/vuno/ST-MEM)). CC BY-NC 4.0 code/weights;
  pretraining includes Chapman-Shaoxing and Ningbo. Role: related work or
  explicitly contaminated sensitivity analysis only, never a clean primary
  comparator.
- **ECG-FM** — JAMIA Open 2025
  ([paper](https://academic.oup.com/jamiaopen/article/8/5/ooaf122/8287827),
  [code/weights](https://github.com/bowang-lab/ecg-fm)). MIT code/weights;
  pretraining includes MIMIC-IV plus PhysioNet Challenge 2021, which overlaps
  PTB-XL and the Ningbo/Chapman/CPSC/Georgia targets. Role: related-only hard
  exclusion from leakage-clean quantitative comparisons.

### Locked Comparison Contract

- Keep two quantitative tables: (1) fixed-backbone mechanisms and (2)
  whole systems. Native pretrained frontends may retain their required sample
  rate only in table (2); they still use identical record IDs, Super5 mapping,
  K500 identities, evaluation records, and selection policy.
- Fixed-backbone main slate: Direct, DeepCORAL, DANN or CDAN from the
  [AdaTime suite](https://github.com/emadeldeen24/AdaTime), TransPL, SSSS-TSA,
  ECGMatch, method ablations, and the full method. Whole-system anchors are
  ECGFounder and the current supervised backbone; they are controls, not new
  competitors. Engineering execution order after provenance gates: MERL and
  HeartLang first, licensed ECGFlow next, then gated/non-commercial D-BETA if
  budget permits. Add ECG-CPC only after its usage permission and HEEDB
  checkpoint provenance are resolved. This order is a reproduction-cost
  recommendation, not an evidence-eligibility or performance ranking.
- **DG-0:** no target signals, labels, or statistics during training/selection.
  **UDA-K500:** K400/K100 signals may be viewed without labels; select from
  source validation plus a frozen unsupervised criterion, then refit on all 500
  unlabeled signals. **SSDA-K500:** K400 target labels train and K100 labels
  select; after freezing the recipe, refit on all 500. No additional held-out
  target record may become an unlabeled training sample.
- Evaluate every arm on the same K500-excluded target records. Report PTB-XL
  fold-10 source floor, clean PN2021 and PN2021-C severity-5 atomic plus
  depth2/depth3/depth23, per center and per class, macro AUROC/AUPRC,
  drop-all-zero primary and all-zero-kept audit.
- Equalize optimizer-step and record-exposure budgets within each backbone;
  freeze preprocessing, checkpoint initialization, K500 hashes, ref exclusion,
  and selection metric. Run three seeds for screening and five seeds for every
  promoted main-table row, with paired uncertainty. Never merge a paper's
  published score into the matched table.
