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
