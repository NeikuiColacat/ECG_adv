# xresnet1d50 Matched K500 Direct vs VAE-LHAT

Date: 2026-06-02

## Protocol

Backbone: `fastai_xresnet1d50`.

Source training:

- PTB-XL Super5 v7.
- `minimal_resample`, `per_sample_global`.
- 500Hz, 5000 samples = 10 seconds.
- Fresh source model trained with the same preprocessing path used by the
  online-AT script.

Matched target adaptation:

- Four PN2021 centers: `ningbo`, `chapman_shaoxing`, `cpsc_2018`, `georgia`.
- K=500 target-center anchors per center.
- Final PN2021 target evaluation excludes the same K=500 ref ids.
- Direct control: supervised target-real stream only.
- VAE-LHAT: same direct setup plus VAE500 latent-hull online AT
  (`diffusets500_v1`, NORM/MI/STTC scope, mixed-soft labels).

## Source Model

```text
/root/autodl-tmp/vae500_benchmark_xresnet50_matched_k500_v7_20260602/source_fastai_xresnet1d50_500hz_perglobal
PTB-XL fold10: 0.9113 / 0.7843
```

## Held-Out Target-Center Results

All values are macro AUROC / macro AUPRC on the target center with K500 ref ids
excluded.

| center | direct K500 | VAE-LHAT K500 | delta |
|---|---:|---:|---:|
| ningbo | 0.8885 / 0.5241 | 0.8886 / 0.5234 | +0.01pp / -0.07pp |
| chapman_shaoxing | 0.8814 / 0.4718 | 0.8799 / 0.4688 | -0.14pp / -0.31pp |
| cpsc_2018 | 0.8529 / 0.5840 | 0.8539 / 0.5856 | +0.11pp / +0.15pp |
| georgia | 0.8286 / 0.6172 | 0.8268 / 0.6162 | -0.17pp / -0.10pp |
| mean | 0.8628 / 0.5493 | 0.8623 / 0.5485 | -0.05pp / -0.08pp |

## Interpretation

The xresnet1d50 matched pair does not meet the goal criterion. K500 internal
validation improved during VAE-LHAT training, but the improvement did not
transfer to ref-excluded held-out PN2021 target evaluation. The only positive
held-out center is `cpsc_2018`, and the gain is very small.

This supports the current diagnosis: the present VAE500 online-AT recipe is not
adding robust information beyond strong direct K500 adaptation. Future work
should change the VAE or the AT objective, not simply transfer this recipe to
more backbones.

## Next Decision

Continue with the controlled VAE replacement branch:

1. Use the newly added `ecgtwin_init_vae500` backend, which reuses ECGTwin's
   original VAE architecture at 5000 points and strictly initializes from the
   author `vae_model.pth`.
2. Run a small training smoke test first.
3. Only launch a longer PTB-XL-only fine-tune if reconstruction and decode
   sanity pass.
