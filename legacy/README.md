# Legacy And Non-Mainline Code

This directory contains historical experiments and exploratory code that are
not part of the `thesis.md` reproduction mainline.

Moved here during thesis archive cleanup:

- PN2021-C robustness and PN2021 v3 re-evaluation helpers.
- TA-OMAT, Latent-Hull, real-anchor, synth-anchor, and PGD experiments.
- AdvDiff and AugMix method experiments.
- Tier-M and older cross-center training routes.
- Old PN2021 center-style prompt-token probes and online adversarial training
  runners.
- Historical advisor summaries and pipeline plans not cited as the final
  thesis reproduction route.

Mainline scripts must not import from `legacy/`. If a moved file is needed
again, copy the required function into a current module with a neutral name and
document why it belongs to the thesis route.
