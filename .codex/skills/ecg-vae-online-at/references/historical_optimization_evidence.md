# Historical VAE-LHAT Optimization Evidence

This file is a routing note, not an active recipe.

The complete pre-simplification Skill, including dated 2026-05 sweeps,
backbone-specific results, old include-anchor/low-lambda recipes, and obsolete
wrapper paths, is frozen at commit
`5b117985089e78a503b47006c03a1a33b0aa3864`:

```bash
git show 5b117985089e78a503b47006c03a1a33b0aa3864:.codex/skills/ecg-vae-online-at/SKILL.md
```

Do not reuse those settings as current defaults. For current performance facts,
start from `configs/active_evidence_registry.yaml` and the managed artifacts it
references. For each historical comparison, verify the backbone, mapping/hash,
K500 identities, source checkpoint, seed, optimizer-step budget, metric view,
checkpoint rule, and whether the comparison is matched.

Historical attack health, a heldout-oracle selection, or a single-center gain
does not establish a paper-ready VAE-LHAT contribution.
