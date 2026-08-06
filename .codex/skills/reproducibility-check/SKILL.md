---
name: reproducibility-check
description: Verify ECG_manual_refactor experiments are traceable and reproducible from YAML config closure, keep manifest, git SHA, seeds, commands, checkpoints, artifacts, and metrics.
---

# Reproducibility Check

Use this before claiming an experiment result, updating paper tables, committing
experiment tooling, or comparing methods.

## Required Evidence

- YAML or managed config path.
- Resolved config or run manifest.
- Exact command and working directory.
- Git SHA and dirty status.
- Python environment path and key package versions.
- Dataset root, mapping version, center list, and split/ref-exclusion rules.
- Seed list and checkpoint selection rule.
- Metrics file path and aggregation view, including all-zero handling.

## Workflow

1. Locate the run manifest and resolved config.
2. Check that command, git SHA, environment, inputs, and outputs are recorded.
3. Confirm label mapping and PN2021 center/ref-exclusion rules.
4. Compare metrics against the declared baseline and note pp deltas.
5. Flag any metric that cannot be replayed from saved artifacts.

## Validation Order

1. Static manifest/config audit.
2. Small replay or eval smoke command when feasible.
3. Broader rerun only when the run is important enough to justify compute.
