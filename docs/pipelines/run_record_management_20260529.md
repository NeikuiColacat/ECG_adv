# Run Record Management, 2026-05-29

This project uses lightweight file-based experiment tracking so future agents
can resume work without replaying chat history or searching old reports.

## Required Per-Run Files

Each managed run directory should contain:

```text
run_card.json
run_file_index.json
summary.md
configs/
manifests/
logs/
checkpoints/
eval/
diagnostics/
reports/
artifacts/
```

`run_card.json` is the agent entry point for a run. It records:

- experiment name, purpose, and description;
- run status, outcome, and result summary;
- mapping version/hash, class order, target centers, K-shot protocol, and
  selection policy;
- center-mean AUROC/AUPRC summaries discovered from `metrics_long.csv`;
- paths to the manifest, file index, and summary.

`run_file_index.json` does not move large outputs. It gives every file in the
run directory a logical category so an agent can quickly find logs, checkpoints,
diagnostics, and evaluation artifacts.

## Commands

For YAML-managed runs, `scripts/run_experiment.py --write-plan` and `--execute`
create the run record automatically.

For older runs:

```bash
micromamba run -n ECGTwin python scripts/agent/finalize_run.py \
  --run-dir /path/to/run_dir \
  --purpose "Why this experiment was run" \
  --result-summary "What the run showed" \
  --outcome provisional
```

Register important finalized runs in the active evidence registry:

```bash
micromamba run -n ECGTwin python scripts/agent/register_run.py \
  --run-dir /path/to/run_dir \
  --status provisional
```

The registration command stores run paths relative to `${paths.output_root}` so
tracked registry YAML does not contain host-specific `/home/...` paths.

## Policy

- Keep checkpoints and generated datasets out of Git.
- Keep `run_manifest.json` at the run root for backward compatibility.
- Use category directories and `run_file_index.json` for navigation; do not
  move legacy child artifacts unless the producing script and tests are updated.
- Always describe both purpose and result summary. A run without those two
  fields is not ready for long-handoff agent work.
