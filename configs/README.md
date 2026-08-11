# Manual-Refactor Configuration Bundles

`configs/` is a portable, copied-as-a-unit experiment contract. Internal YAML
references are resolved relative to the selected bundle root, not relative to
the repository's default configs.

## Sections

| Directory | Contents |
|---|---|
| `data/` | Dataset locations, cache identities, splits, loader/runtime policy |
| `augmentation/` | Five-operator profile and offline PN2021-C cache contract |
| `train/` | Trainer, VAE/LHAT/AugMix, and five finite recipe selectors |
| `eval/` | Canonical PN2021 Clean/PN2021-C evaluation contract |
| `baselines/` | Locked PTB-XL source and Direct+fixed20 evidence registries |
| `experiments/` | Thin executable entry YAMLs |
| `active_scripts.yaml` | Current launcher and executable mainline index |
| `active_evidence_registry.yaml` | Development versus historical evidence boundary |
| `random_seed.yaml` | One base seed and deterministic derivation contract |

## Reference Rules

- Entry YAML paths may be absolute or repository-relative.
- YAML-to-YAML references inside a bundle must be relative to the bundle root,
  such as `train/lhat.yaml` or `augmentation/operators.yaml`.
- Relative references may not escape the bundle.
- Copy the whole `configs/` tree to explore a variant; then pass the copied
  experiment path and `--config-root <copied-configs>`.
- Closure follows schema-owned references only: the top-level `references`
  mapping, finite-recipe resources, and the existing seed/operator config keys.
  YAML-looking prose and checkpoint arguments are not dependencies.
- `config_closure: snapshot_only` and
  `config_closure: managed_run_snapshots_and_sha256` mark evidence registries
  whose nested historical paths are snapshots, not live dependencies.
- Python implementation names come from code-owned allowlists. YAML may not
  dynamically import a module.
- Method YAML selects exactly one of five code-owned RecipeSpec variants. It
  contains only recipe identity and external resource references; exposure,
  objective, resource requirements, and execution behavior remain code-owned.
- Recipe selectors have exactly three root keys: `schema_version`, `recipe`,
  and `resources`. The recipe block has exactly `id`, `kind`,
  `auxiliary_variant`, `scientific_arm`, and `status`; schema version 2 rejects
  every extra key.
- Every accepted file in the transitive closure is hashed into the run record.

## Launcher

The only experiment launcher is:

```bash
/home/linbinhao/miniforge3/envs/ECGTwin/bin/python \
  boot_scripts/run_experiment.py \
  --config configs/experiments/<experiment>.yaml \
  --dry-run
```

A dry-run resolves and hashes the closure, builds the exact delegated command,
and reports output paths without loading data, a checkpoint, or a GPU. When the
configured run directory already exists, dry-run reports
`run_dir_collision: true` and `would_fail_execution: true` without modifying
that directory. Non-dry execution continues to reject the collision.

The experiment schema contains exactly:

```yaml
schema_version: 1
experiment:
  name: unique_experiment_name
  purpose: Human-readable scientific purpose.
entrypoint:
  name: one_code_owned_entrypoint
  config: train/or/eval/config.yaml
  arguments: [--explicit, value]
output:
  run_dir: /external/run/directory
  if_exists: error
  delegate_output_subdir: training
```

The launcher owns `--config`, `--config-root`, `--output-dir`, and
`--dry-run`; these flags cannot be smuggled into `entrypoint.arguments`.
Each code-owned entrypoint also declares its result filename and artifact type;
an exit-zero delegate is marked failed if that exact JSON result is absent or
does not match the action contract.

PN2021 experiment YAML continues to pass a tracked selector via
`--method-config`. The flag name is retained for launch compatibility; its
schema-v2 payload is a finite RecipeSpec declaration, not an extensible method
profile or Python import surface.

## Output and Local Paths

Run directories must be outside the Git worktree and must not already exist.
Large data, checkpoints, waveform caches, TensorBoard events, and generated
samples do not belong in Git.

Host-specific data/checkpoint paths are currently explicit in the tracked
manual-refactor configs because they are part of this server's locked evidence
identity. A copied experimental bundle may change those paths, but the run
record must preserve the resolved values and hashes.

Before GPU execution, inspect `nvidia-smi` and set `CUDA_VISIBLE_DEVICES` to
confirmed free devices. Configuration files never select GPU IDs.

## Evidence Boundary

`active_scripts.yaml` defines what can run. It does not prove any metric.
`active_evidence_registry.yaml` defines which artifacts support a development
or historical result. The current simplified AugMix plus VAE-LHAT recipe is
heldout-tuned single-seed development evidence until prospective multi-seed
replication is registered. Historical `r2` scores remain metric-level evidence
for the pre-migration typed-method-graph runtime and are not evidence that the
current RecipeSpec implementation has been replayed on GPU.
