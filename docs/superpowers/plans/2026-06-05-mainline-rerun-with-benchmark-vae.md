# Mainline Rerun With Benchmark VAE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rerun the v7 SJR/RGQ K500 four-center mainline comparison across EfficientNet1DV2, ECGFounder, and ptbxl_benchmarking backbones, including matched direct and VAE-LHAT training.

**Architecture:** Keep long launch commands behind tracked YAML configs. Reuse the existing VAE-LHAT online adversarial training wrapper, but make it model-zoo aware so benchmark backbones can use the same ref-excluded K500 protocol as EfficientNet. Use 4 GPUs as four independent single-GPU center jobs rather than DDP.

**Tech Stack:** Python, PyTorch, YAML-managed `scripts/run_experiment.py`, `ecg_adv_gen.config` typed adapters, PN2021 ref-excluded evaluation.

---

### Task 1: Freeze Current Refactor SHA

**Files:**
- Commit only already-staged source/config/test/doc changes.
- Leave unstaged `model/*` external handles untouched.

- [x] **Step 1: Run artifact guard checks**

Run:

```bash
micromamba run -n cli-tools git status --short --branch
git diff --cached --check
git diff --check
git diff --cached --name-only | rg '(^model/|\.pt$|\.pth$|\.ckpt$|\.npz$|\.npy$|\.h5$|\.pkl$|(^|/)(runs|outputs|checkpoints|mlruns|wandb|datasets)/)' || true
```

Expected: no staged model paths, large artifacts, or whitespace errors.

- [x] **Step 2: Verify CPU-managed layer**

Run:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python scripts/agent/audit_agent_workspace.py --skip-existing-artifacts
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest util/tests/test_config_loader.py util/tests/test_direct_finetune_adapter.py util/tests/test_pn2021_eval_adapter.py util/tests/test_pn2021c_eval_adapter.py util/tests/test_prompt_token_adapter.py util/tests/test_agent_operating_layer.py util/tests/test_run_record_management.py -q
```

Expected: audit passes with only external-model/source-review warnings; tests pass.

- [x] **Step 3: Commit and push**

Run:

```bash
git commit -m "Refactor paper-safe experiment adapters"
git push origin main
```

Expected: pushed commit `c4ccc5f` or later.

### Task 2: Add Benchmark VAE-LHAT Managed Surface

**Files:**
- Create: `ecg_adv_gen/config/adapters/benchmark_vae_lhat.py`
- Modify: `ecg_adv_gen/config/adapters/registry.py`
- Modify: `ecg_adv_gen/config/adapters/__init__.py`
- Modify: `ecg_adv_gen/config/runner_audit.py`
- Modify: `configs/schemas/experiment_config.schema.json`
- Modify: `ecg_adv_gen/runner/effnet_vae_lhat.py`
- Modify: `scripts/paper/run_effnet_latent_augmix_stage3_20260524.py`
- Create: `configs/experiments/benchmark_resnet1d_vae_lhat_k500_v7_sjr_rgq.yaml`
- Create: `configs/experiments/benchmark_inception1d_vae_lhat_k500_v7_sjr_rgq.yaml`
- Create: `configs/experiments/benchmark_fcn_wang_vae_lhat_k500_v7_sjr_rgq.yaml`
- Modify: `configs/active_scripts.yaml`
- Test: `util/tests/test_benchmark_vae_lhat_adapter.py`
- Test: `util/tests/test_config_loader.py`

- [x] **Step 1: Write failing adapter test**

Create `util/tests/test_benchmark_vae_lhat_adapter.py` that calls `build_benchmark_vae_lhat_argv()` with `model.name=benchmark_resnet1d_wang`, `runtime.run_id=run1`, `matrix.center=ningbo`, and asserts:

```python
argv[argv.index("--center") + 1] == "ningbo"
argv[argv.index("--model_name") + 1] == "benchmark_resnet1d_wang"
argv[argv.index("--init_ckpt") + 1] == "/out/benchmark_resnet1d_wang_direct_k500_v7_sjr_rgq/run1/ningbo/runs/ningbo_K500_direct_ft_benchmark_resnet1d_wang_ep30_seed20260531_val0p2/best_model.pt"
argv[argv.index("--anchor_base") + 1] == "/refs/ningbo/k500_seed20260531/ningbo_real_k500_seed20260531"
argv[argv.index("--out_root") + 1] == "/out/benchmark_resnet1d_wang_vae_lhat_k500_v7_sjr_rgq/run1"
```

Run:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest util/tests/test_benchmark_vae_lhat_adapter.py -q
```

Expected: fails because adapter is missing.

- [x] **Step 2: Implement benchmark adapter and registry**

Implement `build_benchmark_vae_lhat_argv()` by reusing v7/K500 typed fields from `vae_lhat_defaults.yaml`, passing the same LHAT parameters as the legacy benchmark VAE branch:

```text
--model_name ${model.name}
--init_ckpt ${paths.output_root}/${model.name}_direct_k500_v7_sjr_rgq/${runtime.run_id}/${matrix.center}/runs/<benchmark direct leaf>/best_model.pt
--anchor_base ${data.kshot_subset_root}/${matrix.center}/k500_seed20260531/${matrix.center}_real_k500_seed20260531
--quick_eval_source target_real_val
```

Expected: new adapter is registered as `benchmark_vae_lhat`.

- [x] **Step 3: Make VAE wrapper model-zoo aware**

Modify `scripts/paper/run_effnet_latent_augmix_stage3_20260524.py` to accept:

```python
p.add_argument("--model_name", default="efficientnet1dv2", choices=available_model_names())
```

Modify `ecg_adv_gen/runner/effnet_vae_lhat.py` so train and eval command builders pass `args.model_name` instead of hardcoded `efficientnet1dv2`.

Expected: `synth_online_at_super5.py` and `eval_crosscenter.py` load the benchmark backbone.

- [x] **Step 4: Add command audit**

Add `audit_benchmark_vae_lhat_command()` and dispatch it through `runner_audit.py` when `runner.adapter == benchmark_vae_lhat`. It must assert:

```text
script == run_effnet_latent_augmix_stage3_20260524.py
--model_name == config.model.name
--init_ckpt contains ${model.name}_direct_k500_v7_sjr_rgq/${runtime.run_id}/${center}/runs
--anchor_base encodes k500_seed20260531
--quick_eval_source == target_real_val
--device == cuda
```

Expected: config validation catches wrong model/init/ref paths before launch.

- [x] **Step 5: Add three benchmark VAE YAMLs**

Create one YAML per model:

```text
benchmark_resnet1d_vae_lhat_k500_v7_sjr_rgq.yaml
benchmark_inception1d_vae_lhat_k500_v7_sjr_rgq.yaml
benchmark_fcn_wang_vae_lhat_k500_v7_sjr_rgq.yaml
```

Each extends benchmark defaults plus `vae_lhat_defaults.yaml`, sets `runner.adapter: benchmark_vae_lhat`, and uses `runner.matrix.center: ${paper_protocol.centers.target_4}`.

Expected: each config expands to four commands.

- [x] **Step 6: Update active index/schema/config tests**

Add `benchmark_vae_lhat` to schema enum and active scripts. Update `test_tracked_configs_validate_and_expand_commands` with the three new configs, each expecting four commands.

Run:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python -m pytest util/tests/test_benchmark_vae_lhat_adapter.py util/tests/test_config_loader.py util/tests/test_agent_operating_layer.py -q
```

Expected: tests pass.

### Task 3: Dry-Run Launch Surfaces

**Files:**
- No source edits expected after Task 2.

- [x] **Step 1: Dry-run direct and VAE configs**

Run dry-runs with run id `mainline_v7_k500_4gpu_20260605` for:

```text
effnet_direct_k500_v7_sjr_rgq_matrix.yaml
effnet_vae_lhat_k500_v7_sjr_rgq.yaml
ecgfounder_direct_k500_v7_sjr_rgq_matrix.yaml
ecgfounder_vae_lhat_k500_v7_sjr_rgq.yaml
benchmark_resnet1d_direct_k500_v7_sjr_rgq_matrix.yaml
benchmark_resnet1d_vae_lhat_k500_v7_sjr_rgq.yaml
benchmark_inception1d_direct_k500_v7_sjr_rgq_matrix.yaml
benchmark_inception1d_vae_lhat_k500_v7_sjr_rgq.yaml
benchmark_fcn_wang_direct_k500_v7_sjr_rgq_matrix.yaml
benchmark_fcn_wang_vae_lhat_k500_v7_sjr_rgq.yaml
```

Expected: all dry-runs pass, commands are one center per child, no long bash GPU parameters.

- [x] **Step 2: Commit benchmark VAE launch-surface changes**

Run artifact guard checks, then:

```bash
git add docs/superpowers/plans/2026-06-05-mainline-rerun-with-benchmark-vae.md ecg_adv_gen/config/adapters/benchmark_vae_lhat.py ecg_adv_gen/config/adapters/registry.py ecg_adv_gen/config/adapters/__init__.py ecg_adv_gen/config/runner_audit.py configs/schemas/experiment_config.schema.json ecg_adv_gen/runner/effnet_vae_lhat.py scripts/paper/run_effnet_latent_augmix_stage3_20260524.py configs/experiments/benchmark_*_vae_lhat_k500_v7_sjr_rgq.yaml configs/active_scripts.yaml util/tests/test_benchmark_vae_lhat_adapter.py util/tests/test_config_loader.py
git commit -m "Add benchmark VAE-LHAT launch configs"
git push origin main
```

Expected: working tree remains dirty only in `model/*`.

### Task 4: Launch Four-GPU Training Waves

**Files:**
- No source edits expected.

- [ ] **Step 1: GPU and host preflight**

Run:

```bash
sed -n '1,100p' AGENTS.md
nvidia-smi
cat /proc/loadavg
free -h
df -h .
```

Expected: pick four free GPUs explicitly; do not use all 8.

- [ ] **Step 2: Launch direct waves before dependent VAE waves**

Run direct configs first because VAE configs consume same-`run_id` direct checkpoints:

```text
EfficientNet direct
ECGFounder direct
benchmark ResNet1D direct
benchmark Inception1D direct
benchmark FCN-Wang direct
```

Use four independent single-GPU jobs by center (`ningbo`, `chapman_shaoxing`, `cpsc_2018`, `georgia`). Do not pass `CUDA_VISIBLE_DEVICES=0,1,2,3` to one launcher process.

- [ ] **Step 3: Launch VAE waves**

After each model family's direct checkpoints exist, run:

```text
EfficientNet VAE-LHAT
ECGFounder VAE-LHAT
benchmark ResNet1D VAE-LHAT
benchmark Inception1D VAE-LHAT
benchmark FCN-Wang VAE-LHAT
```

Expected: each wave writes run cards, manifests, logs, checkpoints, and eval JSONs under run-id-scoped output roots.

### Task 5: Evaluation And Bundle

**Files:**
- No source edits expected unless evaluation dry-run exposes a real launch-surface bug.

- [ ] **Step 1: Run ref-excluded PN2021 eval and reporting**

Use managed PN2021 ref-excluded eval where possible; otherwise use the eval JSONs emitted by direct/VAE wrappers and export metrics with `scripts/export_metrics_long.py`.

- [ ] **Step 2: Build comparison bundle**

Run:

```bash
/home/linbinhao/micromamba/envs/ECGTwin/bin/python scripts/agent/build_comparison_bundle.py --force
```

Expected: comparison bundle records v7 mapping hash `555ec85d5b51`, K500 ref exclusion, and direct-vs-VAE metrics for all completed model families.
