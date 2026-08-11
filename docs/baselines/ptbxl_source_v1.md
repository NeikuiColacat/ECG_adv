# PTB-XL Super5 Source Baseline v1 Lock Log

## Lock decision

- Status: `LOCKED`
- Locked on: `2026-07-17`
- Machine-readable source of truth: `configs/baselines/ptbxl_source_v1.yaml`
- Historical registry SHA256 at lock: `27443d14e8184739c15c60e0ffe6f4c87f12642bb725bdca62341a33543e445f`
- Current registry SHA256: `e8f1461348e1be35076d175c6151f767e1fb90ed5e5bbebec430b9d8c55f9ea7`
- Historical training config snapshot: `configs/train/PTBXL.yaml` at commit `254c0c80318840430171317c0051497f62232a6e`
- Historical training config SHA256: `ff984500052e0e64e245c68b8c51a1080efd7bd375c70a44dbd49bf9a9cf9f4c`
- Current observer-free replay config SHA256: `85638bbea2c12669dab45973b77223513f5292a11faf12b0c814fe1fe41679c8`
- Seed config SHA256: `8b337606c43b6fec2959dd185477ce32f48b139f5fce5112ee26efe58eb4bbbc`
- Class order: `CD,HYP,MI,NORM,STTC`
- Selection rule: PTB-XL folds 1-8 train, fold 9 macro AUPRC selects
  `best.pt`, restore that checkpoint, then evaluate fold 10 once.

All later PTB-XL-to-PN2021 experiments must use the registered `best.pt` for
their model family. `last.pt` and the ECGFounder 10-epoch probe checkpoint are
not valid source baselines. Replacing either source baseline requires a new
versioned registry and a new lock log; do not edit this v1 identity in place.
The current replay config removes only the retired TensorBoard/waveform
observer. The historical config bytes remain available from the locked Git
object and each managed run's `configs/train/PTBXL.yaml` snapshot; neither the
source checkpoint identity nor the scientific training payload was rewritten.

## Locked checkpoints

| Model | Input contract | Selected epoch | Fold 9 AUROC / AUPRC | Fold 10 AUROC / AUPRC | Checkpoint SHA256 |
|---|---|---:|---:|---:|---|
| EfficientNet1DV2 | `(12,1000)`, 100 Hz | 17 | `0.910160 / 0.779058` | `0.901281 / 0.767951` | `fff9bd293677be54a606e28775c66d410d55b0ad7440a4e7fbde55357ec13e68` |
| ECGFounder full FT | `(12,5000)`, 100 Hz cache linearly expanded to 500 Hz | 4 | `0.928857 / 0.817821` | `0.929407 / 0.824164` | `c23ec4361b097b74a4efed427488c4dd6a3753adfc75f7d1fc281e4df0989948` |

Locked checkpoint paths:

```text
/home/linbinhao/ECG_adv_data/runs/manual_refactor/managed_ptbxl_effnet_source_v1/training/checkpoints/best.pt
/home/linbinhao/ECG_adv_data/runs/manual_refactor/managed_ptbxl_ecgfounder_source_v1/training/checkpoints/best.pt
```

The ECGFounder source model loads the official 12-lead pretrained checkpoint
and replaces its task head with the project Super5 head. Its pretrained
checkpoint identity is:

```text
path:   /home/linbinhao/ECG_adv_data/ecgfounder/checkpoint/12_lead_ECGFounder.pth
sha256: ee199f3781f4ae1f732973267f003da0a759ea12bddb0dd28a77faa60aca7997
scope:  full fine-tuning, 30,670,389 trainable parameters
```

## Locked optimization recipes

EfficientNet1DV2:

```text
initialization: from scratch
epochs: 30
batch size: 128 train / 256 eval
optimizer: AdamW
learning rate: 1e-3
weight decay: 1e-4
scheduler: cosine annealing, minimum ratio 0.01
loss: unweighted BCE with logits
AMP: bfloat16
```

ECGFounder:

```text
initialization: official 12-lead ECGFounder checkpoint
trainable scope: full backbone and new Super5 head
epochs: 5
batch size: 64 train / 128 eval
optimizer: AdamW
learning rate: 1e-4 constant
weight decay: 1e-5
loss: unweighted BCE with logits
AMP: bfloat16
```

The global base seed is `20260501`. The supervised-training derived seed is
`681119983`. `deterministic_algorithms` was `false`, so the registered
checkpoint hashes, rather than an assumption of bitwise-identical retraining,
define the locked weights.

## ECGFounder 10-epoch sensitivity result

The only changed variable in the probe was `epochs: 5 -> 10`. Epochs 6-10 did
not produce a second improvement and instead showed strong overfitting:

| Epoch | Train loss | Fold 9 loss | Fold 9 AUROC | Fold 9 AUPRC |
|---:|---:|---:|---:|---:|
| 3, selected by probe | `0.213778` | `0.269113` | `0.926416` | `0.816635` |
| 5 | `0.172472` | `0.292377` | `0.923099` | `0.806118` |
| 6 | `0.145651` | `0.312771` | `0.920886` | `0.799835` |
| 8 | `0.082093` | `0.405591` | `0.910797` | `0.778857` |
| 10 | `0.039953` | `0.559074` | `0.903402` | `0.774217` |

The probe-selected fold-10 AUROC/AUPRC was `0.926682 / 0.820357`, below the
locked 5-epoch run's `0.929407 / 0.824164`. The probe is retained as negative
sensitivity evidence and must not replace the locked ECGFounder checkpoint.

## Replay commands

Working directory:

```text
/home/linbinhao/ECG_manual_refactor
```

EfficientNet1DV2:

```bash
CUDA_VISIBLE_DEVICES=0 /home/linbinhao/miniforge3/envs/ECGTwin/bin/python -u \
  boot_scripts/run_experiment.py \
  --config configs/experiments/manual_refactor_ptbxl_effnet.yaml
```

ECGFounder locked run:

```bash
CUDA_VISIBLE_DEVICES=0 /home/linbinhao/miniforge3/envs/ECGTwin/bin/python -u \
  boot_scripts/run_experiment.py \
  --config configs/experiments/manual_refactor_ptbxl_ecgfounder.yaml
```

ECGFounder 10-epoch sensitivity probe:

```bash
CUDA_VISIBLE_DEVICES=0 /home/linbinhao/miniforge3/envs/ECGTwin/bin/python -u \
  boot_scripts/run_experiment.py \
  --config configs/experiments/manual_refactor_ptbxl_ecgfounder_e10_probe.yaml
```

These commands must use new output directories if replayed; the managed
launcher intentionally refuses to overwrite the recorded runs.

## Raw evidence

EfficientNet run root:

```text
/home/linbinhao/ECG_adv_data/runs/manual_refactor/managed_ptbxl_effnet_source_v1/
```

ECGFounder locked run root:

```text
/home/linbinhao/ECG_adv_data/runs/manual_refactor/managed_ptbxl_ecgfounder_source_v1/
```

ECGFounder 10-epoch probe root:

```text
/home/linbinhao/ECG_adv_data/runs/manual_refactor/managed_ptbxl_ecgfounder_source_v1_e10_probe/
```

Each run root contains `run_card.json`, `run_manifest.json`,
`run_file_index.json`, `selection.json`, `data_manifest.json`,
`logs/delegate.log`, `training/train_result.json`,
`training/training_history.json`, the TensorBoard event file, and checkpoints.

Consolidated curve artifact:

```text
/home/linbinhao/ECG_adv_data/runs/manual_refactor/ptbxl_source_baselines_v1_report/ptbxl_source_tensorboard_curves.png
```

## Reproducibility note

The runs recorded Git commit `153b5fe3142b175735d837e1e494370df6a448c5`,
branch `refactor/manual-20260715`, Python `3.11.5`, PyTorch `2.1.1+cu118`, and
CUDA runtime `11.8`. The worktree was dirty during training. The managed run
directories preserve the resolved configuration snapshots, exact commands,
environment, selections, and artifact hashes, but the whitelist source-code
state should still be committed before claiming a clean Git-only replay.
