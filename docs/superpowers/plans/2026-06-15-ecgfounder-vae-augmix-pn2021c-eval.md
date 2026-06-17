# ECGFounder VAE AugMix PN2021-C Evaluation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether the ECGTwin VAE-LHAT + latent AugMix K500 method improves ECGFounder PN2021-C robustness over matched ECGFounder direct K500 and VAE-LHAT noAug baselines.

**Architecture:** Treat the current v7 SJR/RGQ managed configs and saved run artifacts as the source of truth. First validate existing four-center clean and PN2021-C JSON outputs; only run GPU evaluation for missing or invalid center/variant outputs.

**Tech Stack:** Python 3 in the `ECGTwin` micromamba environment, `scripts/run_experiment.py`, `scripts/triple_labels/eval_ecgfounder_pn2021_corruptions.py`, managed YAML under `configs/experiments/`, report CSV/JSON under `docs/reports/archive/20260605/`.

---

### Task 1: Confirm The Evaluation Surface

**Files:**
- Read: `configs/active_scripts.yaml`
- Read: `configs/experiments/ecgfounder_direct_k500_v7_sjr_rgq_matrix.yaml`
- Read: `configs/experiments/ecgfounder_vae_lhat_k500_v7_sjr_rgq.yaml`
- Read: `configs/experiments/ecgfounder_vae_lhat_augmix_k500_v7_sjr_rgq.yaml`

- [ ] **Step 1: Confirm variants and mapping**

Run:

```bash
sed -n '340,380p' configs/active_scripts.yaml
sed -n '1,240p' configs/experiments/ecgfounder_direct_k500_v7_sjr_rgq_matrix.yaml
sed -n '1,280p' configs/experiments/ecgfounder_vae_lhat_k500_v7_sjr_rgq.yaml
sed -n '1,300p' configs/experiments/ecgfounder_vae_lhat_augmix_k500_v7_sjr_rgq.yaml
```

Expected:
- `ecgfounder_direct_k500_v7_sjr_rgq_matrix` is active and uses `v7_super5_sjr_rgq_review_20260528`.
- `ecgfounder_vae_lhat_k500_v7_sjr_rgq` consumes same-run direct K500 heads.
- `ecgfounder_vae_lhat_augmix_k500_v7_sjr_rgq` extends the noAug config and only enables `adaptation.latent_augmix`.
- Target centers are `ningbo`, `chapman_shaoxing`, `cpsc_2018`, and `georgia`.

### Task 2: Validate Existing Clean And PN2021-C Artifacts

**Files:**
- Read: `docs/reports/archive/20260605/pn2021_clean_pn2021c_direct_vae_augmix_ablation_completed_per_center_20260605.csv`
- Read: `docs/reports/archive/20260605/pn2021c_dropallzero_eval_manifest_20260605.json`
- Read: `docs/reports/archive/20260605/pn2021c_dropallzero_eval_logs_20260605/ECGFounder_*.log`

- [ ] **Step 1: Check the four-center row matrix**

Run:

```bash
micromamba run -n ECGTwin python - <<'PY'
import csv
from pathlib import Path

csv_path = Path("docs/reports/archive/20260605/pn2021_clean_pn2021c_direct_vae_augmix_ablation_completed_per_center_20260605.csv")
centers = {"ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"}
variants = {"direct", "vae_noaug", "vae_augmix"}
datasets = {"PN2021 clean", "PN2021-C"}
rows = list(csv.DictReader(csv_path.open()))
missing = []
for dataset in sorted(datasets):
    for variant in sorted(variants):
        found = {
            row["center"]
            for row in rows
            if row["dataset"] == dataset
            and row["model"] == "ECGFounder"
            and row["variant"] == variant
        }
        if found != centers:
            missing.append((dataset, variant, sorted(centers - found), sorted(found - centers)))
if missing:
    raise SystemExit(f"missing/inconsistent rows: {missing}")
print("ok: ECGFounder clean and PN2021-C rows cover 3 variants x 4 centers")
PY
```

Expected:

```text
ok: ECGFounder clean and PN2021-C rows cover 3 variants x 4 centers
```

- [ ] **Step 2: Check PN2021-C JSON field coverage**

Run:

```bash
micromamba run -n ECGTwin python - <<'PY'
import csv, json
from pathlib import Path

csv_path = Path("docs/reports/archive/20260605/pn2021_clean_pn2021c_direct_vae_augmix_ablation_completed_per_center_20260605.csv")
required = {
    "n_all_zero_labels",
    "drop_all_zero_macro_auroc",
    "drop_all_zero_macro_auprc",
}
bad = []
for row in csv.DictReader(csv_path.open()):
    if row["dataset"] != "PN2021-C" or row["model"] != "ECGFounder":
        continue
    payload = json.load(open(row["source"]))
    text = json.dumps(payload)
    for key in required:
        if key not in text:
            bad.append((row["variant"], row["center"], key, row["source"]))
if bad:
    raise SystemExit(f"missing drop-all-zero fields: {bad[:5]}")
print("ok: ECGFounder PN2021-C JSONs include drop-all-zero fields")
PY
```

Expected:

```text
ok: ECGFounder PN2021-C JSONs include drop-all-zero fields
```

### Task 3: Run Missing Evaluation Only If Needed

**Files:**
- Run if missing: `scripts/triple_labels/eval_ecgfounder_pn2021_corruptions.py`
- Read: `docs/reports/archive/20260605/pn2021c_dropallzero_eval_logs_20260605/ECGFounder_*.log`

- [ ] **Step 1: Preflight before any GPU eval**

Run:

```bash
sed -n '1,100p' AGENTS.md
git status --short --branch
nvidia-smi
df -h /home/linbinhao /home/linbinhao/ECG_adv_data
free -h
```

Expected:
- No `sudo`, no system changes.
- Choose one low-occupancy GPU and use `CUDA_VISIBLE_DEVICES=<id>`.
- Output paths remain under `/home/linbinhao`.

- [ ] **Step 2: Reuse exact logged commands for missing center/variant jobs**

Run the matching command from:

```bash
sed -n '1p' docs/reports/archive/20260605/pn2021c_dropallzero_eval_logs_20260605/ECGFounder_vae_augmix_ningbo.log
sed -n '1p' docs/reports/archive/20260605/pn2021c_dropallzero_eval_logs_20260605/ECGFounder_vae_augmix_chapman_shaoxing.log
sed -n '1p' docs/reports/archive/20260605/pn2021c_dropallzero_eval_logs_20260605/ECGFounder_vae_augmix_cpsc_2018.log
sed -n '1p' docs/reports/archive/20260605/pn2021c_dropallzero_eval_logs_20260605/ECGFounder_vae_augmix_georgia.log
```

Expected:
- Each rerun writes one `eval_pn2021_c_v7_refexcluded_stream_standard_dropallzero.json`.
- Each rerun processes 25 corruption/severity units for one center.

### Task 4: Aggregate Results And Deltas

**Files:**
- Read: `docs/reports/archive/20260605/pn2021_clean_pn2021c_direct_vae_augmix_ablation_completed_per_center_20260605.csv`
- Produce summary in the final response; no HTML update in this task.

- [ ] **Step 1: Compute clean and PN2021-C means**

Run:

```bash
micromamba run -n ECGTwin python - <<'PY'
import csv
from collections import defaultdict
from pathlib import Path

csv_path = Path("docs/reports/archive/20260605/pn2021_clean_pn2021c_direct_vae_augmix_ablation_completed_per_center_20260605.csv")
rows = [
    row for row in csv.DictReader(csv_path.open())
    if row["model"] == "ECGFounder"
    and row["variant"] in {"direct", "vae_noaug", "vae_augmix"}
    and row["dataset"] in {"PN2021 clean", "PN2021-C"}
]
by_key = defaultdict(list)
for row in rows:
    by_key[(row["dataset"], row["variant"])].append(row)
for key, group in sorted(by_key.items()):
    auroc = sum(float(r["macro_auroc"]) for r in group) / len(group)
    auprc = sum(float(r["macro_auprc"]) for r in group) / len(group)
    print(key[0], key[1], f"auroc={auroc:.6f}", f"auprc={auprc:.6f}", f"n={len(group)}")
PY
```

Expected:
- Six summary rows: 2 datasets x 3 variants.
- Each row has `n=4`.

- [ ] **Step 2: Compute VAE+AugMix deltas**

Run:

```bash
micromamba run -n ECGTwin python - <<'PY'
import csv
from collections import defaultdict
from pathlib import Path

csv_path = Path("docs/reports/archive/20260605/pn2021_clean_pn2021c_direct_vae_augmix_ablation_completed_per_center_20260605.csv")
metrics = {}
for row in csv.DictReader(csv_path.open()):
    if row["model"] != "ECGFounder":
        continue
    if row["variant"] not in {"direct", "vae_noaug", "vae_augmix"}:
        continue
    key = (row["dataset"], row["variant"])
    metrics.setdefault(key, []).append((float(row["macro_auroc"]), float(row["macro_auprc"])))
means = {key: (sum(a for a, _ in vals) / len(vals), sum(p for _, p in vals) / len(vals)) for key, vals in metrics.items()}
for dataset in ["PN2021 clean", "PN2021-C"]:
    aug = means[(dataset, "vae_augmix")]
    direct = means[(dataset, "direct")]
    noaug = means[(dataset, "vae_noaug")]
    print(dataset)
    print("  vae_augmix", f"{aug[0]:.6f}", f"{aug[1]:.6f}")
    print("  delta_vs_direct_pp", f"{(aug[0] - direct[0]) * 100:+.3f}", f"{(aug[1] - direct[1]) * 100:+.3f}")
    print("  delta_vs_noaug_pp", f"{(aug[0] - noaug[0]) * 100:+.3f}", f"{(aug[1] - noaug[1]) * 100:+.3f}")
PY
```

Expected:
- Direct delta answers whether the full VAE+AugMix method beats matched Direct K500.
- noAug delta isolates the latent AugMix branch inside ECGFounder.

### Task 5: Completion Criteria

- [ ] Confirm no missing ECGFounder center/variant PN2021-C JSONs remain.
- [ ] Confirm all reported numbers are ref-excluded, v7 SJR/RGQ, four-center means.
- [ ] Report both all-zero-kept and drop-all-zero if both are available; otherwise state which view was verified.
- [ ] Report whether the evidence is from current files only or includes a fresh GPU rerun.
- [ ] Leave the goal active only if any required center/variant is missing or any JSON fails validation.
