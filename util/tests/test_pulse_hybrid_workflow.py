"""Small synthetic fixtures for the actual hybrid task graph and evidence path."""
import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import yaml

from util.evaluation.pulse_hybrid_development import ARMS, conditions_for, metrics_from_predictions, select_development
from util.pulse_hybrid_contract import validate_config
from util.pulse_hybrid_workflow import make_jobs

ROOT = Path(__file__).resolve().parents[2]


def search():
    return yaml.safe_load((ROOT / "configs/train/pulse_hybrid_search.yaml").read_text())


def test_full_lora_ram_storage_is_explicit_and_bounded():
    from util.pulse_training_contract import validate_config as validate_training
    config = yaml.safe_load((ROOT / "configs/train/pulse_full_lora32_pipeline.yaml").read_text())
    validate_config(config)
    for key, value in (("archive_root", "/data/another-user/experiment"),
                       ("max_workers", 5), ("ram_reserve_gib", 0)):
        bad = copy.deepcopy(config); bad["storage"][key] = value
        with pytest.raises(ValueError):
            validate_config(bad)
    bad = copy.deepcopy(config); del bad["storage"]
    with pytest.raises(ValueError, match="archival contract"):
        validate_config(bad)
    training = yaml.safe_load((ROOT / "configs/train/pulse_full_lora32_template.yaml").read_text())
    training["resume_from"] = config["paths"]["run_root"] + "/smoke/three/training/resume_probe.pt"
    validate_training(training)
    training["resume_from"] = "/dev/shm/another-user/checkpoint.pt"
    with pytest.raises(ValueError, match="scoped RAM"):
        validate_training(training)


def test_live_search_schema_and_unsafe_controls():
    config = search()
    validate_config(config)
    for key, value in (("max_gpu_workers", 8), ("trials", 999), ("evidence_role", "paper_final")):
        bad = copy.deepcopy(config); bad[key] = value
        with pytest.raises(ValueError):
            validate_config(bad)


def test_fixed_recovery_requires_pinned_owned_archive():
    config = yaml.safe_load((ROOT / "configs/train/pulse_full_lora32_pipeline.yaml").read_text())
    config["recovery"] = {"archive_root": "/data/linbinhao/ecg_llm_runs/pulse_full_lora32_interrupted",
                          "archive_receipt_sha256": "a" * 64}
    validate_config(config)
    for key, value in (("archive_root", "/data/another-user/run"),
                       ("archive_receipt_sha256", "unpinned")):
        bad = copy.deepcopy(config); bad["recovery"][key] = value
        with pytest.raises(ValueError, match="recovery archive identity"):
            validate_config(bad)


def test_fixed_recovery_reuses_smoke_and_preserves_checkpoint_identity(tmp_path):
    from util.pulse_hybrid_workflow import prepare_fixed_recovery
    from util.pn2021_artifact_contract import sha256_file
    config = yaml.safe_load((ROOT / "configs/train/pulse_full_lora32_pipeline.yaml").read_text())
    config.pop("recovery", None)
    template = yaml.safe_load((ROOT / "configs/train/pulse_full_lora32_template.yaml").read_text())
    archive = tmp_path / "archive"
    parent = archive / "run"
    files = {}
    def put(relative, value):
        p = parent / relative; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(value if isinstance(value, str) else json.dumps(value))
        files[relative] = {"sha256": sha256_file(p), "size_bytes": p.stat().st_size}
        return p
    put("configs/train/pulse_full_lora32_pipeline.yaml", yaml.safe_dump(config))
    put("configs/train/pulse_full_lora32_template.yaml", yaml.safe_dump(template))
    sources = {"core/pulse_finetune.py": "unchanged", "util/pulse_hybrid_workflow.py": "old"}
    put("state/identity.json", {"sources": sources})
    old_root = Path(config["paths"]["run_root"])
    smokes = {}
    for center in config["centers"]:
        for kind in ("single", "three", "resume", "eval"):
            relative = f"smoke/{center}_{kind}/training/result.json"
            p = put(relative, {"status": "complete"})
            smokes[f"smoke_{center}_{kind}"] = {"path": str(old_root / relative), "sha256": sha256_file(p)}
            if kind == "resume":
                put(f"smoke/{center}_{kind}/training/recovery_audit.json",
                    {"process_resume_verified": True, "resume_max_trainable_delta": 0.0})
    put("workflow/smoke_results.json", smokes)
    checkpoint = put("final/ningbo_single/training/checkpoint.pt", "checkpoint bytes")
    put("final/ningbo_single/training/protocol.json",
        {"implementation_sha256": {"core/pulse_finetune.py": "unchanged"}})
    receipt = archive / "archive_receipt.json"
    receipt.write_text(json.dumps({"schema_version": 1, "status": "verified",
        "source_status": "failed", "source_run_root": str(old_root), "files": files}))
    config["paths"]["run_root"] = str(tmp_path / "new")
    config["recovery"] = {"archive_root": str(archive), "archive_receipt_sha256": sha256_file(receipt)}
    current = {**sources, "util/pulse_hybrid_workflow.py": "new"}
    smoke, checkpoints, completed = prepare_fixed_recovery(config, template, current, tmp_path / "new/state")
    assert completed == {}
    assert len(smoke) == 16 and set(checkpoints) == {"ningbo_single"}
    assert Path(checkpoints["ningbo_single"]).read_bytes() == checkpoint.read_bytes()
    versioned_config = "configs/train/pulse_full_lora32_live_eval_20260928.yaml"
    previous = copy.deepcopy(config)
    previous.pop("recovery")
    previous["paths"]["run_root"] = str(old_root)
    put(versioned_config, yaml.safe_dump(previous))
    put("run_manifest.json", {"config_snapshots": [
        {"role": "delegate_entry_config", "snapshot_path": versioned_config}]})
    receipt.write_text(json.dumps({"schema_version": 1, "status": "verified",
        "source_status": "failed", "source_run_root": str(old_root), "files": files}))
    config["recovery"]["archive_receipt_sha256"] = sha256_file(receipt)
    smoke, checkpoints, completed = prepare_fixed_recovery(config, template, current, tmp_path / "named/state")
    assert completed == {} and len(smoke) == 16
    assert Path(checkpoints["ningbo_single"]).read_bytes() == checkpoint.read_bytes()
    with pytest.raises(ValueError, match="model/data/evaluation code"):
        prepare_fixed_recovery(config, template, {**current, "core/pulse_finetune.py": "drift"}, tmp_path / "bad/state")
    checkpoint.write_text("tampered checkpoint")
    with pytest.raises(ValueError, match="recovery artifact changed"):
        prepare_fixed_recovery(config, template, current, tmp_path / "bad/state")


def test_immutable_copied_task_graph_and_equal_training_budgets(tmp_path, monkeypatch):
    from util import pulse_hybrid_contract as contract
    from util.config_bundle import resolve_yaml_config_closure
    # Permit this test's private temporary directory, not a real run directory.
    monkeypatch.setattr(contract, "RAM", tmp_path)
    config = search()
    bundle = tmp_path / "parent/configs"
    for source in resolve_yaml_config_closure([ROOT / "configs/train/pulse_hybrid_search.yaml"], config_root=ROOT / "configs"):
        destination = bundle / source.relative_to(ROOT / "configs")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    config["paths"].update(run_root=str(tmp_path / "durable"), temporary_root=str(tmp_path / "ram"))
    template = yaml.safe_load((bundle / "train/pulse_hybrid_smoke.yaml").read_text())
    smoke = make_jobs(config, bundle, template, phase="smoke", recipe=None)
    assert len(smoke) == 5
    assert len(smoke["smoke_ningbo_eval"]["dependencies"]) == 4
    resume = yaml.safe_load(smoke["smoke_ningbo_resume"]["plan"].entry_config_path.read_text())
    assert resume["width"] == 3 and resume["resume_from"].endswith("resume_probe.pt")
    recipe = dict(learning_rate=1e-5, projector_learning_rate=1e-5, jsd_weight=3.0, waveform_strength=0.35, image_strength=0.5)
    jobs = make_jobs(config, bundle, template, phase="screen", recipe=recipe, trial_number=0)
    assert len(jobs) == 16
    for name, job in jobs.items():
        plan = job["plan"]
        assert not plan.entry_arguments and plan.data_ledger is not None
        if name.endswith("_eval"):
            assert len(job["dependencies"]) == 3 and plan.run_dir.is_relative_to(tmp_path / "durable")
        else:
            value = yaml.safe_load(plan.entry_config_path.read_text())
            assert (value["training"]["optimizer_steps"], value["training"]["effective_batch"]) == (25, 16)
            assert plan.run_dir.is_relative_to(tmp_path / "ram")
    assert make_jobs(config, bundle, template, phase="screen", recipe=recipe, trial_number=0).keys() == jobs.keys()
    with pytest.raises(ValueError, match="drift"):
        make_jobs(config, bundle, template, phase="screen", recipe={**recipe, "jsd_weight": 6.0}, trial_number=0)


def test_record_exclusion_and_label_independent_prefix():
    rows = [{"hash_id": str(i), "label": [i % 2]} for i in range(512)]
    selected = select_development(rows, [{"hash_id": "train"}], 128)
    assert selected == rows[:128]
    with pytest.raises(ValueError, match="overlapping"):
        select_development(rows, [{"hash_id": "511"}], 4)


def fixture():
    conditions = [{"condition_id": name, "family": name} for name in ("clean", "waveform", "image")]
    samples = [{"sample_key": str(i), "label": [int(i == j) for j in range(5)],
                "label_names": [name]} for i, name in enumerate(("CD", "HYP", "MI", "NORM", "STTC"))]
    rows = [{"sample_key": s["sample_key"], "condition_id": c["condition_id"], "true_labels": s["label_names"],
        "arms": {a: {"predicted_labels": s["label_names"], "parse_valid": True, "hit_max_new_tokens": False} for a in ARMS}}
        for s in samples for c in conditions]
    return rows, samples, conditions


def test_development_metrics_reject_missing_or_duplicate_predictions():
    rows, samples, conditions = fixture()
    result = metrics_from_predictions(rows, samples, conditions)
    assert all(v == 1 for a in result["families"].values() for v in a.values())
    for bad in (rows[:-1], rows + rows[:1]):
        with pytest.raises(ValueError, match="rectangle"):
            metrics_from_predictions(bad, samples, conditions)
    rows[0]["arms"]["three"].update(predicted_labels=[], parse_valid=False, hit_max_new_tokens=True)
    result = metrics_from_predictions(rows, samples, conditions)
    row = next(r for r in result["per_condition"] if r["arm"] == "three" and r["family"] == "clean")
    assert row["macro_f1"] == pytest.approx(0.8) and row["parse_failure_rate"] == pytest.approx(0.2)


def test_fixed_evaluation_families_are_not_tuned_with_training_strength():
    parent = [{"condition_id": "clean" if i == 0 else str(i), "operators": []} for i in range(21)]
    conditions = conditions_for(parent, [1, 10, 11, 20], 1.0)
    assert len(conditions) == 9
    assert sum(c["family"] == "image" for c in conditions) == 4
    assert len(conditions_for(parent, list(range(1, 21)), 1.0)) == 25


def test_scalar_tracking_never_includes_ecg_or_record_labels():
    from util.pulse_tracking import scalar_rows
    payload = {"artifact_type": "pulse_hybrid_result", "mode": "evaluate", "details": {
        "center": "ningbo", "phase": "screen", "records": 128, "development_only": True,
        "families": {a: {f: 0.5 for f in ("clean", "waveform", "image")} for a in ARMS}}}
    config, rows = scalar_rows(payload, Path("unused"))
    assert len(rows[0][1]) == 12 and config["development_only"]
    assert not any(k in json.dumps([config, rows]) for k in ("waveform_sha256", "patient_id", "true_labels"))


def test_real_scheduler_loop_obeys_four_gpu_cap_and_foreign_exclusion(tmp_path, monkeypatch):
    from util import pulse_hybrid_workflow as workflow
    from util.evaluation import ecg_image_elastic as resources
    state = tmp_path / "state"; state.mkdir()
    (state / "control.json").write_text(json.dumps({"paused": False, "max_workers": 4, "allowed_gpus": list(range(8))}))
    config = search(); config["paths"]["temporary_root"] = str(tmp_path)
    jobs = {f"j{i}": {"plan": SimpleNamespace(run_dir=tmp_path / f"run{i}"),
        "experiment": tmp_path / f"j{i}.yaml", "bundle": tmp_path, "dependencies": ()} for i in range(6)}
    clock, finished, children, launched = [0], set(), [], []
    monkeypatch.setattr(workflow.time, "time", lambda: clock[0])
    monkeypatch.setattr(workflow.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0]+seconds))
    monkeypatch.setattr(workflow.shutil, "disk_usage", lambda path: SimpleNamespace(free=100*1024**3))
    monkeypatch.setattr(resources, "resource_snapshot", lambda: {"available_ram_gib": 100, "load1": 1,
        "cpu_affinity_count": 8, "gpus": {i: {"uuid": f"GPU-{i}", "pids": [999] if i == 0 else []} for i in range(8)}})
    monkeypatch.setattr(resources, "free_device", lambda gpu: not gpu["pids"])
    monkeypatch.setattr(resources, "process_identity", lambda pid: {"pid": pid})
    monkeypatch.setattr(workflow, "checked_completion", lambda job: {"result": {}} if next(n for n,j in jobs.items() if j is job) in finished else None)
    monkeypatch.setattr(workflow, "mirror_metrics", lambda *a: None)
    monkeypatch.setattr(workflow.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="{}"))
    class Child:
        def __init__(self, command, **kwargs):
            self.name = Path(command[command.index("--config")+1]).stem
            self.pid = 100+len(children)
            self.uuid = kwargs["env"]["CUDA_VISIBLE_DEVICES"]
            living = [c for c in children if c.name not in finished]
            assert len(living) < 4 and self.uuid not in {c.uuid for c in living}
            assert self.uuid != "GPU-0"
            children.append(self); launched.append(self.name)
        def poll(self):
            finished.add(self.name)
            return 0
    monkeypatch.setattr(workflow.subprocess, "Popen", Child)
    assert len(workflow.run_jobs(jobs, config, state, {})) == 6
    assert set(launched) == set(jobs)


@pytest.mark.parametrize("arms", [ARMS, ("original", "single", "three")])
@pytest.mark.parametrize("verification", ["off", "exact", "restore_drift"])
def test_hybrid_backend_recomputes_clip_after_each_arm_switch(monkeypatch, arms, verification):
    from contextlib import nullcontext
    import torch
    from util.evaluation import pulse_adapters as adapters
    from util.evaluation.pulse_adapters import HybridPulseBackend
    calls = []
    class Base:
        def prepare_inputs_labels_for_multimodal(self, *args, **kwargs):
            return (None, None, None, None, torch.tensor([[ARMS.index(self.arm)]]))
        def generate(self, *args, **kwargs):
            return torch.tensor([[ARMS.index(self.arm)]])
        def get_vision_tower(self):
            def encode(pixels):
                calls.append(self.arm)
                return torch.tensor([[ARMS.index(self.arm)]])
            return encode
    class Parent:
        def __init__(self, evidence, reservation):
            self.base = Base(); self.parameters = {"projector": torch.tensor([1.0])}
            self.prompt_ids = torch.tensor([[1, 2]]); self.tokenizer = None
            self.admissions = {}; self.generation_kwargs = {}; self.original_selections = 0
        def set_pair(self, evidence):
            self.states = dict(evidence)
        def _select(self, arm):
            self.base.arm = arm
            if arm == "original": self.original_selections += 1
        def _generate(self, packed):
            if verification == "restore_drift" and self.original_selections > 1:
                return packed + 1
            return packed
    class Pixels:
        ndim = 5; shape = (1, 5, 3, 336, 336)
        def __len__(self): return 1
        def to(self, **kwargs): return self
        def flatten(self, *args): return self
    for name in ("__init__", "set_pair", "_select", "_generate"):
        if hasattr(Parent, name):
            monkeypatch.setattr(adapters.PairedPulseBackend, name, getattr(Parent, name))
    monkeypatch.setattr(adapters, "pack_shared_prompt_features", lambda base, ids, features: features)
    monkeypatch.setattr(adapters, "decode_generated", lambda tokenizer, generated, **kwargs: generated.tolist())
    monkeypatch.setattr(torch, "autocast", lambda *a, **kwargs: nullcontext())
    backend = HybridPulseBackend({a: {} for a in arms[1:]}, [], arms=arms)
    if verification == "restore_drift":
        with pytest.raises(ValueError, match="restoring adapter"):
            backend.generate_pair(Pixels(), verify_author=True)
    else:
        assert backend.generate_pair(Pixels(), verify_author=verification == "exact") == {a: [[ARMS.index(a)]] for a in arms}
        if verification == "exact":
            assert backend.original_restore_admission == {"tokens_exact_after_adapter_switches": True}
    assert calls == list(arms) + ([] if verification == "off" else ["original"])


def test_tracking_failure_does_not_discard_completed_compute(tmp_path, monkeypatch):
    import subprocess
    from util import pulse_hybrid_workflow as workflow
    entry = {"path": str(tmp_path / "result.json"), "sha256": "a"*64, "result": {}}
    monkeypatch.setattr(workflow.subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(subprocess.TimeoutExpired("trackio", 60)))
    workflow.mirror_metrics("probe", entry, search(), tmp_path)
    receipt = json.loads((tmp_path / "tracking/probe.json").read_text())
    assert receipt["source_sha256"] is None and "error" in receipt


@pytest.mark.parametrize("fail", [False, True])
def test_fp16_inference_restores_exact_fp32_parameter_storage(monkeypatch, fail):
    import torch
    from util.evaluation import pulse_adapters as adapters
    from util.evaluation.pulse_adapters import HybridPulseBackend
    class Parent:
        def __init__(self, evidence, reservation):
            self.parameters = {"model.layers.0.lora_A.default.weight": torch.tensor([0.1234567])}
        def set_pair(self, evidence):
            self.states = dict(evidence)
    for name in ("__init__", "set_pair", "_select", "_generate"):
        if hasattr(Parent, name):
            monkeypatch.setattr(adapters.PairedPulseBackend, name, getattr(Parent, name))
    backend = HybridPulseBackend({"single": {}, "three": {}}, [], arms=("original", "single", "three"), fp16_adapters=True)
    parameter = next(iter(backend.parameters.values()))
    before, pointer = parameter.clone(), parameter.data_ptr()
    def probe(views):
        assert parameter.dtype == torch.float16
        parameter.fill_(7)
        if fail:
            raise RuntimeError("simulated inference failure")
        return "generated"
    backend._generate_views = probe
    if fail:
        with pytest.raises(RuntimeError, match="simulated"):
            backend.generate_views([])
    else:
        assert backend.generate_views([]) == "generated"
    assert parameter.dtype == torch.float32 and parameter.data_ptr() == pointer
    assert torch.equal(parameter, before)


@pytest.mark.parametrize("arms", [ARMS, ("original", "single", "three")])
def test_bootstrap_identical_arms_has_exact_zero_paired_intervals(tmp_path, monkeypatch, arms):
    from util import pulse_hybrid_contract as contract
    from util.evaluation.ecg_image_queue import atomic_json
    from util.evaluation.pulse_hybrid_development import summarize_four_centers
    from util.pn2021_artifact_contract import sha256_file
    from util.pulse_training_contract import CENTERS
    # Test only the bootstrap algebra here, not synthetic final-run admission.
    monkeypatch.setattr(contract, "validate_result", lambda *a: None)
    rows, samples, conditions = fixture()
    for row in rows:
        row["arms"] = {a: row["arms"][a] for a in arms}
    metrics = metrics_from_predictions(rows, samples, conditions, arms=arms)
    entries = {}
    for center in CENTERS:
        directory = tmp_path / center
        atomic_json(directory / "cohort.json", {"samples": samples, "conditions": conditions})
        atomic_json(directory / "batches/0000.json", rows)
        atomic_json(directory / "metrics.json", metrics)
        path = directory / "hybrid_result.json"
        atomic_json(path, {"details": {"phase": "final", "center": center, "model_arms": list(arms)},
                           "files": {"batches/0000.json": "unused"}})
        entries[center] = {"path": str(path), "sha256": sha256_file(path)}
    output = tmp_path / "summary"
    summarize_four_centers(entries, output, seed=5, arms=arms)
    report = json.loads((output / "comparison.json").read_text())
    assert len(report["paired_deltas"]) == 3 * len(arms) * (len(arms) - 1) // 2
    assert all(r["delta_pp"] == 0 and r["ci95_pp"] == [0, 0] for r in report["paired_deltas"])
    assert report["bootstrap"]["selection_bias_corrected"] is False


def test_fixed_pipeline_closure_two_trained_arms_and_all_training_before_evaluation(tmp_path, monkeypatch):
    from util import pulse_hybrid_contract as contract
    from util.config_bundle import resolve_yaml_config_closure
    path = ROOT / "configs/train/pulse_reference_fixed_pipeline.yaml"
    config, _ = contract.load_config(path, ROOT / "configs")
    assert "search_space" not in config and config["mode"] == "fixed"
    monkeypatch.setattr(contract, "DATA", tmp_path)
    bundle = tmp_path / "parent/configs"
    for source in resolve_yaml_config_closure([path], config_root=ROOT / "configs"):
        destination = bundle / source.relative_to(ROOT / "configs")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    config["paths"].update(run_root=str(tmp_path / "durable"), temporary_root=str(tmp_path / "temporary"),
                           source_state=str(tmp_path / "source"))
    template = yaml.safe_load((bundle / "train/pulse_reference_fixed_template.yaml").read_text())
    smoke = make_jobs(config, bundle, template, phase="smoke", recipe=None)
    assert set(smoke) == {"smoke_ningbo_single", "smoke_ningbo_three", "smoke_ningbo_resume", "smoke_ningbo_eval"}
    for arm in ("single", "three", "resume"):
        child = yaml.safe_load(smoke[f"smoke_ningbo_{arm}"]["plan"].entry_config_path.read_text())
        assert child["mode"] == "smoke" and child["training"]["optimizer_steps"] == 2
        assert child["training"]["effective_batch"] == 2 and child["training"]["save_every"] == 1
    jobs = make_jobs(config, bundle, template, phase="final", recipe=None)
    assert len(jobs) == 12
    trains = {name for name, job in jobs.items() if job["plan"].entrypoint_name == "train_ecg_image"}
    assert len(trains) == 8
    for name, job in jobs.items():
        child = yaml.safe_load(job["plan"].entry_config_path.read_text())
        if name.endswith("_eval"):
            assert set(job["dependencies"]) == trains
            assert child["model_arms"] == ["original", "single", "three"]
            assert child["records_per_center"] == 512 and child["image_severity"] == 3
            assert set(child["training_results"]) == {"single", "three"}
            selected = yaml.safe_load((job["bundle"] / child["references"]["training_config"]).read_text())
            assert selected["center"] == child["center"]
        else:
            assert child["training"] == template["training"]
            assert child["image_augmentation"] == template["image_augmentation"]
    for key, value in (("records_per_center", 128), ("image_severity", 5), ("arms", {"three": 3})):
        with pytest.raises(ValueError):
            contract.validate_config({**config, key: value})


def test_three_arm_metrics_exclude_trained_clean_control():
    rows, samples, conditions = fixture()
    for row in rows:
        del row["arms"]["clean"]
    arms = ("original", "single", "three")
    metrics = metrics_from_predictions(rows, samples, conditions, arms=arms)
    assert set(metrics["families"]) == set(arms)
    with pytest.raises(ValueError, match="arm identity"):
        metrics_from_predictions(rows, samples, conditions)


def test_c5_smoke_covers_four_centers_and_known_failure_records(tmp_path, monkeypatch):
    from util import pulse_hybrid_contract as contract
    from util.config_bundle import resolve_yaml_config_closure
    from util.evaluation.pulse_hybrid_development import select_c5_samples
    path = ROOT / "configs/train/pulse_full_lora32_pipeline.yaml"
    config, _ = contract.load_config(path, ROOT / "configs")
    monkeypatch.setattr(contract, "DATA", tmp_path)
    bundle = tmp_path / "parent/configs"
    for source in resolve_yaml_config_closure([path], config_root=ROOT / "configs"):
        destination = bundle / source.relative_to(ROOT / "configs")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    config["paths"].update(run_root=str(tmp_path / "durable"), temporary_root=str(tmp_path / "temporary"),
                           source_state=str(tmp_path / "source"))
    template = yaml.safe_load((bundle / "train/pulse_full_lora32_template.yaml").read_text())
    jobs = make_jobs(config, bundle, template, phase="smoke", recipe=None)
    assert len(jobs) == 16
    for center in config["centers"]:
        for arm in ("single", "three", "resume"):
            child = yaml.safe_load(jobs[f"smoke_{center}_{arm}"]["plan"].entry_config_path.read_text())
            assert child["center"] == center and child["training"]["optimizer_steps"] == 4
        child = yaml.safe_load(jobs[f"smoke_{center}_eval"]["plan"].entry_config_path.read_text())
        assert child["original_baseline_mode"] == "current_protocol_v1"
        assert child["admission_sample_keys"] == config["smoke_sample_keys"][center]
    rows = [{"sample_key": f"cpsc_2018:{i}", "label": i % 2} for i in range(281)]
    assert select_c5_samples(rows, 4, ["cpsc_2018:280"])[0] == rows[280]
    assert select_c5_samples(rows, "full") is rows
    for count, probes in (("full", ["cpsc_2018:280"]), (4, ["cpsc_2018:missing"])):
        with pytest.raises(ValueError):
            select_c5_samples(rows, count, probes)
    bad = copy.deepcopy(child); bad["original_baseline_mode"] = "ignore_parent"
    with pytest.raises(ValueError, match="explicit"):
        contract.validate_config(bad)


def test_four_center_summary_rejects_different_baseline_generation_environment(tmp_path, monkeypatch):
    from util import pulse_hybrid_contract as contract
    from util.evaluation.ecg_image_queue import atomic_json
    from util.evaluation.pulse_hybrid_development import summarize_four_centers
    from util.pn2021_artifact_contract import sha256_file
    from util.pulse_training_contract import CENTERS
    # Isolate cross-center matching; per-result receipts have separate tests.
    monkeypatch.setattr(contract, "validate_result", lambda *a: None)
    arms = ("original", "single", "three")
    rows, samples, conditions = fixture()
    for row in rows: row["arms"] = {a: row["arms"][a] for a in arms}
    metrics = metrics_from_predictions(rows, samples, conditions, arms=arms)
    receipt = {k: "same" for k in ("mode", "batch_size", "precision", "seed",
        "deterministic_algorithms", "generation_kwargs", "prompt_sha256",
        "model_asset_sha256", "environment", "historical_parent_role")}
    entries = {}
    for i, center in enumerate(CENTERS):
        directory = tmp_path / center
        atomic_json(directory / "original_baseline.json", {**receipt, "environment": "changed" if i else "same"})
        atomic_json(directory / "cohort.json", {"samples": samples, "conditions": conditions})
        atomic_json(directory / "batches/0000.json", rows)
        atomic_json(directory / "metrics.json", metrics)
        path = directory / "hybrid_result.json"
        atomic_json(path, {"details": {"phase": "final", "center": center, "model_arms": list(arms),
            "image_suite": "image_c5_gpu_v1"}, "files": {"batches/0000.json": "unused"}})
        entries[center] = {"path": str(path), "sha256": sha256_file(path)}
    with pytest.raises(ValueError, match="baseline generation contracts differ"):
        summarize_four_centers(entries, tmp_path / "summary", seed=5, arms=arms)


def test_failed_scheduler_cleans_siblings_and_clears_stale_active_state(tmp_path, monkeypatch):
    from util import pulse_hybrid_workflow as workflow
    from util.evaluation import ecg_image_elastic as resources
    state = tmp_path / "state"; (state / "launches").mkdir(parents=True)
    jobs = {n: {"plan": SimpleNamespace(run_dir=tmp_path / n)} for n in ("bad", "sibling")}
    for i, n in enumerate(jobs):
        (state / "launches" / f"{n}.json").write_text(json.dumps({"process": {"pid": i+100}, "gpu": i}))
    monkeypatch.setattr(workflow, "checked_completion", lambda job: None)
    monkeypatch.setattr(resources, "process_identity", lambda pid: {"pid": pid})
    from util import pulse_training_queue as queue
    class Child:
        def __init__(self, identity, run_dir): self.pid = identity["pid"]
        def poll(self): return 1 if self.pid == 100 else None
    monkeypatch.setattr(queue, "RecoveredManagedChild", Child)
    cleaned = []
    def cleanup(active, directory):
        cleaned.extend(active)
        return {"signals": [], "survivors": [], "skipped": {}}
    monkeypatch.setattr(workflow, "stop_failed_children", cleanup)
    with pytest.raises(RuntimeError, match="managed child failed"):
        workflow.run_jobs(jobs, search(), state, {})
    assert cleaned == ["bad", "sibling"]
    assert json.loads((state / "status.json").read_text())["active"] == {}


def test_failure_cleanup_stops_owned_session_and_refuses_stale_receipt(tmp_path):
    import os
    import subprocess
    import sys
    import time
    from util.pulse_hybrid_workflow import stop_failed_children
    from util.evaluation.ecg_image_elastic import process_identity
    state = tmp_path / "state"; (state / "launches").mkdir(parents=True)
    child_file = tmp_path / "child.pid"
    code = ("import subprocess,sys,time; from pathlib import Path; "
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
            "Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(60)")
    command = [sys.executable, "-c", code, str(child_file)]
    launcher = subprocess.Popen(command, start_new_session=True)
    members = []
    try:
        deadline = time.monotonic()+5
        while not child_file.exists() and time.monotonic() < deadline: time.sleep(.02)
        assert child_file.exists()
        child_pid = int(child_file.read_text())
        identity = process_identity(launcher.pid)
        members = [process_identity(child_pid), identity]
        receipt = state / "launches/owned.json"
        receipt.write_text(json.dumps({"command": command, "process": {**identity, "start_ticks": "stale"}}))
        active = {"owned": {"process": launcher, "gpu": None}}
        rejected = stop_failed_children(active, state, grace_seconds=0)
        assert not rejected["signals"] and launcher.poll() is None
        receipt.write_text(json.dumps({"command": command, "process": identity}))
        result = stop_failed_children(active, state, grace_seconds=1)
        assert {r["process"]["pid"] for r in result["signals"]} == {launcher.pid, child_pid}
        assert not result["survivors"] and launcher.poll() is not None
    finally:
        for identity in members:
            if identity is not None and process_identity(identity["pid"]) == identity:
                os.kill(identity["pid"], 9)
        launcher.wait(timeout=5)
