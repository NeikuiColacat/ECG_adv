"""CPU block-contract and merger fixtures; these are not trained-model admissions."""
import ast
import copy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import shutil
import util.pulse_hybrid_workflow as workflow
from util.pn2021_artifact_contract import sha256_file

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def candidate():
    return SimpleNamespace(contract=importlib.import_module('util.pulse_hybrid_contract'),
        evaluation=importlib.import_module('util.evaluation.pulse_hybrid_development'),
        workflow=workflow)


@pytest.mark.parametrize('span', [None, [], [0], [False, 512], [0., 512], [0, 0],
    [-512, 0], [1, 513], [0, 511], [0, 513], [18944, 19456], {'start': 0, 'stop': 512}])
def test_record_span_rejects_changed_partition(candidate, span):
    with pytest.raises(ValueError):
        candidate.contract.validate_record_span('ningbo', span)


def test_block_plan_covers_all_39879_records_exactly(candidate):
    total = 0
    blocks = 0
    for center, count in candidate.evaluation.FULL_CENTER_RECORDS.items():
        spans = candidate.workflow.center_block_spans(center)
        assert [i for span in spans for i in range(*span)] == list(range(count))
        for span in spans:
            assert candidate.contract.validate_record_span(center, span) == tuple(span)
        total += count
        blocks += len(spans)
    assert total == 39879 and blocks == 80


def fixed_config():
    config = yaml.safe_load((REPO / 'configs/train/pulse_full_lora32_pipeline.yaml').read_text())
    config.pop('recovery', None)
    config['evaluation_block_size'] = 512
    return config


@pytest.mark.parametrize('phase', ['smoke', 'screen', 'final'])
@pytest.mark.parametrize('records', [4, 128, 512, 'full', 4., 128., 512., True, False, None, '128'])
def test_config_phase_budgets_match_result_contract(candidate, phase, records):
    config = yaml.safe_load((REPO / 'configs/eval/pulse_full_lora32_perf_smoke.yaml').read_text())
    config.pop('performance_smoke', None)
    config.pop('admission_sample_keys', None)
    config.update(phase=phase, records_per_center=records)
    allowed = ((type(records) is int and records in
                {'smoke': (4,), 'screen': (4, 128), 'final': (512,)}[phase])
               or (phase == 'final' and records == 'full'))
    if allowed:
        candidate.contract.validate_config(config)
    else:
        with pytest.raises(ValueError, match='development cohort'):
            candidate.contract.validate_config(config)


@pytest.mark.parametrize('suite', ['image_c5_gpu_v1', 'joint_c15_reference_v1'])
@pytest.mark.parametrize('records', [4, 128, 16, 4., True])
def test_small_screen_result_budget_is_typed_and_c5_only(candidate, tmp_path, suite, records):
    result = {'artifact_type': 'pulse_hybrid_result', 'schema_version': 1,
        'status': 'complete', 'mode': 'evaluate', 'files': {},
        'details': {'development_only': True, 'phase': 'screen', 'center': 'ningbo',
            'image_suite': suite, 'model_arms': ['original', 'single', 'three'],
            'records': records, 'conditions': 6 if suite == 'image_c5_gpu_v1' else 96}}
    allowed = type(records) is int and (records == 128 or records in (4, 16) and suite == 'image_c5_gpu_v1')
    # An admitted budget proceeds to the mandatory cohort read. This fixture
    # deliberately provides no artifacts and cannot establish model admission.
    error = FileNotFoundError if allowed else ValueError
    with pytest.raises(error):
        candidate.contract.validate_result(result, tmp_path / 'result.json')


def test_managed_dag_interleaves_blocks_and_preserves_all_training_dependencies(candidate, tmp_path, monkeypatch):
    config = fixed_config()
    config['paths']['run_root'] = str(tmp_path / 'workflow')
    template = yaml.safe_load((REPO / 'configs/train/pulse_full_lora32_template.yaml').read_text())
    def plan(path, **kwargs):
        value = yaml.safe_load(path.read_text())
        out = Path(value['output']['run_dir'])
        return SimpleNamespace(entrypoint_name=value['entrypoint']['name'], run_dir=out,
            delegate_output_dir=out / value['output']['delegate_output_subdir'])
    monkeypatch.setattr(candidate.workflow, 'load_experiment_plan', plan)
    jobs = candidate.workflow.make_jobs(config, REPO / 'configs', template, phase='final', recipe=None)
    training = [name for name, job in jobs.items() if job['plan'].entrypoint_name == 'train_ecg_image']
    evaluation = [(name, job) for name, job in jobs.items() if job['plan'].entrypoint_name == 'pulse_hybrid']
    assert len(jobs) == 88 and len(training) == 8 and len(evaluation) == 80
    assert [job['merge_config']['center'] for _, job in evaluation[:4]] == list(candidate.workflow.CENTERS)
    assert all(job['record_span'] == [0, 512] for _, job in evaluation[:4])
    for name, job in evaluation:
        assert job['dependencies'] == tuple(training)
        experiment = yaml.safe_load(job['experiment'].read_text())
        child = yaml.safe_load((job['bundle'] / experiment['entrypoint']['config']).read_text())
        candidate.contract.validate_config(child)
        assert child['phase'] == 'block' and child['records_per_center'] == 'full'
        assert child['seed'] == config['seed'] and child['model_arms'] == ['original', 'single', 'three']
        assert len(child['training_results']) == 2


def test_completed_whole_center_is_not_replaced_by_duplicate_blocks(candidate, tmp_path, monkeypatch):
    config = fixed_config()
    config['paths']['run_root'] = str(tmp_path / 'workflow')
    template = yaml.safe_load((REPO / 'configs/train/pulse_full_lora32_template.yaml').read_text())
    def plan(path, **kwargs):
        value = yaml.safe_load(path.read_text())
        out = Path(value['output']['run_dir'])
        return SimpleNamespace(entrypoint_name=value['entrypoint']['name'], run_dir=out,
            delegate_output_dir=out / value['output']['delegate_output_subdir'])
    monkeypatch.setattr(candidate.workflow, 'load_experiment_plan', plan)
    jobs = candidate.workflow.make_jobs(config, REPO / 'configs', template, phase='final', recipe=None,
        precompleted={'final_ningbo_eval': {'path': str(tmp_path / 'archive/ningbo_eval/hybrid_result.json')}})
    assert len(jobs) == 52 and 'final_ningbo_eval' in jobs
    assert not any(name.startswith('final_ningbo_eval_') for name in jobs)


def test_recovery_with_changed_evaluator_reaches_new_smoke_gate_before_final_jobs(candidate, tmp_path, monkeypatch):
    import util.pulse_training_queue as queue
    config = fixed_config()
    config.pop('storage', None)
    config['paths'].update(run_root=str(tmp_path / 'run'), temporary_root=str(tmp_path / 'tmp'))
    config['recovery'] = {'archive_root': '/data/linbinhao/ecg_llm_runs/pulse_full_lora32_fixture',
                          'archive_receipt_sha256': 'a' * 64}
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(queue, 'allowed_gpus', lambda: [2, 3, 4, 6])
    monkeypatch.setattr(candidate.workflow, 'prepare_fixed_recovery', lambda *args: ({}, {}, {}))
    calls = []
    class GateReached(Exception):
        pass
    def run_jobs(jobs, config, state, sources, **kwargs):
        calls.append((list(jobs), kwargs))
        raise GateReached('fresh admission reached')
    monkeypatch.setattr(candidate.workflow, 'run_jobs', run_jobs)
    refs = {'training_config': REPO / 'configs/train/pulse_full_lora32_template.yaml'}
    with pytest.raises(GateReached):
        candidate.workflow.run_fixed(config, refs, REPO / 'configs/train/pulse_full_lora32_pipeline.yaml',
                                     REPO / 'configs', tmp_path / 'run/workflow')
    assert len(calls) == 1 and calls[0][1]['smoke'] is True
    assert len(calls[0][0]) == 16 and all(name.startswith('smoke_') for name in calls[0][0])


@pytest.mark.parametrize('size', [True, 0, 128, 512., '512', 1024])
def test_block_config_rejects_implicit_or_changed_sizes(candidate, size):
    config = fixed_config()
    config['evaluation_block_size'] = size
    with pytest.raises(ValueError):
        candidate.contract.validate_config(config)


def test_block_recovery_still_requires_pinned_archive(candidate):
    config = fixed_config()
    config['recovery'] = {'archive_root': '/data/linbinhao/ecg_llm_runs/pulse_full_lora32_unverified'}
    with pytest.raises(ValueError, match='pinned'):
        candidate.contract.validate_config(config)


def seal_fixture_run(root, *, expected=None):
    """Use real managed file indexes around synthetic inference evidence."""
    from util.evaluation.ecg_image_queue import atomic_json
    from util.pn2021_artifact_contract import sha256_file
    from util.run_record import build_run_file_index, verify_run_file_index
    index = root / 'run_file_index.json'
    atomic_json(index, build_run_file_index(root))
    manifest = {'status': 'complete', 'run_file_index_sha256': sha256_file(index)}
    if expected is not None:
        manifest['expected_result'] = {'path': expected, 'type': 'pulse_hybrid_result'}
    atomic_json(root / 'run_manifest.json', manifest)
    assert verify_run_file_index(root) == []


@pytest.fixture
def block_fixture(candidate, tmp_path, monkeypatch):
    from core.pulse_finetune import MODEL_HASHES, MODEL_CONFIG_HASHES
    from util.evaluation.ecg_image_artifact import parse_response
    from util.evaluation.ecg_image_queue import atomic_json, digest_json
    from util.pn2021_artifact_contract import sha256_file
    import util.pulse_training_contract as training_contract

    # Model evidence is synthetic; managed file-index checks remain real.
    monkeypatch.setattr(training_contract, 'validate_result', lambda *args: None)
    monkeypatch.setattr(candidate.evaluation, 'validate_training', lambda *args: None)
    monkeypatch.setitem(candidate.evaluation.FULL_CENTER_RECORDS, 'ningbo', 513)
    config = yaml.safe_load((REPO / 'configs/eval/pulse_image_c5_gpu_ningbo.yaml').read_text())
    config.update(phase='final', records_per_center='full', execution_mode='arm_major_fp16_adapters_v2')
    config.pop('admission_sample_keys', None)
    config.pop('performance_smoke', None)
    config['training_results'] = {}
    for arm, width in [('single', 1), ('three', 3)]:
        out = tmp_path / arm / 'training'
        out.mkdir(parents=True)
        protocol = {'center': 'ningbo', 'width': width, 'implementation_sha256': {},
            'model_asset_sha256': {**MODEL_HASHES, **MODEL_CONFIG_HASHES},
            'augmentation_topology': 'image_only_gpu_branches_v1', 'mix_residual': 'clean_render',
            'image_gpu': {'implementation': 'augmix_torch_gpu_v2'}, 'image_augmentation': {'waveform_strength': 0}}
        atomic_json(out / 'k500_records.json', [{'hash_id': f'adapt-{i}', 'logical_center': 'ningbo'} for i in range(500)])
        atomic_json(out / 'train_result.json', {'status': 'complete', 'mode': 'train', 'protocol': protocol})
        seal_fixture_run(out.parent)
        config['training_results'][arm] = str(out / 'train_result.json')
    classes = candidate.evaluation.CLASS_ORDER
    samples = []
    for index in range(513):
        digest = hashlib.sha256(f'synthetic-record-{index}'.encode()).hexdigest()
        label = [int(i == index % 5) for i in range(5)]
        samples.append({'hash_id': digest, 'sample_key': 'ningbo:' + digest, 'logical_center': 'ningbo',
                        'label': label, 'label_names': [classes[index % 5]]})
    monkeypatch.setattr(candidate.evaluation, 'checked_full_parent', lambda *args: ({}, samples, []))
    conditions = candidate.evaluation.c5_gpu_conditions_for([{'condition_id': 'clean', 'operators': []}])
    ast_tree = ast.parse((REPO / 'util/pulse_hybrid_contract.py').read_text())
    inventory = next(ast.literal_eval(node.value) for node in ast.walk(ast_tree) if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'required_sources' for t in node.targets))
    entries = []
    for start, stop in [[0, 512], [512, 513]]:
        out = tmp_path / f'block-{start}'
        (out / 'batches').mkdir(parents=True)
        for name in inventory:
            path = out / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('synthetic source snapshot: ' + name)
        selected = samples[start:stop]
        rows = []
        for sample in selected:
            for condition in conditions:
                answers = {}
                for arm in ('original', 'single', 'three'):
                    predicted = sample['label_names'] if (arm == 'original' or start == 0 and arm == 'three') else ['NORM']
                    text = ';'.join(predicted)
                    answers[arm] = {'response': text, 'generated_tokens': 2, 'hit_max_new_tokens': False,
                                    **parse_response(text)}
                rows.append({**condition, 'sample_key': sample['sample_key'], 'hash_id': sample['hash_id'],
                    'true_labels': sample['label_names'], 'arms': answers,
                    'processor_input_hash_mode': candidate.evaluation.PROCESSOR_INPUT_HASH_MODE,
                    'processor_input_identity_sha256': candidate.evaluation.processor_input_identity(
                        seed=config['seed'], sample=sample, condition=condition,
                        image_suite=config['image_suite'], renderer_identity=digest_json({}))})
        metrics = candidate.evaluation.metrics_from_predictions(rows, selected, conditions, arms=tuple(config['model_arms']))
        baseline = {'mode': config['original_baseline_mode'], 'batch_size': 1, 'seed': config['seed'],
            'precision': 'float16', 'deterministic_algorithms': True,
            'generation_kwargs': {'do_sample': False, 'max_new_tokens': 32, 'pad_token_id': 0, 'use_cache': True},
            'model_asset_sha256': {**MODEL_HASHES, **MODEL_CONFIG_HASHES},
            'environment': {'torch': 'synthetic', 'cuda': 'synthetic', 'attention_implementation': 'sdpa'},
            'prompt_sha256': 'b' * 64, 'restore_admission': {'tokens_exact_after_adapter_switches': True},
            'prediction_count': len(rows), 'predictions_sha256': candidate.evaluation.original_prediction_digest(rows),
            'historical_parent_role': 'diagnostic_only_not_metric_baseline'}
        parity = {'mode': 'parent_response_audit', 'baseline_mode': config['original_baseline_mode'],
            'checked': len(selected), 'exact': len(selected), 'drift': 0, 'canonical_equal': 0,
            'canonical_drift': 0, 'canonical_unknown': 0, 'examples': []}
        members = {'batches/0000.json': rows, 'cohort.json': {'samples': selected, 'conditions': conditions,
            'k500_overlap': 0, 'renderer': {}, 'patient_independence_verified': False}, 'metrics.json': metrics,
            'generation_admission.json': {f'{a}_batch1': {'packing_exact': True, 'tokens_exact': True} for a in config['model_arms']},
            'original_parent_parity.json': parity, 'original_baseline.json': baseline,
            'runtime.json': {'seconds': len(selected), 'peak_allocated_bytes': 0},
            'performance_admission.json': {'execution_mode': config['execution_mode'], 'inference_trainable_dtype': 'float16',
                'reference_parameter_storage_restored': True, 'batch_size': 1, 'trainable_vision_outputs_cached': 0,
                'records': [{'sample_key': selected[0]['sample_key'], 'tokens_exact': True, 'views': 6, 'arms': 3}]}}
        for name, value in members.items():
            atomic_json(out / name, value)
        details = {'center': 'ningbo', 'phase': 'block', 'full_cohort': False, 'record_span': [start, stop],
            'parent_cohort_identity': digest_json(samples), 'records': len(selected), 'conditions': 6,
            'cohort_identity': digest_json(selected), 'families': metrics['families'],
            'training_result_paths': config['training_results'],
            'training_results': {arm: sha256_file(Path(path)) for arm, path in config['training_results'].items()},
            'development_only': True, 'image_suite': config['image_suite'], 'image_severity': 5,
            'image_implementation': 'image_c5_torch_gpu_v1', 'model_arms': config['model_arms'],
            'evaluation_seed': config['seed'], 'execution_mode': config['execution_mode'],
            'processor_input_hash_mode': candidate.evaluation.PROCESSOR_INPUT_HASH_MODE,
            'original_baseline_mode': config['original_baseline_mode'], 'original_parent_parity': parity,
            'original_baseline_sha256': sha256_file(out / 'original_baseline.json')}
        candidate.contract.finalize(out, mode='evaluate', details=details)
        path = out / 'hybrid_result.json'
        entries.append({'path': str(path), 'sha256': sha256_file(path)})
    return entries, config, samples, tmp_path / 'merged'


def refresh_entry(entry):
    from util.pn2021_artifact_contract import sha256_file
    path = Path(entry['path'])
    result = json.loads(path.read_text())
    result['files'] = {name: sha256_file(path.parent / name) for name in result['files']}
    path.write_text(json.dumps(result))
    entry['sha256'] = sha256_file(path)


def test_merger_recomputes_whole_population_metrics_and_validates_output(candidate, block_fixture):
    entries, config, samples, output = block_fixture
    result_entry = candidate.evaluation.merge_center_blocks(list(reversed(entries)), config, output)
    result = json.loads(Path(result_entry['path']).read_text())
    candidate.contract.validate_result(result, Path(result_entry['path']))
    assert result['details']['records'] == 513 and result['details']['full_cohort'] is True
    assert result['details']['phase'] == 'final' and result['details']['merged_blocks'] == 2
    assert 'record_span' not in result['details']
    block_scores = [json.loads(Path(e['path']).read_text())['details']['families']['three']['clean'] for e in entries]
    full_score = result['details']['families']['three']['clean']
    assert full_score > 0.99 and abs(full_score - sum(block_scores) / 2) > 0.1
    assert json.loads((output / 'cohort.json').read_text())['samples'] == samples


@pytest.mark.parametrize('change', ['missing', 'extra', 'duplicate', 'different_seed', 'different_parent', 'changed_result_hash', 'changed_snapshot', 'reordered_samples'])
def test_merger_rejects_bad_blocks_before_creating_output(candidate, block_fixture, change):
    from util.evaluation.ecg_image_queue import digest_json
    entries, config, samples, output = block_fixture
    if change == 'missing':
        entries.pop()
    elif change == 'extra':
        entries.append(copy.deepcopy(entries[0]))
    elif change == 'duplicate':
        entries[1] = copy.deepcopy(entries[0])
    elif change == 'changed_result_hash':
        entries[1]['sha256'] = '0' * 64
    else:
        entry = entries[0]
        path = Path(entry['path'])
        result = json.loads(path.read_text())
        if change == 'different_seed':
            result['details']['evaluation_seed'] += 1
        elif change == 'different_parent':
            result['details']['parent_cohort_identity'] = 'a' * 64
        elif change == 'changed_snapshot':
            (path.parent / 'source_snapshot/util/evaluation/pulse_adapters.py').write_text('different implementation')
        elif change == 'reordered_samples':
            cohort_path = path.parent / 'cohort.json'
            cohort = json.loads(cohort_path.read_text())
            cohort['samples'][1], cohort['samples'][2] = cohort['samples'][2], cohort['samples'][1]
            cohort_path.write_text(json.dumps(cohort))
            result['details']['cohort_identity'] = digest_json(cohort['samples'])
        path.write_text(json.dumps(result))
        refresh_entry(entry)
    with pytest.raises(ValueError):
        candidate.evaluation.merge_center_blocks(entries, config, output)
    assert not output.exists()


def test_final_artifact_rejects_changed_merged_rows_even_after_rehash(candidate, block_fixture):
    entries, config, samples, output = block_fixture
    final_entry = candidate.evaluation.merge_center_blocks(entries, config, output)
    batch = output / 'batches/00000.json'
    rows = json.loads(batch.read_text())
    rows[0], rows[1] = rows[1], rows[0]
    batch.write_text(json.dumps(rows))
    refresh_entry(final_entry)
    path = Path(final_entry['path'])
    with pytest.raises(ValueError, match='immutable evaluation blocks'):
        candidate.contract.validate_result(json.loads(path.read_text()), path)


def archive_block_fixture(candidate, block_fixture, monkeypatch):
    """Exercise the real copy, index, receipt and release path on small home fixtures."""
    import util.run_record as records
    import tempfile
    entries, config, samples, output = block_fixture
    final = candidate.evaluation.merge_center_blocks(entries, config, output)
    source = output.parent
    relative = Path(final['path']).relative_to(source)
    seal_fixture_run(source, expected=relative.as_posix())
    # Releasing tmp_path allows pytest to reuse its number in the next case.
    # Keep archive destinations in independently allocated owned directories.
    destination = Path(tempfile.mkdtemp(prefix='archive-', dir=source.parent)) / 'sealed'
    # Only the fixture namespace substitutes for RAM. No live run is touched.
    monkeypatch.setattr(records, 'RAM_ARCHIVE_SOURCE_ROOT', source.parent)
    receipt = records.archive_completed_run(source, destination, release_source=True, min_free_gib=0)
    assert receipt['status'] == 'verified' and receipt['ram_released'] is True
    assert not source.exists()
    return destination, destination / 'run' / relative, config


def test_full_block_merge_survives_real_archive_and_original_source_release(candidate, block_fixture, monkeypatch):
    from util.run_record import verify_run_file_index, archived_reference
    destination, path, config = archive_block_fixture(candidate, block_fixture, monkeypatch)
    assert verify_run_file_index(destination / 'run') == []
    result = json.loads(path.read_text())
    candidate.contract.validate_result(result, path)
    entries = json.loads((path.parent / 'block_results.json').read_text())
    assert all(archived_reference(entry['path'], path).is_relative_to(destination / 'run') for entry in entries)
    assert all(archived_reference(raw, path).is_relative_to(destination / 'run')
               for raw in result['details']['training_result_paths'].values())


@pytest.mark.parametrize('damage', ['missing_block', 'changed_block', 'unverified_receipt', 'missing_receipt'])
def test_archived_merge_rejects_missing_or_changed_dependencies(candidate, block_fixture, monkeypatch, damage):
    from util.run_record import archived_reference
    destination, path, config = archive_block_fixture(candidate, block_fixture, monkeypatch)
    receipt = destination / 'archive_receipt.json'
    reference = json.loads((path.parent / 'block_results.json').read_text())[0]
    block = archived_reference(reference['path'], path)
    if damage == 'missing_block':
        block.unlink()
    elif damage == 'changed_block':
        payload = json.loads(block.read_text())
        payload['details']['evaluation_seed'] += 1
        block.write_text(json.dumps(payload))
    elif damage == 'unverified_receipt':
        payload = json.loads(receipt.read_text())
        payload['status'] = 'pending'
        receipt.write_text(json.dumps(payload))
    else:
        receipt.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        candidate.contract.validate_result(json.loads(path.read_text()), path)


def test_real_archive_rejects_changed_index_and_preserves_original_source(candidate, block_fixture, monkeypatch):
    import util.run_record as records
    entries, config, samples, output = block_fixture
    final = candidate.evaluation.merge_center_blocks(entries, config, output)
    source = output.parent
    seal_fixture_run(source, expected=Path(final['path']).relative_to(source).as_posix())
    (source / 'unindexed_fixture.txt').write_text('simulate a write after sealing')
    destination = source.with_name(source.name + '_archive')
    monkeypatch.setattr(records, 'RAM_ARCHIVE_SOURCE_ROOT', source.parent)
    with pytest.raises(ValueError, match='integrity-checked'):
        records.archive_completed_run(source, destination, release_source=True, min_free_gib=0)
    assert source.exists() and not destination.exists()


def test_mixed_predecessor_and_current_blocks_survive_a_second_verified_archive(candidate, block_fixture, monkeypatch):
    import shutil
    import util.run_record as records
    from util.pn2021_artifact_contract import sha256_file
    first, first_result, config = archive_block_fixture(candidate, block_fixture, monkeypatch)
    predecessor = json.loads((first_result.parent / 'block_results.json').read_text())
    predecessor = [{**entry, 'path': str(records.archived_reference(entry['path'], first_result))}
                   for entry in predecessor]
    config = copy.deepcopy(config)
    config['training_results'] = {arm: str(records.archived_reference(raw, first_result))
                                 for arm, raw in config['training_results'].items()}
    current = first.with_name(first.name + '_next')
    current.mkdir()
    source_block = Path(predecessor[1]['path']).parent
    new_block = current / 'block-512'
    shutil.copytree(source_block, new_block)
    path = new_block / 'hybrid_result.json'
    payload = json.loads(path.read_text())
    payload['details']['training_result_paths'] = config['training_results']
    path.write_text(json.dumps(payload))
    entries = [predecessor[0], {'path': str(path), 'sha256': sha256_file(path)}]
    final = candidate.evaluation.merge_center_blocks(entries, config, current / 'merged')
    seal_fixture_run(current, expected='merged/hybrid_result.json')
    destination = current.with_name(current.name + '_archive')
    records.archive_completed_run(current, destination, release_source=True, min_free_gib=0)
    archived = destination / 'run/merged/hybrid_result.json'
    assert not current.exists() and records.verify_run_file_index(destination / 'run') == []
    candidate.contract.validate_result(json.loads(archived.read_text()), archived)
    # Altering a pinned predecessor must invalidate the later merged result.
    Path(predecessor[0]['path']).write_text('{}')
    with pytest.raises(ValueError, match='evaluation block result changed'):
        candidate.contract.validate_result(json.loads(archived.read_text()), archived)


def recovery_fixture(tmp_path, *, smoke_source):
    from util.pn2021_artifact_contract import sha256_file
    saved = archive(tmp_path / 'archives', 'source')
    parent = saved['receipt'].parent / 'run'
    evaluation_name = 'util/evaluation/pulse_hybrid_development.py'
    inference_name = 'util/evaluation/pulse_adapters.py'
    identity = {'sources': {**SOURCES, evaluation_name: smoke_source,
                            inference_name: 'current-inference'}}
    (parent / 'state/identity.json').write_text(json.dumps(identity))
    smoke_entries = json.loads((parent / 'workflow/smoke_results.json').read_text())
    for key, entry in saved['resolved_smokes'].items():
        if not key.endswith('_eval'):
            continue
        path = Path(entry['path'])
        value = json.loads(path.read_text())
        value['files'] = {'source_snapshot/' + evaluation_name: smoke_source,
                          'source_snapshot/' + inference_name: 'current-inference'}
        path.write_text(json.dumps(value))
        smoke_entries[key]['sha256'] = sha256_file(path)
    (parent / 'workflow/smoke_results.json').write_text(json.dumps(smoke_entries))
    refresh_fixture_receipt(saved)
    sources = {**CURRENT, evaluation_name: 'current-evaluator',
               inference_name: 'current-inference'}
    return saved, sources


@pytest.mark.parametrize('smoke_source,refresh', [('previous-evaluator', True), ('current-evaluator', False)])
def test_evaluator_change_forces_fresh_smokes_but_preserves_numerical_checkpoint(candidate, tmp_path, smoke_source, refresh):
    saved, sources = recovery_fixture(tmp_path, smoke_source=smoke_source)
    config = fixed_config()
    config['recovery'] = saved['pin']
    smoke, checkpoints, completed = candidate.workflow.prepare_fixed_recovery(config, template(), sources, tmp_path / 'fresh/state')
    assert len(smoke) == (0 if refresh else 16)
    assert completed == {} and Path(checkpoints['ningbo_single']).read_bytes() == saved['checkpoint'].read_bytes()
    admission = json.loads((tmp_path / 'fresh/state/recovery_admission.json').read_text())
    assert admission['fresh_smoke_required'] is refresh
    assert admission['training_sources_unchanged'] is True
    assert admission['scientific_sources_unchanged'] is (not refresh)


def test_block_transition_still_rejects_changed_training_code(candidate, tmp_path):
    saved, sources = recovery_fixture(tmp_path, smoke_source='previous-evaluator')
    config = fixed_config()
    config['recovery'] = saved['pin']
    sources['core/pulse_finetune.py'] = 'changed-training'
    with pytest.raises(ValueError, match='model/data/evaluation code'):
        candidate.workflow.prepare_fixed_recovery(config, template(), sources, tmp_path / 'fresh/state')


@pytest.mark.parametrize('change', ['none', 'different_training', 'different_evaluator',
                                  'different_adapter', 'different_span'])
def test_recovery_reuses_only_matching_committed_blocks(candidate, tmp_path, monkeypatch, change):
    import shutil
    from util.pn2021_artifact_contract import sha256_file
    saved, sources = recovery_fixture(tmp_path, smoke_source='current-evaluator')
    trained = {arm: completed_fixture(saved, 'ningbo', arm) for arm in ('single', 'three')}
    hashes = {arm: entry['sha256'] for arm, entry in trained.items()}
    full = completed_fixture(saved, 'ningbo', 'eval', training_hashes=hashes)
    source = Path(full['path']).parent.parent
    block_root = source.parent / 'ningbo_eval_blocks/00000_00512'
    shutil.copytree(source, block_root)
    # An incomplete whole-center leaf must not be confused with its complete block.
    (source / 'run_manifest.json').write_text(json.dumps({'status': 'failed'}))
    path = block_root / 'evaluation/hybrid_result.json'
    result = json.loads(path.read_text())
    result['details'].update(phase='block', full_cohort=False, record_span=[0, 512])
    result['files'] = {
        'source_snapshot/util/evaluation/pulse_hybrid_development.py': 'current-evaluator',
        'source_snapshot/util/evaluation/pulse_adapters.py': 'current-inference',
    }
    if change == 'different_training':
        result['details']['training_results']['single'] = 'other-training'
    elif change == 'different_evaluator':
        result['files']['source_snapshot/util/evaluation/pulse_hybrid_development.py'] = 'old-evaluator'
    elif change == 'different_adapter':
        result['files']['source_snapshot/util/evaluation/pulse_adapters.py'] = 'old-inference'
    elif change == 'different_span':
        result['details']['record_span'] = [512, 1024]
    path.write_text(json.dumps(result))
    refresh_fixture_receipt(saved)
    calls = []
    def checked(job):
        plan = job['plan']
        member = plan.run_dir / plan.expected_result_relative_path
        calls.append(str(member))
        return {'path': str(member), 'sha256': sha256_file(member), 'result': json.loads(member.read_text())}
    # This test isolates reuse decisions; the production block validator has separate fixture tests.
    monkeypatch.setattr(candidate.workflow, 'checked_completion', checked)
    config = fixed_config()
    config['recovery'] = saved['pin']
    if change == 'different_span':
        with pytest.raises(ValueError, match='block identity changed'):
            candidate.workflow.prepare_fixed_recovery(config, template(), sources, tmp_path / 'fresh/state')
        return
    smoke, checkpoints, completed = candidate.workflow.prepare_fixed_recovery(config, template(), sources, tmp_path / 'fresh/state')
    assert len(smoke) == 16 and not checkpoints
    name = 'final_ningbo_eval_00000_00512'
    assert (name in completed) is (change == 'none')
    assert {'final_ningbo_single', 'final_ningbo_three'} <= completed.keys()
    assert str(path) in calls

SOURCES = {"core/pulse_finetune.py": "numerical-identity",
           "util/pulse_hybrid_workflow.py": "old-orchestration"}
CURRENT = {**SOURCES, "util/pulse_hybrid_workflow.py": "candidate-orchestration"}

def base_config():
    value = yaml.safe_load((REPO / "configs/train/pulse_full_lora32_pipeline.yaml").read_text())
    value.pop("recovery", None)
    return value


def template():
    return yaml.safe_load((REPO / "configs/train/pulse_full_lora32_template.yaml").read_text())


def archive(root, name, predecessor=None, entries=None, seed=None, cycle=False):
    location = root / name
    parent = location / "run"
    files = {}
    config = base_config()
    source = root / ("original_" + name)
    config["paths"]["run_root"] = str(source)
    if predecessor is not None:
        config["recovery"] = copy.deepcopy(predecessor["pin"])
    if seed is not None:
        config["seed"] = seed
    if cycle:
        config["recovery"] = {"archive_root": str(location), "archive_receipt_sha256": "0" * 64}

    def put(relative, value):
        path = parent / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value))
        files[relative] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        return path

    put("configs/train/pulse_full_lora32_pipeline.yaml", yaml.safe_dump(config))
    put("configs/train/pulse_full_lora32_template.yaml", yaml.safe_dump(template()))
    put("state/identity.json", {"sources": SOURCES})
    if entries is None and predecessor is not None:
        entries = predecessor["resolved_smokes"]
    resolved = {}
    if entries is None:
        entries = {}
        for center in config["centers"]:
            for kind in ("single", "three", "resume", "eval"):
                relative = f"smoke/{center}_{kind}/training/result.json"
                path = put(relative, {"status": "complete"})
                key = f"smoke_{center}_{kind}"
                entries[key] = {"path": str(source / relative), "sha256": sha256_file(path)}
                resolved[key] = {"path": str(path), "sha256": sha256_file(path)}
                if kind == "resume":
                    put(str(Path(relative).parent / "recovery_audit.json"),
                        {"process_resume_verified": True, "resume_max_trainable_delta": 0.0})
    else:
        resolved = copy.deepcopy(entries)
    put("workflow/smoke_results.json", entries)
    checkpoint = put("final/ningbo_single/training/checkpoint.pt", "immutable checkpoint fixture " + name)
    put("final/ningbo_single/training/protocol.json",
        {"implementation_sha256": {"core/pulse_finetune.py": SOURCES["core/pulse_finetune.py"]}})
    receipt = location / "archive_receipt.json"
    receipt.write_text(json.dumps({"schema_version": 1, "status": "verified",
        "source_status": "failed", "source_run_root": str(source), "files": files}))
    return {"pin": {"archive_root": str(location), "archive_receipt_sha256": sha256_file(receipt)},
            "receipt": receipt, "checkpoint": checkpoint, "resolved_smokes": resolved}


def recover(tmp_path, saved, function=workflow.prepare_fixed_recovery, sources=None):
    config = base_config()
    config["recovery"] = saved["pin"]
    return function(config, template(), CURRENT if sources is None else sources, tmp_path / "fresh/state")


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_pinned_chain_preserves_smokes_and_latest_checkpoint(tmp_path, depth):
    saved = None
    for number in range(depth):
        saved = archive(tmp_path / "archives", f"archive_{number}", predecessor=saved)
    smokes, checkpoints, completed = recover(tmp_path, saved)
    assert len(smokes) == 16
    assert set(checkpoints) == {"ningbo_single"}
    assert completed == {}
    assert Path(checkpoints["ningbo_single"]).read_bytes() == saved["checkpoint"].read_bytes()
    assert all(Path(value["path"]).is_file() for value in smokes.values())
    admission = json.loads((tmp_path / "fresh/state/recovery_admission.json").read_text())
    assert len(admission["pinned_archive_chain"]) == depth




@pytest.mark.parametrize("target", ["receipt", "result", "resume_audit"])
def test_rejects_tampered_ancestor(tmp_path, target):
    first = archive(tmp_path / "archives", "first")
    second = archive(tmp_path / "archives", "second", predecessor=first)
    if target == "receipt":
        path = first["receipt"]
    else:
        key = "smoke_ningbo_resume" if target == "resume_audit" else "smoke_ningbo_single"
        path = Path(first["resolved_smokes"][key]["path"])
        if target == "resume_audit":
            path = path.parent / "recovery_audit.json"
    path.write_text("tampered")
    with pytest.raises(ValueError, match="changed"):
        recover(tmp_path, second)


def test_rejects_existing_but_unpinned_smoke_path(tmp_path):
    first = archive(tmp_path / "archives", "first")
    orphan = tmp_path / "orphan.json"
    orphan.write_text(json.dumps({"status": "complete"}))
    entries = copy.deepcopy(first["resolved_smokes"])
    entries["smoke_ningbo_single"] = {"path": str(orphan), "sha256": sha256_file(orphan)}
    second = archive(tmp_path / "archives", "second", predecessor=first, entries=entries)
    with pytest.raises(ValueError, match="outside the pinned archive chain"):
        recover(tmp_path, second)


def test_rejects_ancestor_scientific_drift(tmp_path):
    first = archive(tmp_path / "archives", "first", seed=base_config()["seed"] + 1)
    second = archive(tmp_path / "archives", "second", predecessor=first)
    with pytest.raises(ValueError, match="scientific workflow"):
        recover(tmp_path, second)


def test_rejects_model_source_drift(tmp_path):
    saved = archive(tmp_path / "archives", "first")
    with pytest.raises(ValueError, match="model/data/evaluation code"):
        recover(tmp_path, saved, sources={**CURRENT, "core/pulse_finetune.py": "drift"})


def test_rejects_archive_cycle(tmp_path):
    saved = archive(tmp_path / "archives", "cycle", cycle=True)
    with pytest.raises(ValueError, match="cycles"):
        recover(tmp_path, saved)


def test_rejects_archive_namespace_escape(tmp_path):
    first = archive(tmp_path / "other_archives", "first")
    second = archive(tmp_path / "archives", "second", predecessor=first)
    with pytest.raises(ValueError, match="escapes its namespace"):
        recover(tmp_path, second)


def refresh_fixture_receipt(saved):
    receipt = json.loads(saved["receipt"].read_text())
    parent = saved["receipt"].parent / "run"
    receipt["files"] = {p.relative_to(parent).as_posix():
                         {"sha256": sha256_file(p), "size_bytes": p.stat().st_size}
                         for p in parent.rglob("*") if p.is_file()}
    saved["receipt"].write_text(json.dumps(receipt))
    saved["pin"]["archive_receipt_sha256"] = sha256_file(saved["receipt"])


def completed_fixture(saved, center, kind, *, status="complete", training_hashes=None):
    """Small control-flow fixtures, not real weights or numerical validation."""
    root = saved["receipt"].parent / "run/final" / f"{center}_{kind}"
    settings = template()
    if kind in ("single", "three"):
        payload = {"status": "complete", "mode": "train",
                   "optimizer_steps": settings["training"]["optimizer_steps"],
                   "protocol": {"center": center, "width": {"single": 1, "three": 3}[kind],
                       "training": settings["training"], "model": settings["model"],
                       "training_scope": settings["training_scope"],
                       "image_augmentation": settings["image_augmentation"],
                       "implementation_sha256": {"core/pulse_finetune.py": SOURCES["core/pulse_finetune.py"]}}}
        relative = "training/train_result.json"
    else:
        config = base_config()
        payload = {"status": "complete", "mode": "evaluate", "details": {
            "center": center, "phase": "final", "image_suite": config["image_suite"],
            "image_severity": config["image_severity"], "evaluation_seed": config["seed"],
            "model_arms": ["original", "single", "three"],
            "full_cohort": config["records_per_center"] == "full",
            "training_results": training_hashes}}
        relative = "evaluation/hybrid_result.json"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    (root / "run_manifest.json").write_text(json.dumps({"status": status}))
    (root / "run_file_index.json").write_text(json.dumps({"fixture": True}))
    refresh_fixture_receipt(saved)
    return {"path": str(path), "sha256": sha256_file(path), "result": payload}


@pytest.fixture
def completed_validator(monkeypatch):
    """Isolate workflow behavior; the production validator is still delegated."""
    calls = []

    def checked(job):
        plan = job["plan"]
        path = plan.run_dir / plan.expected_result_relative_path
        calls.append((plan.entrypoint_name, str(path)))
        return {"path": str(path), "sha256": sha256_file(path),
                "result": json.loads(path.read_text())}

    monkeypatch.setitem(vars(workflow), "checked_completion", checked)
    return calls


def test_completed_training_is_reused_without_copying_terminal_checkpoint(tmp_path, completed_validator):
    saved = archive(tmp_path / "archives", "first")
    entry = completed_fixture(saved, "ningbo", "single")
    _, checkpoints, completed = recover(tmp_path, saved)
    assert checkpoints == {}
    assert completed == {"final_ningbo_single": entry}
    assert len(completed_validator) == 1
    assert not (tmp_path / "fresh/recovery/ningbo_single/checkpoint.pt").exists()


def test_completed_training_from_pinned_ancestor_is_reused(tmp_path, completed_validator):
    first = archive(tmp_path / "archives", "first")
    entry = completed_fixture(first, "ningbo", "single")
    second = archive(tmp_path / "archives", "second", predecessor=first)
    _, checkpoints, completed = recover(tmp_path, second)
    assert checkpoints == {}
    assert completed["final_ningbo_single"] == entry


def test_completed_evaluation_requires_exact_training_result_hashes(tmp_path, completed_validator):
    saved = archive(tmp_path / "archives", "first")
    training = {arm: completed_fixture(saved, "ningbo", arm) for arm in ("single", "three")}
    evaluation = completed_fixture(saved, "ningbo", "eval",
        training_hashes={arm: entry["sha256"] for arm, entry in training.items()})
    _, checkpoints, completed = recover(tmp_path, saved)
    assert checkpoints == {}
    assert len(completed) == 3
    assert completed["final_ningbo_eval"] == evaluation
    assert {kind for kind, _ in completed_validator} == {"train_ecg_image", "pulse_hybrid"}


def test_evaluation_from_different_training_is_not_reused(tmp_path, completed_validator):
    saved = archive(tmp_path / "archives", "first")
    for arm in ("single", "three"):
        completed_fixture(saved, "ningbo", arm)
    completed_fixture(saved, "ningbo", "eval", training_hashes={"single": "wrong", "three": "wrong"})
    _, _, completed = recover(tmp_path, saved)
    assert set(completed) == {"final_ningbo_single", "final_ningbo_three"}
    audit = json.loads((tmp_path / "fresh/state/recovery_admission.json").read_text())
    assert len(audit["skipped_evaluations"]) == 1


def test_failed_manifest_uses_checkpoint_instead_of_claiming_completion(tmp_path, completed_validator):
    saved = archive(tmp_path / "archives", "first")
    completed_fixture(saved, "ningbo", "single", status="failed")
    _, checkpoints, completed = recover(tmp_path, saved)
    assert completed == {} and set(checkpoints) == {"ningbo_single"}
    assert completed_validator == []


def test_completed_validation_failure_is_not_ignored(tmp_path, monkeypatch):
    saved = archive(tmp_path / "archives", "first")
    completed_fixture(saved, "ningbo", "single")

    def invalid(job):
        raise ValueError("completed child file index changed")

    monkeypatch.setitem(vars(workflow), "checked_completion", invalid)
    with pytest.raises(ValueError, match="file index changed"):
        recover(tmp_path, saved)


def test_wrong_completed_optimizer_budget_is_rejected(tmp_path, completed_validator):
    saved = archive(tmp_path / "archives", "first")
    entry = completed_fixture(saved, "ningbo", "single")
    entry["result"]["optimizer_steps"] -= 1
    Path(entry["path"]).write_text(json.dumps(entry["result"]))
    refresh_fixture_receipt(saved)
    with pytest.raises(ValueError, match="completed training identity changed"):
        recover(tmp_path, saved)


def test_scheduler_does_not_relaunch_reused_completed_job(tmp_path, monkeypatch):
    saved = archive(tmp_path / "archives", "first")
    entry = completed_fixture(saved, "ningbo", "single")
    name = "final_ningbo_single"
    jobs = {name: {"plan": SimpleNamespace(run_dir=tmp_path / "never_created")}}

    def forbidden(*args, **kwargs):
        raise AssertionError("a completed job must not launch or request resources")

    import util.evaluation.ecg_image_elastic as elastic
    monkeypatch.setattr(elastic, "resource_snapshot", forbidden)
    monkeypatch.setitem(vars(workflow), "checked_completion", forbidden)
    result = vars(workflow)["run_jobs"](jobs, {"mode": "fixed"}, tmp_path / "state", {},
                                   precompleted={name: entry})
    assert result == {name: entry}
    assert not jobs[name]["plan"].run_dir.exists()


def test_scheduler_rejects_unknown_recovery_job(tmp_path):
    with pytest.raises(ValueError, match="outside the managed job graph"):
        vars(workflow)["run_jobs"]({}, {"mode": "fixed"}, tmp_path / "state", {},
                              precompleted={"unknown": {}})


def test_generated_evaluation_points_to_reused_training_results(tmp_path, monkeypatch):
    from util.config_bundle import resolve_yaml_config_closure
    from util import pulse_hybrid_contract as contract
    path = REPO / "configs/train/pulse_full_lora32_pipeline.yaml"
    bundle = tmp_path / "parent/configs"
    for source in resolve_yaml_config_closure([path], config_root=REPO / "configs"):
        destination = bundle / source.relative_to(REPO / "configs")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    config = base_config()
    config["paths"].update(run_root=str(tmp_path / "durable"),
                           temporary_root=str(tmp_path / "temporary"), source_state=str(tmp_path / "source"))
    monkeypatch.setattr(contract, "DATA", tmp_path)
    reused = {"final_ningbo_single": {"path": str(tmp_path / "archive/run/final/ningbo_single/training/train_result.json")}}
    jobs = vars(workflow)["make_jobs"](config, bundle, template(), phase="final", recipe=None, precompleted=reused)
    assert len(jobs) == 12
    evaluation = yaml.safe_load(jobs["final_ningbo_eval"]["plan"].entry_config_path.read_text())
    assert evaluation["training_results"]["single"] == reused["final_ningbo_single"]["path"]
    assert set(jobs["final_ningbo_eval"]["dependencies"]) == {
        f"final_{center}_{arm}" for center in config["centers"] for arm in ("single", "three")}
