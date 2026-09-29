"""CPU contracts for measured C5 batches and frozen random evaluation subsets."""
import copy
from pathlib import Path

import pytest
import yaml

from util.pulse_hybrid_contract import validate_config
from util.evaluation.pulse_hybrid_development import image_admission_indices, select_image_samples

ROOT = Path(__file__).resolve().parents[2]


def config(mode="evaluate"):
    path = ("configs/eval/pulse_full_lora32_timing_ningbo.yaml" if mode == "evaluate"
            else "configs/train/pulse_full_lora32_pipeline.yaml")
    c = yaml.safe_load((ROOT / path).read_text())
    # Exercise each phase independently of the active timing run's extra probes.
    c.pop("performance_smoke", None)
    c.update(inference_batch_size=2, sampling_method="seeded_hash_v1", records_per_center=16)
    return c


@pytest.mark.parametrize("batch", [1, 2, 3, 4])
@pytest.mark.parametrize("count", [16, 128, 1000])
def test_explicit_random_budget_and_batch_validate(batch, count):
    c = config(); c.update(inference_batch_size=batch, records_per_center=count, phase="final")
    validate_config(c)
    c = config("fixed"); c.update(inference_batch_size=batch, records_per_center=count)
    validate_config(c)


@pytest.mark.parametrize("batch", [True, 0, -1, 5, 2.0, None])
@pytest.mark.parametrize("mode", ["evaluate", "fixed"])
def test_invalid_batch_rejected(batch, mode):
    c = config(mode); c["inference_batch_size"] = batch
    with pytest.raises(ValueError):
        validate_config(c)


@pytest.mark.parametrize("count", [True, 1, 15, 16.0, 20000, "full"])
def test_random_budget_is_finite_typed_and_in_population(count):
    c = config(); c["records_per_center"] = count
    with pytest.raises(ValueError):
        validate_config(c)


@pytest.mark.parametrize("batch", [1, 2, 3, 4])
@pytest.mark.parametrize("count", [4, 16, 17])
def test_author_admission_includes_each_tail_shape(batch, count):
    seen, expected = set(), []
    for start in range(0, count, batch):
        size = min(batch, count-start)
        if size not in seen:
            expected.extend(range(start, start+size))
        seen.add(size)
    assert image_admission_indices(count, batch) == expected
    assert image_admission_indices(count, batch, every_batch=True) == list(range(count))


def test_random_subset_is_reproducible_nested_and_label_independent():
    rows = [{"sample_key": f"ningbo:{i}", "hash_id": str(i), "label": [i % 2]} for i in range(100)]
    before = copy.deepcopy(rows)
    chosen = select_image_samples(rows, 16, sampling_seed=20260924)
    assert chosen == select_image_samples(list(reversed(rows)), 16, sampling_seed=20260924)
    assert chosen == select_image_samples(rows, 32, sampling_seed=20260924)[:16]
    assert chosen != select_image_samples(rows, 16, sampling_seed=20260925)
    altered = [{**r, "label": [0]} for r in rows]
    assert [r["hash_id"] for r in chosen] == [r["hash_id"] for r in select_image_samples(altered, 16, sampling_seed=20260924)]
    assert len({r["hash_id"] for r in chosen}) == 16
    assert rows == before


def test_final_weight_screen_can_compare_all_sixteen_records():
    c = config(); c["performance_smoke"] = True
    validate_config(c)
    c["phase"] = "final"
    with pytest.raises(ValueError):
        validate_config(c)


def test_existing_full_center_blocks_cannot_silently_change_batch_grouping():
    c = config(); c.pop("sampling_method")
    c.update(phase="block", records_per_center="full", record_span=[0, 512])
    with pytest.raises(ValueError, match="batch-one"):
        validate_config(c)
