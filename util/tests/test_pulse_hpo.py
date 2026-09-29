"""CPU-only search/aggregation tests; synthetic values are not experiment results."""
from pathlib import Path

import pytest
import yaml

from core.pulse_hpo import SEARCH_KEYS, development_score, matched_trial_score, training_cell
from util.pulse_training_contract import CENTERS

ROOT = Path(__file__).resolve().parents[2]
SPACE = yaml.safe_load((ROOT / "configs/train/pulse_hybrid_search.yaml").read_text())["search_space"]
RECIPE = dict(learning_rate=1e-5, projector_learning_rate=1e-5,
              jsd_weight=3.0, waveform_strength=0.35, image_strength=0.5)


def test_trial_arms_share_all_scientific_values_except_width():
    template = yaml.safe_load((ROOT / "configs/train/pulse_hybrid_smoke.yaml").read_text())
    before = yaml.safe_dump(template)
    cells = [training_cell(template, center="ningbo", width=w, recipe=RECIPE, steps=25, space=SPACE) for w in (0, 1, 3)]
    assert yaml.safe_dump(template) == before
    for cell in cells:
        cell.pop("width")
        assert cell == cells[0]
    assert "width" not in SEARCH_KEYS


def test_objective_weights_centers_and_families_not_number_of_views():
    centers = {c: dict(clean=0.8, waveform=0.4, image=0.6) for c in CENTERS}
    assert development_score(centers) == pytest.approx(0.65)
    objective, arms = matched_trial_score({a: centers for a in ("clean", "single", "three")})
    assert objective == pytest.approx(0.65) and len(arms) == 3
    centers.pop("georgia")
    with pytest.raises(ValueError, match="four centers"):
        development_score(centers)


def test_nonfinite_and_out_of_space_feedback_is_rejected():
    centers = {c: dict(clean=0.8, waveform=0.4, image=0.6) for c in CENTERS}
    centers["ningbo"]["clean"] = float("nan")
    with pytest.raises(ValueError):
        development_score(centers)
    template = yaml.safe_load((ROOT / "configs/train/pulse_hybrid_smoke.yaml").read_text())
    with pytest.raises(ValueError):
        training_cell(template, center="ningbo", width=1, recipe={**RECIPE, "jsd_weight": 99}, steps=25, space=SPACE)


def test_optuna_cpu_ask_tell_and_storage(tmp_path):
    from importlib.util import find_spec
    import json
    import subprocess
    import sys

    if find_spec("optuna") is None:
        pytest.skip("Optuna is installed only in the coordinator environment")
    # Match the search entrypoint's fresh process and import order. Pytest
    # collection may already have loaded Torch's incompatible libstdc++.
    script = """import optuna
import json
import sys
from pathlib import Path
from core.pulse_hpo import SEARCH_KEYS, open_study, suggest_recipe
directory = Path(sys.argv[1])
study = open_study(directory, seed=20260914)
trial = study.ask()
recipe = suggest_recipe(trial, json.loads(sys.stdin.read()))
assert set(recipe) == SEARCH_KEYS
study.tell(trial, 0.5)
assert open_study(directory, seed=20260914).best_trial.number == trial.number
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)],
                            input=json.dumps(SPACE), cwd=ROOT, text=True,
                            capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
