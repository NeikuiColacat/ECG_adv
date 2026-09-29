"""Finite Optuna recipe selection, separate from GPU scheduling and inference.

All arms receive the same proposed recipe. Width is never a search parameter.
The score is development feedback, not an unbiased external-test estimate.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

from util.pulse_training_contract import CENTERS, validate_config

SEARCH_KEYS = {"learning_rate", "projector_learning_rate", "jsd_weight", "waveform_strength", "image_strength"}


def open_study(directory: Path, *, seed: int):
    import optuna
    directory.mkdir(parents=True, exist_ok=True)
    return optuna.create_study(study_name="pulse_hybrid_development_v1",
        storage=f"sqlite:///{directory / 'optuna.sqlite3'}", load_if_exists=True,
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=4))


def suggest_recipe(trial, space):
    if set(space) != SEARCH_KEYS:
        raise ValueError("invalid finite search keys")
    return {
        key: (trial.suggest_float(key, *values, log=True) if key.endswith("learning_rate")
              else trial.suggest_categorical(key, values))
        for key, values in space.items()
    }


def training_cell(template, *, center, width, recipe, steps, space):
    """Scientific values belong to the resolved YAML, not CLI overrides."""
    if set(recipe) != SEARCH_KEYS or set(space) != SEARCH_KEYS:
        raise ValueError("incomplete finite search recipe")
    for key, value in recipe.items():
        choices = space[key]
        valid = choices[0] <= value <= choices[-1] if key.endswith("learning_rate") else value in choices
        if isinstance(value, bool) or not math.isfinite(value) or not valid:
            raise ValueError(f"trial parameter outside declared space: {key}")
    value = copy.deepcopy(template)
    value.update(mode="train", center=center, width=width, resume_from=None)
    value["training"].update(optimizer_steps=steps, warmup_steps=min(3, steps - 1),
        effective_batch=16, save_every=steps,
        learning_rate=recipe["learning_rate"], projector_learning_rate=recipe["projector_learning_rate"])
    value["image_augmentation"].update(jsd_weight=recipe["jsd_weight"],
        waveform_strength=recipe["waveform_strength"], strength=recipe["image_strength"])
    validate_config(value)
    return value


def development_score(centers):
    """Equal centers; clean receives half, waveform/image families one quarter each."""
    if set(centers) != set(CENTERS):
        raise ValueError("HPO requires all four centers, never a partial-center mean")
    scores = []
    for center in CENTERS:
        row = centers[center]
        if set(row) != {"clean", "waveform", "image"}:
            raise ValueError("missing development metric family")
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in row.values()):
            raise ValueError("invalid development macro-F1")
        scores.append(0.5 * row["clean"] + 0.25 * row["waveform"] + 0.25 * row["image"])
    return sum(scores) / len(scores)


def matched_trial_score(arms):
    if set(arms) != {"clean", "single", "three"}:
        raise ValueError("incomplete matched trial")
    # Symmetric selection: neither chain width gets its own favorable recipe.
    scores = {arm: development_score(metrics) for arm, metrics in arms.items()}
    return sum(scores.values()) / len(scores), scores
