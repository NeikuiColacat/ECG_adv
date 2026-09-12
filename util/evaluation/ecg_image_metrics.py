"""Paired hard-label metrics; records, not corrupted views, are bootstrap units."""
from __future__ import annotations

from typing import Mapping

import numpy as np

CLASS_ORDER = ("CD", "HYP", "MI", "NORM", "STTC")
SCALARS = ("macro_f1", "micro_f1", "exact_match_accuracy", "hamming_loss")


def binary_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape or truth.ndim != 2 or truth.shape[1] != 5 or not len(truth):
        raise ValueError("expected matching nonempty (N,5) hard-label arrays")
    if not np.isin(truth, (0, 1)).all() or not np.isin(prediction, (0, 1)).all():
        raise ValueError("metrics require binary values, not probabilities")
    truth, prediction = truth.astype(bool), prediction.astype(bool)
    tp = (truth & prediction).sum(axis=0)
    positive = truth.sum(axis=0)
    predicted = prediction.sum(axis=0)
    denom = positive + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros(5, dtype=float), where=denom > 0)
    recall = np.divide(tp, positive, out=np.zeros(5, dtype=float), where=positive > 0)
    precision = np.divide(tp, predicted, out=np.zeros(5, dtype=float), where=predicted > 0)
    return {
        "n_records": len(truth), "macro_f1": float(f1.mean()),
        "micro_f1": float(2 * tp.sum() / denom.sum()) if denom.sum() else 0.0,
        "exact_match_accuracy": float(np.all(truth == prediction, axis=1).mean()),
        "hamming_loss": float(np.not_equal(truth, prediction).mean()),
        "per_class": {name: {"f1": float(f1[i]), "recall": float(recall[i]),
                             "precision": float(precision[i]), "positive_count": int(positive[i]),
                             "predicted_positive_count": int(predicted[i]), "true_positive_count": int(tp[i]),
                             "no_true_positives_available": int(positive[i]) == 0}
                      for i, name in enumerate(CLASS_ORDER)},
        "zero_division": 0,
    }


def paired_metrics(truth_by_center: Mapping[str, np.ndarray], prediction_by_center: Mapping[str, np.ndarray],
                   model_names: list[str], condition_ids: list[str]) -> dict:
    """Predictions per center are (models,records,conditions,classes)."""
    if condition_ids[0] != "clean" or len(condition_ids) != 21 or len(set(condition_ids)) != 21:
        raise ValueError("formal comparison requires the complete clean+20 grid")
    centers = list(truth_by_center)
    if list(prediction_by_center) != centers or len(centers) != 4:
        raise ValueError("four matched centers are required")
    output = {"models": {}, "condition_rows": [], "class_rows": [],
              "class_order": list(CLASS_ORDER), "condition_ids": condition_ids,
              "primary_aggregation": "macro_F1_per_view_then_equal_20_views_then_equal_four_centers",
              "macro_denominator": "fixed_five_classes_zero_division_0",
              "auroc_auprc": "not_reported_no_continuous_class_scores"}
    for model_index, model_name in enumerate(model_names):
        per_center = {}
        for center in centers:
            truth = truth_by_center[center]
            prediction = prediction_by_center[center][model_index]
            if prediction.shape != (len(truth), 21, 5):
                raise ValueError("incomplete or malformed prediction tensor")
            scores = [binary_metrics(truth, prediction[:, i]) for i in range(21)]
            clean = {k: scores[0][k] for k in SCALARS}
            corrupt = {k: float(np.mean([s[k] for s in scores[1:]])) for k in SCALARS}
            drop = {k: float(clean[k] - corrupt[k]) for k in SCALARS}
            flip = float(np.not_equal(prediction[:, 1:], prediction[:, :1]).mean())
            per_center[center] = {"n_records": len(truth), "clean": clean,
                                  "corruption_view_mean": corrupt, "clean_minus_corruption": drop,
                                  "label_bit_flip_rate": flip}
            for condition, score in zip(condition_ids, scores, strict=True):
                output["condition_rows"].append({"model": model_name, "center": center,
                                                 "condition": condition, "n_records": len(truth),
                                                 **{k: score[k] for k in SCALARS}})
            for label in CLASS_ORDER:
                clean_class = scores[0]["per_class"][label]
                for metric in ("f1", "recall", "precision"):
                    corrupted = float(np.mean([s["per_class"][label][metric] for s in scores[1:]]))
                    output["class_rows"].append({"model": model_name, "center": center, "class": label,
                                                 "metric": metric, "n_records": len(truth),
                                                 "positive_count": clean_class["positive_count"],
                                                 "clean": clean_class[metric], "corruption_view_mean": corrupted,
                                                 "drop_pp": 100 * (clean_class[metric] - corrupted)})
        primary_clean = {k: float(np.mean([per_center[c]["clean"][k] for c in centers])) for k in SCALARS}
        primary_corrupt = {k: float(np.mean([per_center[c]["corruption_view_mean"][k] for c in centers])) for k in SCALARS}
        pooled_truth = np.concatenate([truth_by_center[c] for c in centers])
        pooled_prediction = np.concatenate([prediction_by_center[c][model_index] for c in centers])
        pooled_clean = binary_metrics(pooled_truth, pooled_prediction[:, 0])
        pooled_corrupt = binary_metrics(np.repeat(pooled_truth, 20, axis=0), pooled_prediction[:, 1:].reshape(-1, 5))
        output["models"][model_name] = {
            "centers": per_center,
            "center_equal": {"clean": primary_clean, "corruption_view_mean": primary_corrupt,
                             "clean_minus_corruption_pp": {k: 100 * (primary_clean[k] - primary_corrupt[k]) for k in SCALARS}},
            "secondary_pooled_all_records_views": {"clean": pooled_clean, "corrupted": pooled_corrupt},
        }
    return output


def paired_bootstrap(truth_by_center: Mapping[str, np.ndarray], prediction_by_center: Mapping[str, np.ndarray],
                     model_names: list[str], *, repeats: int = 1000, seed: int = 20260907) -> dict:
    """Stratified paired record bootstrap; retain every view and model per draw."""
    if repeats < 100:
        raise ValueError("at least 100 bootstrap repeats are required")
    rng = np.random.default_rng(seed)
    draws_by_center = []
    for center, truth in truth_by_center.items():
        prediction = np.asarray(prediction_by_center[center], dtype=float)
        models, count, views, classes = prediction.shape
        if models != len(model_names) or (views, classes) != (21, 5) or count != len(truth):
            raise ValueError("bootstrap requires a complete paired prediction grid")
        weights = rng.multinomial(count, np.full(count, 1 / count), size=repeats).astype(float)
        support = weights @ np.asarray(truth, dtype=float)
        true_positive = prediction * np.asarray(truth)[None, :, None, :]
        tp = (weights @ true_positive.transpose(1, 0, 2, 3).reshape(count, -1)).reshape(repeats, models, views, classes)
        predicted = (weights @ prediction.transpose(1, 0, 2, 3).reshape(count, -1)).reshape(repeats, models, views, classes)
        denom = predicted + support[:, None, None, :]
        f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0).mean(axis=-1)
        draws_by_center.append(f1)
    center_equal = np.mean(draws_by_center, axis=0)
    clean = center_equal[:, :, 0]
    corrupted = center_equal[:, :, 1:].mean(axis=-1)
    drop = (clean - corrupted) * 100
    interval = lambda x: [float(v) for v in np.quantile(x, (0.025, 0.975))]
    result = {"method": "percentile_stratified_paired_record_bootstrap", "unit": "record_not_view",
              "patient_cluster_adjustment": "unavailable_not_claimed", "repeats": repeats, "seed": seed,
              "conditioning": "fixed_cohort_sampling_frame_fixed_20_corruption_conditions_and_seeds",
              "models": {name: {"clean_macro_f1_ci95": interval(clean[:, i]),
                                 "corrupted_macro_f1_ci95": interval(corrupted[:, i]),
                                 "drop_pp_ci95": interval(drop[:, i])} for i, name in enumerate(model_names)},
              "paired_vs_first_model": {}}
    for i, name in enumerate(model_names[1:], 1):
        result["paired_vs_first_model"][name] = {
            "baseline": model_names[0], "clean_delta_pp_ci95": interval((clean[:, i] - clean[:, 0]) * 100),
            "corrupted_delta_pp_ci95": interval((corrupted[:, i] - corrupted[:, 0]) * 100),
            "extra_drop_pp_ci95": interval(drop[:, i] - drop[:, 0]),
        }
    result["centers"] = {}
    for center, draws in zip(truth_by_center, draws_by_center, strict=True):
        center_clean = draws[:, :, 0]
        center_corrupt = draws[:, :, 1:].mean(axis=-1)
        center_drop = (center_clean - center_corrupt) * 100
        result["centers"][center] = {
            "models": {name: {
                "clean_macro_f1_ci95": interval(center_clean[:, i]),
                "corrupted_macro_f1_ci95": interval(center_corrupt[:, i]),
                "drop_pp_ci95": interval(center_drop[:, i]),
            } for i, name in enumerate(model_names)},
            "paired_vs_first_model": {name: {
                "baseline": model_names[0],
                "clean_delta_pp_ci95": interval((center_clean[:, i] - center_clean[:, 0]) * 100),
                "corrupted_delta_pp_ci95": interval((center_corrupt[:, i] - center_corrupt[:, 0]) * 100),
                "extra_drop_pp_ci95": interval(center_drop[:, i] - center_drop[:, 0]),
            } for i, name in enumerate(model_names[1:], 1)},
        }
    return result
