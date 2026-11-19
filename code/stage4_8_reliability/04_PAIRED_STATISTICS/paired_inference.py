from __future__ import annotations

from typing import Mapping, Sequence
import hashlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
)

EPS = 1e-7
LOWER_IS_BETTER = {"brier", "log_loss", "ece"}
PRIMARY_PAIRED_METRICS = ["accuracy", "balanced_accuracy", "f1_macro", "roc_auc", "brier", "log_loss", "ece"]


def probability_columns(n_classes: int, prefix: str = "p_class_") -> list[str]:
    return [f"{prefix}{k}" for k in range(int(n_classes))]


def normalize_probabilities(probability: np.ndarray) -> np.ndarray:
    p = np.asarray(probability, dtype=float)
    if p.ndim != 2:
        raise ValueError("Probability array must be two-dimensional.")
    p = np.clip(p, EPS, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def temperature_scale(probability: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be a positive finite number.")
    p = normalize_probabilities(probability)
    logits = np.log(np.clip(p, EPS, 1.0)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(y_validation: Sequence[int], probability_validation: np.ndarray) -> float:
    y = np.asarray(y_validation, dtype=int)
    p = normalize_probabilities(probability_validation)
    if len(y) != len(p):
        raise ValueError("Validation labels and probabilities have different lengths.")
    n_classes = p.shape[1]
    if set(np.unique(y)) != set(range(n_classes)):
        raise ValueError("Every class must be represented in the temperature-fitting set.")

    def objective(log_temperature: float) -> float:
        calibrated = temperature_scale(p, np.exp(log_temperature))
        return float(log_loss(y, calibrated, labels=np.arange(n_classes)))

    result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    return float(np.exp(result.x))


def expected_calibration_error(y_true: Sequence[int], probability: np.ndarray, n_bins: int = 15) -> float:
    y = np.asarray(y_true, dtype=int)
    p = normalize_probabilities(probability)
    confidence = p.max(axis=1)
    prediction = p.argmax(axis=1)
    correct = (prediction == y).astype(float)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            selected = (confidence >= edges[i]) & (confidence <= edges[i + 1])
        else:
            selected = (confidence >= edges[i]) & (confidence < edges[i + 1])
        if selected.any():
            ece += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return float(ece)


def classification_metrics(y_true: Sequence[int], probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = normalize_probabilities(probability)
    n_classes = p.shape[1]
    if len(y) != len(p):
        raise ValueError("Labels and probabilities have different lengths.")
    prediction = p.argmax(axis=1)
    one_hot = np.eye(n_classes, dtype=float)[y]
    if n_classes == 2:
        roc_auc = roc_auc_score(y, p[:, 1])
        pr_auc = average_precision_score(one_hot[:, 1], p[:, 1])
    else:
        roc_auc = roc_auc_score(one_hot, p, multi_class="ovr", average="macro")
        pr_auc = average_precision_score(one_hot, p, average="macro")
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "f1_macro": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier": float(np.mean(np.sum((one_hot - p) ** 2, axis=1))),
        "log_loss": float(log_loss(y, p, labels=np.arange(n_classes))),
        "ece": expected_calibration_error(y, p),
    }


def stratified_bootstrap_indices(y_true: Sequence[int], rng: np.random.Generator) -> np.ndarray:
    """Resample participants within class, preserving the observed class counts."""
    y = np.asarray(y_true, dtype=int)
    pieces = []
    for label in np.unique(y):
        members = np.flatnonzero(y == label)
        pieces.append(rng.choice(members, size=len(members), replace=True))
    result = np.concatenate(pieces)
    rng.shuffle(result)
    return result.astype(int)


def metric_value(y_true: Sequence[int], probability: np.ndarray, metric: str) -> float:
    y = np.asarray(y_true, dtype=int)
    p = normalize_probabilities(probability)
    n_classes = p.shape[1]
    prediction = p.argmax(axis=1)
    one_hot = np.eye(n_classes, dtype=float)[y]
    if metric == "accuracy":
        return float(accuracy_score(y, prediction))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y, prediction))
    if metric == "f1_macro":
        return float(f1_score(y, prediction, average="macro", zero_division=0))
    if metric == "roc_auc":
        if n_classes == 2:
            return float(roc_auc_score(y, p[:, 1]))
        return float(roc_auc_score(one_hot, p, multi_class="ovr", average="macro"))
    if metric == "pr_auc":
        if n_classes == 2:
            return float(average_precision_score(one_hot[:, 1], p[:, 1]))
        return float(average_precision_score(one_hot, p, average="macro"))
    if metric == "brier":
        return float(np.mean(np.sum((one_hot - p) ** 2, axis=1)))
    if metric == "log_loss":
        return float(log_loss(y, p, labels=np.arange(n_classes)))
    if metric == "ece":
        return expected_calibration_error(y, p)
    raise KeyError(f"Unsupported metric: {metric}")


def stable_seed(*parts) -> int:
    text = "|".join(str(x) for x in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def _oriented_difference(metric_a: float, metric_b: float, metric: str) -> float:
    raw = float(metric_a - metric_b)
    return -raw if metric in LOWER_IS_BETTER else raw


def paired_bootstrap_difference(
    y_true: Sequence[int],
    probabilities_a_by_seed: Sequence[np.ndarray],
    probabilities_b_by_seed: Sequence[np.ndarray],
    metric: str = "roc_auc",
    n_boot: int = 2000,
    random_seed: int = 20261115,
) -> dict[str, float]:
    """Paired participant bootstrap of the mean metric difference across all seeds.

    The same resampled participant indices are used for both models and every seed.
    Positive differences always favor model A (for lower-is-better metrics the sign is reversed).
    """
    y = np.asarray(y_true, dtype=int)
    a_list = [normalize_probabilities(p) for p in probabilities_a_by_seed]
    b_list = [normalize_probabilities(p) for p in probabilities_b_by_seed]
    if len(a_list) == 0 or len(a_list) != len(b_list):
        raise ValueError("The two models must have the same non-zero number of seeds.")
    for a, b in zip(a_list, b_list):
        if a.shape != b.shape or a.shape[0] != len(y):
            raise ValueError("Paired model predictions are not shape-aligned.")
    if metric not in classification_metrics(y, a_list[0]):
        raise KeyError(f"Unsupported metric: {metric}")

    observed_seed_diffs = [
        _oriented_difference(metric_value(y, a, metric), metric_value(y, b, metric), metric)
        for a, b in zip(a_list, b_list)
    ]
    observed = float(np.mean(observed_seed_diffs))

    rng = np.random.default_rng(int(random_seed))
    bootstrap = np.empty(int(n_boot), dtype=float)
    for b_idx in range(int(n_boot)):
        sample_idx = stratified_bootstrap_indices(y, rng)
        diffs = [
            _oriented_difference(
                metric_value(y[sample_idx], a[sample_idx], metric),
                metric_value(y[sample_idx], b[sample_idx], metric),
                metric,
            )
            for a, b in zip(a_list, b_list)
        ]
        bootstrap[b_idx] = float(np.mean(diffs))

    low, high = np.quantile(bootstrap, [0.025, 0.975])
    p_value = min(
        1.0,
        2.0 * min(
            (np.count_nonzero(bootstrap <= 0) + 1) / (len(bootstrap) + 1),
            (np.count_nonzero(bootstrap >= 0) + 1) / (len(bootstrap) + 1),
        ),
    )
    return {
        "estimate_difference_A_favors_positive": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value_bootstrap_two_sided": float(p_value),
        "n_participants": int(len(y)),
        "n_seeds": int(len(a_list)),
        "n_boot": int(n_boot),
    }


def bootstrap_metric_ci(
    y_true: Sequence[int],
    probabilities_by_seed: Sequence[np.ndarray],
    metric: str = "roc_auc",
    n_boot: int = 2000,
    random_seed: int = 20261115,
) -> dict[str, float]:
    """Participant-bootstrap CI for the mean metric over the fixed seed set."""
    y = np.asarray(y_true, dtype=int)
    p_list = [normalize_probabilities(p) for p in probabilities_by_seed]
    if not p_list or any(p.shape[0] != len(y) for p in p_list):
        raise ValueError("Invalid seed prediction collection.")
    values = [metric_value(y, p, metric) for p in p_list]
    estimate = float(np.mean(values))
    seed_sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    rng = np.random.default_rng(int(random_seed))
    boot = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        idx = stratified_bootstrap_indices(y, rng)
        boot[i] = float(np.mean([
            metric_value(y[idx], p[idx], metric) for p in p_list
        ]))
    low, high = np.quantile(boot, [0.025, 0.975])
    return {
        "mean_over_seeds": estimate,
        "sd_over_seeds": seed_sd,
        "participant_bootstrap_ci_low": float(low),
        "participant_bootstrap_ci_high": float(high),
        "n_participants": int(len(y)),
        "n_seeds": int(len(p_list)),
        "n_boot": int(n_boot),
    }


def raps_nonconformity(
    probability: np.ndarray,
    y_true: Sequence[int],
    lam: float = 0.01,
    k_reg: int = 1,
) -> np.ndarray:
    p = normalize_probabilities(probability)
    y = np.asarray(y_true, dtype=int)
    if len(y) != len(p):
        raise ValueError("Labels and probabilities have different lengths.")
    sorted_idx = np.argsort(-p, axis=1, kind="stable")
    sorted_prob = np.take_along_axis(p, sorted_idx, axis=1)
    cumulative = np.cumsum(sorted_prob, axis=1)
    ranks = np.array([np.flatnonzero(sorted_idx[i] == y[i])[0] for i in range(len(y))])
    cumulative_true = cumulative[np.arange(len(y)), ranks]
    penalty = float(lam) * np.maximum(ranks - int(k_reg) + 1, 0)
    return cumulative_true + penalty


def finite_sample_conformal_threshold(calibration_scores: Sequence[float], alpha: float = 0.10) -> float:
    """Exact split-conformal order statistic; avoids interpolated np.quantile."""
    scores = np.asarray(calibration_scores, dtype=float)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("Calibration scores must be a non-empty one-dimensional array.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    k = int(np.ceil((len(scores) + 1) * (1.0 - alpha)))
    if k > len(scores):
        return float("inf")
    return float(np.partition(scores, k - 1)[k - 1])


def legacy_interpolated_conformal_threshold(
    calibration_scores: Sequence[float], alpha: float = 0.10
) -> float:
    """Reproduce the original notebook's interpolated np.quantile convention."""
    scores = np.asarray(calibration_scores, dtype=float)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("Calibration scores must be a non-empty one-dimensional array.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    level = min(float(np.ceil((len(scores) + 1) * (1.0 - alpha)) / len(scores)), 1.0)
    return float(np.quantile(scores, level))



def raps_prediction_sets(
    probability: np.ndarray,
    threshold: float,
    lam: float = 0.01,
    k_reg: int = 1,
    force_nonempty: bool = False,
) -> list[list[int]]:
    """Deterministic RAPS-lite sets with stable tie-breaking.

    `force_nonempty=False` is the primary conformal-style definition.  The
    original pipeline's argmax fallback can be requested explicitly as a
    compatibility sensitivity analysis; it is not silently applied.
    """
    p = normalize_probabilities(probability)
    if not np.isfinite(threshold):
        # An infinite finite-sample threshold legitimately yields the full set.
        if threshold != float("inf"):
            raise ValueError("RAPS threshold must be finite or +infinity.")
    if lam < 0 or int(k_reg) < 1:
        raise ValueError("RAPS requires lam >= 0 and k_reg >= 1.")
    sorted_idx = np.argsort(-p, axis=1, kind="stable")
    sorted_prob = np.take_along_axis(p, sorted_idx, axis=1)
    cumulative = np.cumsum(sorted_prob, axis=1)
    ranks = np.arange(p.shape[1], dtype=int)
    penalty = float(lam) * np.maximum(ranks - int(k_reg) + 1, 0)
    adjusted = cumulative + penalty[None, :]
    prediction_sets = []
    for i in range(len(p)):
        included = sorted_idx[i][adjusted[i] <= threshold].astype(int).tolist()
        if force_nonempty and not included:
            included = [int(sorted_idx[i, 0])]
        prediction_sets.append(sorted(included))
    return prediction_sets



def conformal_test_summary(
    y_calibration: Sequence[int],
    p_calibration: np.ndarray,
    y_test: Sequence[int],
    p_test: np.ndarray,
    alpha: float = 0.10,
    lam: float = 0.01,
    k_reg: int = 1,
    force_nonempty: bool = False,
    threshold_method: str = "finite_sample",
) -> tuple[dict, list[list[int]]]:
    """Empirical validation-calibrated RAPS-lite summary; no formal guarantee claimed."""
    y_cal = np.asarray(y_calibration, dtype=int)
    y = np.asarray(y_test, dtype=int)
    p_cal = normalize_probabilities(p_calibration)
    p = normalize_probabilities(p_test)
    if len(y_cal) != len(p_cal) or len(y) != len(p):
        raise ValueError("RAPS labels and probability rows must be aligned.")
    if p_cal.shape[1] != p.shape[1]:
        raise ValueError("Calibration and test class counts differ.")
    if np.any(y_cal < 0) or np.any(y_cal >= p_cal.shape[1]):
        raise ValueError("Calibration labels are outside the probability class range.")
    if np.any(y < 0) or np.any(y >= p.shape[1]):
        raise ValueError("Test labels are outside the probability class range.")

    scores = raps_nonconformity(p_cal, y_cal, lam=lam, k_reg=k_reg)
    if threshold_method == "finite_sample":
        threshold = finite_sample_conformal_threshold(scores, alpha=alpha)
    elif threshold_method == "legacy_interpolated":
        threshold = legacy_interpolated_conformal_threshold(scores, alpha=alpha)
    else:
        raise ValueError(f"Unknown conformal threshold method: {threshold_method}")

    raw_sets = raps_prediction_sets(p, threshold, lam=lam, k_reg=k_reg, force_nonempty=False)
    raw_empty = np.asarray([len(s) == 0 for s in raw_sets], dtype=bool)
    sets = (
        raps_prediction_sets(p, threshold, lam=lam, k_reg=k_reg, force_nonempty=True)
        if force_nonempty else raw_sets
    )
    sizes = np.asarray([len(s) for s in sets], dtype=int)
    covered = np.asarray([int(y[i] in sets[i]) for i in range(len(y))], dtype=int)
    n_classes = p.shape[1]
    singleton = sizes == 1
    multi_label = sizes > 1
    empty = sizes == 0
    full = sizes == n_classes
    abstention = sizes != 1

    result = {
        "nominal_coverage": float(1.0 - alpha),
        "empirical_coverage": float(covered.mean()),
        "mean_set_size": float(sizes.mean()),
        "singleton_rate": float(singleton.mean()),
        "multi_label_rate": float(multi_label.mean()),
        "non_singleton_rate": float(abstention.mean()),
        "abstention_rate": float(abstention.mean()),
        "full_set_rate": float(full.mean()),
        "empty_set_rate": float(empty.mean()),
        "raw_empty_set_rate_before_fallback": float(raw_empty.mean()),
        "forced_singleton_fallback_rate": float((raw_empty & bool(force_nonempty)).mean()),
        "threshold": float(threshold),
        "calibration_n": int(len(y_cal)),
        "test_n": int(len(y)),
        "alpha": float(alpha),
        "lambda_raps": float(lam),
        "k_reg": int(k_reg),
        "argmax_fallback": bool(force_nonempty),
        "threshold_method": str(threshold_method),
        "formal_guarantee_claimed": False,
        "guarantee_note": (
            "Empirical validation-calibrated RAPS-lite only. The validation "
            "partition was also used for early stopping/model selection and "
            "temperature fitting; no formal distribution-free coverage claim is made."
        ),
    }
    if not np.isclose(
        result["singleton_rate"] + result["multi_label_rate"] + result["empty_set_rate"], 1.0
    ):
        raise AssertionError("Singleton + multi-label + empty-set rates must sum to one.")
    if not np.isclose(
        result["abstention_rate"], result["multi_label_rate"] + result["empty_set_rate"]
    ):
        raise AssertionError("Abstention must equal multi-label plus empty-set rate.")
    for c in range(n_classes):
        mask = y == c
        result[f"class_{c}_coverage"] = float(covered[mask].mean()) if mask.any() else float("nan")
    return result, sets

def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def predict_tree_bundle(model_bundle, design: pd.DataFrame, n_classes: int) -> np.ndarray:
    """Apply the frozen Stage 4.5 train-only preprocessing and tree estimator."""
    from stage4_6_missingness_benchmark import _align_probability
    if isinstance(model_bundle, dict) and {"imputer", "scaler", "model", "feature_cols"}.issubset(model_bundle):
        feature_cols = list(model_bundle["feature_cols"])
        missing = sorted(set(feature_cols) - set(design.columns))
        if missing:
            raise ValueError(f"Tree design is missing frozen feature columns: {missing}")
        raw = design[feature_cols].apply(pd.to_numeric, errors="coerce")
        transformed = model_bundle["imputer"].transform(raw)
        transformed = model_bundle["scaler"].transform(transformed)
        estimator = model_bundle["model"]
        probability = estimator.predict_proba(transformed)
        classes = estimator.classes_
    else:
        probability = model_bundle.predict_proba(design)
        classes = model_bundle.classes_
    return normalize_probabilities(_align_probability(probability, classes, int(n_classes)))


def export_tree_validation_predictions(
    prepared: Mapping,
    tree_output_dir,
    seeds: Sequence[int],
    tasks: Sequence[str] | None = None,
    feature_set: str = "S0_verified_asof",
) -> pd.DataFrame:
    """Export validation probabilities for tree models; row-level output stays local."""
    from pathlib import Path
    from stage4_6_missingness_benchmark import (
        TREE_ALGORITHMS, load_tree_baseline, build_tree_design, _align_probability,
        _write_csv_once,
    )
    output_dir = Path(tree_output_dir)
    tasks = list(tasks) if tasks is not None else [
        "diagnosis_multiclass", "diagnosis_binary", "mci_conversion"
    ]
    rows = []
    for task in tasks:
        pack = prepared[(task, feature_set)]
        idx = pack.idx("validation", eligible_only=True)
        y = pack.frame.iloc[idx][pack.target].to_numpy(dtype=int)
        x = build_tree_design(pack, idx)
        for model_name in TREE_ALGORITHMS:
            for seed in seeds:
                model = load_tree_baseline(output_dir, task, model_name, int(seed))
                probability = predict_tree_bundle(model, x, pack.n_classes)
                for j, row_index in enumerate(idx):
                    record = {
                        "task": task, "model": model_name, "seed": int(seed),
                        "split": "validation", "row_index": int(row_index),
                        "y_true": int(y[j]),
                    }
                    for cls in range(pack.n_classes):
                        record[f"p_class_{cls}"] = float(probability[j, cls])
                    rows.append(record)
    frame = pd.DataFrame(rows)
    _write_csv_once(
        output_dir / "tree_baseline_validation_predictions_LOCAL_ONLY.csv", frame
    )
    return frame




def predict_neural_natural_missingness(
    model, pack, indices: Sequence[int], batch_size: int = 128
) -> np.ndarray:
    """Infer with the participant's naturally observed modality mask.

    Unlike Stage 4.6's controlled missingness benchmark, the S0 full-cohort
    inference must retain participants whose biomarker modality is naturally
    unavailable. No artificial modality removal is applied here.
    """
    import torch
    indices = np.asarray(indices, dtype=int)
    if indices.size == 0:
        raise ValueError("Cannot infer an empty participant index set.")
    inputs_np = {
        modality: np.asarray(pack.matrices[modality][indices], dtype=np.float32).copy()
        for modality in pack.modalities
    }
    masks_np = np.asarray(pack.masks[indices], dtype=np.float32).copy()
    if masks_np.ndim != 2 or masks_np.shape[0] != len(indices):
        raise ValueError("Natural modality mask has an invalid shape.")
    if not np.isfinite(masks_np).all() or np.any(masks_np.sum(axis=1) <= 0):
        raise ValueError("Natural modality mask is invalid or all-zero for a participant.")
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            stop = min(start + batch_size, len(indices))
            inputs = {
                modality: torch.as_tensor(values[start:stop], dtype=torch.float32, device=device)
                for modality, values in inputs_np.items()
            }
            mask = torch.as_tensor(masks_np[start:stop], dtype=torch.float32, device=device)
            logits, _aux = model(inputs, mask, return_aux=True)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return normalize_probabilities(np.concatenate(probabilities, axis=0))


def load_s0_prediction_bundle(
    prepared: Mapping,
    neural_run_dir,
    tree_output_dir,
    seeds: Sequence[int],
    feature_set: str = "S0_verified_asof",
) -> pd.DataFrame:
    """Recreate validation/test probabilities from frozen S0 checkpoints.

    The final Stage 4.5 manifest contains four neural families and three tree
    baselines. Every checkpoint is loaded once per task/model/seed; no model is
    retrained and no participant-level output leaves the local environment.
    """
    from pathlib import Path
    from stage4_6_missingness_benchmark import (
        TREE_ALGORITHMS, NEURAL_ALGORITHMS, load_tree_baseline,
        load_neural_checkpoint, build_tree_design, _align_probability,
        predict_neural_scenario,
    )
    neural_run_dir = Path(neural_run_dir)
    tree_output_dir = Path(tree_output_dir)
    tasks = ["diagnosis_multiclass", "diagnosis_binary", "mci_conversion"]
    expected_neural = ["EarlyFusion", "LateFusion", "AttentionFusion", "AUAF"]
    expected_tree = ["XGBoost", "CatBoost", "ExtraTrees"]
    if list(NEURAL_ALGORITHMS) != expected_neural:
        raise ValueError(f"Unexpected neural model inventory: {NEURAL_ALGORITHMS}")
    if list(TREE_ALGORITHMS) != expected_tree:
        raise ValueError(
            "Tree baseline inventory does not match the final Stage 4.5 manifest. "
            f"Expected {expected_tree}; found {TREE_ALGORITHMS}."
        )
    if not (tree_output_dir / "tree_models").is_dir():
        raise FileNotFoundError(
            f"Expected tree model folder at {tree_output_dir / 'tree_models'}. "
            "Set TREE_OUTPUT_DIR to the Stage 4.5 checkpoints folder."
        )

    rows = []
    for task in tasks:
        pack = prepared[(task, feature_set)]
        split_data = {}
        for split in ["validation", "test"]:
            idx = pack.idx(split, eligible_only=True)
            if len(idx) == 0:
                raise ValueError(f"No eligible participants in {task}/{split}.")
            y = pack.frame.iloc[idx][pack.target].to_numpy(dtype=int)
            split_data[split] = (idx, y)

        # Four neural models: load each immutable checkpoint once, infer both splits.
        for model_name in expected_neural:
            for seed in seeds:
                model = load_neural_checkpoint(
                    pack, neural_run_dir, task, model_name, int(seed)
                )
                for split, (idx, y) in split_data.items():
                    probability = predict_neural_natural_missingness(
                        model, pack, idx, batch_size=128
                    )
                    probability = normalize_probabilities(probability)
                    for j, row_index in enumerate(idx):
                        record = {
                            "task": task, "model": model_name, "seed": int(seed),
                            "split": split, "row_index": int(row_index),
                            "y_true": int(y[j]),
                        }
                        for cls in range(pack.n_classes):
                            record[f"p_class_{cls}"] = float(probability[j, cls])
                        rows.append(record)

        # Three tree baselines: load each final model once, infer both splits.
        for model_name in expected_tree:
            for seed in seeds:
                model = load_tree_baseline(
                    tree_output_dir, task, model_name, int(seed)
                )
                for split, (idx, y) in split_data.items():
                    x = build_tree_design(pack, idx)
                    probability = predict_tree_bundle(model, x, pack.n_classes)
                    for j, row_index in enumerate(idx):
                        record = {
                            "task": task, "model": model_name, "seed": int(seed),
                            "split": split, "row_index": int(row_index),
                            "y_true": int(y[j]),
                        }
                        for cls in range(pack.n_classes):
                            record[f"p_class_{cls}"] = float(probability[j, cls])
                        rows.append(record)

    combined = pd.DataFrame(rows)
    required = {"task", "model", "seed", "split", "row_index", "y_true"}
    missing = required - set(combined.columns)
    if missing:
        raise ValueError(f"Prediction bundle is missing required columns: {sorted(missing)}")

    # Fail closed: same participant rows and labels across all models/seeds.
    expected_models = expected_neural + expected_tree
    for task in tasks:
        n_classes = prepared[(task, feature_set)].n_classes
        expected_prob = probability_columns(n_classes)
        if not set(expected_prob).issubset(combined.columns):
            raise ValueError(f"{task}: missing probability columns {expected_prob}")
        for split in ["validation", "test"]:
            reference = None
            for model in expected_models:
                for seed in seeds:
                    sub = combined[
                        combined["task"].eq(task)
                        & combined["model"].eq(model)
                        & combined["seed"].astype(int).eq(int(seed))
                        & combined["split"].eq(split)
                    ].sort_values("row_index")
                    if sub.empty:
                        raise ValueError(f"Missing predictions: {task}/{model}/{seed}/{split}")
                    indices = sub["row_index"].astype(int).to_numpy()
                    labels = sub["y_true"].astype(int).to_numpy()
                    if reference is None:
                        reference = (indices, labels)
                    elif not (
                        np.array_equal(indices, reference[0])
                        and np.array_equal(labels, reference[1])
                    ):
                        raise ValueError(
                            f"Participant pairing failed for {task}/{model}/{seed}/{split}."
                        )
    return combined.sort_values(
        ["task", "model", "seed", "split", "row_index"]
    ).reset_index(drop=True)

def calibrate_s0_prediction_bundle(
    predictions: pd.DataFrame,
    prepared: Mapping,
    feature_set: str = "S0_verified_asof",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit one temperature per task/model/seed on validation and apply to test."""
    calibration_rows, metric_rows, calibrated_test_rows = [], [], []
    for (task, model, seed), group in predictions.groupby(["task", "model", "seed"]):
        pack = prepared[(task, feature_set)]
        columns = probability_columns(pack.n_classes)
        validation = group[group["split"].eq("validation")].sort_values("row_index")
        test = group[group["split"].eq("test")].sort_values("row_index")
        y_val = validation["y_true"].to_numpy(dtype=int)
        y_test = test["y_true"].to_numpy(dtype=int)
        p_val = normalize_probabilities(validation[columns].to_numpy(float))
        p_test = normalize_probabilities(test[columns].to_numpy(float))
        temperature = fit_temperature(y_val, p_val)
        p_test_cal = temperature_scale(p_test, temperature)

        calibration_rows.append({
            "task": task, "model": model, "seed": int(seed),
            "temperature": temperature,
            "validation_n": int(len(validation)), "test_n": int(len(test)),
        })
        for probability_type, probability in [
            ("pre_temperature", p_test), ("post_temperature", p_test_cal)
        ]:
            metric_rows.append({
                "task": task, "model": model, "seed": int(seed),
                "probability_type": probability_type,
                "test_n": int(len(test)),
                **classification_metrics(y_test, probability),
            })

        for j, (_, row) in enumerate(test.iterrows()):
            record = {
                "task": task, "model": model, "seed": int(seed),
                "row_index": int(row["row_index"]), "y_true": int(y_test[j]),
            }
            for cls in range(pack.n_classes):
                record[f"p_raw_class_{cls}"] = float(p_test[j, cls])
                record[f"p_cal_class_{cls}"] = float(p_test_cal[j, cls])
            calibrated_test_rows.append(record)

    return (
        pd.DataFrame(calibration_rows),
        pd.DataFrame(metric_rows),
        pd.DataFrame(calibrated_test_rows),
    )


def summarize_s0_metrics(
    predictions: pd.DataFrame,
    calibration_parameters: pd.DataFrame,
    calibrated_test: pd.DataFrame,
    prepared: Mapping,
    feature_set: str = "S0_verified_asof",
) -> pd.DataFrame:
    """Mean and SD across seeds for raw and temperature-scaled test metrics."""
    rows = []
    metric_names = [
        "accuracy", "balanced_accuracy", "f1_macro", "roc_auc", "pr_auc",
        "brier", "log_loss", "ece"
    ]
    for (task, model), group in calibrated_test.groupby(["task", "model"]):
        pack = prepared[(task, feature_set)]
        y_reference = group[group["seed"].eq(int(group["seed"].iloc[0]))].sort_values("row_index")["y_true"].to_numpy(int)
        for probability_type, prefix in [("pre_temperature", "p_raw_class_"), ("post_temperature", "p_cal_class_")]:
            per_seed = []
            for seed in sorted(group["seed"].astype(int).unique()):
                sub = group[group["seed"].astype(int).eq(seed)].sort_values("row_index")
                p = sub[probability_columns(pack.n_classes, prefix)].to_numpy(float)
                per_seed.append(classification_metrics(sub["y_true"].to_numpy(int), p))
            for metric in metric_names:
                vals = np.asarray([x[metric] for x in per_seed], dtype=float)
                rows.append({
                    "task": task, "model": model, "probability_type": probability_type,
                    "metric": metric, "n_seeds": int(len(vals)),
                    "mean": float(np.nanmean(vals)),
                    "sd_across_seeds": float(np.nanstd(vals, ddof=1)),
                    "test_n": int(len(y_reference)),
                })
    return pd.DataFrame(rows)



def paired_s0_tests(
    predictions: pd.DataFrame,
    calibrated_test: pd.DataFrame,
    prepared: Mapping,
    seeds: Sequence[int],
    n_boot: int = 2000,
    random_seed: int = 20261115,
    feature_set: str = "S0_verified_asof",
    tasks: Sequence[str] | None = None,
    comparators: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Paired AUAF-versus-all six final comparators across five initializations.

    The five seeds are model initializations on the same frozen outer split.
    Bootstrap resampling is over participants, stratified by outcome class;
    the same sampled participant indices are used for both models and all seeds.
    """
    neural_models = ["EarlyFusion", "LateFusion", "AttentionFusion"]
    tree_models = ["XGBoost", "CatBoost", "ExtraTrees"]
    comparator_list = list(comparators) if comparators is not None else neural_models + tree_models
    raw_metrics = ["accuracy", "balanced_accuracy", "f1_macro", "roc_auc"]
    calibrated_metrics = ["brier", "log_loss", "ece"]
    rows = []

    task_list = list(tasks) if tasks is not None else ["diagnosis_multiclass", "diagnosis_binary", "mci_conversion"]
    for task in task_list:
        pack = prepared[(task, feature_set)]
        n_classes = pack.n_classes
        raw_columns = probability_columns(n_classes)
        cal_columns = probability_columns(n_classes, "p_cal_class_")

        for comparator in comparator_list:
            for metric in raw_metrics + calibrated_metrics:
                a_list, b_list = [], []
                y_reference = None
                source = calibrated_test if metric in calibrated_metrics else predictions[
                    predictions["split"].eq("test")
                ]
                for seed in seeds:
                    a = source[
                        source["task"].eq(task) & source["model"].eq("AUAF")
                        & source["seed"].astype(int).eq(int(seed))
                    ].sort_values("row_index")
                    b = source[
                        source["task"].eq(task) & source["model"].eq(comparator)
                        & source["seed"].astype(int).eq(int(seed))
                    ].sort_values("row_index")
                    if a.empty or b.empty:
                        raise ValueError(f"Missing paired predictions: {task}/AUAF/{comparator}/{seed}")
                    if not (
                        np.array_equal(a["row_index"].to_numpy(int), b["row_index"].to_numpy(int))
                        and np.array_equal(a["y_true"].to_numpy(int), b["y_true"].to_numpy(int))
                    ):
                        raise ValueError(f"Pairing mismatch: {task}/AUAF/{comparator}/{seed}")
                    y = a["y_true"].to_numpy(int)
                    if y_reference is None:
                        y_reference = y
                    elif not np.array_equal(y_reference, y):
                        raise ValueError("Seed-specific test labels/order are inconsistent.")
                    cols = cal_columns if metric in calibrated_metrics else raw_columns
                    a_list.append(a[cols].to_numpy(float))
                    b_list.append(b[cols].to_numpy(float))

                result = paired_bootstrap_difference(
                    y_reference, a_list, b_list, metric=metric,
                    n_boot=n_boot,
                    random_seed=random_seed + stable_seed(task, comparator, metric) % 100000,
                )
                rows.append({
                    "task": task,
                    "proposed": "AUAF",
                    "comparator": comparator,
                    "metric": metric,
                    "probability_source": (
                        "post_temperature" if metric in calibrated_metrics else "pre_temperature"
                    ),
                    "effect_size_metric_difference": result[
                        "estimate_difference_A_favors_positive"
                    ],
                    "effect_direction": "positive_favors_AUAF",
                    **result,
                })

    table = pd.DataFrame(rows)
    table["q_value_BH_global"] = benjamini_hochberg(
        table["p_value_bootstrap_two_sided"].to_numpy(float)
    )
    return table



def conformal_s0_results(
    predictions: pd.DataFrame,
    calibration_parameters: pd.DataFrame,
    prepared: Mapping,
    alpha: float = 0.10,
    lam: float = 0.01,
    k_reg: int = 1,
    feature_set: str = "S0_verified_asof",
    force_nonempty: bool = False,
    threshold_method: str = "finite_sample",
    use_temperature: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute RAPS-lite results for each model/seed using raw or calibrated probabilities."""
    parameter_lookup = {
        (r.task, r.model, int(r.seed)): float(r.temperature)
        for r in calibration_parameters.itertuples(index=False)
    }
    summary_rows, prediction_rows = [], []
    for (task, model, seed), group in predictions.groupby(["task", "model", "seed"]):
        pack = prepared[(task, feature_set)]
        columns = probability_columns(pack.n_classes)
        validation = group[group["split"].eq("validation")].sort_values("row_index")
        test = group[group["split"].eq("test")].sort_values("row_index")
        if validation.empty or test.empty:
            raise ValueError(f"RAPS needs validation and test rows: {task}/{model}/{seed}")
        temperature = parameter_lookup[(task, model, int(seed))]
        p_val_raw = normalize_probabilities(validation[columns].to_numpy(float))
        p_test_raw = normalize_probabilities(test[columns].to_numpy(float))
        p_val = temperature_scale(p_val_raw, temperature) if use_temperature else p_val_raw
        p_test = temperature_scale(p_test_raw, temperature) if use_temperature else p_test_raw
        result, sets = conformal_test_summary(
            validation["y_true"].to_numpy(int), p_val,
            test["y_true"].to_numpy(int), p_test,
            alpha=alpha, lam=lam, k_reg=k_reg,
            force_nonempty=force_nonempty,
            threshold_method=threshold_method,
        )
        variant = (
            f"{'calibrated' if use_temperature else 'raw'}_"
            f"{threshold_method}_"
            f"{'argmax_fallback' if force_nonempty else 'no_fallback'}"
        )
        summary_rows.append({
            "task": task, "model": model, "seed": int(seed),
            "temperature_parameter_fitted_on_validation": temperature,
            "temperature_applied_to_RAPS_probabilities": bool(use_temperature),
            "probability_source": "temperature_scaled" if use_temperature else "raw",
            "raps_variant": variant,
            **result,
        })
        y_test = test["y_true"].to_numpy(int)
        for i, (_, row) in enumerate(test.iterrows()):
            prediction_rows.append({
                "task": task, "model": model, "seed": int(seed),
                "row_index": int(row["row_index"]), "y_true": int(y_test[i]),
                "prediction_set": ";".join(map(str, sets[i])),
                "set_size": int(len(sets[i])),
                "covered": bool(y_test[i] in sets[i]),
                "raps_variant": variant,
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(prediction_rows)

def bootstrap_conformal_intervals(
    y_true: Sequence[int],
    prediction_sets: Sequence[Sequence[int]],
    n_classes: int,
    n_boot: int = 2000,
    random_seed: int = 20261115,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    sets = [list(map(int, x)) for x in prediction_sets]
    if len(y) != len(sets):
        raise ValueError("Labels and prediction sets must have equal length.")
    sizes = np.asarray([len(x) for x in sets], dtype=int)
    covered = np.asarray([int(y[i] in sets[i]) for i in range(len(y))], dtype=float)
    singleton = (sizes == 1).astype(float)
    multi = (sizes > 1).astype(float)
    empty = (sizes == 0).astype(float)
    full = (sizes == int(n_classes)).astype(float)
    abstention = (sizes != 1).astype(float)
    metrics = {
        "coverage": covered,
        "mean_set_size": sizes.astype(float),
        "singleton_rate": singleton,
        "multi_label_rate": multi,
        "empty_set_rate": empty,
        "full_set_rate": full,
        "abstention_rate": abstention,
    }
    rng = np.random.default_rng(int(random_seed))
    boot = {name: np.empty(int(n_boot), dtype=float) for name in metrics}
    for b in range(int(n_boot)):
        idx = stratified_bootstrap_indices(y, rng)
        for name, values in metrics.items():
            boot[name][b] = float(values[idx].mean())
    result = {}
    for name, values in metrics.items():
        result[f"{name}_ci_low"], result[f"{name}_ci_high"] = map(
            float, np.quantile(boot[name], [0.025, 0.975])
        )
    return result


def bootstrap_conformal_intervals_across_seeds(
    y_true: Sequence[int],
    prediction_sets_by_seed: Sequence[Sequence[Sequence[int]]],
    n_classes: int,
    n_boot: int = 2000,
    random_seed: int = 20261115,
) -> dict[str, float]:
    """Participant bootstrap of the mean conformal metric across fixed seeds.

    Each bootstrap draw samples participants once and applies the same sampled
    indices to every seed. Seed-to-seed SD is reported separately; seeds are not
    treated as independent cohorts or resampled as participants.
    """
    y = np.asarray(y_true, dtype=int)
    sets_by_seed = [
        [list(map(int, x)) for x in seed_sets]
        for seed_sets in prediction_sets_by_seed
    ]
    if not sets_by_seed or any(len(x) != len(y) for x in sets_by_seed):
        raise ValueError("Every seed must have prediction sets aligned to y_true.")
    names = [
        "coverage", "mean_set_size", "singleton_rate", "multi_label_rate",
        "empty_set_rate", "full_set_rate", "abstention_rate",
    ]
    per_seed = {name: [] for name in names}
    per_seed_values = []
    for seed_sets in sets_by_seed:
        sizes = np.asarray([len(x) for x in seed_sets], dtype=int)
        covered = np.asarray([int(y[i] in seed_sets[i]) for i in range(len(y))], dtype=float)
        values = {
            "coverage": covered,
            "mean_set_size": sizes.astype(float),
            "singleton_rate": (sizes == 1).astype(float),
            "multi_label_rate": (sizes > 1).astype(float),
            "empty_set_rate": (sizes == 0).astype(float),
            "full_set_rate": (sizes == int(n_classes)).astype(float),
            "abstention_rate": (sizes != 1).astype(float),
        }
        per_seed_values.append(values)
        for name in names:
            per_seed[name].append(float(values[name].mean()))

    rng = np.random.default_rng(int(random_seed))
    boot = {name: np.empty(int(n_boot), dtype=float) for name in names}
    for b in range(int(n_boot)):
        idx = stratified_bootstrap_indices(y, rng)
        for name in names:
            seed_means = [float(values[name][idx].mean()) for values in per_seed_values]
            boot[name][b] = float(np.mean(seed_means))

    result = {
        "n_participants": int(len(y)),
        "n_seeds": int(len(sets_by_seed)),
        "n_boot": int(n_boot),
    }
    for name in names:
        result[f"{name}_mean_over_seeds"] = float(np.mean(per_seed[name]))
        result[f"{name}_sd_across_seeds"] = (
            float(np.std(per_seed[name], ddof=1)) if len(per_seed[name]) > 1 else 0.0
        )
        low, high = np.quantile(boot[name], [0.025, 0.975])
        result[f"{name}_participant_bootstrap_ci_low"] = float(low)
        result[f"{name}_participant_bootstrap_ci_high"] = float(high)
    return result

