from __future__ import annotations

import json
from typing import Mapping, Sequence, Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from paired_inference import (
    classification_metrics,
    paired_bootstrap_difference,
    benjamini_hochberg,
    stable_seed,
)


def _task_pack(prepared: Mapping, task: str, feature_set: str):
    return prepared[(task, feature_set)]


def _mc_predict(model, pack, indices, n_passes: int = 30, random_seed: int = 20261115):
    indices = np.asarray(indices, dtype=int)
    inputs = {
        modality: torch.as_tensor(
            pack.matrices[modality][indices], dtype=torch.float32, device="cpu"
        )
        for modality in pack.modalities
    }
    mask = torch.as_tensor(pack.masks[indices], dtype=torch.float32, device="cpu")
    model = model.to("cpu")
    model.eval()
    torch.manual_seed(int(random_seed))
    with torch.no_grad():
        out = model.mc_forward(inputs, mask, n_passes=int(n_passes))
    required = {"mean_prob", "std_prob", "entropy", "mean_gates"}
    missing = required - set(out)
    if missing:
        raise KeyError(f"AUAF mc_forward output missing keys: {sorted(missing)}")
    return {
        "mean_prob": np.asarray(out["mean_prob"], dtype=float),
        "std_prob": np.asarray(out["std_prob"], dtype=float),
        "entropy": np.asarray(out["entropy"], dtype=float).reshape(-1),
        "mean_gates": np.asarray(out["mean_gates"], dtype=float),
    }


def _bootstrap_mean_ci(values, n_boot=2000, random_seed=20261115):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(int(random_seed))
    draws = rng.integers(0, len(x), size=(int(n_boot), len(x)))
    means = x[draws].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"mean": float(x.mean()), "ci_low": float(lo), "ci_high": float(hi)}



def replace_auaf_with_mc_probabilities(
    prediction_bundle: pd.DataFrame,
    mc_predictions: pd.DataFrame,
    n_classes_by_task: Mapping[str, int],
) -> pd.DataFrame:
    """Replace only AUAF probabilities with the 30-pass MC mean, after strict pairing."""
    out = prediction_bundle.copy()
    for task, n_classes in n_classes_by_task.items():
        for seed in sorted(mc_predictions["seed"].astype(int).unique()):
            for split in ["validation", "test"]:
                mc = mc_predictions[
                    mc_predictions["task"].eq(task)
                    & mc_predictions["seed"].astype(int).eq(int(seed))
                    & mc_predictions["split"].eq(split)
                ].sort_values("row_index")
                target_mask = (
                    out["task"].eq(task) & out["model"].eq("AUAF")
                    & out["seed"].astype(int).eq(int(seed))
                    & out["split"].eq(split)
                )
                target = out.loc[target_mask].sort_values("row_index")
                if mc.empty or target.empty:
                    raise ValueError(f"Missing AUAF MC probabilities: {task}/{seed}/{split}")
                if not (
                    np.array_equal(mc["row_index"].to_numpy(int), target["row_index"].to_numpy(int))
                    and np.array_equal(mc["y_true"].to_numpy(int), target["y_true"].to_numpy(int))
                ):
                    raise ValueError(f"AUAF MC participant/label mismatch: {task}/{seed}/{split}")
                for cls in range(int(n_classes)):
                    out.loc[target.index, f"p_class_{cls}"] = mc[
                        f"p_class_{cls}"
                    ].to_numpy(float)
    return out.sort_values(
        ["task", "model", "seed", "split", "row_index"]
    ).reset_index(drop=True)


def run_auaf_mc_uncertainty_and_gates(
    prepared: Mapping,
    neural_run_dir,
    seeds: Sequence[int],
    mc_passes: int = 30,
    feature_set: str = "S0_verified_asof",
    n_boot: int = 2000,
    random_seed: int = 20261115,
):
    """Run 30-pass MC dropout for AUAF only; row-level output is local-only."""
    from stage4_6_missingness_benchmark import load_neural_checkpoint

    prediction_rows, summary_rows, selective_rows = [], [], []
    tasks = ["diagnosis_multiclass", "diagnosis_binary", "mci_conversion"]

    for task in task_list:
        pack = _task_pack(prepared, task, feature_set)
        for seed in seeds:
            model = load_neural_checkpoint(pack, neural_run_dir, task, "AUAF", int(seed))
            for split in ["validation", "test"]:
                idx = pack.idx(split, eligible_only=True)
                y = pack.frame.iloc[idx][pack.target].to_numpy(dtype=int)
                mc = _mc_predict(
                    model, pack, idx, n_passes=mc_passes,
                    random_seed=stable_seed(task, seed, split, "mc_dropout")
                )
                if len(y) != len(mc["mean_prob"]):
                    raise AssertionError(f"{task}/{seed}/{split}: MC rows and labels differ.")
                for j, row_index in enumerate(idx):
                    record = {
                        "task": task, "model": "AUAF", "seed": int(seed),
                        "split": split, "row_index": int(row_index), "y_true": int(y[j]),
                        "entropy": float(mc["entropy"][j]),
                        "mean_class_probability_sd": float(mc["std_prob"][j].mean()),
                        "predicted_class": int(mc["mean_prob"][j].argmax()),
                    }
                    for cls in range(pack.n_classes):
                        record[f"p_class_{cls}"] = float(mc["mean_prob"][j, cls])
                        record[f"std_class_{cls}"] = float(mc["std_prob"][j, cls])
                    for m_idx, modality in enumerate(pack.modalities):
                        record[f"gate_{modality}"] = float(mc["mean_gates"][j, m_idx])
                    prediction_rows.append(record)

                if split != "test":
                    continue

                pred = mc["mean_prob"].argmax(axis=1)
                correct = (pred == y).astype(float)
                entropy_ci = _bootstrap_mean_ci(
                    mc["entropy"], n_boot,
                    random_seed + stable_seed(task, seed, "entropy") % 100000,
                )
                sd_ci = _bootstrap_mean_ci(
                    mc["std_prob"].mean(axis=1), n_boot,
                    random_seed + stable_seed(task, seed, "prob_sd") % 100000,
                )
                err_ci = _bootstrap_mean_ci(
                    1.0 - correct, n_boot,
                    random_seed + stable_seed(task, seed, "error") % 100000,
                )
                summary = {
                    "task": task, "seed": int(seed), "n_test": int(len(idx)),
                    "mc_passes": int(mc_passes),
                    "accuracy": float(correct.mean()),
                    "mean_entropy": entropy_ci["mean"],
                    "mean_entropy_ci_low": entropy_ci["ci_low"],
                    "mean_entropy_ci_high": entropy_ci["ci_high"],
                    "mean_probability_sd": sd_ci["mean"],
                    "mean_probability_sd_ci_low": sd_ci["ci_low"],
                    "mean_probability_sd_ci_high": sd_ci["ci_high"],
                    "error_rate": err_ci["mean"],
                    "error_rate_ci_low": err_ci["ci_low"],
                    "error_rate_ci_high": err_ci["ci_high"],
                    "entropy_error_spearman": (
                        float(spearmanr(mc["entropy"], 1.0 - correct).statistic)
                        if np.std(mc["entropy"]) > 0 and np.std(1.0-correct) > 0
                        else float("nan")
                    ),
                }

                # Gate summaries are restricted to participants with both modalities
                # available; missing-modality zeros are not interpreted as preference.
                if len(pack.modalities) >= 2:
                    both = (pack.masks[idx] > 0).all(axis=1)
                else:
                    both = np.zeros(len(idx), dtype=bool)
                summary["gate_complete_case_n"] = int(both.sum())
                if both.any():
                    for m_idx, modality in enumerate(pack.modalities):
                        vals = mc["mean_gates"][both, m_idx]
                        ci = _bootstrap_mean_ci(
                            vals, n_boot,
                            random_seed + stable_seed(task, seed, modality, "gate") % 100000,
                        )
                        summary[f"gate_{modality}_mean_complete_case"] = ci["mean"]
                        summary[f"gate_{modality}_ci_low"] = ci["ci_low"]
                        summary[f"gate_{modality}_ci_high"] = ci["ci_high"]
                summary_rows.append(summary)

                # Selective prediction: retain lowest-entropy participants.
                order = np.argsort(mc["entropy"], kind="stable")
                for retention in np.round(np.arange(0.50, 1.001, 0.05), 2):
                    keep_n = max(1, int(np.ceil(float(retention) * len(idx))))
                    chosen = order[:keep_n]
                    yy = y[chosen]
                    pp = mc["mean_prob"][chosen]
                    counts = {
                        str(c): int((yy == c).sum()) for c in range(pack.n_classes)
                    }
                    selective_rows.append({
                        "task": task, "seed": int(seed),
                        "retention_fraction": float(retention),
                        "retained_n": int(keep_n),
                        "accuracy": float((pp.argmax(axis=1) == yy).mean()),
                        "macro_f1": float(
                            classification_metrics(yy, pp)["f1_macro"]
                        ),
                        "class_counts_retained": json.dumps(counts, sort_keys=True),
                    })

    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(summary_rows),
        pd.DataFrame(selective_rows),
    )


def run_auaf_modality_occlusion(
    prepared: Mapping,
    neural_run_dir,
    seeds: Sequence[int],
    n_boot: int = 2000,
    random_seed: int = 20261115,
    mc_passes: int = 30,
    feature_set: str = "S0_verified_asof",
    tasks: Sequence[str] | None = None,
):
    """Matched complete-case AUAF occlusion: both, clinical-only, biomarker-only."""
    from stage4_6_missingness_benchmark import load_neural_checkpoint

    task_list = list(tasks) if tasks is not None else ["diagnosis_multiclass", "diagnosis_binary", "mci_conversion"]
    row_records, metric_rows, test_rows = [], [], []
    scenarios = ["both_modalities", "clinical_only", "biomarker_only"]

    for task in task_list:
        pack = _task_pack(prepared, task, feature_set)
        modality_names = list(pack.modalities)
        if "clinical_cognitive" not in modality_names or "fluid_biomarker" not in modality_names:
            raise AssertionError(f"{task}: expected clinical_cognitive and fluid_biomarker.")
        c_idx = modality_names.index("clinical_cognitive")
        b_idx = modality_names.index("fluid_biomarker")
        idx = pack.paired_modality_idx("test")
        if len(idx) == 0:
            raise ValueError(f"{task}: no complete-case paired test participants.")
        y = pack.frame.iloc[idx][pack.target].to_numpy(dtype=int)
        seed_probabilities = {scenario: [] for scenario in scenarios}

        for seed in seeds:
            model = load_neural_checkpoint(pack, neural_run_dir, task, "AUAF", int(seed))
            base_inputs = {
                modality: torch.as_tensor(
                    pack.matrices[modality][idx], dtype=torch.float32, device="cpu"
                )
                for modality in modality_names
            }
            base_mask = np.asarray(pack.masks[idx], dtype=np.float32)
            for scenario in scenarios:
                inputs = {k: v.clone() for k, v in base_inputs.items()}
                mask = base_mask.copy()
                if scenario == "clinical_only":
                    mask[:, b_idx] = 0.0
                    inputs["fluid_biomarker"] = torch.zeros_like(inputs["fluid_biomarker"])
                elif scenario == "biomarker_only":
                    mask[:, c_idx] = 0.0
                    inputs["clinical_cognitive"] = torch.zeros_like(inputs["clinical_cognitive"])
                mask_t = torch.as_tensor(mask, dtype=torch.float32, device="cpu")
                model = model.to("cpu")
                model.eval()
                torch.manual_seed(stable_seed(task, seed, "occlusion_common_random_numbers"))
                with torch.no_grad():
                    out = model.mc_forward(inputs, mask_t, n_passes=int(mc_passes))
                p = np.asarray(out["mean_prob"], dtype=float)
                if p.shape != (len(idx), pack.n_classes):
                    raise AssertionError(f"{task}/{seed}/{scenario}: probability shape mismatch.")
                seed_probabilities[scenario].append(p)
                metrics = classification_metrics(y, p)
                metric_rows.append({
                    "task": task, "seed": int(seed), "scenario": scenario,
                    "n_matched_test": int(len(idx)), "mc_passes": int(mc_passes),
                    **metrics,
                })
                for j, row_index in enumerate(idx):
                    row_records.append({
                        "task": task, "seed": int(seed), "scenario": scenario,
                        "row_index": int(row_index), "y_true": int(y[j]),
                        **{f"p_class_{c}": float(p[j, c]) for c in range(pack.n_classes)},
                    })

        # Paired comparisons average the metric difference across the same five
        # initializations while resampling the same matched participants.
        for comparator_scenario in ["clinical_only", "biomarker_only"]:
            for metric in ["accuracy", "balanced_accuracy", "f1_macro", "roc_auc", "brier"]:
                result = paired_bootstrap_difference(
                    y,
                    seed_probabilities["both_modalities"],
                    seed_probabilities[comparator_scenario],
                    metric=metric,
                    n_boot=n_boot,
                    random_seed=random_seed + stable_seed(task, comparator_scenario, metric) % 100000,
                )
                test_rows.append({
                    "task": task,
                    "contrast": f"both_modalities_minus_{comparator_scenario}",
                    "metric": metric,
                    "effect_size_metric_difference": result["estimate_difference_A_favors_positive"],
                    "effect_direction": "positive_favors_both_modalities",
                    **result,
                })

    tests = pd.DataFrame(test_rows)
    if not tests.empty:
        tests["q_value_BH_global"] = benjamini_hochberg(
            tests["p_value_bootstrap_two_sided"].to_numpy(float)
        )
    return pd.DataFrame(metric_rows), tests, pd.DataFrame(row_records)


def run_catboost_shap(
    prepared: Mapping,
    tree_output_dir,
    seeds: Sequence[int],
    feature_set: str = "S0_verified_asof",
    tasks: Sequence[str] | None = None,
):
    """Global CatBoost TreeSHAP summaries; never attributed to AUAF."""
    import shap
    from stage4_6_missingness_benchmark import load_tree_baseline, build_tree_design

    rows = []
    task_list = list(tasks) if tasks is not None else ["diagnosis_multiclass", "diagnosis_binary", "mci_conversion"]
    for task in task_list:
        pack = _task_pack(prepared, task, feature_set)
        idx = pack.idx("test", eligible_only=True)
        x = build_tree_design(pack, idx)
        for seed in seeds:
            baseline = load_tree_baseline(
                tree_output_dir, task, "CatBoost", int(seed)
            )
            if isinstance(baseline, dict) and {"imputer", "scaler", "model", "feature_cols"}.issubset(baseline):
                frozen_cols = list(baseline["feature_cols"])
                raw = x[frozen_cols].apply(pd.to_numeric, errors="coerce")
                imputed = baseline["imputer"].transform(raw)
                transformed = baseline["scaler"].transform(imputed)
                estimator = baseline["model"]
                try:
                    feature_names = list(baseline["imputer"].get_feature_names_out(frozen_cols))
                except Exception:
                    indicator_features = list(getattr(getattr(baseline["imputer"], "indicator_", None), "features_", []))
                    feature_names = frozen_cols + [f"missingindicator_{frozen_cols[int(j)]}" for j in indicator_features]
            else:
                transformed = baseline._transform(x)
                estimator = baseline.estimator
                feature_names = list(x.columns)
            explainer = shap.TreeExplainer(estimator)
            values = explainer.shap_values(transformed)
            if isinstance(values, list):
                arr = np.stack([np.asarray(v) for v in values], axis=-1)
                importance = np.mean(np.abs(arr), axis=(0, 2))
            else:
                arr = np.asarray(values)
                if arr.ndim == 2:
                    importance = np.mean(np.abs(arr), axis=0)
                elif arr.ndim == 3 and arr.shape[0] == len(idx) and arr.shape[1] == len(feature_names):
                    importance = np.mean(np.abs(arr), axis=(0, 2))
                elif arr.ndim == 3 and arr.shape[1] == len(idx) and arr.shape[2] == len(feature_names):
                    importance = np.mean(np.abs(arr), axis=(0, 1))
                else:
                    raise ValueError(
                        f"Unexpected CatBoost SHAP shape for {task}/{seed}: {arr.shape}"
                    )
            if len(importance) != len(feature_names):
                raise AssertionError("SHAP feature count does not match model input columns.")
            for feature, value in zip(feature_names, importance):
                rows.append({
                    "task": task, "model": "CatBoost",
                    "seed": int(seed), "feature": feature,
                    "mean_abs_shap": float(value),
                    "interpretation_scope": "CatBoost feature attribution only; not AUAF and not causal",
                })
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["task", "feature"], as_index=False)["mean_abs_shap"]
        .agg(mean="mean", sd="std")
        .sort_values(["task", "mean"], ascending=[True, False])
    )
    summary["rank_within_task"] = summary.groupby("task")["mean"].rank(
        method="first", ascending=False
    ).astype(int)
    return frame, summary


def run_conversion_followup_selection_audit(
    all_mci_path,
    eligible_conversion_path,
    features: Sequence[str],
    expected_all_n: int = 1753,
    expected_eligible_n: int = 911,
    rid_column: str = "RID",
):
    """Descriptive included-vs-excluded baseline audit for follow-up eligibility.

    This does not infer outcomes for excluded participants and does not use
    their post-index information. It reports missingness and standardized
    mean differences for verified baseline predictors only.
    """
    all_mci = pd.read_csv(all_mci_path)
    eligible = pd.read_csv(eligible_conversion_path)
    for frame, label in [(all_mci, "all_mci"), (eligible, "eligible_conversion")]:
        if rid_column not in frame.columns:
            raise KeyError(f"{label}: missing participant key {rid_column}.")
        if frame[rid_column].isna().any() or frame[rid_column].duplicated().any():
            raise ValueError(f"{label}: RID must be nonmissing and unique.")
    if len(all_mci) != int(expected_all_n):
        raise ValueError(f"Expected {expected_all_n} first-MCI participants; found {len(all_mci)}.")
    if len(eligible) != int(expected_eligible_n):
        raise ValueError(f"Expected {expected_eligible_n} eligible participants; found {len(eligible)}.")
    all_ids = set(all_mci[rid_column].astype(str))
    eligible_ids = set(eligible[rid_column].astype(str))
    if not eligible_ids.issubset(all_ids):
        raise ValueError("Eligible conversion cohort contains RIDs absent from all-MCI file.")
    included = all_mci[rid_column].astype(str).isin(eligible_ids).to_numpy()
    if int(included.sum()) != expected_eligible_n:
        raise AssertionError("Included cohort count does not match the 911-person manifest.")
    if int((~included).sum()) != expected_all_n - expected_eligible_n:
        raise AssertionError("Excluded cohort count does not match the 842-person flow.")

    rows = []
    for feature in features:
        if feature not in all_mci.columns:
            rows.append({
                "feature": feature,
                "status": "MISSING_IN_ALL_MCI_SOURCE",
                "included_n_nonmissing": 0,
                "excluded_n_nonmissing": 0,
                "included_missing_rate": float("nan"),
                "excluded_missing_rate": float("nan"),
                "missing_rate_difference_included_minus_excluded": float("nan"),
                "included_mean": float("nan"),
                "excluded_mean": float("nan"),
                "included_sd": float("nan"),
                "excluded_sd": float("nan"),
                "standardized_mean_difference": float("nan"),
            })
            continue
        values = pd.to_numeric(all_mci[feature], errors="coerce").replace([-1, -4], np.nan)
        a = values[included].dropna().to_numpy(float)
        b = values[~included].dropna().to_numpy(float)
        miss_a = float(values[included].isna().mean())
        miss_b = float(values[~included].isna().mean())
        mean_a = float(np.mean(a)) if len(a) else float("nan")
        mean_b = float(np.mean(b)) if len(b) else float("nan")
        sd_a = float(np.std(a, ddof=1)) if len(a) > 1 else float("nan")
        sd_b = float(np.std(b, ddof=1)) if len(b) > 1 else float("nan")
        pooled = np.sqrt(
            (((len(a)-1) * sd_a**2) + ((len(b)-1) * sd_b**2))
            / max(len(a) + len(b) - 2, 1)
        ) if len(a) > 1 and len(b) > 1 else float("nan")
        smd = (mean_a - mean_b) / pooled if np.isfinite(pooled) and pooled > 0 else float("nan")
        rows.append({
            "feature": feature,
            "status": "COMPUTED",
            "included_n_nonmissing": int(len(a)),
            "excluded_n_nonmissing": int(len(b)),
            "included_missing_rate": miss_a,
            "excluded_missing_rate": miss_b,
            "missing_rate_difference_included_minus_excluded": miss_a - miss_b,
            "included_mean": mean_a,
            "excluded_mean": mean_b,
            "included_sd": sd_a,
            "excluded_sd": sd_b,
            "standardized_mean_difference": float(smd),
        })
    summary = pd.DataFrame(rows)
    flow = pd.DataFrame([{
        "all_first_mci_n": int(len(all_mci)),
        "included_36m_eligible_n": int(included.sum()),
        "excluded_insufficient_followup_n": int((~included).sum()),
        "interpretation": (
            "Descriptive baseline selection audit only; excluded participants "
            "do not receive inferred stable/converter labels."
        ),
    }])
    return flow, summary


def build_checkpoint_hash_audit(
    neural_run_dir,
    tree_output_dir,
    seeds: Sequence[int],
):
    """Hash/identity audit for all frozen Stage 4.5 checkpoints; no model/data rows exported."""
    from pathlib import Path
    from stage4_6_missingness_benchmark import (
        NEURAL_ALGORITHMS, TREE_ALGORITHMS, _resolve_neural_job_dir, sha256_file
    )
    neural_root = Path(neural_run_dir)
    tree_root = Path(tree_output_dir)
    rows = []
    for task in ["diagnosis_multiclass", "diagnosis_binary", "mci_conversion"]:
        for model_name in NEURAL_ALGORITHMS:
            for seed in seeds:
                job_key = f"{task}__S0_verified_asof__{model_name}__seed{int(seed)}"
                job_dir = _resolve_neural_job_dir(neural_root, job_key)
                best_path = job_dir / "best.pt"
                manifest_path = job_dir / "job_manifest.json"
                status_path = job_dir / "status.json"
                wrapper = json.loads(manifest_path.read_text(encoding="utf-8"))
                status = json.loads(status_path.read_text(encoding="utf-8"))
                actual_hash = sha256_file(best_path)
                expected_hash = status.get("output_sha256", {}).get("best.pt")
                job_manifest = wrapper["job_manifest"]
                identity_ok = (
                    job_manifest.get("task") == task
                    and job_manifest.get("feature_set") == "S0_verified_asof"
                    and job_manifest.get("model_variant") == model_name
                    and int(job_manifest.get("seed")) == int(seed)
                )
                passed = (
                    status.get("status") == "completed"
                    and expected_hash == actual_hash
                    and identity_ok
                )
                try:
                    relative_path = str(best_path.relative_to(neural_root.resolve()))
                except Exception:
                    relative_path = best_path.name
                rows.append({
                    "task": task, "model": model_name, "seed": int(seed),
                    "family": "neural", "checkpoint_relative_path": relative_path,
                    "checkpoint_sha256": actual_hash,
                    "expected_sha256": expected_hash,
                    "manifest_sha256": wrapper.get("job_manifest_sha256"),
                    "identity_check": bool(identity_ok),
                    "status": "PASS" if passed else "FAIL",
                })

        for model_name in TREE_ALGORITHMS:
            for seed in seeds:
                stem = f"{model_name}__seed{int(seed)}"
                nested_path = tree_root / "tree_models" / task / f"{stem}.joblib"
                nested_meta = tree_root / "tree_models" / task / f"{stem}.json"
                flat_path = tree_root / "tree_models" / f"{task}__{stem}.joblib"
                if nested_path.is_file() and nested_meta.is_file():
                    model_path = nested_path
                    meta = json.loads(nested_meta.read_text(encoding="utf-8"))
                    expected_hash = meta.get("model_sha256")
                    manifest_hash = meta.get("model_manifest_sha256")
                    identity_ok = True
                elif flat_path.is_file():
                    summary_path = tree_root.parent / "reports" / "tree_training_summary.csv"
                    if not summary_path.is_file():
                        raise FileNotFoundError(f"Missing Stage 4.5 tree training summary: {summary_path}")
                    summary = pd.read_csv(summary_path)
                    row = summary[
                        summary["task"].eq(task)
                        & summary["model"].eq(model_name)
                        & summary["seed"].astype(int).eq(int(seed))
                    ]
                    if len(row) != 1:
                        raise Stage46IntegrityError(f"Expected one tree manifest row for {task}/{model_name}/{seed}.")
                    model_path = flat_path
                    expected_hash = str(row.iloc[0]["checkpoint_sha256"])
                    manifest_hash = None
                    identity_ok = bool(row.iloc[0].get("train_only_preprocessing", False))
                else:
                    raise FileNotFoundError(f"Missing final tree checkpoint: {nested_path} or {flat_path}")
                actual_hash = sha256_file(model_path)
                passed = (expected_hash == actual_hash) and identity_ok
                try:
                    relative_path = str(model_path.relative_to(tree_root.resolve()))
                except Exception:
                    relative_path = model_path.name
                rows.append({
                    "task": task, "model": model_name, "seed": int(seed),
                    "family": "tree", "checkpoint_relative_path": relative_path,
                    "checkpoint_sha256": actual_hash,
                    "expected_sha256": expected_hash,
                    "manifest_sha256": manifest_hash,
                    "identity_check": bool(identity_ok),
                    "status": "PASS" if passed else "FAIL",
                })
    audit = pd.DataFrame(rows)
    if len(audit) != 3 * (4 + 3) * len(seeds):
        raise AssertionError("Checkpoint audit row count does not match final model inventory.")
    if not audit["status"].eq("PASS").all():
        raise AssertionError("One or more Stage 4.5 checkpoints failed hash/identity verification.")
    return audit


def load_cached_test_and_infer_validation_mc(
    prepared: Mapping,
    neural_run_dir,
    cached_test_path,
    validation_mc_path,
    seeds: Sequence[int],
    mc_passes: int = 30,
    feature_set: str = "S0_verified_asof",
    n_boot: int = 2000,
    random_seed: int = 20261115,
    batch_size: int = 128,
):
    """Reuse the verified Stage 4.8 test MC run and infer only validation MC.

    The cached file was produced from the frozen Stage 4.5 AUAF checkpoints
    with 30 MC-dropout passes. This avoids needlessly rerunning the identical
    held-out test inference while still producing validation probabilities for
    temperature/RAPS calibration. Participant IDs/labels are rechecked.
    """
    from pathlib import Path
    cached = pd.read_csv(Path(cached_test_path))
    validation_cached = pd.read_csv(Path(validation_mc_path))
    required_cached = {
        "task", "seed", "RID", "y_true", "mc_predictive_entropy",
        "mc_mean_class_sd", "mc_pred", "mc_gate_clinical_cognitive",
        "mc_gate_fluid_biomarker",
    }
    missing = required_cached - set(cached.columns)
    if missing:
        raise ValueError(f"Cached Stage 4.8 MC file is missing columns: {sorted(missing)}")
    required_validation = {
        "task", "model", "seed", "split", "row_index", "y_true", "entropy",
        "mean_class_probability_sd", "predicted_class", "gate_clinical_cognitive",
        "gate_fluid_biomarker",
    }
    missing_validation = required_validation - set(validation_cached.columns)
    if missing_validation:
        raise ValueError(f"Validation MC cache is missing columns: {sorted(missing_validation)}")

    prediction_rows, summary_rows, selective_rows = [], [], []
    tasks = ["diagnosis_multiclass", "diagnosis_binary", "mci_conversion"]
    for task_index, task in enumerate(tasks):
        pack = _task_pack(prepared, task, feature_set)
        rid_to_row = {int(rid): int(i) for i, rid in enumerate(pack.frame["RID"].astype(int))}
        for seed in seeds:
            sub = cached[cached["task"].eq(task) & cached["seed"].astype(int).eq(int(seed))].copy()
            if len(sub) != len(pack.idx("test", eligible_only=True)):
                raise AssertionError(f"{task}/{seed}: cached MC test row count mismatch.")
            sub["row_index"] = sub["RID"].astype(int).map(rid_to_row)
            if sub["row_index"].isna().any():
                raise AssertionError(f"{task}/{seed}: cached MC contains unknown RID.")
            sub["row_index"] = sub["row_index"].astype(int)
            sub = sub.sort_values("row_index").reset_index(drop=True)
            test_idx = pack.idx("test", eligible_only=True)
            if not np.array_equal(sub["row_index"].to_numpy(int), test_idx):
                raise AssertionError(f"{task}/{seed}: cached MC test RID order differs from frozen test split.")
            y_test = pack.frame.iloc[test_idx][pack.target].to_numpy(int)
            if not np.array_equal(sub["y_true"].to_numpy(int), y_test):
                raise AssertionError(f"{task}/{seed}: cached MC test labels differ from frozen labels.")

            # Cached test MC probabilities/gates.
            for _, row in sub.iterrows():
                rec = {
                    "task": task, "model": "AUAF", "seed": int(seed),
                    "split": "test", "row_index": int(row["row_index"]),
                    "y_true": int(row["y_true"]),
                    "entropy": float(row["mc_predictive_entropy"]),
                    "mean_class_probability_sd": float(row["mc_mean_class_sd"]),
                    "predicted_class": int(row["mc_pred"]),
                    "gate_clinical_cognitive": float(row["mc_gate_clinical_cognitive"]),
                    "gate_fluid_biomarker": float(row["mc_gate_fluid_biomarker"]),
                }
                for cls in range(pack.n_classes):
                    rec[f"p_class_{cls}"] = float(row[f"mc_mean_p_class_{cls}"])
                    rec[f"std_class_{cls}"] = float(row[f"mc_sd_p_class_{cls}"])
                prediction_rows.append(rec)

            # Validation MC was computed from the same frozen checkpoints in
            # bounded batches and is loaded from its local-only cache here.
            val_idx = pack.idx("validation", eligible_only=True)
            y_val = pack.frame.iloc[val_idx][pack.target].to_numpy(int)
            val_sub = validation_cached[
                validation_cached["task"].eq(task)
                & validation_cached["seed"].astype(int).eq(int(seed))
                & validation_cached["split"].eq("validation")
            ].sort_values("row_index").reset_index(drop=True)
            if len(val_sub) != len(val_idx):
                raise AssertionError(f"{task}/{seed}: validation MC cache row count mismatch.")
            if not np.array_equal(val_sub["row_index"].to_numpy(int), val_idx):
                raise AssertionError(f"{task}/{seed}: validation MC cache roster differs from frozen split.")
            if not np.array_equal(val_sub["y_true"].to_numpy(int), y_val):
                raise AssertionError(f"{task}/{seed}: validation MC cache labels differ from frozen split.")
            prediction_rows.extend(val_sub.to_dict("records"))
            print(f"MC validation cache verified: {task}/{seed} n={len(val_idx)}", flush=True)

            # Test-set uncertainty, gates, and selective prediction summaries.
            p_test = np.column_stack([sub[f"mc_mean_p_class_{c}"].to_numpy(float) for c in range(pack.n_classes)])
            entropy = sub["mc_predictive_entropy"].to_numpy(float)
            prob_sd = sub["mc_mean_class_sd"].to_numpy(float)
            pred = sub["mc_pred"].to_numpy(int)
            correct = (pred == y_test).astype(float)
            entropy_ci = _bootstrap_mean_ci(entropy, n_boot, random_seed + stable_seed(task, seed, "entropy") % 100000)
            sd_ci = _bootstrap_mean_ci(prob_sd, n_boot, random_seed + stable_seed(task, seed, "prob_sd") % 100000)
            err_ci = _bootstrap_mean_ci(1.0 - correct, n_boot, random_seed + stable_seed(task, seed, "error") % 100000)
            summary = {
                "task": task, "seed": int(seed), "n_test": int(len(test_idx)),
                "mc_passes": int(mc_passes), "accuracy": float(correct.mean()),
                "mean_entropy": entropy_ci["mean"], "mean_entropy_ci_low": entropy_ci["ci_low"],
                "mean_entropy_ci_high": entropy_ci["ci_high"],
                "mean_probability_sd": sd_ci["mean"], "mean_probability_sd_ci_low": sd_ci["ci_low"],
                "mean_probability_sd_ci_high": sd_ci["ci_high"],
                "error_rate": err_ci["mean"], "error_rate_ci_low": err_ci["ci_low"],
                "error_rate_ci_high": err_ci["ci_high"],
                "entropy_error_spearman": (
                    float(spearmanr(entropy, 1.0 - correct).statistic)
                    if np.std(entropy) > 0 and np.std(1.0-correct) > 0 else float("nan")
                ),
            }
            both = (pack.masks[test_idx] > 0).all(axis=1)
            summary["gate_complete_case_n"] = int(both.sum())
            for gate_col, modality in [
                ("mc_gate_clinical_cognitive", "clinical_cognitive"),
                ("mc_gate_fluid_biomarker", "fluid_biomarker"),
            ]:
                vals = sub.loc[both, gate_col].to_numpy(float)
                ci = _bootstrap_mean_ci(vals, n_boot, random_seed + stable_seed(task, seed, modality, "gate") % 100000)
                summary[f"gate_{modality}_mean_complete_case"] = ci["mean"]
                summary[f"gate_{modality}_ci_low"] = ci["ci_low"]
                summary[f"gate_{modality}_ci_high"] = ci["ci_high"]
            summary_rows.append(summary)

            order = np.argsort(entropy, kind="stable")
            for retention in np.round(np.arange(0.50, 1.001, 0.05), 2):
                keep_n = max(1, int(np.ceil(float(retention) * len(test_idx))))
                chosen = order[:keep_n]
                yy, pp = y_test[chosen], p_test[chosen]
                counts = {str(c): int((yy == c).sum()) for c in range(pack.n_classes)}
                selective_rows.append({
                    "task": task, "seed": int(seed), "retention_fraction": float(retention),
                    "retained_n": int(keep_n), "accuracy": float((pp.argmax(axis=1) == yy).mean()),
                    "macro_f1": float(classification_metrics(yy, pp)["f1_macro"]),
                    "class_counts_retained": json.dumps(counts, sort_keys=True),
                })

    predictions = pd.DataFrame(prediction_rows).sort_values(
        ["task", "seed", "split", "row_index"]
    ).reset_index(drop=True)
    return predictions, pd.DataFrame(summary_rows), pd.DataFrame(selective_rows)
