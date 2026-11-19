from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from auaf_model_factory import build_model
from auaf_safe_training_engine import classification_metrics, sha256_file


# Frozen final Stage 4.5 inventory: three tree baselines only.
TREE_ALGORITHMS = ["XGBoost", "CatBoost", "ExtraTrees"]
NEURAL_ALGORITHMS = ["EarlyFusion", "LateFusion", "AttentionFusion", "AUAF"]
DEFAULT_MISSINGNESS_RATES = [0.00, 0.25, 0.50, 0.75, 1.00]
DEFAULT_TUNING_TRIALS = 8
DEFAULT_INNER_FOLDS = 3
DEFAULT_SCENARIO_SEED = 20261115


class Stage46IntegrityError(RuntimeError):
    pass


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _write_csv_once(path: Path, frame: pd.DataFrame) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = frame.to_csv(index=False, lineterminator="\n")
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise Stage46IntegrityError(
                f"Refusing to overwrite a different existing result: {path}. Use a new Stage 4.6 RUN_ID."
            )
        return "EXISTING_IDENTICAL"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)
    return "CREATED"


def modality_indices(pack: Any) -> tuple[int, int]:
    names = list(pack.modalities)
    if "clinical_cognitive" not in names or "fluid_biomarker" not in names:
        raise Stage46IntegrityError(
            f"Expected clinical_cognitive and fluid_biomarker modalities; found {names}"
        )
    return names.index("clinical_cognitive"), names.index("fluid_biomarker")


def paired_test_indices(pack: Any) -> np.ndarray:
    clinical_idx, biomarker_idx = modality_indices(pack)
    test_idx = pack.idx("test", eligible_only=False)
    masks = np.asarray(pack.masks[test_idx], dtype=float)
    keep = (masks[:, clinical_idx] > 0) & (masks[:, biomarker_idx] > 0)
    selected = test_idx[keep]
    if len(selected) == 0:
        raise Stage46IntegrityError("No paired-modality test participants are available.")
    return selected.astype(int)


def nested_missingness_schedule(
    rids: Sequence[Any],
    rates: Sequence[float] = DEFAULT_MISSINGNESS_RATES,
    task_name: str = "task",
    scenario_seed: int = DEFAULT_SCENARIO_SEED,
) -> dict[float, np.ndarray]:
    """Create nested, deterministic participant-level biomarker-drop masks.

    The returned masks are aligned to `rids`. The same schedule is reused across
    all model families and model-initialization seeds.
    """
    rids = [str(x) for x in rids]
    if len(set(rids)) != len(rids):
        raise Stage46IntegrityError("RID values must be unique in the paired test cohort.")
    if not rates or any(float(r) < 0 or float(r) > 1 for r in rates):
        raise ValueError("Missingness rates must be between 0 and 1.")
    order = sorted(
        range(len(rids)),
        key=lambda i: hashlib.sha256(
            f"{task_name}|{rids[i]}|{scenario_seed}".encode("utf-8")
        ).hexdigest()
    )
    schedule = {}
    for rate in sorted(set(float(x) for x in rates)):
        n_drop = int(np.floor(rate * len(rids) + 0.5))
        mask = np.zeros(len(rids), dtype=bool)
        if n_drop:
            mask[np.asarray(order[:n_drop], dtype=int)] = True
        schedule[rate] = mask
    ordered_rates = sorted(schedule)
    for left, right in zip(ordered_rates[:-1], ordered_rates[1:]):
        if np.any(schedule[left] & ~schedule[right]):
            raise Stage46IntegrityError("Missingness scenarios are not nested.")
    return schedule


def apply_neural_missingness(
    pack: Any,
    indices: Sequence[int],
    dropped_biomarker: Sequence[bool],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Copy neural inputs, zero the removed biomarker vector, and update its mask."""
    indices = np.asarray(indices, dtype=int)
    dropped = np.asarray(dropped_biomarker, dtype=bool)
    if len(indices) != len(dropped):
        raise ValueError("indices and dropped_biomarker must have equal length.")
    clinical_idx, biomarker_idx = modality_indices(pack)
    inputs = {
        modality: np.asarray(pack.matrices[modality][indices], dtype=np.float32).copy()
        for modality in pack.modalities
    }
    masks = np.asarray(pack.masks[indices], dtype=np.float32).copy()
    if not np.all(masks[:, clinical_idx] > 0):
        raise Stage46IntegrityError("Paired test cohort unexpectedly lacks clinical modality.")
    if not np.all(masks[:, biomarker_idx] > 0):
        raise Stage46IntegrityError("Paired test cohort unexpectedly lacks biomarkers before imposed missingness.")
    inputs["fluid_biomarker"][dropped, :] = 0.0
    masks[dropped, biomarker_idx] = 0.0
    if np.any(masks.sum(axis=1) <= 0):
        raise Stage46IntegrityError("An imposed scenario created an all-missing participant.")
    return inputs, masks


def tree_feature_columns(pack: Any) -> tuple[list[str], list[str], list[str]]:
    clinical = list(pack.features_by_modality["clinical_cognitive"])
    biomarkers = list(pack.features_by_modality["fluid_biomarker"])
    features = clinical + biomarkers
    if len(features) != len(set(features)):
        raise Stage46IntegrityError("Duplicate tree feature columns.")
    missing = sorted(set(features) - set(pack.frame.columns))
    if missing:
        raise Stage46IntegrityError(f"Raw feature columns missing from task frame: {missing}")
    return features, clinical, biomarkers


def build_tree_design(
    pack: Any,
    indices: Sequence[int],
    dropped_biomarker: Sequence[bool] | None = None,
) -> pd.DataFrame:
    """Build raw-feature tree inputs with explicit modality-availability flags."""
    indices = np.asarray(indices, dtype=int)
    features, clinical, biomarkers = tree_feature_columns(pack)
    clinical_idx, biomarker_idx = modality_indices(pack)
    x = pack.frame.iloc[indices][features].copy().reset_index(drop=True)
    x = x.apply(pd.to_numeric, errors="coerce")

    masks = np.asarray(pack.masks[indices], dtype=np.float32).copy()
    if dropped_biomarker is not None:
        dropped = np.asarray(dropped_biomarker, dtype=bool)
        if len(dropped) != len(indices):
            raise ValueError("dropped_biomarker length does not match indices.")
        x.loc[dropped, biomarkers] = np.nan
        masks[dropped, biomarker_idx] = 0.0

    x["__available_clinical"] = masks[:, clinical_idx].astype(np.float32)
    x["__available_biomarker"] = masks[:, biomarker_idx].astype(np.float32)
    return x


def _suggest_parameters(trial: optuna.Trial, algorithm: str) -> dict:
    # Search spaces are preserved from the original manuscript pipeline.
    if algorithm == "LogisticRegression":
        return {"C": trial.suggest_float("C", 1e-3, 100.0, log=True)}
    if algorithm in {"RandomForest", "ExtraTrees"}:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 700, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.7]),
        }
    if algorithm in {"XGBoost", "LightGBM"}:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if algorithm == "CatBoost":
        return {
            "iterations": trial.suggest_int("iterations", 150, 600, step=50),
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 20.0, log=True),
        }
    raise KeyError(algorithm)


def _build_classifier(algorithm: str, params: Mapping[str, Any], n_classes: int, seed: int):
    common = {"random_state": int(seed)}
    params = dict(params)
    if algorithm == "LogisticRegression":
        return LogisticRegression(
            **common, **params, class_weight="balanced", max_iter=3000, solver="lbfgs"
        )
    if algorithm == "RandomForest":
        return RandomForestClassifier(
            **common, **params, class_weight="balanced", n_jobs=1
        )
    if algorithm == "ExtraTrees":
        return ExtraTreesClassifier(
            **common, **params, class_weight="balanced", n_jobs=1
        )
    if algorithm == "XGBoost":
        kwargs = {
            **common, **params,
            "objective": "binary:logistic" if n_classes == 2 else "multi:softprob",
            "eval_metric": "logloss", "n_jobs": 1, "verbosity": 0,
        }
        if n_classes > 2:
            kwargs["num_class"] = int(n_classes)
        return XGBClassifier(**kwargs)
    if algorithm == "LightGBM":
        kwargs = {
            **common, **params,
            "objective": "binary" if n_classes == 2 else "multiclass",
            "class_weight": "balanced", "verbosity": -1, "n_jobs": 1,
        }
        if n_classes > 2:
            kwargs["num_class"] = int(n_classes)
        return LGBMClassifier(**kwargs)
    if algorithm == "CatBoost":
        return CatBoostClassifier(
            **common, **params,
            loss_function="Logloss" if n_classes == 2 else "MultiClass",
            verbose=False, allow_writing_files=False,
            auto_class_weights="Balanced", thread_count=1,
        )
    raise KeyError(algorithm)


class TabularBaseline:
    """Small serializable preprocessing + classifier wrapper.

    This avoids relying on third-party estimator sklearn tags (notably CatBoost
    across some sklearn versions) while keeping preprocessing fold-specific.
    """
    def __init__(self, algorithm: str, params: Mapping[str, Any], n_classes: int, seed: int):
        self.algorithm = algorithm
        self.params = dict(params)
        self.n_classes = int(n_classes)
        self.seed = int(seed)
        self.imputer = SimpleImputer(strategy="median", add_indicator=True)
        self.scaler = StandardScaler() if algorithm == "LogisticRegression" else None
        self.estimator = _build_classifier(algorithm, self.params, self.n_classes, self.seed)
        self.classes_ = None

    def _transform_fit(self, x: pd.DataFrame) -> np.ndarray:
        values = self.imputer.fit_transform(x)
        if self.scaler is not None:
            values = self.scaler.fit_transform(values)
        return np.asarray(values, dtype=np.float32)

    def _transform(self, x: pd.DataFrame) -> np.ndarray:
        values = self.imputer.transform(x)
        if self.scaler is not None:
            values = self.scaler.transform(values)
        return np.asarray(values, dtype=np.float32)

    def fit(self, x: pd.DataFrame, y: np.ndarray):
        y = np.asarray(y, dtype=int)
        values = self._transform_fit(x)
        weights = compute_sample_weight(class_weight="balanced", y=y)
        self.estimator.fit(values, y, sample_weight=weights)
        self.classes_ = np.asarray(self.estimator.classes_, dtype=int)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("Baseline has not been fitted.")
        return np.asarray(self.estimator.predict_proba(self._transform(x)), dtype=float)


def _build_pipeline(algorithm: str, params: Mapping[str, Any], n_classes: int, seed: int):
    return TabularBaseline(algorithm, params, n_classes, seed)


def _fit_pipeline(pipeline: TabularBaseline, x: pd.DataFrame, y: np.ndarray) -> TabularBaseline:
    return pipeline.fit(x, y)


def _align_probability(probability: np.ndarray, classes: Sequence[int], n_classes: int) -> np.ndarray:
    probability = np.asarray(probability, dtype=float)
    output = np.zeros((probability.shape[0], n_classes), dtype=float)
    for source_col, class_value in enumerate(classes):
        class_value = int(class_value)
        if 0 <= class_value < n_classes:
            output[:, class_value] = probability[:, source_col]
    row_sum = output.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0):
        raise Stage46IntegrityError("Classifier produced an all-zero probability row.")
    return output / row_sum


def _tree_contract(
    pack: Any, task_name: str, algorithm: str, seed: int,
    n_trials: int, inner_folds: int, input_hashes: Mapping[str, str] | None,
    module_sha256: str
) -> dict:
    features, clinical, biomarkers = tree_feature_columns(pack)
    return {
        "schema_version": 2,
        "module_sha256": module_sha256,
        "task": task_name,
        "feature_set": "S0_verified_asof",
        "algorithm": algorithm,
        "seed": int(seed),
        "features": features,
        "clinical_features": clinical,
        "biomarker_features": biomarkers,
        "n_trials": int(n_trials),
        "inner_folds": int(inner_folds),
        "input_hashes": dict(input_hashes or {}),
        "missingness_indicators": ["__available_clinical", "__available_biomarker"],
        "objective": "mean inner-CV macro-F1",
        "preprocessing": "fold-specific median imputation; StandardScaler for LogisticRegression only",
    }


def fit_tree_baseline(
    pack: Any,
    task_name: str,
    algorithm: str,
    seed: int,
    output_dir: Path,
    input_hashes: Mapping[str, str] | None = None,
    n_trials: int = DEFAULT_TUNING_TRIALS,
    inner_folds: int = DEFAULT_INNER_FOLDS,
    resume: bool = True,
) -> tuple[TabularBaseline, dict]:
    if algorithm not in TREE_ALGORITHMS:
        raise KeyError(algorithm)
    output_dir = Path(output_dir)
    model_dir = output_dir / "tree_models" / task_name
    model_dir.mkdir(parents=True, exist_ok=True)

    module_sha256 = sha256_file(Path(__file__))
    contract = _tree_contract(
        pack, task_name, algorithm, seed, n_trials, inner_folds,
        input_hashes, module_sha256
    )
    contract_hash = _sha256_text(json.dumps(contract, sort_keys=True, separators=(",", ":")))
    stem = f"{algorithm}__seed{int(seed)}"
    model_path = model_dir / f"{stem}.joblib"
    meta_path = model_dir / f"{stem}.json"
    study_path = model_dir / f"{stem}.sqlite3"
    contract_path = model_dir / f"{stem}.contract.json"

    def _load_study():
        storage = f"sqlite:///{study_path.resolve().as_posix()}"
        return optuna.create_study(
            direction="maximize",
            study_name=f"{task_name}_{algorithm}_seed{int(seed)}",
            storage=storage,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=int(seed)),
        )

    def _trial_counts(study):
        states = [trial.state.name for trial in study.trials]
        return {
            "complete": int(sum(state == "COMPLETE" for state in states)),
            "failed": int(sum(state == "FAIL" for state in states)),
            "pruned": int(sum(state == "PRUNED" for state in states)),
            "total": int(len(states)),
        }

    if meta_path.exists():
        if not (model_path.exists() and contract_path.exists()):
            raise Stage46IntegrityError(f"Incomplete existing tree-baseline artifact: {stem}")
        old_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if old_contract.get("contract_sha256") != contract_hash:
            raise Stage46IntegrityError(f"Tree baseline contract mismatch for {stem}; use a new RUN_ID.")
        if not resume:
            raise Stage46IntegrityError(f"Tree baseline already exists and resume=False: {stem}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if sha256_file(model_path) != meta["model_sha256"]:
            raise Stage46IntegrityError(f"Tree model hash mismatch: {model_path}")
        return joblib.load(model_path), meta

    # Recover a crash after the atomic model rename but before metadata writing.
    if model_path.exists() and not meta_path.exists():
        if not (contract_path.exists() and study_path.exists()):
            raise Stage46IntegrityError(f"Unverifiable orphan model artifact: {model_path}")
        old_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if old_contract.get("contract_sha256") != contract_hash:
            raise Stage46IntegrityError(f"Orphan model contract mismatch for {stem}.")
        study = _load_study()
        counts_trials = _trial_counts(study)
        if counts_trials["complete"] < int(n_trials):
            raise Stage46IntegrityError(
                f"Model exists but only {counts_trials['complete']} complete Optuna trials; "
                "cannot safely reconstruct metadata."
            )
        train_idx = pack.idx("train", eligible_only=True)
        y_train = pack.frame.iloc[train_idx][pack.target].to_numpy(dtype=int)
        class_counts = np.bincount(y_train, minlength=pack.n_classes)
        model = joblib.load(model_path)
        meta = {
            "contract_sha256": contract_hash,
            "task": task_name,
            "algorithm": algorithm,
            "seed": int(seed),
            "best_params": dict(study.best_params),
            "best_inner_cv_macro_f1": float(study.best_value),
            "train_n": int(len(train_idx)),
            "train_class_counts": {str(i): int(class_counts[i]) for i in range(pack.n_classes)},
            "model_sha256": sha256_file(model_path),
            "tuning_complete_trials": counts_trials["complete"],
            "tuning_failed_trials": counts_trials["failed"],
            "tuning_pruned_trials": counts_trials["pruned"],
            "tuning_total_trials": counts_trials["total"],
            "input_hashes": dict(input_hashes or {}),
            "module_sha256": module_sha256,
            "recovered_after_atomic_model_save": True,
        }
        _atomic_json(meta_path, meta)
        return model, meta

    if not contract_path.exists() and study_path.exists():
        raise Stage46IntegrityError(
            f"Optuna database exists without its contract file for {stem}; refusing unsafe reuse."
        )
    if contract_path.exists():
        old = json.loads(contract_path.read_text(encoding="utf-8"))
        if old.get("contract_sha256") != contract_hash:
            raise Stage46IntegrityError(f"Optuna study contract mismatch for {stem}; use a new RUN_ID.")
    else:
        _atomic_json(contract_path, {
            "contract_sha256": contract_hash,
            "contract": contract,
        })

    train_idx = pack.idx("train", eligible_only=True)
    y_train = pack.frame.iloc[train_idx][pack.target].to_numpy(dtype=int)
    counts = np.bincount(y_train, minlength=pack.n_classes)
    if np.any(counts < inner_folds):
        raise Stage46IntegrityError(
            f"{task_name}: at least one training class has fewer than {inner_folds} participants."
        )
    x_train = build_tree_design(pack, train_idx)
    folds = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=int(seed))

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_parameters(trial, algorithm)
        scores = []
        for fit_local, valid_local in folds.split(np.zeros(len(y_train)), y_train):
            pipeline = _build_pipeline(algorithm, params, pack.n_classes, seed)
            _fit_pipeline(pipeline, x_train.iloc[fit_local], y_train[fit_local])
            probability = _align_probability(
                pipeline.predict_proba(x_train.iloc[valid_local]),
                pipeline.classes_,
                pack.n_classes,
            )
            prediction = probability.argmax(axis=1)
            scores.append(float(f1_score(
                y_train[valid_local], prediction,
                average="macro", zero_division=0
            )))
        return float(np.mean(scores))

    study = _load_study()
    counts_trials = _trial_counts(study)
    # Target COMPLETE trials, not total attempts. Failed/interrupted trials are
    # retained in SQLite for audit but do not satisfy the 8-trial budget.
    remaining = max(0, int(n_trials) - counts_trials["complete"])
    if remaining:
        study.optimize(
            objective, n_trials=remaining, n_jobs=1,
            show_progress_bar=False, gc_after_trial=True
        )

    counts_trials = _trial_counts(study)
    if counts_trials["complete"] < int(n_trials):
        raise Stage46IntegrityError(
            f"{stem}: only {counts_trials['complete']}/{n_trials} Optuna trials completed; "
            "rerun with the same RUN_ID to resume."
        )

    best_params = dict(study.best_params)
    final_pipeline = _build_pipeline(algorithm, best_params, pack.n_classes, seed)
    _fit_pipeline(final_pipeline, x_train, y_train)

    tmp_model = model_path.with_name(f".{model_path.name}.{os.getpid()}.tmp")
    joblib.dump(final_pipeline, tmp_model)
    os.replace(tmp_model, model_path)

    meta = {
        "contract_sha256": contract_hash,
        "task": task_name,
        "algorithm": algorithm,
        "seed": int(seed),
        "best_params": best_params,
        "best_inner_cv_macro_f1": float(study.best_value),
        "train_n": int(len(train_idx)),
        "train_class_counts": {str(i): int(counts[i]) for i in range(pack.n_classes)},
        "model_sha256": sha256_file(model_path),
        "tuning_complete_trials": counts_trials["complete"],
        "tuning_failed_trials": counts_trials["failed"],
        "tuning_pruned_trials": counts_trials["pruned"],
        "tuning_total_trials": counts_trials["total"],
        "input_hashes": dict(input_hashes or {}),
        "module_sha256": module_sha256,
        "recovered_after_atomic_model_save": False,
    }
    _atomic_json(meta_path, meta)
    return final_pipeline, meta

def run_tree_baselines(
    prepared: Mapping[tuple[str, str], Any],
    output_dir: Path,
    seeds: Sequence[int],
    input_hashes: Mapping[str, str] | None = None,
    n_trials: int = DEFAULT_TUNING_TRIALS,
    inner_folds: int = DEFAULT_INNER_FOLDS,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Fit the three final Stage 4.5 tree baselines per task and seed; outputs stay local."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = {}
    metric_rows, prediction_rows = [], []

    for task_name in ["diagnosis_multiclass", "diagnosis_binary", "mci_conversion"]:
        pack = prepared[(task_name, "S0_verified_asof")]
        for algorithm in TREE_ALGORITHMS:
            for seed in seeds:
                model, meta = fit_tree_baseline(
                    pack, task_name, algorithm, int(seed), output_dir,
                    input_hashes=input_hashes, n_trials=n_trials,
                    inner_folds=inner_folds, resume=True,
                )
                models[(task_name, algorithm, int(seed))] = model

                for split in ["validation", "test"]:
                    idx = pack.idx(split, eligible_only=True)
                    x = build_tree_design(pack, idx)
                    y = pack.frame.iloc[idx][pack.target].to_numpy(dtype=int)
                    probability = _align_probability(
                        model.predict_proba(x),
                        model.classes_,
                        pack.n_classes,
                    )
                    metrics = classification_metrics(y, probability)
                    metric_rows.append({
                        "task": task_name, "feature_set": "S0_verified_asof",
                        "model": algorithm, "seed": int(seed), "split": split,
                        "n": int(len(idx)), "inner_cv_macro_f1": meta["best_inner_cv_macro_f1"],
                        **metrics,
                    })
                    if split == "validation":
                        continue
                    for j, row_index in enumerate(idx):
                        record = {
                            "task": task_name, "model": algorithm, "seed": int(seed),
                            "split": split, "row_index": int(row_index),
                            "y_true": int(y[j]), "y_pred": int(probability[j].argmax()),
                        }
                        for cls in range(pack.n_classes):
                            record[f"p_class_{cls}"] = float(probability[j, cls])
                        prediction_rows.append(record)

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    _write_csv_once(output_dir / "tree_baseline_metrics.csv", metrics_df)
    _write_csv_once(output_dir / "tree_baseline_test_predictions_LOCAL_ONLY.csv", predictions_df)
    return models, metrics_df, predictions_df


def load_tree_baseline(
    output_dir: Path, task_name: str, algorithm: str, seed: int
) -> Pipeline:
    output_dir = Path(output_dir)
    model_dir = output_dir / "tree_models" / task_name
    stem = f"{algorithm}__seed{int(seed)}"
    model_path = model_dir / f"{stem}.joblib"
    meta_path = model_dir / f"{stem}.json"

    # Stage 4.5 delivery ZIP stores the 45 immutable tree models flat under
    # checkpoints/tree_models and records their SHA-256 values in reports/tree_training_summary.csv.
    # Accept that canonical delivery layout without inventing sidecar metadata.
    if not model_path.is_file():
        flat_path = output_dir / "tree_models" / f"{task_name}__{stem}.joblib"
        summary_path = output_dir.parent / "reports" / "tree_training_summary.csv"
        if not flat_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"Missing trained tree baseline: {model_path} (flat fallback: {flat_path})")
        summary = pd.read_csv(summary_path)
        row = summary[
            summary["task"].eq(task_name)
            & summary["model"].eq(algorithm)
            & summary["seed"].astype(int).eq(int(seed))
        ]
        if len(row) != 1:
            raise Stage46IntegrityError(f"Expected one Stage 4.5 tree manifest row for {task_name}/{algorithm}/{seed}.")
        expected_hash = str(row.iloc[0]["checkpoint_sha256"])
        if sha256_file(flat_path) != expected_hash:
            raise Stage46IntegrityError(f"Tree model checksum mismatch: {flat_path}")
        return joblib.load(flat_path)

    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing trained tree baseline metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if sha256_file(model_path) != meta.get("model_sha256"):
        raise Stage46IntegrityError(f"Tree model checksum mismatch: {model_path}")
    return joblib.load(model_path)



def _resolve_neural_job_dir(neural_run_dir: Path, job_key: str) -> Path:
    """Resolve the immutable neural job folder from common Stage 4.5 layouts."""
    root = Path(neural_run_dir).expanduser().resolve()
    candidates = [
        root / "jobs" / job_key,
        root / "outputs" / "stage4_5_neural_training_v1" / "jobs" / job_key,
        root / "stage4_5_neural_training_v1" / "jobs" / job_key,
        root / "checkpoints" / "neural_best" / job_key,
        root / "neural_best" / job_key,
        root.parent / "jobs" / job_key,
        root.parent / "outputs" / "stage4_5_neural_training_v1" / "jobs" / job_key,
    ]
    valid = []
    for candidate in candidates:
        if (candidate / "best.pt").is_file():
            valid.append(candidate)
    if not valid and root.is_dir():
        # The package may have one additional wrapper directory. Search only
        # for the exact immutable job key, never for a generic best.pt.
        valid = [
            p for p in root.rglob(job_key)
            if p.is_dir() and (p / "best.pt").is_file()
        ]
    unique = sorted({p.resolve() for p in valid})
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        # Prefer the canonical outputs/.../jobs path when it is one of the hits.
        canonical = [p for p in unique if p.name == job_key and p.parent.name == "jobs"]
        if len(canonical) == 1:
            return canonical[0]
        raise Stage46IntegrityError(
            f"Ambiguous neural job folders for {job_key}: {[str(x) for x in unique]}"
        )
    raise FileNotFoundError(
        f"Could not locate best.pt for neural job {job_key} under {root}. "
        "Set NEURAL_RUN_DIR to the extracted Stage 4.5 folder containing jobs/."
    )


def load_neural_checkpoint(
    pack: Any, neural_run_dir: Path, task_name: str,
    model_variant: str, seed: int
) -> nn.Module:
    job_key = f"{task_name}__S0_verified_asof__{model_variant}__seed{int(seed)}"
    job_dir = _resolve_neural_job_dir(Path(neural_run_dir), job_key)
    manifest_path = job_dir / "job_manifest.json"
    status_path = job_dir / "status.json"
    best_path = job_dir / "best.pt"
    for p in [manifest_path, status_path, best_path]:
        if not p.is_file():
            raise FileNotFoundError(f"Required completed neural job artifact is missing: {p}")

    wrapper = json.loads(manifest_path.read_text(encoding="utf-8"))
    job_manifest = wrapper["job_manifest"]
    job_hash = wrapper["job_manifest_sha256"]
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "completed":
        raise Stage46IntegrityError(f"Neural job is not completed: {job_key}")
    expected_hash = status.get("output_sha256", {}).get("best.pt")
    if not expected_hash or sha256_file(best_path) != expected_hash:
        raise Stage46IntegrityError(f"Neural best checkpoint checksum mismatch: {job_key}")
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    if checkpoint.get("job_manifest_sha256") != job_hash:
        raise Stage46IntegrityError(f"Neural checkpoint manifest hash mismatch: {job_key}")
    if (
        job_manifest.get("task") != task_name
        or job_manifest.get("feature_set") != "S0_verified_asof"
        or job_manifest.get("model_variant") != model_variant
        or int(job_manifest.get("seed")) != int(seed)
    ):
        raise Stage46IntegrityError(f"Neural checkpoint identity mismatch: {job_key}")

    dimensions = {m: int(pack.matrices[m].shape[1]) for m in pack.modalities}
    model = build_model(model_variant, dimensions, pack.n_classes)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model

def predict_neural_scenario(
    model: nn.Module, pack: Any, indices: Sequence[int],
    dropped_biomarker: Sequence[bool], batch_size: int = 128
) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    dropped = np.asarray(dropped_biomarker, dtype=bool)
    inputs_np, masks_np = apply_neural_missingness(pack, indices, dropped)
    output_probabilities = []
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        inputs = {
            modality: torch.as_tensor(values[start:stop], dtype=torch.float32, device=device)
            for modality, values in inputs_np.items()
        }
        mask = torch.as_tensor(masks_np[start:stop], dtype=torch.float32, device=device)
        logits, _aux = model(inputs, mask, return_aux=True)
        output_probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(output_probabilities, axis=0)


def run_missingness_benchmark(
    prepared: Mapping[tuple[str, str], Any],
    neural_run_dir: Path,
    tree_output_dir: Path,
    output_dir: Path,
    seeds: Sequence[int],
    rates: Sequence[float] = DEFAULT_MISSINGNESS_RATES,
    scenario_seed: int = DEFAULT_SCENARIO_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate four neural and three final tree models on one paired test cohort per task."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows, prediction_rows, scenario_rows = [], [], []

    for task_name in ["diagnosis_multiclass", "diagnosis_binary", "mci_conversion"]:
        pack = prepared[(task_name, "S0_verified_asof")]
        indices = paired_test_indices(pack)
        rids = pack.frame.iloc[indices]["RID"].astype(str).tolist()
        y = pack.frame.iloc[indices][pack.target].to_numpy(dtype=int)
        schedule = nested_missingness_schedule(
            rids, rates=rates, task_name=task_name, scenario_seed=scenario_seed
        )

        # Load and checksum each trained model once per task; reuse it across all five scenarios.
        neural_models = {
            (model_name, int(seed)): load_neural_checkpoint(
                pack, neural_run_dir, task_name, model_name, int(seed)
            )
            for model_name in NEURAL_ALGORITHMS
            for seed in seeds
        }
        tree_models = {
            (model_name, int(seed)): load_tree_baseline(
                tree_output_dir, task_name, model_name, int(seed)
            )
            for model_name in TREE_ALGORITHMS
            for seed in seeds
        }

        for rate, dropped in schedule.items():
            scenario_rows.append({
                "task": task_name,
                "requested_biomarker_missingness": float(rate),
                "paired_test_n": int(len(indices)),
                "test_class_counts": json.dumps(
                    {str(k): int(v) for k, v in pd.Series(y).value_counts().sort_index().items()},
                    sort_keys=True,
                ),
                "biomarker_removed_n": int(dropped.sum()),
                "biomarker_missingness_realized": float(dropped.mean()),
                "dropped_row_index_sha256": _sha256_text(
                    ",".join(map(str, indices[dropped].tolist()))
                ),
                "scenario_seed": int(scenario_seed),
            })

            # Neural fusion models: identical participants and exact same mask per scenario.
            for model_name in NEURAL_ALGORITHMS:
                for seed in seeds:
                    model = neural_models[(model_name, int(seed))]
                    probability = predict_neural_scenario(
                        model, pack, indices, dropped, batch_size=128
                    )
                    metrics = classification_metrics(y, probability)
                    metric_rows.append({
                        "task": task_name, "model": model_name, "family": "neural",
                        "seed": int(seed), "missingness_requested": float(rate),
                        "missingness_realized": float(dropped.mean()),
                        "n_test_paired": int(len(indices)), **metrics,
                    })
                    for j, row_index in enumerate(indices):
                        record = {
                            "task": task_name, "model": model_name, "family": "neural",
                            "seed": int(seed), "missingness_requested": float(rate),
                            "row_index": int(row_index), "y_true": int(y[j]),
                            "y_pred": int(probability[j].argmax()),
                            "biomarker_removed": bool(dropped[j]),
                        }
                        for cls in range(pack.n_classes):
                            record[f"p_class_{cls}"] = float(probability[j, cls])
                        prediction_rows.append(record)

            # Classical baselines: use the same paired cohort and missingness schedule.
            for model_name in TREE_ALGORITHMS:
                for seed in seeds:
                    model = tree_models[(model_name, int(seed))]
                    x = build_tree_design(pack, indices, dropped_biomarker=dropped)
                    probability = _align_probability(
                        model.predict_proba(x),
                        model.classes_,
                        pack.n_classes,
                    )
                    metrics = classification_metrics(y, probability)
                    metric_rows.append({
                        "task": task_name, "model": model_name, "family": "classical",
                        "seed": int(seed), "missingness_requested": float(rate),
                        "missingness_realized": float(dropped.mean()),
                        "n_test_paired": int(len(indices)), **metrics,
                    })
                    for j, row_index in enumerate(indices):
                        record = {
                            "task": task_name, "model": model_name, "family": "classical",
                            "seed": int(seed), "missingness_requested": float(rate),
                            "row_index": int(row_index), "y_true": int(y[j]),
                            "y_pred": int(probability[j].argmax()),
                            "biomarker_removed": bool(dropped[j]),
                        }
                        for cls in range(pack.n_classes):
                            record[f"p_class_{cls}"] = float(probability[j, cls])
                        prediction_rows.append(record)

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    scenarios_df = pd.DataFrame(scenario_rows)

    # Participant-level predictions are local-only artifacts; never upload or publish them.
    _write_csv_once(output_dir / "missingness_metrics.csv", metrics_df)
    _write_csv_once(output_dir / "missingness_scenarios.csv", scenarios_df)
    _write_csv_once(output_dir / "missingness_predictions_LOCAL_ONLY.csv", predictions_df)

    manifest = {
        "stage": "4.6_missingness_benchmark",
        "models": NEURAL_ALGORITHMS + TREE_ALGORITHMS,
        "seeds": [int(x) for x in seeds],
        "missingness_rates": [float(x) for x in sorted(set(rates))],
        "scenario_seed": int(scenario_seed),
        "paired_test_cohort_policy": "same participants with both modalities available at baseline, reused across every model and missingness rate",
        "tree_tuning": {
            "trials": DEFAULT_TUNING_TRIALS,
            "inner_folds": DEFAULT_INNER_FOLDS,
            "objective": "macro-F1",
        },
        "participant_level_outputs": "local only; do not upload or publicly release",
    }
    _atomic_json(output_dir / "stage4_6_manifest.json", manifest)
    return metrics_df, scenarios_df, predictions_df


def synthetic_smoke_test() -> dict:
    """Small synthetic-only test of paired cohorts, nested masks, and model inputs."""
    class ToyPack:
        def __init__(self):
            rng = np.random.default_rng(17)
            n = 40
            self.modalities = ["clinical_cognitive", "fluid_biomarker"]
            self.features_by_modality = {
                "clinical_cognitive": ["c1", "c2"],
                "fluid_biomarker": ["b1", "b2"],
            }
            self.frame = pd.DataFrame({
                "RID": np.arange(1000, 1000 + n),
                "split": ["train"] * 20 + ["validation"] * 8 + ["test"] * 12,
                "target": np.array([0, 1] * 20),
                "c1": rng.normal(size=n), "c2": rng.normal(size=n),
                "b1": rng.normal(size=n), "b2": rng.normal(size=n),
            })
            self.target = "target"
            self.n_classes = 2
            self.matrices = {
                "clinical_cognitive": rng.normal(size=(n, 2)).astype(np.float32),
                "fluid_biomarker": rng.normal(size=(n, 2)).astype(np.float32),
            }
            self.masks = np.ones((n, 2), dtype=np.float32)
        def idx(self, split, eligible_only=False):
            idx = np.flatnonzero(self.frame["split"].eq(split).to_numpy())
            if eligible_only:
                idx = idx[self.masks[idx].sum(axis=1) > 0]
            return idx

    pack = ToyPack()
    indices = paired_test_indices(pack)
    rids = pack.frame.iloc[indices]["RID"].tolist()
    schedule = nested_missingness_schedule(rids, task_name="toy")
    assert len(indices) == 12
    assert [int(schedule[r].sum()) for r in sorted(schedule)] == [0, 3, 6, 9, 12]
    ordered = sorted(schedule)
    assert all(np.all(schedule[a] <= schedule[b]) for a, b in zip(ordered[:-1], ordered[1:]))

    inputs, masks = apply_neural_missingness(pack, indices, schedule[0.5])
    assert np.all(masks[schedule[0.5], 1] == 0)
    assert np.all(inputs["fluid_biomarker"][schedule[0.5]] == 0)
    assert np.all(masks[~schedule[0.5], 1] == 1)

    tree_x = build_tree_design(pack, indices, schedule[0.5])
    assert tree_x.loc[schedule[0.5], ["b1", "b2"]].isna().all().all()
    assert np.all(tree_x.loc[schedule[0.5], "__available_biomarker"] == 0)
    assert np.all(tree_x.loc[~schedule[0.5], "__available_biomarker"] == 1)

    return {
        "status": "PASS",
        "test_data": "synthetic only",
        "paired_test_n": len(indices),
        "missingness_counts": {str(k): int(v.sum()) for k, v in schedule.items()},
        "same_cohort_across_scenarios": True,
        "nested_scenarios": True,
        "neural_inputs_zeroed_when_biomarkers_removed": True,
        "tree_inputs_nan_and_availability_flagged": True,
    }
