from __future__ import annotations

import os
import re
import json
import time
import uuid
import hashlib
import random
import tempfile
import platform
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score,
    average_precision_score, log_loss
)


class ResumeMismatchError(RuntimeError):
    """Raised when a run/checkpoint does not match its immutable manifest."""


class ExistingRunError(RuntimeError):
    """Raised when an existing run would otherwise be overwritten."""


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, (float,)):
        return value if np.isfinite(value) else None
    if isinstance(value, (np.ndarray,)):
        return _json_safe(value.tolist())
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def canonical_json(payload: Any) -> str:
    return json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with tmp.open("w", encoding="utf-8", newline="") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(_json_safe(payload), indent=2, sort_keys=True,
                                       ensure_ascii=False, allow_nan=False))


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_write_text(path, frame.to_csv(index=False, lineterminator="\n"))


def atomic_torch_save(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        torch.save(payload, tmp)
        with tmp.open("rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _acquire_lock(lock_path: Path) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": os.getpid(),
        "hostname": platform.node(),
        "created_unix": time.time(),
    }
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(str(lock_path), flags, 0o600)
    except FileExistsError as exc:
        try:
            detail = lock_path.read_text(encoding="utf-8")
        except Exception:
            detail = "unreadable lock metadata"
        raise RuntimeError(
            f"Job is locked by another process or a previous interrupted process: "
            f"{lock_path}. Inspect the lock before removing it. Details: {detail}"
        ) from exc
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(payload, indent=2))


def _release_lock(lock_path: Path) -> None:
    try:
        Path(lock_path).unlink()
    except FileNotFoundError:
        pass


@dataclass(frozen=True)
class TrainingConfig:
    max_epochs: int = 20
    patience: int = 6
    batch_size: int = 64
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    min_delta: float = 1e-5
    gradient_clip_norm: float = 5.0
    num_workers: int = 0
    num_threads: int = 2
    device: Optional[str] = None

    def validate(self) -> None:
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be >= 1")
        if self.patience < 1:
            raise ValueError("patience must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer configuration.")
        if self.num_workers != 0:
            raise ValueError("For deterministic resume, num_workers must remain 0.")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive.")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)


def choose_device(config: TrainingConfig) -> torch.device:
    if config.device:
        device = torch.device(config.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _hash_id_list(values: Sequence[Any]) -> str:
    normalized = sorted(str(v) for v in values)
    return sha256_bytes(("\n".join(normalized) + "\n").encode("utf-8"))


def _hash_frame(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    subset = frame[list(columns)].copy()
    if "RID" in subset.columns:
        subset["RID"] = subset["RID"].astype(str)
        subset = subset.sort_values("RID", kind="mergesort")
    text = subset.to_csv(index=False, date_format="%Y-%m-%dT%H:%M:%S", lineterminator="\n")
    return sha256_bytes(text.encode("utf-8"))


def _hash_preprocessors(pack: Any) -> str:
    parts = []
    for modality, processor in sorted(getattr(pack, "processors", {}).items()):
        record = {"modality": modality, "columns": list(processor.columns)}
        for attr in ("medians_", "means_", "scales_", "all_missing_train_"):
            value = getattr(processor, attr, None)
            if value is not None:
                arr = np.asarray(value)
                record[attr] = {
                    "dtype": str(arr.dtype),
                    "shape": list(arr.shape),
                    "sha256": sha256_bytes(arr.tobytes()),
                }
        record["fit_n"] = getattr(processor, "fit_n_", None)
        record["fit_rids_sha256"] = getattr(processor, "audit", lambda: {})().get("fit_RID_sha256")
        parts.append(record)
    return sha256_bytes(canonical_json(parts).encode("utf-8"))


def initialize_run_directory(run_dir: Path, immutable_manifest: Mapping[str, Any],
                             resume: bool = True) -> str:
    run_dir = Path(run_dir)
    manifest_path = run_dir / "run_manifest.json"
    payload = dict(immutable_manifest)
    payload_hash = sha256_bytes(canonical_json(payload).encode("utf-8"))
    wrapped = {"immutable_manifest": payload, "immutable_manifest_sha256": payload_hash}

    if manifest_path.exists():
        if not resume:
            raise ExistingRunError(
                f"Run directory already exists. Use resume=True with the same manifest "
                f"or choose a new RUN_ID: {run_dir}"
            )
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("immutable_manifest_sha256") != payload_hash:
            raise ResumeMismatchError(
                "Existing run manifest differs from the requested configuration/data/code. "
                "Do not reuse this RUN_ID; create a new immutable RUN_ID."
            )
        return payload_hash

    if run_dir.exists() and any(run_dir.iterdir()):
        raise ExistingRunError(
            f"Run directory is non-empty but has no run_manifest.json: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(manifest_path, wrapped)
    return payload_hash


def _metric_ece(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    prediction = probability.argmax(1)
    confidence = probability.max(1)
    correct = prediction == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        selected = (confidence > left) & (confidence <= right)
        if selected.any():
            value += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return float(value)


def classification_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    eps = 1e-7
    y_true = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), eps, 1.0 - eps)
    probability = probability / probability.sum(axis=1, keepdims=True)
    prediction = probability.argmax(1)
    n_classes = probability.shape[1]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, prediction, labels=np.arange(n_classes),
        average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, prediction, labels=np.arange(n_classes))
    specificities = []
    for k in range(n_classes):
        tp = cm[k, k]
        fn = cm[k].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = cm.sum() - tp - fn - fp
        specificities.append(tn / max(tn + fp, 1))

    truth = np.eye(n_classes, dtype=float)[y_true]
    try:
        if n_classes == 2:
            roc_auc = roc_auc_score(y_true, probability[:, 1])
        else:
            roc_auc = roc_auc_score(truth, probability, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc = float("nan")
    try:
        if n_classes == 2:
            pr_auc = average_precision_score(truth[:, 1], probability[:, 1])
        else:
            pr_auc = average_precision_score(truth, probability, average="macro")
    except ValueError:
        pr_auc = float("nan")

    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "sensitivity_macro": float(recall),
        "specificity_macro": float(np.mean(specificities)),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier": float(np.mean(np.sum((truth - probability) ** 2, axis=1))),
        "log_loss": float(log_loss(y_true, probability, labels=np.arange(n_classes))),
        "ece": _metric_ece(y_true, probability),
    }


def build_training_plan(prepared: Mapping[tuple[str, str], Any],
                        feature_set_names: Sequence[str],
                        model_variants: Sequence[str],
                        seeds: Sequence[int],
                        tasks: Optional[Sequence[str]] = None) -> pd.DataFrame:
    task_names = list(tasks) if tasks is not None else sorted({k[0] for k in prepared})
    rows = []
    for task_name in task_names:
        for feature_set_name in feature_set_names:
            key = (task_name, feature_set_name)
            if key not in prepared:
                continue
            pack = prepared[key]
            split_info = {}
            class_ok = True
            for split in ("train", "validation", "test"):
                idx = pack.idx(split, eligible_only=True)
                y = pack.frame.iloc[idx][pack.target].to_numpy(dtype=int)
                counts = {str(int(k)): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
                split_info[split] = {
                    "n": int(len(idx)),
                    "class_counts": counts,
                    "rid_hash": _hash_id_list(pack.frame.iloc[idx]["RID"].astype(str).tolist()),
                }
                if split == "train" and set(np.unique(y).tolist()) != set(range(pack.n_classes)):
                    class_ok = False
            if not class_ok:
                raise RuntimeError(
                    f"{task_name}/{feature_set_name}: training split does not contain every target class."
                )
            dimensions = {m: int(pack.matrices[m].shape[1]) for m in pack.modalities}
            for model_variant in model_variants:
                for seed in seeds:
                    rows.append({
                        "task": task_name,
                        "feature_set": feature_set_name,
                        "model_variant": model_variant,
                        "seed": int(seed),
                        "modalities": ";".join(pack.modalities),
                        "dimensions": json.dumps(dimensions, sort_keys=True),
                        "train_n": split_info["train"]["n"],
                        "validation_n": split_info["validation"]["n"],
                        "test_n": split_info["test"]["n"],
                        "train_class_counts": json.dumps(split_info["train"]["class_counts"], sort_keys=True),
                        "validation_class_counts": json.dumps(split_info["validation"]["class_counts"], sort_keys=True),
                        "test_class_counts": json.dumps(split_info["test"]["class_counts"], sort_keys=True),
                        "train_RID_sha256": split_info["train"]["rid_hash"],
                        "validation_RID_sha256": split_info["validation"]["rid_hash"],
                        "test_RID_sha256": split_info["test"]["rid_hash"],
                        "status": "READY",
                    })
    return pd.DataFrame(rows)


def _make_loader(pack: Any, indices: np.ndarray, modalities: Sequence[str],
                 seed: int, epoch: int, config: TrainingConfig) -> DataLoader:
    tensors = [
        torch.as_tensor(pack.matrices[m][indices], dtype=torch.float32)
        for m in modalities
    ]
    tensors.append(torch.as_tensor(pack.masks[indices], dtype=torch.float32))
    tensors.append(torch.as_tensor(
        pack.frame.iloc[indices][pack.target].to_numpy(dtype=int), dtype=torch.long
    ))
    generator_seed = (int(seed) * 1_000_003 + int(epoch) * 9_176 + 13) % (2**63 - 1)
    generator = torch.Generator()
    generator.manual_seed(generator_seed)
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )


def _predict_split(model: nn.Module, pack: Any, indices: np.ndarray,
                   modalities: Sequence[str], device: torch.device,
                   batch_size: int) -> dict[str, Any]:
    model.eval()
    logits_all, y_all, rows_all = [], [], []
    gates_all = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            rows = indices[start:start + batch_size]
            inputs = {
                m: torch.as_tensor(pack.matrices[m][rows], dtype=torch.float32, device=device)
                for m in modalities
            }
            mask = torch.as_tensor(pack.masks[rows], dtype=torch.float32, device=device)
            y = pack.frame.iloc[rows][pack.target].to_numpy(dtype=int)
            output = model(inputs, mask, return_aux=True)
            logits, aux = output if isinstance(output, tuple) else (output, {})
            logits_all.append(logits.detach().cpu().numpy())
            y_all.append(y)
            rows_all.append(rows)
            if isinstance(aux, Mapping) and "gates" in aux:
                gates_all.append(aux["gates"].detach().cpu().numpy())

    if not rows_all:
        raise RuntimeError("Cannot predict an empty split.")
    logits = np.concatenate(logits_all, axis=0)
    y_true = np.concatenate(y_all, axis=0)
    rows = np.concatenate(rows_all, axis=0)
    logits_t = torch.as_tensor(logits, dtype=torch.float64)
    probability = torch.softmax(logits_t, dim=1).cpu().numpy()
    result = {"y": y_true, "probability": probability, "rows": rows}
    if gates_all:
        result["gates"] = np.concatenate(gates_all, axis=0)
    return result


def _prediction_frame(pack: Any, prediction: Mapping[str, Any], task_name: str,
                      feature_set_name: str, model_variant: str, seed: int,
                      run_id: str, split: str, modalities: Sequence[str]) -> pd.DataFrame:
    rows = prediction["rows"]
    source = pack.frame.iloc[rows]
    frame = pd.DataFrame({
        "run_id": run_id,
        "task": task_name,
        "feature_set": feature_set_name,
        "model": model_variant,
        "seed": int(seed),
        "split": split,
        "row_index": rows.astype(int),
        "RID": source["RID"].astype(str).to_numpy(),
        "INDEX_DATE": pd.to_datetime(source["INDEX_DATE"], errors="coerce").dt.strftime("%Y-%m-%d").to_numpy(),
        "y_true": prediction["y"].astype(int),
        "y_pred": prediction["probability"].argmax(axis=1).astype(int),
    })
    for j in range(prediction["probability"].shape[1]):
        frame[f"p_class_{j}"] = prediction["probability"][:, j]
    for j, modality in enumerate(pack.modalities):
        frame[f"available_{modality}"] = pack.masks[rows, j].astype(int)
    if "gates" in prediction:
        for j, modality in enumerate(modalities):
            frame[f"gate_{modality}"] = prediction["gates"][:, j]
    return frame


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def run_training_job(
    pack: Any,
    task_name: str,
    feature_set_name: str,
    model_variant: str,
    seed: int,
    run_dir: Path,
    run_manifest_sha256: str,
    model_factory: Callable[[str, Mapping[str, int], int], nn.Module],
    config: Optional[TrainingConfig] = None,
    input_hashes: Optional[Mapping[str, str]] = None,
    model_code_sha256: str = "",
    resume: bool = True,
    stop_after_epoch: Optional[int] = None,
) -> dict[str, Any]:
    config = config or TrainingConfig()
    config.validate()
    run_dir = Path(run_dir)
    device = choose_device(config)
    if config.num_threads:
        torch.set_num_threads(config.num_threads)

    modalities = list(pack.modalities)
    dimensions = {m: int(pack.matrices[m].shape[1]) for m in modalities}
    train_idx = pack.idx("train", eligible_only=True)
    val_idx = pack.idx("validation", eligible_only=True)
    test_idx = pack.idx("test", eligible_only=True)
    if not len(train_idx) or not len(val_idx) or not len(test_idx):
        raise RuntimeError(f"{task_name}/{feature_set_name}: empty eligible split.")
    y_train = pack.frame.iloc[train_idx][pack.target].to_numpy(dtype=int)
    observed_classes = set(np.unique(y_train).tolist())
    if observed_classes != set(range(pack.n_classes)):
        raise RuntimeError(
            f"{task_name}/{feature_set_name}: training class coverage is incomplete: {observed_classes}"
        )

    task_columns = ["RID", "split", pack.target, "INDEX_DATE"]
    frame_hash = _hash_frame(pack.frame, task_columns)
    split_hash = _hash_frame(pack.frame, ["RID", "split"])
    preprocessor_hash = _hash_preprocessors(pack)
    feature_manifest = {
        m: list(getattr(pack, "features_by_modality", {}).get(m, []))
        for m in modalities
    }
    train_rids = pack.frame.iloc[train_idx]["RID"].astype(str).tolist()
    val_rids = pack.frame.iloc[val_idx]["RID"].astype(str).tolist()
    test_rids = pack.frame.iloc[test_idx]["RID"].astype(str).tolist()

    job_key = "__".join([
        re.sub(r"[^A-Za-z0-9_.-]+", "-", str(task_name)),
        re.sub(r"[^A-Za-z0-9_.-]+", "-", str(feature_set_name)),
        re.sub(r"[^A-Za-z0-9_.-]+", "-", str(model_variant)),
        f"seed{int(seed)}",
    ])
    job_dir = run_dir / "jobs" / job_key
    lock_path = job_dir / ".job.lock"
    job_dir_existed = job_dir.exists()
    if job_dir_existed and not resume:
        raise ExistingRunError(f"Job directory already exists and resume=False: {job_dir}")
    job_dir.mkdir(parents=True, exist_ok=True)
    _acquire_lock(lock_path)

    started = time.time()
    try:
        job_manifest = {
            "schema_version": 1,
            "job_key": job_key,
            "run_manifest_sha256": run_manifest_sha256,
            "task": task_name,
            "feature_set": feature_set_name,
            "model_variant": model_variant,
            "seed": int(seed),
            "modalities": modalities,
            "dimensions": dimensions,
            "feature_manifest": feature_manifest,
            "training_config": config.to_dict(),
            "device": str(device),
            "input_hashes": dict(input_hashes or {}),
            "task_frame_sha256": frame_hash,
            "split_sha256": split_hash,
            "preprocessor_sha256": preprocessor_hash,
            "train_n": int(len(train_idx)),
            "validation_n": int(len(val_idx)),
            "test_n": int(len(test_idx)),
            "train_RID_sha256": _hash_id_list(train_rids),
            "validation_RID_sha256": _hash_id_list(val_rids),
            "test_RID_sha256": _hash_id_list(test_rids),
            "model_code_sha256": model_code_sha256,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "python_version": platform.python_version(),
        }
        job_manifest_text = canonical_json(job_manifest)
        job_manifest_hash = sha256_bytes(job_manifest_text.encode("utf-8"))
        manifest_path = job_dir / "job_manifest.json"

        if manifest_path.exists():
            if not resume:
                raise ExistingRunError(f"Job already exists and resume=False: {job_dir}")
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("job_manifest_sha256") != job_manifest_hash:
                raise ResumeMismatchError(
                    f"Job manifest mismatch for {job_key}. Use a new RUN_ID or restore exact inputs/config."
                )
        else:
            other_files = [p.name for p in job_dir.iterdir() if p.name != lock_path.name]
            if other_files:
                raise ExistingRunError(
                    f"Job directory contains files but no manifest: {job_dir}: {other_files}"
                )
            atomic_write_json(manifest_path, {
                "job_manifest": job_manifest,
                "job_manifest_sha256": job_manifest_hash,
            })

        status_path = job_dir / "status.json"
        last_path = job_dir / "last.pt"
        best_path = job_dir / "best.pt"

        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") == "completed":
                expected_outputs = status.get("output_sha256", {})
                for rel, expected in expected_outputs.items():
                    path = job_dir / rel
                    if not path.is_file() or sha256_file(path) != expected:
                        raise ResumeMismatchError(
                            f"Completed output hash mismatch; refusing silent repair: {path}"
                        )
                return {
                    "job_key": job_key,
                    "status": "SKIPPED_COMPLETE",
                    "job_dir": str(job_dir),
                    "best_epoch": status.get("best_epoch"),
                    "test_metrics": status.get("test_metrics", {}),
                }

        seed_everything(seed)
        model = model_factory(model_variant, dimensions, pack.n_classes).to(device)
        counts = np.bincount(y_train, minlength=pack.n_classes)
        class_weights = counts.sum() / np.maximum(counts, 1)
        class_weights = class_weights / class_weights.mean()
        class_weights_t = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_t)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=config.scheduler_factor,
            patience=config.scheduler_patience
        )

        history = []
        best_state = None
        best_score = -np.inf
        best_epoch = -1
        stale = 0
        start_epoch = 1
        elapsed_before = 0.0

        if last_path.exists():
            if not resume:
                raise ExistingRunError(f"Checkpoint exists and resume=False: {last_path}")
            resume_status = (
                json.loads(status_path.read_text(encoding="utf-8"))
                if status_path.exists() else {}
            )
            checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
            if checkpoint.get("job_manifest_sha256") != job_manifest_hash:
                raise ResumeMismatchError(f"Checkpoint manifest mismatch: {last_path}")
            checkpoint_epoch = int(checkpoint["epoch"])
            status_epoch = resume_status.get("last_completed_epoch")
            if status_epoch is not None and checkpoint_epoch < int(status_epoch):
                raise ResumeMismatchError(
                    "last.pt is older than status.json; refusing to resume from stale state."
                )
            if (
                status_epoch is not None
                and checkpoint_epoch == int(status_epoch)
                and resume_status.get("last_checkpoint_sha256")
                and sha256_file(last_path) != resume_status["last_checkpoint_sha256"]
            ):
                raise ResumeMismatchError(
                    "last.pt hash differs from status.json at the same epoch."
                )
            if best_path.exists():
                best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
                if best_checkpoint.get("job_manifest_sha256") != job_manifest_hash:
                    raise ResumeMismatchError(f"Best-checkpoint manifest mismatch: {best_path}")
                if (
                    resume_status.get("best_epoch") is not None
                    and int(best_checkpoint.get("best_epoch", -1)) == int(resume_status["best_epoch"])
                    and resume_status.get("best_checkpoint_sha256")
                    and sha256_file(best_path) != resume_status["best_checkpoint_sha256"]
                ):
                    raise ResumeMismatchError(
                        "best.pt hash differs from status.json for the recorded best epoch."
                    )
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            _move_optimizer_state(optimizer, device)
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            history = list(checkpoint["history"])
            best_state = checkpoint["best_state"]
            best_score = float(checkpoint["best_score"])
            best_epoch = int(checkpoint["best_epoch"])
            stale = int(checkpoint["stale"])
            start_epoch = int(checkpoint["epoch"]) + 1
            elapsed_before = float(checkpoint.get("elapsed_sec", 0.0))
            restore_rng_state(checkpoint["rng_state"])
            if not best_path.exists() and best_state is not None:
                atomic_torch_save(best_path, {
                    "job_manifest_sha256": job_manifest_hash,
                    "model_state": best_state,
                    "dimensions": dimensions,
                    "n_classes": int(pack.n_classes),
                    "model_variant": model_variant,
                    "modalities": modalities,
                    "seed": int(seed),
                    "best_epoch": int(best_epoch),
                    "best_validation_macro_f1": float(best_score),
                })
        elif status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") == "running":
                raise ResumeMismatchError(
                    "Status says running but last.pt is missing. Refusing to initialize a fresh model."
                )

        if best_state is None and start_epoch > 1:
            raise ResumeMismatchError("Resume checkpoint lacks best_state.")

        epoch_range = range(start_epoch, config.max_epochs + 1) if stale < config.patience else range(0)
        for epoch in epoch_range:
            model.train()
            loader = _make_loader(pack, train_idx, modalities, seed, epoch, config)
            total_loss = 0.0
            total_seen = 0
            for batch in loader:
                feature_tensors = batch[:len(modalities)]
                mask_t = batch[len(modalities)].to(device, non_blocking=True)
                y_t = batch[len(modalities) + 1].to(device, non_blocking=True)
                inputs = {
                    modality: feature_tensors[j].to(device, non_blocking=True)
                    for j, modality in enumerate(modalities)
                }
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs, mask_t)
                loss = criterion(logits, y_t)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.step()
                batch_n = int(y_t.shape[0])
                total_loss += float(loss.detach().cpu().item()) * batch_n
                total_seen += batch_n

            train_loss = total_loss / max(total_seen, 1)
            validation = _predict_split(model, pack, val_idx, modalities, device, config.batch_size)
            validation_metrics = classification_metrics(
                validation["y"], validation["probability"]
            )
            score = float(validation_metrics["f1_macro"])
            scheduler.step(score)

            if score > best_score + config.min_delta:
                best_score = score
                best_epoch = epoch
                stale = 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                atomic_torch_save(best_path, {
                    "job_manifest_sha256": job_manifest_hash,
                    "model_state": best_state,
                    "dimensions": dimensions,
                    "n_classes": int(pack.n_classes),
                    "model_variant": model_variant,
                    "modalities": modalities,
                    "seed": int(seed),
                    "best_epoch": int(best_epoch),
                    "best_validation_macro_f1": float(best_score),
                })
            else:
                stale += 1

            history_row = {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "validation_macro_f1": float(score),
                "validation_roc_auc": float(validation_metrics["roc_auc"]),
                "validation_brier": float(validation_metrics["brier"]),
                "validation_ece": float(validation_metrics["ece"]),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "best_validation_macro_f1": float(best_score),
                "best_epoch": int(best_epoch),
                "stale_epochs": int(stale),
            }
            history.append(history_row)

            last_checkpoint = {
                "job_manifest_sha256": job_manifest_hash,
                "epoch": int(epoch),
                "model_state": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_state": best_state,
                "best_score": float(best_score),
                "best_epoch": int(best_epoch),
                "stale": int(stale),
                "history": history,
                "rng_state": capture_rng_state(),
                "elapsed_sec": float(elapsed_before + (time.time() - started)),
            }
            atomic_torch_save(last_path, last_checkpoint)
            atomic_write_csv(job_dir / "history.csv", pd.DataFrame(history))
            atomic_write_json(status_path, {
                "status": "running",
                "last_completed_epoch": int(epoch),
                "best_epoch": int(best_epoch),
                "best_validation_macro_f1": float(best_score),
                "last_checkpoint_sha256": sha256_file(last_path),
                "best_checkpoint_sha256": sha256_file(best_path) if best_path.exists() else None,
            })

            if stop_after_epoch is not None and epoch >= int(stop_after_epoch):
                return {
                    "job_key": job_key,
                    "status": "PAUSED_BY_TEST_HOOK",
                    "last_completed_epoch": int(epoch),
                    "job_dir": str(job_dir),
                }
            if stale >= config.patience:
                break

        if best_state is None or best_epoch < 1:
            raise RuntimeError(f"No valid best checkpoint was produced for {job_key}.")

        model.load_state_dict(best_state)
        model.to(device)
        model.eval()

        validation_best = _predict_split(model, pack, val_idx, modalities, device, config.batch_size)
        test_best = _predict_split(model, pack, test_idx, modalities, device, config.batch_size)
        validation_metrics = classification_metrics(
            validation_best["y"], validation_best["probability"]
        )
        test_metrics = classification_metrics(
            test_best["y"], test_best["probability"]
        )

        run_id = run_dir.name
        val_frame = _prediction_frame(
            pack, validation_best, task_name, feature_set_name, model_variant,
            seed, run_id, "validation", modalities
        )
        test_frame = _prediction_frame(
            pack, test_best, task_name, feature_set_name, model_variant,
            seed, run_id, "test", modalities
        )
        atomic_write_csv(job_dir / "predictions_validation.csv", val_frame)
        atomic_write_csv(job_dir / "predictions_test.csv", test_frame)

        metrics_payload = {
            "run_id": run_id,
            "job_key": job_key,
            "task": task_name,
            "feature_set": feature_set_name,
            "model": model_variant,
            "seed": int(seed),
            "best_epoch": int(best_epoch),
            "best_validation_macro_f1": float(best_score),
            "train_n": int(len(train_idx)),
            "validation_n": int(len(val_idx)),
            "test_n": int(len(test_idx)),
            "epochs_completed": int(len(history)),
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "training_config": config.to_dict(),
            "input_hashes": dict(input_hashes or {}),
            "preprocessor_sha256": preprocessor_hash,
            "split_sha256": split_hash,
            "model_code_sha256": model_code_sha256,
            "elapsed_sec": float(elapsed_before + (time.time() - started)),
        }
        atomic_write_json(job_dir / "metrics.json", metrics_payload)

        output_names = [
            "best.pt", "last.pt", "history.csv", "predictions_validation.csv",
            "predictions_test.csv", "metrics.json"
        ]
        output_hashes = {
            name: sha256_file(job_dir / name)
            for name in output_names if (job_dir / name).is_file()
        }
        final_status = {
            "status": "completed",
            "last_completed_epoch": int(history[-1]["epoch"]),
            "best_epoch": int(best_epoch),
            "best_validation_macro_f1": float(best_score),
            "test_metrics": test_metrics,
            "output_sha256": output_hashes,
        }
        atomic_write_json(status_path, final_status)
        return {
            "job_key": job_key,
            "status": "COMPLETED",
            "best_epoch": int(best_epoch),
            "best_validation_macro_f1": float(best_score),
            "test_metrics": test_metrics,
            "job_dir": str(job_dir),
        }
    finally:
        _release_lock(lock_path)


def run_training_plan(
    prepared: Mapping[tuple[str, str], Any],
    plan: pd.DataFrame,
    run_dir: Path,
    run_manifest_sha256: str,
    model_factory: Callable[[str, Mapping[str, int], int], nn.Module],
    config: Optional[TrainingConfig] = None,
    input_hashes: Optional[Mapping[str, str]] = None,
    model_code_sha256: str = "",
    resume: bool = True,
    max_jobs: Optional[int] = None,
) -> pd.DataFrame:
    results = []
    jobs = plan.to_dict(orient="records")
    if max_jobs is not None:
        jobs = jobs[:int(max_jobs)]
    for position, job in enumerate(jobs, start=1):
        key = (job["task"], job["feature_set"])
        result = run_training_job(
            pack=prepared[key],
            task_name=job["task"],
            feature_set_name=job["feature_set"],
            model_variant=job["model_variant"],
            seed=int(job["seed"]),
            run_dir=run_dir,
            run_manifest_sha256=run_manifest_sha256,
            model_factory=model_factory,
            config=config,
            input_hashes=input_hashes,
            model_code_sha256=model_code_sha256,
            resume=resume,
        )
        result["plan_position"] = position
        results.append(result)
        print(
            f"[{position}/{len(jobs)}] {result['status']} | {result['job_key']}",
            flush=True
        )
    return pd.DataFrame(results)
