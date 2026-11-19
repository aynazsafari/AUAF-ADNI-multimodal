from __future__ import annotations

import json
import shutil
import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from auaf_safe_training_engine import (
    TrainingConfig,
    initialize_run_directory,
    run_training_job,
    ResumeMismatchError,
    ExistingRunError,
    sha256_file,
)


class _ToyPreparedTask:
    def __init__(self, seed: int = 19):
        rng = np.random.default_rng(seed)
        n = 60
        split = np.array(["train"] * 36 + ["validation"] * 12 + ["test"] * 12)
        y = np.concatenate([
            np.array([0, 1] * 18),
            np.array([0, 1] * 6),
            np.array([0, 1] * 6),
        ])
        self.frame = pd.DataFrame({
            "RID": np.arange(9000, 9000 + n),
            "INDEX_DATE": pd.date_range("2020-01-01", periods=n, freq="D"),
            "split": split,
            "target": y,
        })
        self.target = "target"
        self.n_classes = 2
        self.modalities = ["clinical_cognitive", "fluid_biomarker"]
        self.matrices = {
            self.modalities[0]: rng.normal(size=(n, 5)).astype(np.float32),
            self.modalities[1]: rng.normal(size=(n, 3)).astype(np.float32),
        }
        self.masks = np.ones((n, 2), dtype=np.float32)
        self.masks[::3, 1] = 0
        self.masks[1::4, 0] = 0
        self.masks[self.masks.sum(axis=1) == 0, 0] = 1
        self.features_by_modality = {
            self.modalities[0]: [f"c{i}" for i in range(5)],
            self.modalities[1]: [f"b{i}" for i in range(3)],
        }
        self.processors = {}

    def idx(self, split: str, eligible_only: bool = False) -> np.ndarray:
        idx = np.flatnonzero(self.frame["split"].astype(str).eq(split).to_numpy())
        if eligible_only:
            idx = idx[self.masks[idx].sum(axis=1) > 0]
        return idx


class _TinyFusionModel(nn.Module):
    def __init__(self, dimensions, n_classes):
        super().__init__()
        self.modalities = list(dimensions)
        width = sum(dimensions.values()) + len(dimensions)
        self.net = nn.Sequential(
            nn.Linear(width, 16), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(16, n_classes),
        )

    def forward(self, inputs, mask, return_aux=False):
        x = torch.cat([inputs[m] for m in self.modalities] + [mask], dim=1)
        logits = self.net(x)
        return (logits, {}) if return_aux else logits


def _factory(variant, dimensions, n_classes):
    if variant != "TinyTest":
        raise KeyError(variant)
    return _TinyFusionModel(dimensions, n_classes)


def _state_hash(path: Path) -> str:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    h = hashlib.sha256()
    for key in sorted(checkpoint["model_state"]):
        h.update(key.encode("utf-8"))
        h.update(checkpoint["model_state"][key].contiguous().numpy().tobytes())
    return h.hexdigest()


def run_qualification_tests(output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "safe_resume_qualification.json"
    work_root = Path(tempfile.mkdtemp(prefix="auaf_safe_resume_"))

    pack = _ToyPreparedTask()
    config = TrainingConfig(
        max_epochs=3, patience=6, batch_size=8,
        learning_rate=8e-4, weight_decay=1e-4,
        scheduler_patience=2, num_threads=1, device="cpu",
    )
    manifest = {
        "test_name": "deterministic safe-resume qualification",
        "test_data": "synthetic only; not research data",
        "training_config": config.to_dict(),
        "seed": 42,
        "model": "TinyTest",
    }

    resumed_dir = work_root / "resume_path"
    resumed_hash = initialize_run_directory(resumed_dir, manifest, resume=False)
    paused = run_training_job(
        pack, "toy_task", "toy_features", "TinyTest", 42,
        resumed_dir, resumed_hash, _factory, config,
        input_hashes={"synthetic_input": "fixed"},
        model_code_sha256="tiny-model-test-v1",
        resume=True, stop_after_epoch=1,
    )
    assert paused["status"] == "PAUSED_BY_TEST_HOOK"
    assert paused["last_completed_epoch"] == 1

    resumed = run_training_job(
        pack, "toy_task", "toy_features", "TinyTest", 42,
        resumed_dir, resumed_hash, _factory, config,
        input_hashes={"synthetic_input": "fixed"},
        model_code_sha256="tiny-model-test-v1",
        resume=True,
    )
    assert resumed["status"] == "COMPLETED"

    full_dir = work_root / "uninterrupted_path"
    full_hash = initialize_run_directory(full_dir, manifest, resume=False)
    full = run_training_job(
        pack, "toy_task", "toy_features", "TinyTest", 42,
        full_dir, full_hash, _factory, config,
        input_hashes={"synthetic_input": "fixed"},
        model_code_sha256="tiny-model-test-v1",
        resume=True,
    )
    assert full["status"] == "COMPLETED"

    resumed_best = resumed_dir / "jobs/toy_task__toy_features__TinyTest__seed42/best.pt"
    full_best = full_dir / "jobs/toy_task__toy_features__TinyTest__seed42/best.pt"
    resumed_hash_state = _state_hash(resumed_best)
    full_hash_state = _state_hash(full_best)
    assert resumed_hash_state == full_hash_state, "Resumed and uninterrupted model states differ."

    skipped = run_training_job(
        pack, "toy_task", "toy_features", "TinyTest", 42,
        resumed_dir, resumed_hash, _factory, config,
        input_hashes={"synthetic_input": "fixed"},
        model_code_sha256="tiny-model-test-v1",
        resume=True,
    )
    assert skipped["status"] == "SKIPPED_COMPLETE"

    mismatch_rejected = False
    bad_config = TrainingConfig(
        max_epochs=4, patience=6, batch_size=8,
        learning_rate=8e-4, weight_decay=1e-4,
        scheduler_patience=2, num_threads=1, device="cpu",
    )
    try:
        run_training_job(
            pack, "toy_task", "toy_features", "TinyTest", 42,
            resumed_dir, resumed_hash, _factory, bad_config,
            input_hashes={"synthetic_input": "fixed"},
            model_code_sha256="tiny-model-test-v1",
            resume=True,
        )
    except ResumeMismatchError:
        mismatch_rejected = True
    assert mismatch_rejected

    no_overwrite_rejected = False
    try:
        run_training_job(
            pack, "toy_task", "toy_features", "TinyTest", 42,
            resumed_dir, resumed_hash, _factory, config,
            input_hashes={"synthetic_input": "fixed"},
            model_code_sha256="tiny-model-test-v1",
            resume=False,
        )
    except ExistingRunError:
        no_overwrite_rejected = True
    assert no_overwrite_rejected

    run_manifest_mismatch_rejected = False
    try:
        initialize_run_directory(
            resumed_dir, {**manifest, "changed": True}, resume=True
        )
    except ResumeMismatchError:
        run_manifest_mismatch_rejected = True
    assert run_manifest_mismatch_rejected

    report = {
        "status": "PASS",
        "test_data": "synthetic only; no research participants or labels used",
        "checks": {
            "interrupted_after_epoch_1": True,
            "resume_completed": True,
            "resumed_equals_uninterrupted_best_state": True,
            "best_state_sha256": resumed_hash_state,
            "completed_job_skipped_on_repeat": True,
            "config_mismatch_rejected": mismatch_rejected,
            "existing_job_overwrite_rejected": no_overwrite_rejected,
            "run_manifest_mismatch_rejected": run_manifest_mismatch_rejected,
        },
        "resumed_status": resumed["status"],
        "uninterrupted_status": full["status"],
        "completed_repeat_status": skipped["status"],
    }

    # Remove all synthetic checkpoints after verification; retain only the compact report.
    shutil.rmtree(work_root, ignore_errors=True)
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if report_path.exists():
        existing = report_path.read_text(encoding="utf-8")
        if existing != serialized:
            raise RuntimeError("Existing safe-resume report differs; refusing to overwrite it.")
    else:
        report_path.write_text(serialized, encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./outputs/safe_resume_qualification")
    args = parser.parse_args()
    result = run_qualification_tests(Path(args.output_dir))
    print(json.dumps(result, indent=2))
