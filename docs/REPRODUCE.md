# Reproducing Stage 4.6

The package includes the exact frozen Stage 4.5 input snapshot under `data/stage4_5_frozen_inputs/`.

From the package root:

```bash
python code/test_auaf_ablation_factory.py

python code/train_stage4_6_ablation.py \
  --data-dir data/stage4_5_frozen_inputs \
  --reference-dir data/stage4_5_frozen_inputs/reference_splits \
  --output-dir outputs/stage4_6_ablation_finalcohort_20261116_v1
```

The training script uses a fixed participant split, five initialization seeds, train-only preprocessing, validation Macro-F1 checkpoint selection, and resumable per-job checkpoints. Re-running the command against the same output directory verifies and skips completed jobs rather than overwriting or retraining them.

Primary artifacts:
- `run_manifest.json`
- `outputs/stage4_6_ablation_finalcohort_20261116_v1/execution_results.csv`
- `outputs/stage4_6_ablation_finalcohort_20261116_v1/jobs/*/best.pt`
- `outputs/stage4_6_ablation_finalcohort_20261116_v1/jobs/*/predictions_test.csv`
- `reports/seed_level_metrics.csv`
- `reports/summary_mean_sd.csv`
- `reports/paired_AUAF_differences_descriptive.csv`
- `reports/integrity_audit.csv`
- `FINAL_AUDIT_FA.md`

The `No Both Gates / Uniform Gate` condition is implemented by removing both learned gate branches and assigning equal weights to available modalities. No separate eighth run is implied.
