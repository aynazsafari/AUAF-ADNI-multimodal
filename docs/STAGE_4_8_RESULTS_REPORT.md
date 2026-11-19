# AUAF Stage 4.8 — Execution report

**Status:** Core Stage 4.8 analyses executed; participant-level selection-bias audit for the 842 excluded MCI cases remains pending the complete 1,753-row index file.

## Integrity and reproducibility
The frozen cohorts were retained (3,700 three-class diagnosis; 3,700 binary diagnosis; 911 MCI conversion). Fifteen diagnostic index records without a date were preserved. All 105 Stage 4.5 checkpoints passed SHA-256 verification. All 105 deterministic held-out test prediction files reproduced the Stage 4.5 archived `miss000` predictions on RID roster and labels, with maximum absolute probability difference 9.3561e-08. No retraining or resplitting occurred. The Stage 4.7 packaged CSV hashes differ from the original Stage 4.5 input hashes, so byte-level file identity is not claimed; the model-output compatibility gate passed.

## Main results
- **MC dropout:** 30 passes. Mean entropy: 0.4031 (three-class), 0.2238 (binary), 0.5465 (conversion). Mean probability SD: 0.0485, 0.0464, 0.0802.
- **Gate weights (clinical/cognitive vs fluid biomarker):** 0.786/0.214 (three-class), 0.791/0.209 (binary), 0.552/0.448 (conversion). Descriptive only; not causal importance.
- **Calibration:** AUAF Brier/ECE pre/post temperature scaling: 0.1978/0.0299 to 0.1987/0.0360 (three-class); 0.2040/0.0744 to 0.1857/0.0410 (binary); 0.3639/0.0787 to 0.3625/0.0824 (conversion). Effects were task-dependent.
- **RAPS-lite primary:** AUAF coverage 0.9230 (95% CI 0.9058–0.9385), mean set size 1.3973 (three-class); 0.9516 (0.9431–0.9607), mean set size 1.5981 (binary); 1.0000, mean set size 2.0000 (conversion). All conversion prediction sets contained both classes (100% full-set; 0% singleton), so coverage was conservative but non-selective. No formal distribution-free guarantee is claimed.
- **Paired comparisons:** 126 comparisons, 500 participant-bootstrap resamples, global Benjamini–Hochberg adjustment; 23 remained significant at q<0.05 (11 three-class, 7 binary, 5 conversion). Results do not establish blanket AUAF superiority; CatBoost and XGBoost were better on selected metrics.
- **Modality occlusion:** matched complete-case n=81, 67, and 29 for the three tasks. Both modalities did not show a consistent discrimination advantage over clinical/cognitive-only input; binary Brier improved by -0.0713 (95% CI -0.1316 to -0.0217).
- **CatBoost TreeSHAP:** 32 verified base predictors plus missingness indicators (64 transformed columns). SHAP is CatBoost-only and non-causal.

## Bootstrap settings
Paired model comparisons: 500; descriptive metric confidence intervals: 200 (matching the manuscript's original protocol); RAPS confidence intervals: 500; modality-occlusion intervals: 10,000.

## Pending item
The MCI conversion cohort was selected from 1,753 first-MCI participants: 911 eligible and 842 excluded for insufficient follow-up. The individual-level selection-bias audit is not run because `mci_first_index_all_1753.csv` could not be retrieved as raw bytes from the Library in this session. User authorization is confirmed; upload the file directly to complete this final audit.

## Limitations
ADNI-only retrospective evaluation; no independent external, prospective, or multi-center validation. Validation data were also used for early stopping/model selection and temperature fitting; RAPS coverage is therefore reported empirically, without a formal distribution-free guarantee. Gate weights, SHAP values, and occlusion effects are not causal evidence.
