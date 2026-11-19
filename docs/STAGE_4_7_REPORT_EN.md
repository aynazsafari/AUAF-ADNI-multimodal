# AUAF Stage 4.7 — Statistical Validation and Multivariable Circularity Audit

## Execution status
Completed on the final Stage 4.4 as-of-index cohorts and Stage 4.5 participant-level predictions. No participant resplitting was performed. The five seeds are repeated initializations on the same frozen split, not five independent folds.

## Statistical validation
- Audited 1,050 prediction files (525 full-test and 525 matched-complete-case); all rosters, labels, probability normalization and finite-value checks passed.
- Recomputed 12 metrics and generated 1,000-replicate stratified participant-bootstrap 95% CIs for all task/model/cohort/missingness conditions.
- Reproduced the archived Stage 4.5 metrics to numerical precision except for small ROC-AUC/PR-AUC differences in ExtraTrees. The largest was 0.000306 ROC-AUC and 0.000101 PR-AUC in one matched multiclass condition; this is explicitly recorded.
- Ran 1,440 paired AUAF-vs-key-baseline comparisons and 2,016 paired missingness-vs-0% effects, with BH-FDR. No missingness effect survived FDR. At 0% missingness, none of the four key metrics (F1, ROC-AUC, Brier, ECE) showed an FDR-significant AUAF advantage over the key comparators.
- Temperature scaling was fitted on validation only and assessed at 0% and 100% additional biomarker missingness. No calibration change survived BH-FDR. Youden thresholds for the two binary tasks were selected on validation only.

## Multivariable circularity sensitivity
Twenty raw predictors from CDR, FAQ, MMSE and MoCA-derived blocks were removed before preprocessing. The effective input set fell from 32 predictors (29 clinical/cognitive + 3 CSF biomarkers) to 12 (9 clinical/cognitive + 3 CSF biomarkers). Fifteen AUAF and fifteen best-tree-baseline models were retrained on the frozen training participants only. Best tree baselines were selected on validation macro-F1: XGBoost for multiclass, CatBoost for binary diagnosis and MCI conversion.

The strict sensitivity analysis showed substantial performance reductions on the full test set, consistent with the high predictive contribution of diagnosis-proximal clinical scales. This is a sensitivity analysis, not proof that all information leakage has been eliminated. Matched test sizes were 81, 67 and 29, so the conversion comparison is especially imprecise.

## Conformal audit and provenance
Manuscript Table 6 aggregate RAPS-lite values were transcribed and checked for internal consistency. Participant-level prediction sets, validation scores/q-hats and the exact RAPS-lite implementation/configuration were not present in the Stage 4.4/4.5 artifacts, so set-level coverage could not be recomputed. The reported multiclass abstention rate (0.0000) differs from the complement of its singleton rate (0.4211); the manuscript definition needs clarification.

Source table, instrument, feature description, missing-code handling and temporal alignment were taken from the Stage 4.4 feature manifest. Feature-specific raw source-row IDs/visit codes are not preserved in the integrated predictor matrix. The final Stage 4.5 model has two modalities only (clinical/cognitive and fluid biomarkers); no imaging modality was present.

## Main limitations
ADNI-only evaluation, no external validation, small matched subsets, no participant-level conformal artifacts, and incomplete feature-level source-row provenance. Do not describe the model as leakage-free or clinically deployment-ready.
