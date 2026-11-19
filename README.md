<div align="center">

# 🧠 AUAF — Adaptive Uncertainty-Aware Multimodal Fusion
### Alzheimer's Disease Diagnosis and Early MCI-to-AD Conversion (ADNI)

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![Data](https://img.shields.io/badge/Data-ADNI%20(DUA%20required)-orange)
![Status](https://img.shields.io/badge/Status-Research--stage-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

**Code, configuration, manifests and complete aggregate outputs for the manuscript**
*“Adaptive Uncertainty-Aware Multimodal Fusion for Alzheimer's Disease Diagnosis and Early MCI-to-AD Conversion.”*

</div>

---

> ⚠️ **Research-stage tabular framework.** AUAF uses only clinical/cognitive variables and CSF biomarkers (Aβ40, Aβ42, PTAU), has no imaging, and has **not** been externally validated. It is **not** a clinical decision-support tool.

## 📌 Overview

| | |
|---|---|
| **Tasks** | CN / MCI / AD (3-class) · AD vs non-AD · 36-month MCI→AD conversion |
| **Cohorts** | 3,700 index-diagnosis participants · 911 first-observed-MCI participants |
| **Inputs** | 29 clinical/cognitive + 3 CSF biomarkers, aligned to the index date |
| **Models** | AUAF · AttentionFusion · LateFusion · EarlyFusion · XGBoost · CatBoost · ExtraTrees |
| **Protocol** | One frozen subject-disjoint split (64/16/20) × 5 initialisation seeds |
| **Inference** | 1,000-resample participant bootstrap · paired tests · Benjamini–Hochberg FDR |

### Headline results (test set, mean of 5 seeds)

| Task | AUAF ROC-AUC | Best baseline ROC-AUC |
|---|---|---|
| CN/MCI/AD | 0.956 | CatBoost 0.963 |
| AD vs non-AD | 0.930 | ExtraTrees 0.940 |
| MCI→AD (36 mo) | 0.799 | CatBoost 0.829 |

**Key findings, reported transparently:**
- AUAF has **no** FDR-significant discrimination advantage over AttentionFusion or the gradient-boosted trees.
- Under controlled biomarker removal (0–100%), **no** model changed significantly.
- The learned mask tokens are **inert** in this implementation: removing them gives identical predictions.
- Removing 20 diagnosis-proximal variables (CDR, FAQ, MMSE, MoCA) drops 3-class ROC-AUC from **0.956 to 0.671**.

---

## 🗂️ Repository structure

```
AUAF-ADNI/
├── code/
│   ├── stage4_4_temporal_alignment/     # index-date (as-of) cohort & feature reconstruction
│   ├── stage4_5_training_missingness/   # model factory, safe training engine, missing-modality benchmark
│   ├── stage4_6_ablation/               # 7-variant AUAF component ablation (105 runs)
│   ├── stage4_7_statistics_circularity/ # bootstrap CIs, paired tests, FDR, circularity retraining
│   ├── stage4_8_reliability/            # calibration, MC dropout, RAPS-lite conformal, SHAP, occlusion
│   └── manuscript_figures/              # selection-bias audit + manuscript figure scripts
├── notebooks/                           # source notebooks (outputs cleared)
├── results/                             # ALL aggregate tables used in the paper (CSV/XLSX)
├── figures/                             # ALL figures (PNG/PDF)
├── manifests/                           # run manifests, feature inventory, hash & roster audits
├── environment/                         # requirements files and exact package versions
├── docs/                                # stage reports (EN/FA) and reproduction notes
├── DATA_ACCESS.md                       # how to obtain ADNI data
├── CITATION.cff
└── LICENSE
```

## 🔁 Pipeline

```
ADNI tables ──► 4.3 Cohort/index freeze ──► 4.4 Temporal (as-of) alignment ──► 4.5 Training (7 models × 5 seeds)
                                                                                      │
          4.8 Calibration · MC-dropout · Conformal · SHAP · Occlusion ◄── 4.7 Statistics & circularity ◄── 4.6 Ablation
```

## 📊 Where each manuscript result lives

| Manuscript item | File |
|---|---|
| Table 4 – selection bias (911 vs 842) | `results/cohort_selection/selection_bias_audit_911_vs_842.csv` |
| Table 5 / Fig 5 – main performance + CIs | `results/stage4_7_statistics_circularity/stage47_model_metrics_bootstrap_CI_all_conditions.csv` |
| Table 6 – sensitivity / specificity | `results/stage4_7_statistics_circularity/stage47_validation_selected_thresholds_summary.csv` |
| Table 7 / Fig 6 – missing-modality benchmark | `stage47_model_metrics_…`, `stage47_missingness_paired_effects_vs_0pct.csv` |
| Table 8 / Fig 7 – ablation | `results/stage4_6_ablation/summary_mean_sd.csv` |
| Table 9 / Fig 8 – calibration pre/post | `results/stage4_7_statistics_circularity/stage47_calibration_pre_post_bootstrap_CI.csv` |
| Fig 9 – selective prediction | `results/stage4_8_reliability/auaf_selective_curve_summary.csv` |
| Table 10 / Fig 10 – conformal | `results/stage4_8_reliability/conformal_summary_by_seed.csv` |
| Figs 11–13 – SHAP, gates, occlusion | `results/stage4_8_reliability/catboost_shap_summary.csv`, `auaf_gate_summary_pooled.csv`, `auaf_modality_occlusion_*` |
| Table 11 / Supp S2 – paired tests | `results/stage4_7_statistics_circularity/stage47_AUAF_vs_key_baselines_paired_bootstrap.csv` |
| Table 12 / Fig 14 – circularity | `results/stage4_7_statistics_circularity/stage47_circularity_standard_vs_reduced_summary_mean_sd.csv` |
| Table 13 – severity strata | `results/stage4_7_statistics_circularity/stage47_conversion_severity_stratified_performance.csv` |
| Supp S1 – feature inventory | `manifests/stage47_feature_source_and_circularity_inventory.csv` |

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/AUAF-ADNI.git
cd AUAF-ADNI
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r environment/requirements-analysis.txt
```

Exact versions used in the experiments are recorded in `environment/environment_versions.json`.

## ▶️ Reproducing the analysis

1. Obtain ADNI access (see [`DATA_ACCESS.md`](DATA_ACCESS.md)) and place the ADNI source tables in a local `data/` folder. This folder is git-ignored and is never committed.
2. **Temporal alignment:** `python code/stage4_4_temporal_alignment/AUAF_Stage4_4_Temporal_Cleaning.py`
3. **Training and missing-modality benchmark:** run `notebooks/AUAF_Stage4_5_RealData_FinalCohort.ipynb`, then `…_Part2_…ipynb`.
4. **Ablation:** `python code/stage4_6_ablation/train_stage4_6_ablation.py`
5. **Statistics and circularity:** `python code/stage4_7_statistics_circularity/run_stage47_all.py`
6. **Reliability:** run `notebooks/AUAF_Stage4_8_StatisticalInference_FINAL_SOURCE.ipynb`
7. **Manuscript figures:** `python code/manuscript_figures/make_manuscript_figures.py`

Each stage verifies the SHA-256 hashes of its inputs against `manifests/`. See `docs/REPRODUCE.md` for details. Paths in the scripts may need adjusting to your local `data/` folder.

## 🔒 Data and privacy

Under the ADNI Data Use Agreement, this repository contains **no participant-level data**: no ADNI tables, no RIDs, no split rosters, no per-participant predictions and no trained weights. Only code, configuration, hashes and **aggregate** results are shared. Split manifests, prediction files and checkpoints can be provided to researchers who hold an approved ADNI account, on request to the corresponding author.

## 📖 Citation

```bibtex
@article{auaf2026,
  title   = {Adaptive Uncertainty-Aware Multimodal Fusion for Alzheimer's Disease Diagnosis and Early MCI-to-AD Conversion},
  author  = {Safari Shateri, Aynaz and Salmanpour Fard, Parya and Samadi Gharehveran, Sina and Shirini, Kimia},
  year    = {2026},
  note    = {Manuscript under review}
}
```

## 🙏 Acknowledgements

Data collection and sharing for this project was funded by the Alzheimer's Disease Neuroimaging Initiative (ADNI). Data used in preparation of this article were obtained from the ADNI database (adni.loni.usc.edu). The ADNI investigators contributed to the design and implementation of ADNI and/or provided data but did not participate in the analysis or writing of this work.

## ✉️ Contact

Corresponding author: **Kimia Shirini** — k.shirini@tabrizu.ac.ir
