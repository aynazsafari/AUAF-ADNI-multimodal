from pathlib import Path
import pandas as pd, numpy as np, json, hashlib, shutil, os, re, sys, zipfile, csv
ROOT=Path(os.environ.get('AUAF_WORKSPACE', Path.cwd())); OUT=ROOT/'output'/'stage47'
REMOVED={'CDMEMORY','CDRSB','MMSEASON','TRAILS','CLOCKCON','CLOCKNO','CLOCKHAN','FFLUENCY','MOCA','FAQFINAN','FAQFORM','FAQSHOP','FAQGAME','FAQBEVG','FAQMEAL','FAQEVENT','FAQTV','FAQREM','FAQTRAVL','FAQTOTAL'}

# 1) Enrich circularity run manifest from immutable job outputs.
run_path=OUT/'manifests'/'stage47_circularity_model_run_manifest.csv'
runs=pd.read_csv(run_path)
sel=pd.read_csv(OUT/'tables'/'stage47_best_tree_baseline_validation_selection.csv')
for i,r in runs.iterrows():
    if r.model=='AUAF':
        mp=Path(r.job_dir)/'metrics.json'
        if mp.exists():
            m=json.loads(mp.read_text())
            for col,key in [('train_n','train_n'),('validation_n','validation_n'),('test_n','test_n'),('epochs_completed','epochs_completed'),('elapsed_sec','elapsed_sec')]:
                runs.loc[i,col]=m.get(key)
            runs.loc[i,'best_validation_macro_f1']=m.get('best_validation_macro_f1')
    else:
        z=sel[(sel.task==r.task)&(sel.model==r.model)&(sel.seed.astype(str)=='MEAN')]
        if not z.empty: runs.loc[i,'best_validation_macro_f1']=float(z.iloc[0].validation_macro_f1)
        # retrieve counts from frozen task dataset and the reduced job's model metadata
        f={'diagnosis_multiclass':'diagnosis_multiclass_3700_asof.csv','diagnosis_binary':'diagnosis_binary_3700_asof.csv','mci_conversion':'mci_conversion_911_asof.csv'}[r.task]
        df=pd.read_csv(ROOT/'stage4_data'/f)
        runs.loc[i,'train_n']=int((df.split=='train').sum())
        runs.loc[i,'validation_n']=int((df.split=='validation').sum())
        runs.loc[i,'test_n']=int((df.split=='test').sum())
# Reconstruct reduced-model split counts and validation metrics from the saved reduced predictions.
red_raw=pd.read_csv(OUT/'predictions'/'stage47_circularity_reduced_predictions_uncalibrated.csv.gz',dtype={'RID':str})
from sys import path as _sys_path
_sys_path.insert(0,str(ROOT/'input45'/'code'))
from auaf_safe_training_engine import classification_metrics as _metrics
val_metric_rows=[]
for (task,model,seed),g in red_raw[red_raw.split.eq('validation')].groupby(['task','model','seed']):
    k={'diagnosis_multiclass':3,'diagnosis_binary':2,'mci_conversion':2}[task]
    cols=[f'p_class_{j}' for j in range(k)]
    mm=_metrics(g.y_true.to_numpy(dtype=int),g[cols].to_numpy(dtype=float))
    val_metric_rows.append({'task':task,'model':model,'seed':int(seed),'n_validation_reduced':len(g),'reduced_validation_macro_f1':float(mm['f1_macro'])})
val_red=pd.DataFrame(val_metric_rows)
val_red.to_csv(OUT/'tables'/'stage47_circularity_reduced_validation_metrics_seed_level.csv',index=False)
feature_inv_for_counts=pd.read_csv(OUT/'manifests'/'stage47_feature_source_and_circularity_inventory.csv')
retained_for_counts=feature_inv_for_counts.loc[~feature_inv_for_counts.removed_in_strict_sensitivity,'feature'].astype(str).tolist()
counts_by_task={}
for task in ['diagnosis_multiclass','diagnosis_binary','mci_conversion']:
    fname={'diagnosis_multiclass':'diagnosis_multiclass_3700_asof.csv','diagnosis_binary':'diagnosis_binary_3700_asof.csv','mci_conversion':'mci_conversion_911_asof.csv'}[task]
    df0=pd.read_csv(ROOT/'stage4_data'/fname)
    eligible=df0[retained_for_counts].notna().any(axis=1)
    counts_by_task[task]={split:int((df0.split.astype(str).str.lower().eq(split)&eligible).sum()) for split in ['train','validation','test']}
for i,r in runs.iterrows():
    g=red_raw[(red_raw.task==r.task)&(red_raw.model==r.model)&(red_raw.seed==r.seed)]
    counts=g.groupby('split').size().to_dict()
    runs.loc[i,'train_n']=counts_by_task[r.task]['train']
    runs.loc[i,'validation_n']=counts.get('validation',np.nan)
    runs.loc[i,'test_n']=counts.get('test',np.nan)
    v=val_red[(val_red.task==r.task)&(val_red.model==r.model)&(val_red.seed==r.seed)]
    runs.loc[i,'reduced_validation_macro_f1']=float(v.iloc[0].reduced_validation_macro_f1) if len(v) else np.nan
    if r.model!='AUAF':
        z=sel[(sel.task==r.task)&(sel.model==r.model)&(sel.seed.astype(str)=='MEAN')]
        runs.loc[i,'selection_validation_macro_f1']=float(z.iloc[0].validation_macro_f1) if len(z) else np.nan
    else:
        runs.loc[i,'selection_validation_macro_f1']=runs.loc[i,'reduced_validation_macro_f1']
runs['model_family']=np.where(runs.model.eq('AUAF'),'neural','tree')
runs['features_removed_n']=20
runs['retraining_data_scope']='frozen Stage 4.4 training participants only'
runs.to_csv(run_path,index=False)

# 2) Standard-vs-reduced summary across seeds.
perf=pd.read_csv(OUT/'tables'/'stage47_circularity_standard_vs_reduced_seed_metrics.csv')
eff=pd.read_csv(OUT/'tables'/'stage47_circularity_standard_vs_reduced_paired_effects.csv')
rows=[]
for (cohort,task,model),g in perf.groupby(['cohort','task','model'],sort=True):
    for metric in ['accuracy','balanced_accuracy','f1_macro','roc_auc','pr_auc','brier','log_loss','ece']:
        sm=g[f'{metric}_standard']; rm=g[f'{metric}_reduced']
        ee=eff[(eff.cohort==cohort)&(eff.task==task)&(eff.model==model)&(eff.metric==metric)]
        e=ee.iloc[0] if len(ee) else None
        rows.append({'cohort':cohort,'task':task,'model':model,'metric':metric,
                     'standard_mean_5seeds':float(sm.mean()),'standard_sd_5seeds':float(sm.std(ddof=1)),
                     'reduced_mean_5seeds':float(rm.mean()),'reduced_sd_5seeds':float(rm.std(ddof=1)),
                     'delta_reduced_minus_standard':float((rm-sm).mean()),
                     'paired_bootstrap_ci95_low':float(e.ci95_low) if e is not None else np.nan,
                     'paired_bootstrap_ci95_high':float(e.ci95_high) if e is not None else np.nan,
                     'p_value_bootstrap':float(e.p_value_bootstrap) if e is not None else np.nan,
                     'p_fdr_bh':float(e.p_fdr_bh) if e is not None else np.nan,
                     'significant_after_FDR_0_05':bool(e.significant_after_FDR_0_05) if e is not None else False,
                     'n_paired':int(e.n_paired) if e is not None else int(g.n_paired.min())})
summary=pd.DataFrame(rows)
summary.to_csv(OUT/'tables'/'stage47_circularity_standard_vs_reduced_summary_mean_sd.csv',index=False)

# 3) Explicit kept feature list.
inv=pd.read_csv(OUT/'manifests'/'stage47_feature_source_and_circularity_inventory.csv')
inv[~inv.removed_in_strict_sensitivity].to_csv(OUT/'manifests'/'stage47_strict_sensitivity_retained_features.csv',index=False)

# 4) Biomarker temporal audit summary and feature inventory details already produced; record the index-visit identifiers.
inv['index_visit_code_fields']=np.where(inv.modality.eq('biomarker'),'PHASE + VISCODE2 (biomarker source match); participant index date from INDEX_DATE','VISCODE/VISCODE2 in participant-level index provenance')
inv['feature_level_source_row_id_retained']=False
inv['provenance_interpretation']='Source table, instrument, and temporal rule are from the Stage 4.4 feature manifest; per-feature raw source row identifier/visit code is not in the integrated predictor matrix.'
inv.to_csv(OUT/'manifests'/'stage47_feature_source_and_circularity_inventory.csv',index=False)

# 5) Update statistical manifest with completed circularity stage.
sm_path=OUT/'manifests'/'stage47_statistical_reproducibility_manifest.json'
sm=json.loads(sm_path.read_text())
cm=json.loads((OUT/'manifests'/'stage47_circularity_retraining_manifest.json').read_text())
sm['status']='COMPLETED_STATISTICAL_VALIDATION_AND_MULTIVARIABLE_CIRCULARITY_SENSITIVITY'
sm['circularity_retraining']=cm
sm['paired_model_comparisons_fdr_significant_rows']=int(pd.read_csv(OUT/'tables'/'stage47_AUAF_vs_key_baselines_paired_bootstrap.csv').significant_after_FDR_0_05.sum())
sm['missingness_effects_fdr_significant_rows']=int(pd.read_csv(OUT/'tables'/'stage47_missingness_paired_effects_vs_0pct.csv').significant_after_FDR_0_05.sum())
sm['calibration_effects_fdr_significant_rows']=int(pd.read_csv(OUT/'tables'/'stage47_calibration_pre_post_bootstrap_CI.csv').significant_after_FDR_0_05.sum())
sm['conformal_audit_status']='AGGREGATE_TABLE_REVIEWED; participant-level RAPS-lite sets/q-hats not available for reproduction'
sm_path.write_text(json.dumps(sm,indent=2,ensure_ascii=False),encoding='utf-8')

# 6) Master workbook including all Stage 4.7 deliverables.
files={
 'Cohort sizes':'stage47_cohort_denominators.csv',
 'Model CI all levels':'stage47_model_metrics_bootstrap_CI_all_conditions.csv',
 'Seed metrics':'stage47_seed_level_metrics_all_conditions.csv',
 'AUAF paired tests':'stage47_AUAF_vs_key_baselines_paired_bootstrap.csv',
 'Missingness effects':'stage47_missingness_paired_effects_vs_0pct.csv',
 'Calibration':'stage47_calibration_pre_post_bootstrap_CI.csv',
 'Temp parameters':'stage47_temperature_scaling_parameters.csv',
 'Thresholds':'stage47_validation_selected_thresholds_summary.csv',
 'McNemar':'stage47_mcnemar_accuracy_seed_level_supplement.csv',
 'Feature inventory':'stage47_feature_source_and_circularity_inventory.csv',
 'Removed features':'stage47_diagnosis_proximal_features_removed.csv',
 'Retained features':'stage47_strict_sensitivity_retained_features.csv',
 'Conformal audit':'stage47_conformal_summary_reproduction_gap_audit.csv',
 'Metric roundtrip':'stage47_metric_roundtrip_audit.csv',
 'Baseline selection':'stage47_best_tree_baseline_validation_selection.csv',
 'Reduced validation':'stage47_circularity_reduced_validation_metrics_seed_level.csv',
 'Circularity summary':'stage47_circularity_standard_vs_reduced_summary_mean_sd.csv',
 'Circularity effects':'stage47_circularity_standard_vs_reduced_paired_effects.csv',
 'Severity strata':'stage47_conversion_severity_stratified_performance.csv',
 'Biomarker timing':'stage47_biomarker_temporal_lag_summary.csv'
}
with pd.ExcelWriter(OUT/'reports'/'AUAF_Stage4_7_Master_Results.xlsx',engine='openpyxl') as w:
    for sheet,f in files.items():
        p=OUT/'tables'/f
        if not p.exists(): p=OUT/'manifests'/f
        if p.exists(): pd.read_csv(p).to_excel(w,sheet_name=sheet[:31],index=False)

# 7) Human-readable report (Persian) with grounded key results.
metrics=pd.read_csv(OUT/'tables'/'stage47_model_metrics_bootstrap_CI_all_conditions.csv')
paired=pd.read_csv(OUT/'tables'/'stage47_AUAF_vs_key_baselines_paired_bootstrap.csv')
missing=pd.read_csv(OUT/'tables'/'stage47_missingness_paired_effects_vs_0pct.csv')
cal=pd.read_csv(OUT/'tables'/'stage47_calibration_pre_post_bootstrap_CI.csv')
conf=pd.read_csv(OUT/'tables'/'stage47_conformal_summary_reproduction_gap_audit.csv')

def mval(cohort,task,model,metric,lev=0.0):
    x=metrics[(metrics.cohort==cohort)&(metrics.task==task)&(metrics.model==model)&(metrics.metric==metric)&(metrics.missingness==lev)]
    return float(x.iloc[0].mean_across_5_seeds) if len(x) else float('nan')

def cirval(cohort,task,model,metric):
    x=summary[(summary.cohort==cohort)&(summary.task==task)&(summary.model==model)&(summary.metric==metric)]
    return x.iloc[0] if len(x) else None

lines=[]
lines += ['# گزارش نهایی مرحله ۴.۷ — اعتبارسنجی آماری و ممیزی چندمتغیره Circularity','',
'## وضعیت اجرا','',
'- تحلیل روی خروجی‌های نهایی Stage 4.5 و داده‌های as-of-index مرحله 4.4 اجرا شد؛ هیچ split جدیدی ساخته نشد.',
'- ۱٬۰۵۰ فایل پیش‌بینی فردبه‌فرد (۵۲۵ full-test و ۵۲۵ matched) بررسی شد؛ همه rosterها، شناسه‌ها، برچسب‌ها و احتمال‌ها از ممیزی یکپارچگی عبور کردند.',
'- ۱٬۰۰۰ تکرار bootstrap طبقه‌بندی‌شده بر حسب کلاس برای فاصله اطمینان و مقایسه‌های paired استفاده شد.',
'- پنج seed، پنج initialization روی همان split ثابت هستند؛ پنج fold یا پنج cohort مستقل نیستند.',
'- ۱۵ مدل AUAF و ۱۵ مدل baseline در آزمایش حساسیت circularity، با همان train/validation/testهای منجمد، دوباره آموزش داده شدند.',
'', '## ۱. بازتولید معیارهای Stage 4.5','',
'- معیارهای accuracy، F1، Brier، log-loss و ECE در تمام ۱٬۰۵۰ شرط با خروجی آرشیوی عملاً دقیق بازتولید شدند.',
'- برای ROC-AUC/PR-AUC اختلاف کوچک عددی در خروجی ExtraTrees مشاهده شد؛ بیشترین اختلاف در matched diagnosis multiclass برای یک شرط seed/missingness برابر 0.000306 در ROC-AUC و 0.000101 در PR-AUC بود. این اختلاف در گزارش round-trip جداگانه علامت‌گذاری شده و پنهان نشده است.',
'', '## ۲. Bootstrap و مقایسه مدل‌ها','',
'- برای هر مدل، task، cohort و پنج سطح missingness، میانگین و SD پنج seed به‌همراه ۹۵٪ bootstrap CI محاسبه شد.',
'- در مقایسه AUAF با AttentionFusion و سه baseline درختی، ۱٬۴۴۰ مقایسه paired ثبت شد. ۱۳۷ ردیف پس از BH-FDR در کل خانواده مقایسه‌ها معنادار شدند؛ این شمار به معنی برتری کلی AUAF نیست و باید در سطح task/metric/شرط خوانده شود.',
'- در سطح ۰٪ حذف اضافه biomarker، برای چهار معیار کلیدی F1، ROC-AUC، Brier و ECE هیچ مقایسه AUAF در برابر comparator پس از FDR معنادار نشد.',
'- در تحلیل اثر ۲۵/۵۰/۷۵/۱۰۰٪ حذف اضافه biomarker نسبت به ۰٪، هیچ‌یک از ۲٬۰۱۶ اثر پس از BH-FDR معنادار نماند. این نتیجه نبود اثر قطعی را اثبات نمی‌کند؛ matched cohortها کوچک‌اند، به‌ویژه conversion با n=29.',
'', '### چند مقدار نمونه در full test، سطح ۰٪','',
'| Task | مدل | Macro-F1 | ROC-AUC |',
'|---|---|---:|---:|']
for task in ['diagnosis_multiclass','diagnosis_binary','mci_conversion']:
    best={'diagnosis_multiclass':'XGBoost','diagnosis_binary':'CatBoost','mci_conversion':'CatBoost'}[task]
    lines.append(f"| {task} | AUAF | {mval('full_test',task,'AUAF','f1_macro'):.3f} | {mval('full_test',task,'AUAF','roc_auc'):.3f} |")
    lines.append(f"| {task} | {best} (best tree by validation) | {mval('full_test',task,best,'f1_macro'):.3f} | {mval('full_test',task,best,'roc_auc'):.3f} |")
lines += ['', '## ۳. Temperature scaling و thresholding','',
'- دمای calibration برای هر task/model/seed فقط روی validation fit شد؛ همان دما برای test به کار رفت.',
'- pre/post calibration در دو نقطه ۰٪ و ۱۰۰٪ missingness گزارش شده است. هیچ اثر calibration پس از BH-FDR در ۱٬۰۰۸ ردیف مقایسه باقی نماند.',
'- برای diagnosis_binary و mci_conversion، threshold با Youden J فقط روی validation انتخاب و برای test در پنج سطح missingness اعمال شد.',
'', '## ۴. ممیزی چندمتغیره diagnostic circularity','',
'- بر اساس feature manifest و reconciliation واقعی Stage 4.4، ۲۰ متغیر از بلوک‌های CDR، FAQ، MMSE و MoCA پیش از preprocessing حذف شدند.',
'- فهرست حذف‌شده: ' + ', '.join(sorted(REMOVED)) + '.',
'- مجموعه ورودی strict sensitivity از ۳۲ ویژگی مؤثر به ۱۲ ویژگی رسید: ۹ ویژگی بالینی/شناختی غیر از بلوک‌های فوق و ۳ CSF biomarker.',
'- baseline برتر فقط از بین سه baseline موجود در بسته Stage 4.5 (XGBoost، CatBoost، ExtraTrees) و با میانگین Validation Macro-F1 انتخاب شد: multiclass = XGBoost، binary = CatBoost، conversion = CatBoost.',
'- در full test، حذف بلوک‌های تشخیصی باعث افت محسوس عملکرد شد؛ این نتیجه نشان می‌دهد بخشی از قدرت پیش‌بینی به مقیاس‌های بالینی نزدیک به تعریف تشخیص وابسته است. این آزمایش گواه نبود leakage نیست و به‌عنوان sensitivity analysis گزارش می‌شود.',
'', '### Macro-F1 استاندارد در برابر strict sensitivity (میانگین پنج seed، full test)','',
'| Task | مدل | همه ویژگی‌ها | پس از حذف ۲۰ ویژگی |',
'|---|---|---:|---:|']
for task,model in [('diagnosis_multiclass','AUAF'),('diagnosis_binary','AUAF'),('mci_conversion','AUAF'),('diagnosis_multiclass','XGBoost'),('diagnosis_binary','CatBoost'),('mci_conversion','CatBoost')]:
    x=cirval('full_test',task,model,'f1_macro')
    if x is not None: lines.append(f"| {task} | {model} | {x.standard_mean_5seeds:.3f} | {x.reduced_mean_5seeds:.3f} |")
lines += ['',
'- برای matched test، مقایسه روی همان افراد مشترک انجام شد: n=81 برای multiclass، n=67 برای binary و n=29 برای conversion. در conversion و زیرگروه‌های شدت، CIها عریض و برآوردها ناپایدارند.',
'- تحلیل severity با CDRSB و FAQTOTAL از median محاسبه‌شده روی train استفاده کرد. در matched conversion، CDRSB برای ۲۸ از ۲۹ نفر missing بود؛ بنابراین این strata قابل تفسیر قوی نیست. FAQTOTAL نیز گروه‌های بسیار کوچک دارد.',
'', '## ۵. Conformal prediction — ممیزی قابلیت بازتولید','',
'- مقادیر Table 6 از Manuscript استخراج و از نظر سازگاری داخلی بررسی شدند؛ اما prediction setهای فردبه‌فرد، calibration scores/q-hatها و پیاده‌سازی دقیق RAPS-lite در آرشیو Stage 4.4/4.5 موجود نبود. بنابراین پوشش و اندازه setها در این مرحله از نو محاسبه نشدند.',
'- مقدار گزارش‌شده coverage: CN/MCI/AD = 0.9259، AD/Non-AD = 0.8851 (پایین‌تر از هدف 0.90)، conversion = 0.9705.',
'- برای multiclass، abstention_rate گزارش‌شده 0 است، درحالی‌که مکمل singleton_rate برابر 0.4211 است؛ تعریف abstention در manuscript باید روشن شود. این اختلاف به‌عنوان ابهام تعریف ثبت شد، نه اینکه خودسرانه اصلاح شود.',
'', '## ۶. Provenance و محدودیت‌ها','',
'- source table، instrument، شرح ویژگی، missing-code handling و temporal alignment از `feature_manifest_34.csv` مرحله 4.4 ثبت شد.',
'- برای ویژگی‌های بالینی، انتخاب روی index visit و برای CSF biomarkers تطبیق PHASE + RID + VISCODE2 همراه شرط EXAMDATE_BIO ≤ INDEX_DATE ثبت شده است. در temporal audit، ۲۰۹ رکورد biomarker پس از index در cohort تشخیصی و ۱۰۳ رکورد در conversion شناسایی شدند و از ورودی as-of کنار گذاشته شدند.',
'- در audit زمانی، biomarker قابل استفاده در index برای ۳۱۶/۳۷۰۰ نفر در cohort تشخیصی و ۱۶۹/۹۱۱ نفر در conversion ثبت شده است؛ توزیع lag همه رکوردهای تاریخ‌دار به تفکیک در جدول temporal lag آمده است.',
'- شناسه ردیف خام/visit code اختصاصی هر feature در integrated predictor matrix نگهداری نشده؛ این محدودیت در inventory صریحاً علامت‌گذاری شده است.',
'- نسخه نهایی Stage 4.5 فقط دو modality دارد: clinical/cognitive و fluid biomarkers؛ هیچ modality تصویربرداری در ورودی مدل نهایی وجود ندارد.',
'- مطالعه همچنان ADNI-only و بدون external validation است. از واژه leakage-free یا ادعای آمادگی بالینی استفاده نشود.',
'', '## فایل‌های کلیدی','',
'- `reports/AUAF_Stage4_7_Master_Results.xlsx`: همه جدول‌های اصلی در یک workbook.',
'- `tables/`: جدول‌های کامل seed-level، bootstrap CI، paired tests، missingness effects، calibration، thresholds و circularity.',
'- `predictions/`: calibrated predictions و reduced-feature predictions در CSV.GZ.',
'- `models/`: best checkpoints مدل‌های حساسیت AUAF و مدل‌های درختی reduced؛ last.pt در ZIP نهایی نگهداری نمی‌شود.',
'- `manifests/`: hashها، roster audit، feature provenance، فهرست حذف/نگهداشت و run manifest.',
'', '## جمع‌بندی','',
'مرحله ۴.۷ از نظر تحلیل آماری Stage 4.5، calibration، thresholding، feature provenance و بازآموزی حساسیت چندمتغیره اجرا شد. تنها بخش conformal در سطح participant قابل بازتولید نبود، چون prediction setها و q-hatهای اصلی در ورودی‌های در دسترس وجود نداشتند؛ این مورد به‌عنوان gap صریح تحویل شده است.']
(OUT/'reports'/'STAGE_4_7_REPORT_FA.md').write_text('\n'.join(lines),encoding='utf-8')

# English concise report.
en='''# AUAF Stage 4.7 — Statistical Validation and Multivariable Circularity Audit\n\n## Execution status\nCompleted on the final Stage 4.4 as-of-index cohorts and Stage 4.5 participant-level predictions. No participant resplitting was performed. The five seeds are repeated initializations on the same frozen split, not five independent folds.\n\n## Statistical validation\n- Audited 1,050 prediction files (525 full-test and 525 matched-complete-case); all rosters, labels, probability normalization and finite-value checks passed.\n- Recomputed 12 metrics and generated 1,000-replicate stratified participant-bootstrap 95% CIs for all task/model/cohort/missingness conditions.\n- Reproduced the archived Stage 4.5 metrics to numerical precision except for small ROC-AUC/PR-AUC differences in ExtraTrees. The largest was 0.000306 ROC-AUC and 0.000101 PR-AUC in one matched multiclass condition; this is explicitly recorded.\n- Ran 1,440 paired AUAF-vs-key-baseline comparisons and 2,016 paired missingness-vs-0% effects, with BH-FDR. No missingness effect survived FDR. At 0% missingness, none of the four key metrics (F1, ROC-AUC, Brier, ECE) showed an FDR-significant AUAF advantage over the key comparators.\n- Temperature scaling was fitted on validation only and assessed at 0% and 100% additional biomarker missingness. No calibration change survived BH-FDR. Youden thresholds for the two binary tasks were selected on validation only.\n\n## Multivariable circularity sensitivity\nTwenty raw predictors from CDR, FAQ, MMSE and MoCA-derived blocks were removed before preprocessing. The effective input set fell from 32 predictors (29 clinical/cognitive + 3 CSF biomarkers) to 12 (9 clinical/cognitive + 3 CSF biomarkers). Fifteen AUAF and fifteen best-tree-baseline models were retrained on the frozen training participants only. Best tree baselines were selected on validation macro-F1: XGBoost for multiclass, CatBoost for binary diagnosis and MCI conversion.\n\nThe strict sensitivity analysis showed substantial performance reductions on the full test set, consistent with the high predictive contribution of diagnosis-proximal clinical scales. This is a sensitivity analysis, not proof that all information leakage has been eliminated. Matched test sizes were 81, 67 and 29, so the conversion comparison is especially imprecise.\n\n## Conformal audit and provenance\nManuscript Table 6 aggregate RAPS-lite values were transcribed and checked for internal consistency. Participant-level prediction sets, validation scores/q-hats and the exact RAPS-lite implementation/configuration were not present in the Stage 4.4/4.5 artifacts, so set-level coverage could not be recomputed. The reported multiclass abstention rate (0.0000) differs from the complement of its singleton rate (0.4211); the manuscript definition needs clarification.\n\nSource table, instrument, feature description, missing-code handling and temporal alignment were taken from the Stage 4.4 feature manifest. Feature-specific raw source-row IDs/visit codes are not preserved in the integrated predictor matrix. The final Stage 4.5 model has two modalities only (clinical/cognitive and fluid biomarkers); no imaging modality was present.\n\n## Main limitations\nADNI-only evaluation, no external validation, small matched subsets, no participant-level conformal artifacts, and incomplete feature-level source-row provenance. Do not describe the model as leakage-free or clinically deployment-ready.\n'''
(OUT/'reports'/'STAGE_4_7_REPORT_EN.md').write_text(en,encoding='utf-8')

# 8) Create circularity plots.
import matplotlib.pyplot as plt
for cohort in ['full_test','matched_complete_case']:
  for task in ['diagnosis_multiclass','diagnosis_binary','mci_conversion']:
    g=summary[(summary.cohort==cohort)&(summary.task==task)&(summary.metric=='f1_macro')]
    if g.empty: continue
    fig,ax=plt.subplots(figsize=(7.5,4.5))
    x=np.arange(len(g)); width=.34
    ax.bar(x-width/2,g.standard_mean_5seeds,width,label='All Stage 4.5 features')
    ax.bar(x+width/2,g.reduced_mean_5seeds,width,label='Strict proximal-feature removal')
    ax.set_xticks(x,g.model.tolist());ax.set_ylabel('Macro-F1');ax.set_ylim(0,1)
    ax.set_title(f'{task} — {cohort.replace("_"," ")}')
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.25);fig.tight_layout()
    fig.savefig(OUT/'figures'/f'circularity_macro_f1_{cohort}_{task}.png',dpi=180);plt.close(fig)

# 9) Update statistical execution summary and add circularity counts.
ss=json.loads((OUT/'reports'/'stage47_statistical_execution_summary.json').read_text())
ss['circularity_neural_jobs_completed']=int((runs.model=='AUAF').sum())
ss['circularity_tree_jobs_completed']=int((runs.model!='AUAF').sum())
ss['circularity_paired_effect_rows']=len(eff)
ss['circularity_best_baselines']=json.loads((OUT/'manifests'/'stage47_circularity_best_baselines.json').read_text())
ss['circularity_retained_features']=inv[~inv.removed_in_strict_sensitivity].feature.tolist()
ss['circularity_removed_features']=sorted(REMOVED)
(OUT/'reports'/'stage47_statistical_execution_summary.json').write_text(json.dumps(ss,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(ss,indent=2,ensure_ascii=False))
