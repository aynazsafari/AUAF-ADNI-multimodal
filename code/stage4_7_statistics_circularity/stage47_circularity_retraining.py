from __future__ import annotations
import os
import sys, json, hashlib, time, re
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

ROOT=Path(os.environ.get('AUAF_WORKSPACE', Path.cwd())); PKG=ROOT/'input45'; DATA=ROOT/'stage4_data'; OUT=ROOT/'output'/'stage47'
for d in ['tables','predictions','manifests','reports','models','logs']:(OUT/d).mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(PKG/'code'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from auaf_model_factory import build_model
from auaf_safe_training_engine import TrainingConfig, initialize_run_directory, run_training_job, sha256_file
from stage47_complete_analysis import (TASKS, MODELS, SEEDS, N_CLASSES, METRICS, BOOTSTRAP_N,
    classification_metrics, fit_temperature, apply_temperature, make_stratified_weights,
    bootstrap_metrics, ci, p_from_bootstrap_distribution, bh_adjust)

# Exact diagnosis-proximal instrument blocks identified in the Stage 4.4 feature reconciliation.
CIRCULARITY_GROUPS={
 'CDR':['CDMEMORY','CDRSB'],
 'FAQ':['FAQFINAN','FAQFORM','FAQSHOP','FAQGAME','FAQBEVG','FAQMEAL','FAQEVENT','FAQTV','FAQREM','FAQTRAVL','FAQTOTAL'],
 'MMSE':['MMSEASON'],
 'MOCA':['TRAILS','CLOCKCON','CLOCKNO','CLOCKHAN','FFLUENCY','MOCA']
}
REMOVED=set(sum(CIRCULARITY_GROUPS.values(),[]))

class TrainOnlyNumericPreprocessor:
    def __init__(self,columns):
        self.columns=list(columns); self.medians_=self.means_=self.scales_=self.all_missing_train_=None
        self.fit_n_=None; self.fit_rids_sha256_=None
    @staticmethod
    def _array(frame,columns):
        x=frame[list(columns)].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=np.float64)
        x[~np.isfinite(x)]=np.nan
        return x
    def fit(self,frame,train_mask):
        train_mask=np.asarray(train_mask,bool); x=self._array(frame.loc[train_mask],self.columns)
        med=np.zeros(x.shape[1]); miss=np.zeros(x.shape[1],bool)
        for j in range(x.shape[1]):
            v=x[:,j][np.isfinite(x[:,j])]
            if len(v): med[j]=np.median(v)
            else: miss[j]=True; med[j]=0.
        xi=np.where(np.isfinite(x),x,med[None,:]); mean=xi.mean(axis=0); scale=xi.std(axis=0,ddof=0)
        scale[~np.isfinite(scale)|(scale<1e-12)]=1.
        self.medians_,self.means_,self.scales_,self.all_missing_train_=med,mean,scale,miss
        self.fit_n_=int(train_mask.sum())
        ids=frame.loc[train_mask,'RID'].astype(str).sort_values().tolist()
        self.fit_rids_sha256_=hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest()
        return self
    def effective_columns(self): return [c for c,m in zip(self.columns,self.all_missing_train_) if not m]
    def transform(self,frame):
        x=self._array(frame,self.columns); x[:,self.all_missing_train_]=np.nan
        xi=np.where(np.isfinite(x),x,self.medians_[None,:]); z=(xi-self.means_[None,:])/self.scales_[None,:]
        z[~np.isfinite(z)]=0.
        return z.astype(np.float32)
    def audit(self): return {'fit_RID_sha256':self.fit_rids_sha256_,'fit_n':self.fit_n_}

class PreparedTask:
    def __init__(self,name,frame,target,n_classes,modalities,matrices,masks,features_by_modality,processors):
        self.name=name; self.frame=frame; self.target=target; self.n_classes=n_classes
        self.modalities=list(modalities); self.matrices=matrices; self.masks=masks
        self.features_by_modality=features_by_modality; self.processors=processors
    def idx(self,split,eligible_only=False):
        idx=np.flatnonzero(self.frame['split'].astype(str).str.lower().eq(str(split).lower()).to_numpy())
        if eligible_only: idx=idx[self.masks[idx].sum(axis=1)>0]
        return idx


def sha256_text(s):return hashlib.sha256(str(s).encode()).hexdigest()

def prepare_reduced_task(task):
    specs={
      'diagnosis_multiclass':('diagnosis_multiclass_3700_asof.csv','TARGET_MULTICLASS',3),
      'diagnosis_binary':('diagnosis_binary_3700_asof.csv','TARGET_BINARY',2),
      'mci_conversion':('mci_conversion_911_asof.csv','TARGET_CONVERSION',2)}
    filename,target,k=specs[task]
    frame=pd.read_csv(DATA/filename,parse_dates=['INDEX_DATE'])
    frame['split']=frame['split'].astype(str).str.lower()
    fm=pd.read_csv(DATA/'feature_manifest_34_predictors.csv')
    effective=[f for f in fm.feature.astype(str).tolist() if f not in {'PTENGSPKAGE','PTIMMAGE'}]
    columns_by={
      'clinical_cognitive':[f for f in effective if f in set(fm.loc[fm.modality.eq('clinical_cognitive'),'feature'].astype(str)) and f not in REMOVED],
      'fluid_biomarker':[f for f in effective if f in set(fm.loc[fm.modality.eq('fluid_biomarker'),'feature'].astype(str))]
    }
    train_mask=frame['split'].eq('train').to_numpy()
    matrices={}; processors={}; features={}; masks=[]; modalities=[]
    for modality,cols in columns_by.items():
        if not cols: continue
        pre0=TrainOnlyNumericPreprocessor(cols).fit(frame,train_mask)
        effective_cols=pre0.effective_columns()
        if not effective_cols: continue
        pre=TrainOnlyNumericPreprocessor(effective_cols).fit(frame,train_mask)
        values=pre.transform(frame)
        avail=frame[effective_cols].notna().any(axis=1).to_numpy(dtype=np.float32)
        matrices[modality]=values; processors[modality]=pre; features[modality]=effective_cols
        masks.append(avail); modalities.append(modality)
    mask=np.column_stack(masks).astype(np.float32)
    if not np.isfinite(mask).all():raise RuntimeError(f'{task}: non-finite reduced modality masks')
    if frame[target].isna().any():raise RuntimeError(f'{task}: target missing')
    pack=PreparedTask(task,frame,target,k,modalities,matrices,mask,features,processors)
    for split in ['train','validation','test']:
        idx=pack.idx(split,eligible_only=True)
        if len(idx)==0:raise RuntimeError(f'{task}: no eligible {split} rows after proximal feature removal')
        if split=='train' and set(frame.iloc[idx][target].astype(int).unique())!=set(range(k)):
            raise RuntimeError(f'{task}: train classes missing after proximal feature removal')
    return pack,columns_by


def choose_best_tree_baseline(task,val):
    rows=[]; k=N_CLASSES[task]; cols=[f'p_class_{j}' for j in range(k)]
    for model in ['XGBoost','CatBoost','ExtraTrees']:
        vals=[]
        for seed in SEEDS:
            d=val[(val.task==task)&(val.model==model)&(val.seed==seed)].sort_values('RID',kind='mergesort')
            met=classification_metrics(d.y_true.to_numpy(dtype=int),d[cols].to_numpy(dtype=float))
            vals.append(float(met['f1_macro']))
            rows.append({'task':task,'model':model,'seed':seed,'validation_macro_f1':float(met['f1_macro'])})
        rows.append({'task':task,'model':model,'seed':'MEAN','validation_macro_f1':float(np.mean(vals))})
    summary=pd.DataFrame(rows)
    means=summary[summary.seed.astype(str).eq('MEAN')].sort_values('validation_macro_f1',ascending=False)
    return str(means.iloc[0].model),summary


def fit_reduced_tree(task,model_name,seed,pack,feature_cols,tree_dir):
    bundle_path=PKG/'checkpoints'/'tree_models'/f'{task}__{model_name}__seed{seed}.joblib'
    original=joblib.load(bundle_path)
    estimator=clone(original['model'])
    params=estimator.get_params()
    updates={}
    if 'random_state' in params: updates['random_state']=int(seed)
    if 'random_seed' in params: updates['random_seed']=int(seed)
    if 'n_jobs' in params: updates['n_jobs']=2
    if 'thread_count' in params: updates['thread_count']=2
    if 'verbose' in params: updates['verbose']=False
    if 'allow_writing_files' in params: updates['allow_writing_files']=False
    if updates: estimator.set_params(**updates)
    train_idx=pack.idx('train',eligible_only=True)
    Xtr=pack.frame.iloc[train_idx][feature_cols].apply(pd.to_numeric,errors='coerce')
    ytr=pack.frame.iloc[train_idx][pack.target].to_numpy(dtype=int)
    imputer=SimpleImputer(strategy='median',add_indicator=True)
    xtr_i=imputer.fit_transform(Xtr)
    scaler=StandardScaler().fit(xtr_i)
    xtr=scaler.transform(xtr_i)
    estimator.fit(xtr,ytr)
    artifact=tree_dir/f'{task}__{model_name}__seed{seed}__strict_proximal_removed.joblib'
    joblib.dump({'imputer':imputer,'scaler':scaler,'model':estimator,'feature_cols':list(feature_cols)},artifact,compress=3)
    return {'imputer':imputer,'scaler':scaler,'model':estimator,'feature_cols':list(feature_cols)},artifact


def predict_tree(bundle,frame,k):
    x=frame[bundle['feature_cols']].apply(pd.to_numeric,errors='coerce')
    xi=bundle['imputer'].transform(x); xs=bundle['scaler'].transform(xi)
    raw=bundle['model'].predict_proba(xs); classes=np.asarray(bundle['model'].classes_,dtype=int)
    p=np.zeros((len(frame),k),dtype=float)
    for j,c in enumerate(classes):p[:,int(c)]=raw[:,j]
    p=np.clip(p,1e-8,None);return p/p.sum(axis=1,keepdims=True)


def make_pred_frame(task,model,seed,split,frame,target,p):
    out=pd.DataFrame({'task':task,'model':model,'seed':int(seed),'split':split,'RID':frame.RID.astype(str).to_numpy(),
                      'y_true':frame[target].to_numpy(dtype=int),'y_pred':p.argmax(axis=1)})
    for j in range(p.shape[1]):out[f'p_class_{j}']=p[:,j]
    return out


def main():
    t0=time.time()
    val=pd.read_csv(ROOT/'validation_predictions_uncalibrated.csv',dtype={'RID':str})
    val['RID']=val.RID.astype(str); val['seed']=val.seed.astype(int)
    neural_manifest=json.loads((PKG/'neural_training_run_manifest.json').read_text())['immutable_manifest']
    config=TrainingConfig(**neural_manifest['training_config'])
    model_code_hash=sha256_text(sha256_file(PKG/'code'/'auaf_model_factory.py')+sha256_file(PKG/'code'/'auaf_safe_training_engine.py'))
    data_hashes={p.name:sha256_file(p) for p in DATA.glob('*.csv')}
    run_manifest={'stage':'4.7_circularity_sensitivity','setting':'strict_diagnosis_proximal_removed',
                  'removed_features':sorted(REMOVED),'seeds':SEEDS,'tasks':TASKS,
                  'training_config':config.to_dict(),'split_policy':'reuse frozen Stage 4.4 participant split',
                  'preprocessing':'train-only numeric median imputation and z-score scaling; per-modality availability mask',
                  'data_hashes':data_hashes,'model_code_sha256':model_code_hash}
    run_dir=OUT/'models'/'circularity_neural_runs'/'strict_proximal_removed'
    run_hash=initialize_run_directory(run_dir,run_manifest,resume=True)

    all_pred=[]; run_rows=[]; baseline_selection=[]; best_baselines={}; packs={}; feature_cols_by_task={}
    tree_dir=OUT/'models'/'circularity_tree_models';tree_dir.mkdir(parents=True,exist_ok=True)
    for task in TASKS:
        pack,columns_by=prepare_reduced_task(task);packs[task]=pack
        feature_cols=[f for mod in pack.modalities for f in pack.features_by_modality[mod]]
        feature_cols_by_task[task]=feature_cols
        best_tree,sel=choose_best_tree_baseline(task,val);best_baselines[task]=best_tree
        baseline_selection.append(sel)
        print(f'[{task}] reduced inputs: {len(feature_cols)}; best validation tree baseline: {best_tree}',flush=True)
        for seed in SEEDS:
            result=run_training_job(pack=pack,task_name=task,feature_set_name='S0_verified_asof_strict_proximal_removed',
                model_variant='AUAF',seed=seed,run_dir=run_dir,run_manifest_sha256=run_hash,
                model_factory=build_model,config=config,input_hashes=data_hashes,model_code_sha256=model_code_hash,resume=True)
            job_dir=Path(result['job_dir'])
            run_rows.append({'task':task,'model':'AUAF','seed':seed,'status':result['status'],
                             'best_epoch':result.get('best_epoch'),'best_validation_macro_f1':result.get('best_validation_macro_f1'),
                             'test_n':result.get('test_metrics',{}).get('n_test',None),'job_dir':str(job_dir)})
            for split in ['validation','test']:
                f=job_dir/f'predictions_{split}.csv'
                d=pd.read_csv(f,dtype={'RID':str});d['setting']='strict_proximal_removed';d['model']='AUAF';d['family']='neural'
                all_pred.append(d)
            # Best tree baseline is selected using validation predictions only; retain original Stage 4.5 hyperparameters.
            tree_bundle,tree_artifact=fit_reduced_tree(task,best_tree,seed,pack,feature_cols,tree_dir)
            for split in ['validation','test']:
                idx=pack.idx(split,eligible_only=True); frame=pack.frame.iloc[idx].reset_index(drop=True)
                p=predict_tree(tree_bundle,frame,N_CLASSES[task])
                d=make_pred_frame(task,best_tree,seed,split,frame,pack.target,p)
                d['setting']='strict_proximal_removed';d['family']='tree';all_pred.append(d)
            run_rows.append({'task':task,'model':best_tree,'seed':seed,'status':'TRAINED_ON_TRAIN_SPLIT_WITH_FROZEN_STAGE45_HYPERPARAMETERS',
                             'best_epoch':None,'best_validation_macro_f1':None,'test_n':len(pack.idx('test',eligible_only=True)),
                             'job_dir':str(tree_artifact)})
        print(f'[{task}] circularity retraining complete',flush=True)

    reduced=pd.concat(all_pred,ignore_index=True,sort=False)
    reduced.to_csv(OUT/'predictions'/'stage47_circularity_reduced_predictions_uncalibrated.csv.gz',index=False,compression='gzip')
    pd.DataFrame(run_rows).to_csv(OUT/'manifests'/'stage47_circularity_model_run_manifest.csv',index=False)
    pd.concat(baseline_selection,ignore_index=True).to_csv(OUT/'tables'/'stage47_best_tree_baseline_validation_selection.csv',index=False)
    (OUT/'manifests'/'stage47_circularity_best_baselines.json').write_text(json.dumps(best_baselines,indent=2),encoding='utf-8')

    # Fit temperature on reduced-model validation predictions only, then calibrate held-out test probabilities.
    cal_chunks=[]; reduced_metric_rows=[]; reduced_temp_rows=[]
    for task in TASKS:
      k=N_CLASSES[task]; cols=[f'p_class_{j}' for j in range(k)]
      task_data=reduced[reduced.task.eq(task)]
      for (model,seed),g in task_data.groupby(['model','seed'],sort=True):
        vd=g[g.split.eq('validation')].sort_values('RID',kind='mergesort')
        td=g[g.split.eq('test')].sort_values('RID',kind='mergesort')
        temp=fit_temperature(vd.y_true.to_numpy(dtype=int),vd[cols].to_numpy(dtype=float))
        pcal=apply_temperature(td[cols].to_numpy(dtype=float),temp)
        reduced_temp_rows.append({'task':task,'model':model,'seed':int(seed),'temperature':temp,
                                  'validation_n':len(vd),'test_eligible_n':len(td),'fit_split':'validation_only'})
        out=td[['task','model','seed','split','RID','y_true']].copy();out['setting']='strict_proximal_removed';out['temperature']=temp
        for j in range(k):out[f'p_class_{j}']=pcal[:,j]
        cal_chunks.append(out)
        met=classification_metrics(td.y_true.to_numpy(dtype=int),pcal)
        reduced_metric_rows.append({'task':task,'model':model,'seed':int(seed),'n_test_eligible':len(td),
                                    'temperature':temp,**met})
    red_cal=pd.concat(cal_chunks,ignore_index=True)
    red_cal.to_csv(OUT/'predictions'/'stage47_circularity_reduced_predictions_calibrated.csv.gz',index=False,compression='gzip')
    pd.DataFrame(reduced_temp_rows).to_csv(OUT/'tables'/'stage47_circularity_reduced_temperature_parameters.csv',index=False)
    pd.DataFrame(reduced_metric_rows).to_csv(OUT/'tables'/'stage47_circularity_reduced_test_metrics_seed_level.csv',index=False)

    # Compare standard and reduced models on exactly the common held-out participants.
    standard=pd.read_csv(OUT/'predictions'/'stage47_all_models_calibrated_test_predictions.csv.gz',dtype={'RID':str})
    standard['RID']=standard.RID.astype(str)
    effects=[]; perf=[]
    for cohort in ['full_test','matched_complete_case']:
      for task in TASKS:
        k=N_CLASSES[task]; cols=[f'p_class_{j}' for j in range(k)]
        for model in ['AUAF',best_baselines[task]]:
          for seed in SEEDS:
            r=red_cal[(red_cal.task==task)&(red_cal.model==model)&(red_cal.seed==seed)&(red_cal.split=='test')].copy()
            s=standard[(standard.cohort==cohort)&(standard.task==task)&(standard.model==model)&(standard.seed==seed)&(standard.missingness_requested==0.0)].copy()
            m=r[['RID','y_true',*cols]].merge(s[['RID','y_true',*cols]],on=['RID','y_true'],how='inner',validate='one_to_one',suffixes=('_reduced','_standard'))
            if m.empty:raise RuntimeError(f'No paired rows for circularity comparison {cohort}/{task}/{model}/{seed}')
            y=m.y_true.to_numpy(dtype=int);pr=m[[f'{c}_reduced' for c in cols]].to_numpy(dtype=float);ps=m[[f'{c}_standard' for c in cols]].to_numpy(dtype=float)
            mr=classification_metrics(y,pr);ms=classification_metrics(y,ps)
            perf.append({'cohort':cohort,'task':task,'model':model,'seed':seed,'n_paired':len(m),'standard_setting':'all_Stage45_features',
                         'reduced_setting':'strict_proximal_removed',**{f'{z}_standard':ms[z] for z in METRICS},**{f'{z}_reduced':mr[z] for z in METRICS}})
          # bootstrap mean effect over five seeds on common participants (same bootstrap weights per seed and model)
          common_ids=None; y_ref=None; seed_arrays=[]
          for seed in SEEDS:
            r=red_cal[(red_cal.task==task)&(red_cal.model==model)&(red_cal.seed==seed)&(red_cal.split=='test')]
            s=standard[(standard.cohort==cohort)&(standard.task==task)&(standard.model==model)&(standard.seed==seed)&(standard.missingness_requested==0.0)]
            m=r[['RID','y_true',*cols]].merge(s[['RID','y_true',*cols]],on=['RID','y_true'],how='inner',validate='one_to_one',suffixes=('_reduced','_standard')).sort_values('RID',kind='mergesort')
            ids=m.RID.astype(str).to_numpy(); y=m.y_true.to_numpy(dtype=int)
            if common_ids is None:common_ids=ids;y_ref=y
            elif not np.array_equal(ids,common_ids) or not np.array_equal(y,y_ref):raise RuntimeError('Circularity paired participant roster differs by seed')
            seed_arrays.append((m[[f'{c}_reduced' for c in cols]].to_numpy(dtype=float),m[[f'{c}_standard' for c in cols]].to_numpy(dtype=float)))
          W=make_stratified_weights(y_ref,BOOTSTRAP_N,20261116+TASKS.index(task)*100+(0 if cohort=='full_test' else 100))
          point_diff={m:[] for m in METRICS}; boot_diff={m:[] for m in METRICS}
          for pr,ps in seed_arrays:
            mr=classification_metrics(y_ref,pr); ms=classification_metrics(y_ref,ps)
            br=bootstrap_metrics(y_ref,pr,W); bs=bootstrap_metrics(y_ref,ps,W)
            for metric in METRICS:
              point_diff[metric].append(mr[metric]-ms[metric]); boot_diff[metric].append(br[metric]-bs[metric])
          for metric in METRICS:
            estimate=float(np.mean(point_diff[metric])); bd=np.mean(np.stack(boot_diff[metric]),axis=0);lo,hi=ci(bd)
            effects.append({'cohort':cohort,'task':task,'model':model,'comparison':'strict_proximal_removed_minus_all_features',
                            'metric':metric,'delta_reduced_minus_standard':estimate,'ci95_low':lo,'ci95_high':hi,
                            'p_value_bootstrap':p_from_bootstrap_distribution(bd),'n_paired':len(y_ref),
                            'seed_n':5,'bootstrap_repetitions':BOOTSTRAP_N})
    perfdf=pd.DataFrame(perf);perfdf.to_csv(OUT/'tables'/'stage47_circularity_standard_vs_reduced_seed_metrics.csv',index=False)
    eff=pd.DataFrame(effects);eff['p_fdr_bh']=bh_adjust(eff.p_value_bootstrap);eff['significant_after_FDR_0_05']=eff.p_fdr_bh<.05
    eff.to_csv(OUT/'tables'/'stage47_circularity_standard_vs_reduced_paired_effects.csv',index=False)

    # Baseline severity strata for conversion, using training medians only.
    conv=pd.read_csv(DATA/'mci_conversion_911_asof.csv',dtype={'RID':str})
    conv['split']=conv['split'].astype(str).str.lower()
    train=conv[conv.split.eq('train')];test=conv[conv.split.eq('test')]
    severity_rows=[]
    for feature in ['CDRSB','FAQTOTAL']:
      med=float(pd.to_numeric(train[feature],errors='coerce').median())
      tmp=test[['RID',feature]].copy();tmp[feature]=pd.to_numeric(tmp[feature],errors='coerce')
      tmp['severity_stratum']=np.where(tmp[feature].isna(),'severity_missing',np.where(tmp[feature]<=med,'at_or_below_training_median','above_training_median'))
      for cohort in ['full_test','matched_complete_case']:
        for model in ['AUAF',best_baselines['mci_conversion']]:
          for seed in SEEDS:
            p=standard[(standard.cohort==cohort)&(standard.task=='mci_conversion')&(standard.model==model)&(standard.seed==seed)&(standard.missingness_requested==0.0)].copy()
            m=p.merge(tmp[['RID',feature,'severity_stratum']],on='RID',how='inner',validate='one_to_one')
            for stratum,g in m.groupby('severity_stratum'):
              y=g.y_true.to_numpy(dtype=int);pro=g[['p_class_0','p_class_1']].to_numpy(dtype=float)
              met=classification_metrics(y,pro)
              severity_rows.append({'cohort':cohort,'task':'mci_conversion','severity_feature':feature,'training_median_threshold':med,
                                    'model':model,'seed':seed,'severity_stratum':stratum,'n':len(g),
                                    'stable_n':int((y==0).sum()),'converter_n':int((y==1).sum()),**met})
    pd.DataFrame(severity_rows).to_csv(OUT/'tables'/'stage47_conversion_severity_stratified_performance.csv',index=False)

    # Biomarker timing summary from the Stage 4.4 audit, with negative lag = pre-index sample.
    bio=pd.read_csv(ROOT/'input44'/'AUAF_Stage4_4_Temporal_Clean_Package'/'outputs'/'biomarker_temporal_audit.csv')
    bio_rows=[]
    for task,g in bio.groupby('task'):
      lag=pd.to_numeric(g.BIOMARKER_LAG_DAYS,errors='coerce').dropna()
      bio_rows.append({'task':task,'rows':len(g),'biomarker_available_asof_n':int(g.BIOMARKER_AVAILABLE_AS_OF_INDEX.astype(bool).sum()),
                       'after_index_n':int(g.BIOMARKER_AFTER_INDEX.astype(bool).sum()),'dated_biomarker_rows':len(lag),
                       'lag_days_median':float(lag.median()) if len(lag) else np.nan,'lag_days_q1':float(lag.quantile(.25)) if len(lag) else np.nan,
                       'lag_days_q3':float(lag.quantile(.75)) if len(lag) else np.nan,'lag_days_min':float(lag.min()) if len(lag) else np.nan,
                       'lag_days_max':float(lag.max()) if len(lag) else np.nan})
    pd.DataFrame(bio_rows).to_csv(OUT/'tables'/'stage47_biomarker_temporal_lag_summary.csv',index=False)

    # Update final manifest.
    manifest={'stage':'4.7','status':'EXECUTED_WITH_STATISTICAL_VALIDATION_AND_CIRCULARITY_SENSITIVITY',
              'strict_circularity_removed_features':sorted(REMOVED),'best_tree_baselines_selected_on_validation':best_baselines,
              'neural_sensitivity_jobs':15,'tree_sensitivity_jobs':15,'seeds':SEEDS,'tasks':TASKS,
              'training_config':config.to_dict(),'model_selection':'Validation macro-F1; test not used for selection',
              'tree_selection':'highest mean validation macro-F1 across the three Stage 4.5 tree baselines; same original Stage 4.5 hyperparameters frozen for reduced-feature retraining',
              'reduced_preprocessing':'train-only imputation/scaling; refit after raw proximal-feature removal',
              'calibration':'reduced model temperature fitted on reduced validation predictions only; standard model temperature fitted on original validation predictions only',
              'paired_comparison':'same held-out participants common to standard and reduced; participant stratified bootstrap; five seeds averaged, not independent folds',
              'severity_strata':'CDRSB and FAQTOTAL thresholds are medians calculated on training participants only',
              'limitations':['No external validation cohort.','Strict feature removal addresses named CDR, FAQ, MMSE and MoCA instrument blocks only; it does not prove absence of every possible information leak.','Feature-specific visit/source-row identifiers are not available; source table and instrument mapping come from Stage 4.4 feature reconciliation.','Small matched-complete-case test cohorts, especially conversion, make subgroup estimates imprecise.']}
    (OUT/'manifests'/'stage47_circularity_retraining_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print('CIRCULARITY SENSITIVITY COMPLETE')
    print(json.dumps({'best_baselines':best_baselines,'neural_jobs':15,'tree_jobs':15,'paired_effect_rows':len(eff),
                      'elapsed_seconds':round(time.time()-t0,2)},indent=2),flush=True)

if __name__=='__main__':main()
