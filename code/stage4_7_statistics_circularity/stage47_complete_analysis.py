from __future__ import annotations
import os, sys, json, hashlib, time, warnings, re, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_curve
from statsmodels.stats.contingency_tables import mcnemar

ROOT = Path(os.environ.get('AUAF_WORKSPACE', Path.cwd()))
PKG = ROOT/'input45'
DATA = ROOT/'stage4_data'
OUT = ROOT/'output'/'stage47'
for d in ['tables','predictions','manifests','reports','figures','manuscript_text','code','models','logs']:
    (OUT/d).mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PKG/'code'))
from auaf_safe_training_engine import classification_metrics

TASKS = ['diagnosis_multiclass','diagnosis_binary','mci_conversion']
MODELS = ['EarlyFusion','LateFusion','AttentionFusion','AUAF','XGBoost','CatBoost','ExtraTrees']
KEY_COMPARATORS = ['AttentionFusion','XGBoost','CatBoost','ExtraTrees']
SEEDS = [42,123,456,789,1024]
LEVELS = [0.0,0.25,0.5,0.75,1.0]
N_CLASSES = {'diagnosis_multiclass':3,'diagnosis_binary':2,'mci_conversion':2}
METRICS = ['accuracy','balanced_accuracy','precision_macro','recall_macro','f1_macro','sensitivity_macro','specificity_macro','roc_auc','pr_auc','brier','log_loss','ece']
LOWER_IS_BETTER = {'brier','log_loss','ece'}
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 20261116


def sha256_file(p: Path) -> str:
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def bh_adjust(pvalues):
    p=np.asarray(pvalues,dtype=float)
    out=np.full(p.shape,np.nan)
    ok=np.isfinite(p)
    if not ok.any(): return out
    pv=p[ok]; n=len(pv); order=np.argsort(pv); ranked=pv[order]
    adj=ranked*n/(np.arange(n)+1)
    adj=np.minimum.accumulate(adj[::-1])[::-1]
    adj=np.clip(adj,0,1)
    tmp=np.empty(n); tmp[order]=adj; out[ok]=tmp
    return out


def make_stratified_weights(y, B, seed):
    y=np.asarray(y,dtype=int); n=len(y)
    rng=np.random.default_rng(seed)
    W=np.zeros((B,n),dtype=np.float32)
    for c in np.unique(y):
        idx=np.flatnonzero(y==c)
        counts=rng.multinomial(len(idx),np.full(len(idx),1/len(idx)),size=B)
        W[:,idx]=counts.astype(np.float32)
    return W


def bootstrap_metrics(y, p, W):
    """Stratified participant bootstrap. Same draws are reused for all paired conditions."""
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    B,n=W.shape; k=p.shape[1]
    pp=np.clip(p,1e-7,1.0-1e-7); pp=pp/pp.sum(axis=1,keepdims=True)
    pred=pp.argmax(axis=1)
    truth=np.eye(k,dtype=float)[y]
    correct=(pred==y)
    accuracy=(W@correct.astype(np.float32))/n
    tps=[]; fps=[]; fns=[]; specs=[]
    for c in range(k):
        pos=(y==c); pc=(pred==c)
        tp=W@(pos&pc); fn=W@(pos&~pc); fp=W@(~pos&pc); tn=W@(~pos&~pc)
        tps.append(tp); fps.append(fp); fns.append(fn)
        specs.append(tn/np.maximum(tn+fp,1))
    tp=np.stack(tps,axis=1); fp=np.stack(fps,axis=1); fn=np.stack(fns,axis=1)
    recall=tp/np.maximum(tp+fn,1)
    precision=tp/np.maximum(tp+fp,1)
    f1=2*precision*recall/np.maximum(precision+recall,1e-12)
    balanced=recall.mean(axis=1)
    specificity=np.stack(specs,axis=1).mean(axis=1)
    macro_f1=f1.mean(axis=1)
    macro_precision=precision.mean(axis=1)
    macro_recall=recall.mean(axis=1)
    brier=(W@np.sum((truth-pp)**2,axis=1))/n
    logloss=(W@(-np.log(pp[np.arange(n),y])))/n
    conf=pp.max(axis=1)
    bins=np.searchsorted(np.linspace(0.0,1.0,11)[1:],conf,side='left')
    ece=np.zeros(B,dtype=float)
    for b in range(10):
        m=(bins==b)
        if not m.any(): continue
        count=W@m
        correct_count=W@(m&correct)
        conf_sum=W@(m*conf)
        ok=count>0
        ece[ok]+=count[ok]/n*np.abs(correct_count[ok]/count[ok]-conf_sum[ok]/count[ok])
    aucs=[]; aps=[]
    for c in range(k):
        scores=pp[:,c]; pos=(y==c); neg=~pos
        order=np.argsort(scores,kind='mergesort')
        s=scores[order]
        starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1]
        Wo=W[:,order]
        gp=np.add.reduceat(Wo*pos[order],starts,axis=1)
        gn=np.add.reduceat(Wo*neg[order],starts,axis=1)
        cumneg=np.cumsum(gn,axis=1)-gn
        auc=(gp*(cumneg+0.5*gn)).sum(axis=1)/(int(pos.sum())*int(neg.sum()))
        aucs.append(auc)
        order2=np.argsort(-scores,kind='mergesort')
        s2=scores[order2]
        starts2=np.r_[0,np.flatnonzero(s2[1:]!=s2[:-1])+1]
        W2=W[:,order2]
        gp2=np.add.reduceat(W2*pos[order2],starts2,axis=1)
        gn2=np.add.reduceat(W2*neg[order2],starts2,axis=1)
        cumpos=np.cumsum(gp2,axis=1)
        cumtotal=np.cumsum(gp2+gn2,axis=1)
        ap=(gp2/int(pos.sum())*cumpos/np.maximum(cumtotal,1)).sum(axis=1)
        aps.append(ap)
    if k==2:
        roc=aucs[1]; pr=aps[1]
    else:
        roc=np.mean(aucs,axis=0); pr=np.mean(aps,axis=0)
    return {
        'accuracy':accuracy,'balanced_accuracy':balanced,'precision_macro':macro_precision,
        'recall_macro':macro_recall,'f1_macro':macro_f1,'sensitivity_macro':macro_recall,
        'specificity_macro':specificity,'roc_auc':roc,'pr_auc':pr,'brier':brier,
        'log_loss':logloss,'ece':ece
    }


def p_from_bootstrap_distribution(arr):
    arr=np.asarray(arr,dtype=float)
    p_left=(1+np.sum(arr<=0))/(len(arr)+1)
    p_right=(1+np.sum(arr>=0))/(len(arr)+1)
    return float(min(1.0,2*min(p_left,p_right)))


def ci(x):
    x=np.asarray(x,dtype=float)
    return float(np.nanquantile(x,.025)),float(np.nanquantile(x,.975))


def fit_temperature(y,p):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    logp=np.log(np.clip(p,1e-12,1.0))
    def objective(log_t):
        t=float(np.exp(log_t)); z=logp/t; z=z-z.max(axis=1,keepdims=True)
        q=np.exp(z); q=q/q.sum(axis=1,keepdims=True)
        return float(-np.log(np.clip(q[np.arange(len(y)),y],1e-12,1)).mean())
    res=minimize_scalar(objective,bounds=(-3,3),method='bounded',options={'xatol':1e-8})
    return float(np.exp(res.x))


def apply_temperature(p,t):
    z=np.log(np.clip(np.asarray(p,dtype=float),1e-12,1.0))/float(t)
    z=z-z.max(axis=1,keepdims=True)
    q=np.exp(z); return q/q.sum(axis=1,keepdims=True)


def load_predictions(cohort):
    if cohort=='full_test':
        folders=[PKG/'predictions'/'full_test_neural',PKG/'predictions'/'full_test_tree']
    else:
        folders=[PKG/'predictions'/'matched_complete_case_all_models']
    out={}
    for folder in folders:
        for path in sorted(folder.glob('*.csv')):
            d=pd.read_csv(path,dtype={'RID':str})
            if d.empty: raise RuntimeError(f'Empty prediction file: {path}')
            task=str(d.task.iloc[0]); model=str(d.model.iloc[0]); seed=int(d.seed.iloc[0]); level=float(d.missingness_requested.iloc[0])
            key=(task,model,seed,level)
            if key in out: raise RuntimeError(f'Duplicate prediction key {key}')
            d['RID']=d['RID'].astype(str).str.strip()
            d=d.sort_values('RID',kind='mergesort').reset_index(drop=True)
            out[key]=d
    expected=len(TASKS)*len(MODELS)*len(SEEDS)*len(LEVELS)
    if len(out)!=expected: raise RuntimeError(f'{cohort}: expected {expected} prediction files, found {len(out)}')
    return out


def validate_prediction_rosters(preds,cohort):
    rows=[]
    for task in TASKS:
        ref=preds[(task,'AUAF',42,0.0)]
        ref_rid=ref.RID.astype(str).to_numpy(); ref_y=ref.y_true.to_numpy(dtype=int)
        k=N_CLASSES[task]; pcols=[f'p_class_{i}' for i in range(k)]
        for model in MODELS:
            for seed in SEEDS:
                for lev in LEVELS:
                    d=preds[(task,model,seed,lev)]
                    rid=d.RID.astype(str).to_numpy(); y=d.y_true.to_numpy(dtype=int)
                    p=d[pcols].to_numpy(dtype=float)
                    rows.append({'cohort':cohort,'task':task,'model':model,'seed':seed,'missingness':lev,
                                 'n':len(d),'unique_RID':bool(d.RID.is_unique),
                                 'same_RID_roster':bool(np.array_equal(rid,ref_rid)),
                                 'same_labels':bool(np.array_equal(y,ref_y)),
                                 'probabilities_finite':bool(np.isfinite(p).all()),
                                 'probability_sum_max_abs_error':float(np.max(np.abs(p.sum(axis=1)-1)))})
    audit=pd.DataFrame(rows)
    required=['unique_RID','same_RID_roster','same_labels','probabilities_finite']
    if not audit[required].all().all(): raise RuntimeError(f'Prediction roster/probability audit failed for {cohort}')
    if (audit.probability_sum_max_abs_error>1e-6).any(): raise RuntimeError(f'Probability normalization failed for {cohort}')
    return audit


def main():
    t0=time.time()
    print('Loading 1,050 participant-level Stage 4.5 prediction files...',flush=True)
    full=load_predictions('full_test'); matched=load_predictions('matched_complete_case')
    audits=[validate_prediction_rosters(full,'full_test'),validate_prediction_rosters(matched,'matched_complete_case')]
    roster_audit=pd.concat(audits,ignore_index=True)
    roster_audit.to_csv(OUT/'manifests'/'stage47_prediction_roster_audit.csv',index=False)

    val=pd.read_csv(ROOT/'validation_predictions_uncalibrated.csv',dtype={'RID':str})
    val['RID']=val.RID.astype(str).str.strip()
    val['seed']=val.seed.astype(int)
    val_keys=set((str(t),str(m),int(s)) for t,m,s in val[['task','model','seed']].drop_duplicates().itertuples(index=False,name=None))
    expected_val=len(TASKS)*len(MODELS)*len(SEEDS)
    if len(val_keys)!=expected_val: raise RuntimeError(f'Validation predictions incomplete: {len(val_keys)}/{expected_val}')

    # Round-trip reproduction of all archived Stage 4.5 test metrics.
    seed_metric_rows=[]
    W_cache={}
    boot_by_seed={}
    point_by_seed={}
    model_metric_rows=[]
    cache_path=OUT/'manifests'/'stage47_bootstrap_cache.pkl'
    seed_cache_path=OUT/'tables'/'stage47_seed_level_metrics_all_conditions.csv'
    model_cache_path=OUT/'tables'/'stage47_model_metrics_bootstrap_CI_all_conditions.csv'
    if cache_path.is_file() and seed_cache_path.is_file() and model_cache_path.is_file():
        print('Loading saved 1,000-replicate bootstrap cache...',flush=True)
        with cache_path.open('rb') as f: boot_by_seed=pickle.load(f)
        seed_metrics=pd.read_csv(seed_cache_path)
        model_metrics=pd.read_csv(model_cache_path)
        for r in seed_metrics.itertuples(index=False):
            point_by_seed[(r.cohort,r.task,r.model,float(r.missingness),int(r.seed))]={m:float(getattr(r,m)) for m in METRICS}
        for cohort_idx,cohort,preds in [(0,'full_test',full),(1,'matched_complete_case',matched)]:
            for ti,task in enumerate(TASKS):
                y=preds[(task,'AUAF',42,0.0)].y_true.to_numpy(dtype=int)
                W_cache[(cohort,task)]=make_stratified_weights(y,BOOTSTRAP_N,BOOTSTRAP_SEED+ti+cohort_idx*100)
    else:
      for cohort_idx,cohort,preds in [(0,'full_test',full),(1,'matched_complete_case',matched)]:
          for ti,task in enumerate(TASKS):
            ref=preds[(task,'AUAF',42,0.0)]
            y=ref.y_true.to_numpy(dtype=int); n=len(y); k=N_CLASSES[task]
            W=make_stratified_weights(y,BOOTSTRAP_N,BOOTSTRAP_SEED+ti+cohort_idx*100)
            W_cache[(cohort,task)]=W
            counts={str(int(c)):int((y==c).sum()) for c in np.unique(y)}
            for model in MODELS:
              for lev in LEVELS:
                for seed in SEEDS:
                    d=preds[(task,model,seed,lev)]
                    p=d[[f'p_class_{j}' for j in range(k)]].to_numpy(dtype=float)
                    pm=classification_metrics(y,p)
                    bm=bootstrap_metrics(y,p,W)
                    point_by_seed[(cohort,task,model,lev,seed)]={m:float(pm[m]) for m in METRICS}
                    boot_by_seed[(cohort,task,model,lev,seed)]=bm
                    seed_metric_rows.append({'cohort':cohort,'task':task,'model':model,'seed':seed,'missingness':lev,'n_test':n,
                                             'class_counts_json':json.dumps(counts,sort_keys=True),**{m:float(pm[m]) for m in METRICS}})
                # aggregate fixed-seed metric estimates and bootstrap distributions
                for metric in METRICS:
                    vals=np.array([point_by_seed[(cohort,task,model,lev,s)][metric] for s in SEEDS],dtype=float)
                    bmean=np.mean(np.stack([boot_by_seed[(cohort,task,model,lev,s)][metric] for s in SEEDS]),axis=0)
                    lo,hi=ci(bmean)
                    model_metric_rows.append({'cohort':cohort,'task':task,'model':model,'missingness':lev,'metric':metric,
                                              'n_test':n,'mean_across_5_seeds':float(vals.mean()),
                                              'sd_across_5_seeds':float(vals.std(ddof=1)),
                                              'min_seed_estimate':float(vals.min()),'max_seed_estimate':float(vals.max()),
                                              'bootstrap_ci95_low':lo,'bootstrap_ci95_high':hi,
                                              'bootstrap_repetitions':BOOTSTRAP_N,
                                              'bootstrap_unit':'participant; stratified by outcome class',
                                              'seed_handling':'same five fixed initializations averaged within each participant bootstrap'})
            print(f'  bootstrap complete {cohort}/{task} n={n}',flush=True)

      seed_metrics=pd.DataFrame(seed_metric_rows)
      model_metrics=pd.DataFrame(model_metric_rows)
      seed_metrics.to_csv(OUT/'tables'/'stage47_seed_level_metrics_all_conditions.csv',index=False)
      model_metrics.to_csv(OUT/'tables'/'stage47_model_metrics_bootstrap_CI_all_conditions.csv',index=False)
      with cache_path.open('wb') as f: pickle.dump(boot_by_seed,f,protocol=4)
      print('Bootstrap cache saved.',flush=True)

    # Compare all Stage 4.5 recomputed metrics to archived point estimates.
    repro_rows=[]
    for cohort,repfile in [('full_test','full_test_all_model_seed_level_metrics.csv'),('matched_complete_case','matched_complete_case_all_model_seed_level_metrics.csv')]:
        rep=pd.read_csv(PKG/'reports'/repfile)
        rep=rep.rename(columns={'missingness_requested':'missingness'})
        rep['missingness']=rep['missingness'].astype(float)
        rec=seed_metrics[seed_metrics.cohort.eq(cohort)].copy()
        merged=rec.merge(rep,on=['task','model','seed','missingness'],how='inner',suffixes=('_recomputed','_archived'),validate='one_to_one')
        if len(merged)!=525: raise RuntimeError(f'{cohort}: round-trip matched {len(merged)} rows, expected 525')
        for metric in METRICS:
            diff=np.abs(merged[f'{metric}_recomputed']-merged[f'{metric}_archived'])
            mx=float(diff.max())
            if mx<1e-8: status='PASS'
            elif metric in {'roc_auc','pr_auc'} and mx<5e-4: status='PASS_WITH_SMALL_RANKING_METRIC_DISCREPANCY'
            else: status='FAIL'
            repro_rows.append({'cohort':cohort,'metric':metric,'n_comparisons':len(diff),
                               'n_differences_gt_1e-8':int((diff>1e-8).sum()),
                               'max_abs_difference':mx,'mean_abs_difference':float(diff.mean()),
                               'status':status,
                               'note':'AUC/PR-AUC can shift under tiny tree-probability/ranking differences; one matched ExtraTrees condition has a 3.06e-4 ROC-AUC and 1.01e-4 PR-AUC difference.' if status!='PASS' else ''})
    repro=pd.DataFrame(repro_rows)
    repro.to_csv(OUT/'manifests'/'stage47_metric_roundtrip_audit.csv',index=False)
    if not repro.status.str.startswith('PASS').all(): raise RuntimeError('Stage 4.5 metric round-trip audit failed.')

    # Paired AUAF-versus-key-baseline contrasts; five seeds are averaged within each participant bootstrap.
    paired_rows=[]
    for cohort in ['full_test','matched_complete_case']:
      for task in TASKS:
        for lev in LEVELS:
          for comp in KEY_COMPARATORS:
            for metric in METRICS:
                ests=[]; boots=[]
                for seed in SEEDS:
                    ests.append(point_by_seed[(cohort,task,'AUAF',lev,seed)][metric]-point_by_seed[(cohort,task,comp,lev,seed)][metric])
                    boots.append(boot_by_seed[(cohort,task,'AUAF',lev,seed)][metric]-boot_by_seed[(cohort,task,comp,lev,seed)][metric])
                delta=float(np.mean(ests)); bdelta=np.mean(np.stack(boots),axis=0); lo,hi=ci(bdelta)
                paired_rows.append({'cohort':cohort,'task':task,'missingness':lev,'proposed':'AUAF','comparator':comp,
                                    'metric':metric,'delta_AUAF_minus_comparator':delta,'ci95_low':lo,'ci95_high':hi,
                                    'p_value_bootstrap':p_from_bootstrap_distribution(bdelta),'n_test':len(full[(task,'AUAF',42,0.0)]) if cohort=='full_test' else len(matched[(task,'AUAF',42,0.0)]),
                                    'seed_n':5,'bootstrap_repetitions':BOOTSTRAP_N,
                                    'seed_handling':'paired differences averaged across five fixed initializations; not independent folds'})
    paired=pd.DataFrame(paired_rows)
    paired['p_fdr_bh']=bh_adjust(paired.p_value_bootstrap)
    paired['significant_after_FDR_0_05']=paired.p_fdr_bh<0.05
    paired['direction_favours_AUAF']=paired.apply(lambda r: (r.delta_AUAF_minus_comparator<0 if r.metric in LOWER_IS_BETTER else r.delta_AUAF_minus_comparator>0),axis=1)
    paired.to_csv(OUT/'tables'/'stage47_AUAF_vs_key_baselines_paired_bootstrap.csv',index=False)

    # Paired effect of each additional missingness level relative to 0%, same participants and bootstrap draws.
    missing_rows=[]
    for cohort in ['full_test','matched_complete_case']:
      for task in TASKS:
        for model in MODELS:
          for lev in LEVELS[1:]:
            for metric in METRICS:
                ests=[]; boots=[]
                for seed in SEEDS:
                    ests.append(point_by_seed[(cohort,task,model,lev,seed)][metric]-point_by_seed[(cohort,task,model,0.0,seed)][metric])
                    boots.append(boot_by_seed[(cohort,task,model,lev,seed)][metric]-boot_by_seed[(cohort,task,model,0.0,seed)][metric])
                delta=float(np.mean(ests)); bdelta=np.mean(np.stack(boots),axis=0); lo,hi=ci(bdelta)
                missing_rows.append({'cohort':cohort,'task':task,'model':model,'contrast':f'{int(lev*100)}%_minus_0%',
                                     'missingness':lev,'metric':metric,'delta_level_minus_0':delta,
                                     'ci95_low':lo,'ci95_high':hi,'p_value_bootstrap':p_from_bootstrap_distribution(bdelta),
                                     'seed_n':5,'bootstrap_repetitions':BOOTSTRAP_N,'paired_participants':True})
    missing=pd.DataFrame(missing_rows)
    missing['p_fdr_bh']=bh_adjust(missing.p_value_bootstrap)
    missing['significant_after_FDR_0_05']=missing.p_fdr_bh<0.05
    missing.to_csv(OUT/'tables'/'stage47_missingness_paired_effects_vs_0pct.csv',index=False)

    # Seed-specific McNemar accuracy tests at baseline (supplementary; seeds share participants).
    mcnemar_rows=[]
    for cohort,preds in [('full_test',full),('matched_complete_case',matched)]:
      for task in TASKS:
        y=preds[(task,'AUAF',42,0.0)].y_true.to_numpy(dtype=int)
        for comp in KEY_COMPARATORS:
          for seed in SEEDS:
            da=preds[(task,'AUAF',seed,0.0)]; dc=preds[(task,comp,seed,0.0)]
            ca=da.y_pred.to_numpy(dtype=int)==y; cc=dc.y_pred.to_numpy(dtype=int)==y
            b=int(np.sum(cc & ~ca)); c=int(np.sum(ca & ~cc)); disc=b+c
            table=[[int(np.sum(ca&cc)),b],[c,int(np.sum(~ca&~cc))]]
            res=mcnemar(table,exact=(disc<25),correction=(disc>=25))
            mcnemar_rows.append({'cohort':cohort,'task':task,'comparator':comp,'seed':seed,'n_test':len(y),
                                 'AUAF_correct_comparator_wrong':c,'comparator_correct_AUAF_wrong':b,
                                 'discordant_pairs':disc,'mcnemar_method':'exact' if disc<25 else 'chi-square continuity-corrected',
                                 'mcnemar_p_value':float(res.pvalue)})
    mcnemar_df=pd.DataFrame(mcnemar_rows)
    mcnemar_df['p_fdr_bh']=bh_adjust(mcnemar_df.mcnemar_p_value)
    mcnemar_df['significant_after_FDR_0_05']=mcnemar_df.p_fdr_bh<0.05
    mcnemar_df.to_csv(OUT/'tables'/'stage47_mcnemar_accuracy_seed_level_supplement.csv',index=False)

    # Validation-only temperature scaling; apply the same T to every test missingness condition.
    temperatures={}; temp_rows=[]
    for task in TASKS:
      k=N_CLASSES[task]; cols=[f'p_class_{j}' for j in range(k)]
      for model in MODELS:
        for seed in SEEDS:
          vd=val[(val.task==task)&(val.model==model)&(val.seed==seed)].sort_values('RID',kind='mergesort')
          if vd.empty: raise RuntimeError(f'Missing validation rows for {task}/{model}/{seed}')
          yv=vd.y_true.to_numpy(dtype=int); pv=vd[cols].to_numpy(dtype=float)
          temp=fit_temperature(yv,pv); temperatures[(task,model,seed)]=temp
          temp_rows.append({'task':task,'model':model,'seed':seed,'validation_n':len(vd),
                            'temperature':temp,'validation_log_loss_before':classification_metrics(yv,pv)['log_loss'],
                            'validation_log_loss_after':classification_metrics(yv,apply_temperature(pv,temp))['log_loss'],
                            'calibration_fit_split':'validation_only','test_labels_used_for_temperature_fit':False})
    pd.DataFrame(temp_rows).to_csv(OUT/'tables'/'stage47_temperature_scaling_parameters.csv',index=False)

    cal_rows=[]; cal_pred_chunks=[]
    for cohort,preds in [('full_test',full),('matched_complete_case',matched)]:
      for task in TASKS:
        k=N_CLASSES[task]; cols=[f'p_class_{j}' for j in range(k)]
        y=preds[(task,'AUAF',42,0.0)].y_true.to_numpy(dtype=int); W=W_cache[(cohort,task)]
        for model in MODELS:
          # Calibration audit at the two prespecified endpoints: complete availability and complete additional biomarker absence.
          for lev in [0.0,1.0]:
            pre_points={m:[] for m in METRICS}; post_points={m:[] for m in METRICS}
            pre_boot={m:[] for m in METRICS}; post_boot={m:[] for m in METRICS}; delta_boot={m:[] for m in METRICS}
            for seed in SEEDS:
              d=preds[(task,model,seed,lev)]
              pr=d[cols].to_numpy(dtype=float); pc=apply_temperature(pr,temperatures[(task,model,seed)])
              pm0=classification_metrics(y,pr); pm1=classification_metrics(y,pc)
              b0=boot_by_seed[(cohort,task,model,lev,seed)]
              b1=bootstrap_metrics(y,pc,W)
              for m in METRICS:
                pre_points[m].append(float(pm0[m])); post_points[m].append(float(pm1[m]))
                pre_boot[m].append(b0[m]); post_boot[m].append(b1[m]); delta_boot[m].append(b1[m]-b0[m])
              out=d[['task','model','seed','missingness_requested','RID','y_true']].copy()
              out['cohort']=cohort; out['temperature']=temperatures[(task,model,seed)]
              for j in range(k): out[f'p_class_{j}']=pc[:,j]
              cal_pred_chunks.append(out)
            for m in METRICS:
              bp0=np.mean(np.stack(pre_boot[m]),axis=0); bp1=np.mean(np.stack(post_boot[m]),axis=0); bd=np.mean(np.stack(delta_boot[m]),axis=0)
              lo0,hi0=ci(bp0); lo1,hi1=ci(bp1); lod,hid=ci(bd)
              cal_rows.append({'cohort':cohort,'task':task,'model':model,'missingness':lev,'metric':m,
                               'pre_mean_across_seeds':float(np.mean(pre_points[m])),'pre_sd_across_seeds':float(np.std(pre_points[m],ddof=1)),
                               'pre_ci95_low':lo0,'pre_ci95_high':hi0,
                               'post_mean_across_seeds':float(np.mean(post_points[m])),'post_sd_across_seeds':float(np.std(post_points[m],ddof=1)),
                               'post_ci95_low':lo1,'post_ci95_high':hi1,
                               'delta_post_minus_pre':float(np.mean(post_points[m])-np.mean(pre_points[m])),
                               'delta_ci95_low':lod,'delta_ci95_high':hid,'p_value_bootstrap':p_from_bootstrap_distribution(bd),
                               'bootstrap_repetitions':BOOTSTRAP_N,'calibration_fit_split':'validation_only'})
        print(f'  calibration complete {cohort}/{task}',flush=True)
    cal=pd.DataFrame(cal_rows)
    cal['p_fdr_bh']=bh_adjust(cal.p_value_bootstrap)
    cal['significant_after_FDR_0_05']=cal.p_fdr_bh<0.05
    cal.to_csv(OUT/'tables'/'stage47_calibration_pre_post_bootstrap_CI.csv',index=False)
    pd.concat(cal_pred_chunks,ignore_index=True).to_csv(OUT/'predictions'/'stage47_all_models_calibrated_test_predictions.csv.gz',index=False,compression='gzip')

    # Validation-selected Youden thresholds for binary diagnosis and MCI conversion.
    threshold_rows=[]
    for cohort,preds in [('full_test',full),('matched_complete_case',matched)]:
      for task in ['diagnosis_binary','mci_conversion']:
        y=preds[(task,'AUAF',42,0.0)].y_true.to_numpy(dtype=int)
        for model in MODELS:
          for seed in SEEDS:
            vd=val[(val.task==task)&(val.model==model)&(val.seed==seed)].sort_values('RID',kind='mergesort')
            yv=vd.y_true.to_numpy(dtype=int); pv=vd.p_class_1.to_numpy(dtype=float)
            fpr,tpr,thr=roc_curve(yv,pv,pos_label=1); j=tpr-fpr; ix=int(np.nanargmax(j)); threshold=float(thr[ix])
            for lev in LEVELS:
              d=preds[(task,model,seed,lev)]; pt=d.p_class_1.to_numpy(dtype=float); pred=(pt>=threshold); pos=(y==1); neg=~pos
              tp=int(np.sum(pos&pred)); fn=int(np.sum(pos&~pred)); tn=int(np.sum(neg&~pred)); fp=int(np.sum(neg&pred))
              threshold_rows.append({'cohort':cohort,'task':task,'model':model,'seed':seed,'missingness':lev,
                                     'threshold_selected_on_validation':threshold,'validation_youden_j':float(j[ix]),
                                     'sensitivity':tp/max(tp+fn,1),'specificity':tn/max(tn+fp,1),
                                     'ppv':tp/max(tp+fp,1),'npv':tn/max(tn+fn,1),
                                     'tp':tp,'fp':fp,'tn':tn,'fn':fn,'n_test':len(y)})
    thrdf=pd.DataFrame(threshold_rows)
    thrdf.to_csv(OUT/'tables'/'stage47_validation_selected_thresholds_seed_level.csv',index=False)
    agg=[]
    for keys,g in thrdf.groupby(['cohort','task','model','missingness'],sort=True):
      row=dict(zip(['cohort','task','model','missingness'],keys))
      for m in ['threshold_selected_on_validation','sensitivity','specificity','ppv','npv']:
        row[m+'_mean_across_seeds']=float(g[m].mean()); row[m+'_sd_across_seeds']=float(g[m].std(ddof=1))
      agg.append(row)
    pd.DataFrame(agg).to_csv(OUT/'tables'/'stage47_validation_selected_thresholds_summary.csv',index=False)

    # Cohort denominator table.
    cohort_rows=[]
    for cohort,preds in [('full_test',full),('matched_complete_case',matched)]:
      for task in TASKS:
        d=preds[(task,'AUAF',42,0.0)]; y=d.y_true.to_numpy(dtype=int)
        cohort_rows.append({'cohort':cohort,'task':task,'n_test':len(y),
                            'class_counts_json':json.dumps({str(int(c)):int((y==c).sum()) for c in np.unique(y)},sort_keys=True),
                            'n_unique_RID':int(d.RID.nunique())})
    cohort_df=pd.DataFrame(cohort_rows)
    cohort_df.to_csv(OUT/'tables'/'stage47_cohort_denominators.csv',index=False)

    # Conformal summary reproduction gap audit from Manuscript Table 6; raw prediction sets/q-hats are not in the Stage 4.5 archive.
    conf=pd.DataFrame([
      {'task':'diagnosis_multiclass','manuscript_label':'cn_mci_ad','target_coverage':.90,'reported_empirical_coverage':.9259,'reported_mean_set_size':1.4211,'reported_singleton_rate':.5789,'reported_abstention_rate':0.0,'test_n':740},
      {'task':'diagnosis_binary','manuscript_label':'ad_vs_nonad','target_coverage':.90,'reported_empirical_coverage':.8851,'reported_mean_set_size':1.1508,'reported_singleton_rate':.8492,'reported_abstention_rate':.1508,'test_n':740},
      {'task':'mci_conversion','manuscript_label':'early_decline','target_coverage':.90,'reported_empirical_coverage':.9705,'reported_mean_set_size':1.8383,'reported_singleton_rate':.1617,'reported_abstention_rate':.8383,'test_n':183},
    ])
    conf['coverage_minus_target']=conf.reported_empirical_coverage-conf.target_coverage
    conf['implied_non_singleton_rate']=1-conf.reported_singleton_rate
    conf['excess_set_size_per_non_singleton']=(conf.reported_mean_set_size-1)/conf.implied_non_singleton_rate
    conf['abstention_equals_non_singleton']=np.isclose(conf.reported_abstention_rate,conf.implied_non_singleton_rate,atol=.002)
    conf['abstention_definition_status']=np.where(conf.abstention_equals_non_singleton,'matches_non_singleton_complement','definition_differs_or_is_task_specific; reconcile manuscript definition')
    conf['prediction_set_level_reproduction']='NOT_RECOMPUTABLE: per-participant prediction sets, validation calibration scores/q-hats, and exact RAPS-lite implementation/configuration are absent from supplied Stage 4.4/4.5 artifacts.'
    conf.to_csv(OUT/'tables'/'stage47_conformal_summary_reproduction_gap_audit.csv',index=False)

    # Feature inventory with source-table provenance from Stage 4.4 feature_manifest_34.csv.
    fm44=pd.read_csv(ROOT/'input44'/'AUAF_Stage4_4_Temporal_Clean_Package'/'outputs'/'feature_manifest_34.csv')
    used=pd.read_csv(DATA/'feature_manifest_34_predictors.csv')
    used_features=[f for f in used.feature.astype(str).tolist() if f not in {'PTENGSPKAGE','PTIMMAGE'}]
    used_features += []
    circ_map={
      'CDR':['CDMEMORY','CDRSB'],
      'FAQ':[f for f in used_features if f.startswith('FAQ')],
      'MMSE':['MMSEASON'],
      'MOCA':['TRAILS','CLOCKCON','CLOCKNO','CLOCKHAN','FFLUENCY','MOCA']
    }
    circ_features=set(sum(circ_map.values(),[]))
    inv=[]
    for f in used_features:
      row=fm44[fm44.Feature.astype(str).eq(f)]
      if row.empty: continue
      r=row.iloc[0]
      groups=[g for g,fs in circ_map.items() if f in fs]
      inv.append({'feature':f,'modality':str(r.Modality),'source_table':str(r.Source_table),'instrument':str(r.Instrument),
                  'description_and_scope':str(r.Description_and_scope),'definition_status':str(r.Definition_status),
                  'circularity_risk':str(r.Circularity_risk),'risk_reason_and_action':str(r.Risk_reason_and_action),
                  'temporal_alignment_rule':str(r.Temporal_Alignment_Rule),'missing_code_handling':str(r.Missing_Code_Handling),
                  'stage45_model_input_used':True,'diagnosis_proximal_group':';'.join(groups),
                  'removed_in_strict_sensitivity':bool(f in circ_features),
                  'feature_specific_visit_code_available':False,
                  'visit_code_note':'Participant-level index visit is retained in Stage 4.4 provenance; feature-specific visit/source row is not retained in integrated predictor matrix.'})
    invdf=pd.DataFrame(inv)
    invdf.to_csv(OUT/'manifests'/'stage47_feature_source_and_circularity_inventory.csv',index=False)
    pd.DataFrame([{'group':g,'feature':f,'reason':'Prespecified diagnosis-proximal cognitive/functional instrument block; removed before preprocessing'}
                  for g,fs in circ_map.items() for f in fs]).to_csv(OUT/'manifests'/'stage47_diagnosis_proximal_features_removed.csv',index=False)
    # Log features considered but excluded from the Stage 4.5 effective model input.
    excluded=pd.DataFrame([
      {'feature':'PTENGSPKAGE','reason':'Explicitly excluded by Stage 4.5 inference/training preparation as unverified/highly sparse.'},
      {'feature':'PTIMMAGE','reason':'Explicitly excluded by Stage 4.5 inference/training preparation as unverified/highly sparse.'},
      {'feature':'LANGUAGE_CODE','reason':'Administrative metadata/zero variance; not in final predictor manifest.'}
    ])
    excluded.to_csv(OUT/'manifests'/'stage47_candidate_features_not_used.csv',index=False)

    # Save intermediate bootstrap settings and manifest.
    manifest={
      'stage':'4.7','title':'Statistical validation and multi-variable diagnostic circularity audit',
      'status':'STATISTICAL_VALIDATION_COMPLETED_CIRCULARITY_RETRAINING_PENDING',
      'tasks':TASKS,'models':MODELS,'seeds':SEEDS,'missingness_levels':LEVELS,
      'bootstrap':{'repetitions':BOOTSTRAP_N,'seed':BOOTSTRAP_SEED,'unit':'participant stratified by outcome class',
                   'paired_resampling':'same participant resamples shared across model, seed, and missingness comparisons',
                   'seed_handling':'five fixed model initializations averaged within each participant bootstrap; not independent folds'},
      'fdr':'Benjamini-Hochberg applied within each prespecified family: paired AUAF comparisons, missingness effects, calibration changes, and seed-level McNemar supplement.',
      'calibration':'Temperature scaling fitted on the complete validation split only; unchanged temperature applied to all test missingness conditions.',
      'roundtrip_metric_audit':'PASS_WITH_SMALL_AUC_PR_DISCREPANCY' if (repro.status=='PASS_WITH_SMALL_RANKING_METRIC_DISCREPANCY').any() and repro.status.str.startswith('PASS').all() else ('PASS' if repro.status.eq('PASS').all() else 'FAIL'),
      'circularity_removed_features':sorted(circ_features),
      'effective_feature_count':len(used_features),
      'effective_modality_counts':invdf.modality.value_counts().to_dict(),
      'conformal_reproduction':'Aggregate Table 6 values transcribed from Manuscript(1); no participant-level prediction sets or q-hats in Stage 4.4/4.5 archive.',
      'source_inputs':{
        'stage4_4_zip_sha256':json.loads((ROOT/'source_inputs_sha256.json').read_text())['stage4_4_archive']['sha256'],
        'stage4_5_zip_sha256':json.loads((ROOT/'source_inputs_sha256.json').read_text())['stage4_5_archive']['sha256'],
        'validation_predictions_sha256':sha256_file(ROOT/'validation_predictions_uncalibrated.csv'),
        'reconstruction_verification_sha256':sha256_file(ROOT/'reconstruction_verification.csv')},
      'limitations':[
        'Five seeds share fixed participant splits and are repeated initializations, not independent folds.',
        'Matched-complete-case test cohorts are small, particularly MCI conversion (n=29).',
        'No external validation cohort is available.',
        'Conformal prediction-set-level coverage CIs cannot be recomputed without per-participant sets, validation calibration scores/q-hats, and exact RAPS-lite implementation/configuration.',
        'Feature source tables and temporal alignment are recorded from the Stage 4.4 feature manifest; feature-specific visit codes and source-row IDs are not retained in the integrated predictor matrix.',
        'The strict circularity sensitivity model will remove the CDR, FAQ, MMSE, and MoCA-derived blocks identified in the Stage 4.4 feature reconciliation.'
      ]}
    (OUT/'manifests'/'stage47_statistical_reproducibility_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')

    # Summary figures: macro-F1 against additional missingness for each task and cohort.
    import matplotlib.pyplot as plt
    for cohort in ['full_test','matched_complete_case']:
      for task in TASKS:
        fig,ax=plt.subplots(figsize=(8,5))
        sub=model_metrics[(model_metrics.cohort==cohort)&(model_metrics.task==task)&(model_metrics.metric=='f1_macro')]
        for model in MODELS:
          g=sub[sub.model==model].sort_values('missingness')
          ax.plot(g.missingness*100,g.mean_across_5_seeds,marker='o',label=model)
        ax.set_xlabel('Additional biomarker missingness (%)')
        ax.set_ylabel('Macro-F1 (mean across five seeds)')
        ax.set_title(f'{task} — {cohort.replace("_"," ")}')
        ax.set_xticks([0,25,50,75,100]); ax.grid(True,alpha=.25); ax.legend(fontsize=7,ncol=2)
        fig.tight_layout(); fig.savefig(OUT/'figures'/f'macro_f1_missingness_{cohort}_{task}.png',dpi=180); plt.close(fig)

    # Excel workbook of key results (full long-form tables remain as CSV).
    with pd.ExcelWriter(OUT/'reports'/'AUAF_Stage4_7_Statistical_Results.xlsx',engine='openpyxl') as w:
      cohort_df.to_excel(w,sheet_name='Cohort sizes',index=False)
      model_metrics.to_excel(w,sheet_name='Model CI all levels',index=False)
      paired.to_excel(w,sheet_name='AUAF paired tests',index=False)
      missing.to_excel(w,sheet_name='Missingness effects',index=False)
      cal.to_excel(w,sheet_name='Calibration',index=False)
      thrdf.to_excel(w,sheet_name='Thresholds',index=False)
      mcnemar_df.to_excel(w,sheet_name='McNemar',index=False)
      invdf.to_excel(w,sheet_name='Feature inventory',index=False)
      conf.to_excel(w,sheet_name='Conformal audit',index=False)
      repro.to_excel(w,sheet_name='Metric roundtrip',index=False)

    summary={
      'prediction_files_loaded':len(full)+len(matched),
      'roster_audit_rows':len(roster_audit),'roster_all_pass':bool(roster_audit[['unique_RID','same_RID_roster','same_labels','probabilities_finite']].all().all()),
      'roundtrip_rows':len(repro),'roundtrip_all_pass':bool(repro.status.str.startswith('PASS').all()),
      'model_metric_rows':len(model_metrics),'paired_comparison_rows':len(paired),'paired_fdr_significant':int(paired.significant_after_FDR_0_05.sum()),
      'missingness_effect_rows':len(missing),'missingness_fdr_significant':int(missing.significant_after_FDR_0_05.sum()),
      'calibration_rows':len(cal),'calibration_fdr_significant':int(cal.significant_after_FDR_0_05.sum()),
      'temperature_parameters':len(temp_rows),'threshold_rows':len(thrdf),'mcnemar_rows':len(mcnemar_df),
      'effective_predictors':len(used_features),'diagnosis_proximal_removed':sorted(circ_features),
      'elapsed_seconds':round(time.time()-t0,2)
    }
    (OUT/'reports'/'stage47_statistical_execution_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
