from pathlib import Path
from types import SimpleNamespace
import json, pandas as pd, numpy as np
from paired_inference import conformal_s0_results,bootstrap_conformal_intervals,bootstrap_conformal_intervals_across_seeds,stable_seed
from scipy.stats import spearmanr
ROOT=Path('/mnt/data/AUAF_Stage4_8_AUTHORIZED_RUN')
OUT=ROOT/'outputs'/'stage4_8_authorized_reliability_v2'/'intermediate'
PRED=pd.read_csv(OUT/'prediction_bundle_mc.csv.gz')
CAL=pd.read_csv(OUT/'calibration_parameters.csv')
MC=pd.read_csv(OUT/'auaf_mc_predictions.csv.gz')
seeds=[42,123,456,789,1024]
class_map={'diagnosis_multiclass':3,'diagnosis_binary':2,'mci_conversion':2}
PREP={(task,'S0_verified_asof'):SimpleNamespace(n_classes=n) for task,n in class_map.items()}
alpha=.10;lam=.01;kreg=1;nboot=500;seed0=20261115
summary,preds=conformal_s0_results(PRED,CAL,PREP,alpha=alpha,lam=lam,k_reg=kreg,feature_set='S0_verified_asof',force_nonempty=False,threshold_method='finite_sample',use_temperature=True)
auaf=PRED[PRED.model.eq('AUAF')].copy();auafcal=CAL[CAL.model.eq('AUAF')].copy()
legacy_summary,legacy_preds=conformal_s0_results(auaf,auafcal,PREP,alpha=alpha,lam=lam,k_reg=kreg,feature_set='S0_verified_asof',force_nonempty=True,threshold_method='legacy_interpolated',use_temperature=False)
cirows=[]; pooledrows=[]
for (task,model,seed),g in preds.groupby(['task','model','seed']):
 y=g.sort_values('row_index').y_true.to_numpy(int); o=g.sort_values('row_index')
 sets=[[int(x) for x in str(v).split(';') if str(v).strip()!=''] for v in o.prediction_set]
 ci=bootstrap_conformal_intervals(y,sets,class_map[task],n_boot=nboot,random_seed=seed0+stable_seed(task,model,seed,'raps')%100000)
 cirows.append({'task':task,'model':model,'seed':int(seed),**ci})
for (task,model),g in preds.groupby(['task','model']):
 sets_by=[]; yref=None
 for seed in seeds:
  o=g[g.seed.astype(int).eq(seed)].sort_values('row_index'); y=o.y_true.to_numpy(int)
  if yref is None:yref=y
  elif not np.array_equal(yref,y):raise RuntimeError('RAPS test labels differ across seeds')
  sets_by.append([[int(x) for x in str(v).split(';') if str(v).strip()!=''] for v in o.prediction_set])
 ci=bootstrap_conformal_intervals_across_seeds(yref,sets_by,class_map[task],n_boot=nboot,random_seed=seed0+stable_seed(task,model,'pooled_raps')%100000)
 pooledrows.append({'task':task,'model':model,**ci})
ci=pd.DataFrame(cirows); pooled=pd.DataFrame(pooledrows)
summary_seed=summary.merge(ci,on=['task','model','seed'],how='left',validate='one_to_one')
# Association between AUAF MC entropy and RAPS set size.
auafsets=preds[preds.model.eq('AUAF')]
entropy_sets=auafsets.merge(MC[MC.split.eq('test')][['task','seed','row_index','entropy']],on=['task','seed','row_index'],how='inner',validate='one_to_one')
assoc=[]
for (task,seed),g in entropy_sets.groupby(['task','seed']):
 r=spearmanr(g.entropy,g.set_size)
 assoc.append({'task':task,'seed':int(seed),'n':len(g),'spearman_entropy_vs_set_size':float(r.statistic),'p_value_descriptive':float(r.pvalue)})
assoc=pd.DataFrame(assoc)
for name,df in [('conformal_summary_by_seed.csv',summary_seed),('conformal_predictions_LOCAL_ONLY.csv.gz',preds),('conformal_ci_by_seed.csv',ci),('conformal_pooled_five_seed_ci.csv',pooled),('conformal_legacy_auaf_summary.csv',legacy_summary),('conformal_legacy_auaf_predictions_LOCAL_ONLY.csv.gz',legacy_preds),('raps_entropy_association.csv',assoc)]:
 df.to_csv(OUT/name,index=False,compression='gzip' if name.endswith('.gz') else None)
print('primary summary',len(summary_seed),'pooled',len(pooled),'legacy AUAF',len(legacy_summary),'predictions',len(preds))
print(pooled[['task','model','coverage_mean_over_seeds','coverage_participant_bootstrap_ci_low','coverage_participant_bootstrap_ci_high','mean_set_size_mean_over_seeds','singleton_rate_mean_over_seeds','multi_label_rate_mean_over_seeds','empty_set_rate_mean_over_seeds','abstention_rate_mean_over_seeds']].round(4).to_string(index=False))
print('\nlegacy AUAF')
print(legacy_summary[['task','seed','empirical_coverage','mean_set_size','singleton_rate','multi_label_rate','empty_set_rate','full_set_rate','abstention_rate','threshold']].round(4).to_string(index=False))
print('\nentropy association')
print(assoc.round(4).to_string(index=False))
