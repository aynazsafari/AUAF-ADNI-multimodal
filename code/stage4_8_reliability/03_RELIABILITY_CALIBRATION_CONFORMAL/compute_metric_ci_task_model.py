from pathlib import Path
import sys,pandas as pd,numpy as np
from paired_inference import bootstrap_metric_ci,stable_seed
ROOT=Path('/mnt/data/AUAF_Stage4_8_AUTHORIZED_RUN')
OUT=ROOT/'outputs'/'stage4_8_authorized_reliability_v2'/'intermediate'
task,model=sys.argv[1:3]
pred=pd.read_csv(OUT/'prediction_bundle_mc.csv.gz')
cal=pd.read_csv(OUT/'calibrated_test.csv.gz')
seeds=[42,123,456,789,1024]
nc={'diagnosis_multiclass':3,'diagnosis_binary':2,'mci_conversion':2}[task]
raw_cols=[f'p_class_{c}' for c in range(nc)]
raw_cal_cols=[f'p_raw_class_{c}' for c in range(nc)]
cal_cols=[f'p_cal_class_{c}' for c in range(nc)]
metrics=['accuracy','balanced_accuracy','f1_macro','roc_auc','pr_auc','brier','log_loss','ece']
rows=[]
for metric in metrics:
    sources=['pre_temperature','post_temperature'] if metric in {'brier','log_loss','ece'} else ['pre_temperature']
    for source_name in sources:
        p_by_seed=[]; y_ref=None
        for seed in seeds:
            if source_name=='post_temperature' or metric in {'brier','log_loss','ece'}:
                sub=cal[(cal.task==task)&(cal.model==model)&(cal.seed.astype(int)==seed)].sort_values('row_index')
                cols=cal_cols if source_name=='post_temperature' else raw_cal_cols
            else:
                sub=pred[(pred.task==task)&(pred.model==model)&(pred.seed.astype(int)==seed)&(pred.split=='test')].sort_values('row_index')
                cols=raw_cols
            y=sub.y_true.to_numpy(int)
            if y_ref is None:y_ref=y
            elif not np.array_equal(y_ref,y):raise RuntimeError('test label/order mismatch')
            p_by_seed.append(sub[cols].to_numpy(float))
        ci=bootstrap_metric_ci(y_ref,p_by_seed,metric=metric,n_boot=200,random_seed=20261115+stable_seed(task,model,metric,source_name)%100000)
        rows.append({'task':task,'model':model,'metric':metric,'probability_source':source_name,**ci})
out=pd.DataFrame(rows)
p=OUT/f'metric_ci_{task}_{model}.csv'
out.to_csv(p,index=False)
print('SAVED',task,model,len(out),p,flush=True)
print(out[['metric','probability_source','mean_over_seeds','participant_bootstrap_ci_low','participant_bootstrap_ci_high']].to_string(index=False),flush=True)
