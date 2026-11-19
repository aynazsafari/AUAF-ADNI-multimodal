import os
from pathlib import Path
import sys, json, numpy as np, pandas as pd, torch, joblib, glob, hashlib
root=Path(os.environ.get('AUAF_WORKSPACE', Path.cwd()))
pkg=root/'input45'; data_dir=root/'stage4_data'
sys.path.insert(0,str(pkg/'code'))
import run_stage4_5_missing_modality_inference as inf
from auaf_model_factory import build_model
fm=pd.read_csv(data_dir/'feature_manifest_34_predictors.csv')
tasks=['diagnosis_multiclass','diagnosis_binary','mci_conversion']
models=['EarlyFusion','LateFusion','AttentionFusion','AUAF','XGBoost','CatBoost','ExtraTrees']
seeds=[42,123,456,789,1024]
rows=[]; val_rows=[]
for task in tasks:
 pack=inf.prepare_task(data_dir,task,fm)
 frame=pack['frame']; target=pack['target']; nclass=pack['n_classes']
 val_idx=np.flatnonzero(frame['split'].astype(str).str.lower().eq('validation').to_numpy())
 test_idx=np.flatnonzero(frame['split'].astype(str).str.lower().eq('test').to_numpy())
 yval=frame.iloc[val_idx][target].to_numpy(dtype=int)
 for model_name in models:
  for seed in seeds:
   if model_name in ['EarlyFusion','LateFusion','AttentionFusion','AUAF']:
    cp=pkg/'checkpoints'/'neural_best'/f'{task}__S0_verified_asof__{model_name}__seed{seed}'/'best.pt'
    payload=torch.load(cp,map_location='cpu',weights_only=False)
    model=build_model(payload['model_variant'],payload['dimensions'],payload['n_classes'])
    model.load_state_dict(payload['model_state']); model.eval()
    inputs={m:torch.as_tensor(pack['matrices'][m][val_idx],dtype=torch.float32) for m in pack['modalities']}
    mask=torch.as_tensor(pack['masks'][val_idx],dtype=torch.float32)
    with torch.no_grad():
     logits=model(inputs,mask)
     pval=torch.softmax(logits.to(torch.float64),dim=1).cpu().numpy()
    # compare validation macro-F1 to checkpoint's stored selection metric
    from sklearn.metrics import f1_score
    f1=f1_score(yval,pval.argmax(1),labels=np.arange(nclass),average='macro',zero_division=0)
    stored=float(payload.get('best_validation_macro_f1',np.nan))
    status='PASS' if abs(f1-stored)<1e-6 else 'CHECK'
   else:
    tp=pkg/'checkpoints'/'tree_models'/f'{task}__{model_name}__seed{seed}.joblib'
    bundle=joblib.load(tp)
    pval=inf.predict_tree(bundle,frame.iloc[val_idx],bundle['feature_cols'])
    stored=np.nan; f1=np.nan; status='NOT_APPLICABLE'
   for j,ri in enumerate(val_idx):
    rec={'task':task,'model':model_name,'seed':seed,'RID':str(frame.iloc[ri]['RID']),'y_true':int(yval[j]),'split':'validation'}
    for k,v in enumerate(pval[j]): rec[f'p_class_{k}']=float(v)
    val_rows.append(rec)
   # Compare 0% full-test predictions against package CSV.
   if model_name in ['EarlyFusion','LateFusion','AttentionFusion','AUAF']:
    # reuse model for test inference
    inputs_t={m:torch.as_tensor(pack['matrices'][m][test_idx],dtype=torch.float32) for m in pack['modalities']}
    mask_t=torch.as_tensor(pack['masks'][test_idx],dtype=torch.float32)
    with torch.no_grad():
     logits_t=model(inputs_t,mask_t); ptest=torch.softmax(logits_t.to(torch.float64),dim=1).cpu().numpy()
   else:
    ptest=inf.predict_tree(bundle,frame.iloc[test_idx],bundle['feature_cols'])
   fn=pkg/'predictions'/('full_test_neural' if model_name in ['EarlyFusion','LateFusion','AttentionFusion','AUAF'] else 'full_test_tree')/f'{task}__{model_name}__seed{seed}__miss000.csv'
   d=pd.read_csv(fn).sort_values('RID')
   order=np.argsort(frame.iloc[test_idx]['RID'].astype(int).to_numpy())
   ptest=ptest[order]
   maxdiff=float(np.max(np.abs(ptest-d[[f'p_class_{k}' for k in range(nclass)]].to_numpy())))
   rid_equal=np.array_equal(np.sort(frame.iloc[test_idx]['RID'].astype(int).to_numpy()),d['RID'].astype(int).to_numpy())
   y_equal=np.array_equal(frame.iloc[test_idx].iloc[order][target].to_numpy(dtype=int),d['y_true'].to_numpy(dtype=int))
   rows.append({'task':task,'model':model_name,'seed':seed,'validation_macro_f1_recomputed':f1,'checkpoint_best_validation_macro_f1':stored,'validation_f1_status':status,'test_probability_max_abs_diff_vs_stage45_0pct':maxdiff,'test_roster_equal':rid_equal,'test_labels_equal':y_equal})
 print('completed',task,flush=True)
res=pd.DataFrame(rows); val=pd.DataFrame(val_rows)
res.to_csv(root/'reconstruction_verification.csv',index=False)
val.to_csv(root/'validation_predictions_uncalibrated.csv',index=False)
print(res.groupby(['model'])['test_probability_max_abs_diff_vs_stage45_0pct'].agg(['max','mean']).to_string())
print('Neural val f1 mismatches',res[(res.model.isin(['EarlyFusion','LateFusion','AttentionFusion','AUAF']))&(res.validation_f1_status!='PASS')][['task','model','seed','validation_macro_f1_recomputed','checkpoint_best_validation_macro_f1']].to_string(index=False))
print('test roster all',res.test_roster_equal.all(),'labels all',res.test_labels_equal.all())
