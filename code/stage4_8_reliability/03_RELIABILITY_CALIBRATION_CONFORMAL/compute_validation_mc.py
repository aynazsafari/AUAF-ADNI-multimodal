import os,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from stage48_prepare_data import prepare_task
from paired_inference import stable_seed
from stage4_6_missingness_benchmark import load_neural_checkpoint

ROOT=Path('/mnt/data/AUAF_Stage4_8_AUTHORIZED_RUN')
STAGE45=ROOT/'AUAF_Stage4_5_MissingModality_FinalCohort'
task=sys.argv[1]
seeds=[42,123,456,789,1024]
pack=prepare_task(task)
rows=[]
torch.set_num_threads(1)
for seed in seeds:
    model=load_neural_checkpoint(pack,STAGE45,task,'AUAF',seed).to('cpu')
    model.eval()
    idx=pack.idx('validation',eligible_only=True)
    y=pack.frame.iloc[idx][pack.target].to_numpy(dtype=int)
    torch.manual_seed(stable_seed(task,seed,'validation','mc_dropout'))
    parts={k:[] for k in ['mean_prob','std_prob','entropy','mean_gates']}
    with torch.no_grad():
        for start in range(0,len(idx),128):
            batch_idx=idx[start:start+128]
            inputs={m:torch.as_tensor(pack.matrices[m][batch_idx],dtype=torch.float32,device='cpu') for m in pack.modalities}
            mask=torch.as_tensor(pack.masks[batch_idx],dtype=torch.float32,device='cpu')
            out=model.mc_forward(inputs,mask,n_passes=30)
            for k in parts: parts[k].append(np.asarray(out[k]))
    mc={k:np.concatenate(v,axis=0) for k,v in parts.items()}
    if len(mc['mean_prob'])!=len(idx): raise RuntimeError(f'{task}/{seed}: validation row count mismatch')
    for j,row_index in enumerate(idx):
        rec={'task':task,'model':'AUAF','seed':seed,'split':'validation','row_index':int(row_index),'y_true':int(y[j]),
             'entropy':float(mc['entropy'][j]),'mean_class_probability_sd':float(mc['std_prob'][j].mean()),
             'predicted_class':int(mc['mean_prob'][j].argmax()),
             'gate_clinical_cognitive':float(mc['mean_gates'][j,pack.modalities.index('clinical_cognitive')]),
             'gate_fluid_biomarker':float(mc['mean_gates'][j,pack.modalities.index('fluid_biomarker')])}
        for c in range(pack.n_classes):
            rec[f'p_class_{c}']=float(mc['mean_prob'][j,c]);rec[f'std_class_{c}']=float(mc['std_prob'][j,c])
        rows.append(rec)
    print(f'DONE {task}/{seed} validation n={len(idx)}',flush=True)
out=pd.DataFrame(rows).sort_values(['task','seed','row_index']).reset_index(drop=True)
path=ROOT/'outputs'/f'validation_mc_{task}.csv.gz'
path.parent.mkdir(parents=True,exist_ok=True)
out.to_csv(path,index=False,compression='gzip')
print('SAVED',path,'rows',len(out),flush=True)
