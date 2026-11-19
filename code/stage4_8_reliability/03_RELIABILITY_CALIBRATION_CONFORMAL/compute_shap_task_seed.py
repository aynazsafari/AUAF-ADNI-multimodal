from pathlib import Path
import sys,pandas as pd
from stage48_prepare_data import prepare_task
from stage48_reliability import run_catboost_shap
ROOT=Path('/mnt/data/AUAF_Stage4_8_AUTHORIZED_RUN');STAGE45=ROOT/'AUAF_Stage4_5_MissingModality_FinalCohort';OUT=ROOT/'outputs'/'stage4_8_authorized_reliability_v2'/'intermediate'
task,seed=sys.argv[1],int(sys.argv[2])
prepared={(task,'S0_verified_asof'):prepare_task(task)}
frame,summary=run_catboost_shap(prepared,STAGE45/'checkpoints',[seed],feature_set='S0_verified_asof',tasks=[task])
frame.to_csv(OUT/f'catboost_shap_{task}_{seed}.csv',index=False)
summary.to_csv(OUT/f'catboost_shap_summary_{task}_{seed}.csv',index=False)
print('SAVED',task,seed,len(frame),flush=True)
print(summary.head(10).to_string(index=False),flush=True)
