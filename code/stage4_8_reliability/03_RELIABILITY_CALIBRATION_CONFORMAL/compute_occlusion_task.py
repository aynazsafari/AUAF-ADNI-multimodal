from pathlib import Path
import sys,pandas as pd
from stage48_prepare_data import prepare_task
from stage48_reliability import run_auaf_modality_occlusion
ROOT=Path('/mnt/data/AUAF_Stage4_8_AUTHORIZED_RUN');STAGE45=ROOT/'AUAF_Stage4_5_MissingModality_FinalCohort';OUT=ROOT/'outputs'/'stage4_8_authorized_reliability_v2'/'intermediate'
task=sys.argv[1]
prepared={(task,'S0_verified_asof'):prepare_task(task)}
metrics,tests,rows=run_auaf_modality_occlusion(prepared,STAGE45,[42,123,456,789,1024],n_boot=500,random_seed=20261115,mc_passes=30,feature_set='S0_verified_asof',tasks=[task])
for name,df in [(f'occlusion_metrics_{task}.csv',metrics),(f'occlusion_tests_{task}.csv',tests),(f'occlusion_patient_level_{task}.csv.gz',rows)]:df.to_csv(OUT/name,index=False,compression='gzip' if name.endswith('.gz') else None)
print('SAVED',task,len(metrics),len(tests),len(rows),flush=True)
print(metrics.to_string(index=False),flush=True)
print(tests[['contrast','metric','effect_size_metric_difference','ci_low','ci_high','p_value_bootstrap_two_sided']].to_string(index=False),flush=True)
