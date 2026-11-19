from pathlib import Path
from types import SimpleNamespace
import sys, pandas as pd
from paired_inference import paired_s0_tests
ROOT=Path('/mnt/data/AUAF_Stage4_8_AUTHORIZED_RUN')
OUT=ROOT/'outputs'/'stage4_8_authorized_reliability_v2'/'intermediate'
task,comparator=sys.argv[1:3]
pred=pd.read_csv(OUT/'prediction_bundle_mc.csv.gz')
cal=pd.read_csv(OUT/'calibrated_test.csv.gz')
nc={'diagnosis_multiclass':3,'diagnosis_binary':2,'mci_conversion':2}[task]
prepared={(task,'S0_verified_asof'):SimpleNamespace(n_classes=nc)}
r=paired_s0_tests(pred,cal,prepared,seeds=[42,123,456,789,1024],n_boot=500,random_seed=20261115,feature_set='S0_verified_asof',tasks=[task],comparators=[comparator])
p=OUT/f'paired_tests_{task}_{comparator}.csv'
r.to_csv(p,index=False)
print('SAVED',task,comparator,len(r),p,flush=True)
print(r[['metric','effect_size_metric_difference','ci_low','ci_high','p_value_bootstrap_two_sided']].to_string(index=False),flush=True)
