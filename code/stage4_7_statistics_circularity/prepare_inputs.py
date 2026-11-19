import os
from pathlib import Path
import pandas as pd, numpy as np, hashlib, json, shutil
root=Path(os.environ.get('AUAF_WORKSPACE', Path.cwd()))
p44=root/'input44'/'AUAF_Stage4_4_Temporal_Clean_Package'
out=root/'stage4_data'
out.mkdir(exist_ok=True)
# Build model-input manifest from final Stage 4.4 manifest.
fm=pd.read_csv(p44/'outputs'/'feature_manifest_34.csv')
fm2=fm[['Modality','Feature','Included_In_Stage4_4_Model_Input']].copy()
fm2=fm2[fm2['Included_In_Stage4_4_Model_Input'].astype(bool)].copy()
fm2['modality']=fm2['Modality'].map({'clinical':'clinical_cognitive','biomarker':'fluid_biomarker'})
fm2['feature']=fm2['Feature'].astype(str)
fm2=fm2[['modality','feature']]
fm2.to_csv(out/'feature_manifest_34_predictors.csv',index=False)
# Stage4.4 participant-level index provenance.
prov_diag=pd.read_csv(p44/'outputs'/'diagnostic_index_provenance.csv')
prov_conv=pd.read_csv(p44/'outputs'/'mci_conversion_index_provenance.csv')

def build(src, dst, target_old, target_new, prov):
    df=pd.read_csv(p44/'outputs'/src)
    df=df.rename(columns={'_split':'split',target_old:target_new})
    add_cols=[c for c in ['RID','INDEX_DATE','BIOMARKER_AFTER_INDEX','INDEX_VISCODE','INDEX_VISCODE2'] if c in prov.columns]
    meta=prov[add_cols].drop_duplicates('RID')
    df=df.merge(meta,on='RID',how='left',validate='one_to_one')
    df['split']=df['split'].astype(str).str.lower()
    df.to_csv(out/dst,index=False)
    return df

build('cn_mci_ad_asof.csv','diagnosis_multiclass_3700_asof.csv','target_multiclass','TARGET_MULTICLASS',prov_diag)
build('ad_vs_nonad_asof.csv','diagnosis_binary_3700_asof.csv','target_binary','TARGET_BINARY',prov_diag)
build('early_decline_asof.csv','mci_conversion_911_asof.csv','target_decline','TARGET_CONVERSION',prov_conv)
# Copy exact frozen split manifests.
shutil.copy2(p44/'reference_inputs'/'subject_split_manifest__cn_mci_ad.csv',out/'subject_split_manifest__cn_mci_ad.csv')
shutil.copy2(p44/'reference_inputs'/'subject_split_manifest__ad_vs_nonad.csv',out/'subject_split_manifest__ad_vs_nonad.csv')
shutil.copy2(p44/'reference_inputs'/'subject_split_manifest__early_decline.csv',out/'subject_split_manifest__early_decline.csv')
# hashes and shape summary
summary={'feature_manifest_rows':len(fm2),'features_by_modality':fm2.groupby('modality').size().to_dict(),'datasets':{}}
for f in ['diagnosis_multiclass_3700_asof.csv','diagnosis_binary_3700_asof.csv','mci_conversion_911_asof.csv']:
 d=pd.read_csv(out/f)
 summary['datasets'][f]={'n':len(d),'columns':len(d.columns),'missing_index_dates':int(pd.isna(d['INDEX_DATE']).sum()),'split_counts':d['split'].value_counts().to_dict(),'rid_unique':bool(d.RID.is_unique)}
summary['hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}
(out/'RECONSTRUCTED_INPUTS_AUDIT.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
