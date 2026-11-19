from dataclasses import dataclass
from typing import Dict, List, Sequence
import numpy as np
import pandas as pd

class TrainOnlyNumericPreprocessor:
    def __init__(self, columns: Sequence[str]):
        self.columns=list(columns); self.medians_=None; self.means_=None; self.scales_=None; self.all_missing_train_=None
    @staticmethod
    def _array(frame, columns):
        values=frame[list(columns)].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=np.float64)
        values[~np.isfinite(values)]=np.nan
        return values
    def fit(self,frame,train_mask):
        train_mask=np.asarray(train_mask,dtype=bool); x_train=self._array(frame.loc[train_mask],self.columns)
        medians=np.zeros(x_train.shape[1],dtype=np.float64); all_missing=np.zeros(x_train.shape[1],dtype=bool)
        for j in range(x_train.shape[1]):
            observed=x_train[:,j][np.isfinite(x_train[:,j])]
            if observed.size==0: all_missing[j]=True; medians[j]=0.0
            else: medians[j]=float(np.median(observed))
        x_imp=np.where(np.isfinite(x_train),x_train,medians[None,:]); means=x_imp.mean(axis=0); scales=x_imp.std(axis=0,ddof=0)
        scales[~np.isfinite(scales)|(scales<1e-12)]=1.0
        self.medians_=medians; self.means_=means; self.scales_=scales; self.all_missing_train_=all_missing
        return self
    def transform(self,frame):
        x=self._array(frame,self.columns); x[:,self.all_missing_train_]=np.nan
        x_imp=np.where(np.isfinite(x),x,self.medians_[None,:]); z=(x_imp-self.means_[None,:])/self.scales_[None,:]
        z[~np.isfinite(z)]=0.0
        return z.astype(np.float32)
    def effective_columns(self): return [c for c,m in zip(self.columns,self.all_missing_train_) if not m]

@dataclass
class PreparedTask:
    name: str
    feature_set: str
    frame: pd.DataFrame
    target: str
    n_classes: int
    modalities: List[str]
    features_by_modality: Dict[str,List[str]]
    matrices: Dict[str,np.ndarray]
    masks: np.ndarray
    def idx(self,split,eligible_only=False):
        idx=np.flatnonzero(self.frame['split'].astype(str).str.lower().eq(split).to_numpy())
        if eligible_only: idx=idx[self.masks[idx].sum(axis=1)>0]
        return idx
    def paired_modality_idx(self,split):
        idx=self.idx(split,eligible_only=True)
        return idx[(self.masks[idx] > 0).all(axis=1)]

ROOT='/mnt/data/AUAF_Stage4_8_AUTHORIZED_RUN'
DATA=f'{ROOT}/stage4_data'
FEATURE_MANIFEST=pd.read_csv(f'{DATA}/feature_manifest_34_predictors.csv')
VERIFIED_FEATURES=[f for f in FEATURE_MANIFEST['feature'].astype(str).tolist() if f not in ['PTENGSPKAGE','PTIMMAGE']]
MODALITY_ORDER=['clinical_cognitive','fluid_biomarker']
TASKS={
 'diagnosis_multiclass':('diagnosis_multiclass_3700_asof.csv','TARGET_MULTICLASS',3),
 'diagnosis_binary':('diagnosis_binary_3700_asof.csv','TARGET_BINARY',2),
 'mci_conversion':('mci_conversion_911_asof.csv','TARGET_CONVERSION',2),
}
def prepare_task(task_name):
    filename,target,n_classes=TASKS[task_name]
    frame=pd.read_csv(f'{DATA}/{filename}',parse_dates=['INDEX_DATE']).reset_index(drop=True)
    train_mask=frame['split'].astype(str).str.lower().eq('train').to_numpy()
    features_by_modality={}; matrices={}; mask_columns=[]; active=[]
    for modality in MODALITY_ORDER:
        requested=[f for f in VERIFIED_FEATURES if f in set(FEATURE_MANIFEST.loc[FEATURE_MANIFEST['modality'].eq(modality),'feature'].astype(str))]
        if not requested: continue
        proc=TrainOnlyNumericPreprocessor(requested).fit(frame,train_mask)
        effective=proc.effective_columns()
        if not effective: continue
        proc=TrainOnlyNumericPreprocessor(effective).fit(frame,train_mask)
        matrices[modality]=proc.transform(frame)
        features_by_modality[modality]=effective
        mask_columns.append(frame[effective].notna().any(axis=1).to_numpy(dtype=np.float32))
        active.append(modality)
    masks=np.column_stack(mask_columns).astype(np.float32)
    return PreparedTask(task_name,'S0_verified_asof',frame,target,n_classes,active,features_by_modality,matrices,masks)
