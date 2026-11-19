#!/usr/bin/env python3
"""Run the prespecified Stage 4.6 AUAF component-ablation experiment.

Design: 7 variants x 3 tasks x 5 initialization seeds = 105 jobs.
All models use the frozen Stage 4.3/4.4 participant split and S0 verified
as-of-index features. Preprocessing is fit on training participants only.
"""
from __future__ import annotations
import argparse, hashlib, json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

from auaf_ablation_factory import (
    ABLATION_VARIANTS, VARIANT_DEFINITIONS, build_ablation_model
)
from auaf_safe_training_engine import (
    TrainingConfig, build_training_plan, initialize_run_directory,
    run_training_plan, atomic_write_csv, sha256_file, sha256_bytes
)

TASKS = {
    "diagnosis_multiclass": ("diagnosis_multiclass_3700_asof.csv","TARGET_MULTICLASS",3700,3,
                             "subject_split_manifest__cn_mci_ad.csv","target_multiclass"),
    "diagnosis_binary": ("diagnosis_binary_3700_asof.csv","TARGET_BINARY",3700,2,
                         "subject_split_manifest__ad_vs_nonad.csv","target_binary"),
    "mci_conversion": ("mci_conversion_911_asof.csv","TARGET_CONVERSION",911,2,
                       "subject_split_manifest__early_decline.csv","target_decline"),
}
SEEDS=[42,123,456,789,1024]
UNVERIFIED_FIELDS={"PTENGSPKAGE","PTIMMAGE"}
MODALITY_ORDER=["clinical_cognitive","fluid_biomarker"]

class TrainOnlyNumericPreprocessor:
    def __init__(self, columns):
        self.columns=list(columns)
        self.medians_=self.means_=self.scales_=self.all_missing_train_=None
        self.fit_rids_=None
        self.fit_n_=None
    @staticmethod
    def _array(frame,columns):
        x=frame[list(columns)].apply(pd.to_numeric,errors="coerce").to_numpy(dtype=np.float64)
        x[~np.isfinite(x)]=np.nan
        return x
    def fit(self,frame,train_mask):
        x=self._array(frame.loc[np.asarray(train_mask,bool)],self.columns)
        med=np.zeros(x.shape[1],dtype=np.float64)
        missing=np.zeros(x.shape[1],dtype=bool)
        for j in range(x.shape[1]):
            observed=x[:,j][np.isfinite(x[:,j])]
            if observed.size: med[j]=float(np.median(observed))
            else: missing[j]=True
        xi=np.where(np.isfinite(x),x,med[None,:])
        means=xi.mean(axis=0)
        scales=xi.std(axis=0,ddof=0)
        scales[~np.isfinite(scales)|(scales<1e-12)]=1.0
        self.medians_,self.means_,self.scales_,self.all_missing_train_=med,means,scales,missing
        self.fit_rids_=sorted(frame.loc[train_mask,"RID"].astype(int).tolist())
        self.fit_n_=int(train_mask.sum())
        return self
    def transform(self,frame):
        x=self._array(frame,self.columns)
        x[:,self.all_missing_train_]=np.nan
        xi=np.where(np.isfinite(x),x,self.medians_[None,:])
        z=(xi-self.means_[None,:])/self.scales_[None,:]
        z[~np.isfinite(z)]=0.0
        return z.astype(np.float32)
    def effective_columns(self):
        return [c for c,m in zip(self.columns,self.all_missing_train_) if not m]

@dataclass
class PreparedTask:
    name:str
    feature_set:str
    frame:pd.DataFrame
    target:str
    n_classes:int
    modalities:List[str]
    features_by_modality:Dict[str,List[str]]
    matrices:Dict[str,np.ndarray]
    masks:np.ndarray
    processors:Dict[str,TrainOnlyNumericPreprocessor]
    fit_rids:List[int]
    def idx(self,split,eligible_only=False):
        ix=np.flatnonzero(self.frame["split"].astype(str).str.lower().eq(split).to_numpy())
        if eligible_only: ix=ix[self.masks[ix].sum(axis=1)>0]
        return ix

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",required=True)
    ap.add_argument("--reference-dir",required=True)
    ap.add_argument("--output-dir",required=True)
    ap.add_argument("--max-jobs",type=int,default=None)
    args=ap.parse_args()
    data=Path(args.data_dir); ref=Path(args.reference_dir); out=Path(args.output_dir)
    package=Path(__file__).resolve().parent.parent
    source_hashes=json.loads((package/"source_and_input_hashes.json").read_text(encoding="utf-8"))
    input_hashes={}
    for fn,expected in source_hashes["stage4_4_input_sha256"].items():
        actual=sha256_file(data/fn)
        if actual!=expected: raise RuntimeError(f"Stage 4.4 input hash mismatch: {fn}")
        input_hashes[fn]=actual
    for fn,expected in source_hashes["frozen_split_sha256"].items():
        actual=sha256_file(ref/fn)
        if actual!=expected: raise RuntimeError(f"Frozen split hash mismatch: {fn}")
        input_hashes[f"reference_splits/{fn}"]=actual

    feature_manifest=pd.read_csv(data/"feature_manifest_34_predictors.csv")
    features=feature_manifest["feature"].astype(str).tolist()
    verified=[f for f in features if f not in UNVERIFIED_FIELDS]
    if len(features)!=34 or len(verified)!=32:
        raise RuntimeError("Expected 34 candidate fields and 32 verified model predictors.")
    if set(UNVERIFIED_FIELDS)&set(verified):
        raise RuntimeError("Unverified fields must not enter the ablation models.")

    tasks={}
    split_checks=[]
    for task,(filename,target,n,nc,split_file,split_target) in TASKS.items():
        frame=pd.read_csv(data/filename,parse_dates=["INDEX_DATE"])
        if len(frame)!=n or frame.RID.isna().any() or frame.RID.duplicated().any():
            raise RuntimeError(f"{task}: cohort/RID validation failed.")
        if frame[target].isna().any(): raise RuntimeError(f"{task}: target missing.")
        expected_missing_dates=15 if task in ("diagnosis_multiclass","diagnosis_binary") else 0
        if int(frame.INDEX_DATE.isna().sum())!=expected_missing_dates:
            raise RuntimeError(f"{task}: Index Date missingness changed.")
        biom=[f for f in verified if f in set(feature_manifest.loc[
            feature_manifest.modality.eq("fluid_biomarker"),"feature"].astype(str))]
        if frame.loc[frame.INDEX_DATE.isna(),biom].notna().any(axis=None):
            raise RuntimeError(f"{task}: biomarkers exist for missing Index Dates.")
        if "BIOMARKER_AFTER_INDEX" in frame and frame.loc[
            frame.BIOMARKER_AFTER_INDEX.astype(bool),biom].notna().any(axis=None):
            raise RuntimeError(f"{task}: post-index biomarkers remain.")
        split_sets={s:set(frame.loc[frame.split.astype(str).str.lower().eq(s),"RID"])
                    for s in ["train","validation","test"]}
        if any(split_sets[a]&split_sets[b] for a,b in [
            ("train","validation"),("train","test"),("validation","test")]):
            raise RuntimeError(f"{task}: participant split overlap.")
        ref_frame=pd.read_csv(ref/split_file).rename(columns={"_split":"split"})
        merged=ref_frame[["RID","split",split_target]].merge(
            frame[["RID","split",target]],on="RID",how="outer",
            suffixes=("_ref","_current"),indicator=True,validate="one_to_one")
        if not merged._merge.eq("both").all(): raise RuntimeError(f"{task}: split roster mismatch.")
        if not merged.split_ref.astype(str).str.lower().equals(merged.split_current.astype(str).str.lower()):
            raise RuntimeError(f"{task}: split assignment mismatch.")
        if not pd.to_numeric(merged[split_target]).equals(pd.to_numeric(merged[target])):
            raise RuntimeError(f"{task}: target label mismatch.")
        tasks[task]={"frame":frame,"target":target,"n_classes":nc}
        split_checks.append({"task":task,"participants":len(frame),"split_and_target":"PASS"})

    prepared={}
    preprocessor_rows=[]
    for task,spec in tasks.items():
        frame=spec["frame"].reset_index(drop=True).copy()
        train_mask=frame.split.astype(str).str.lower().eq("train").to_numpy()
        matrices={}; processors={}; features_by_modality={}; masks=[]; active=[]; fit_rids=None
        for modality in MODALITY_ORDER:
            requested=[f for f in verified if f in set(feature_manifest.loc[
                feature_manifest.modality.eq(modality),"feature"].astype(str))]
            pre=TrainOnlyNumericPreprocessor(requested).fit(frame,train_mask)
            effective=pre.effective_columns()
            if not effective: continue
            pre=TrainOnlyNumericPreprocessor(effective).fit(frame,train_mask)
            matrices[modality]=pre.transform(frame)
            processors[modality]=pre
            features_by_modality[modality]=effective
            masks.append(frame[effective].notna().any(axis=1).to_numpy(dtype=np.float32))
            active.append(modality)
            if fit_rids is None: fit_rids=pre.fit_rids_
            elif fit_rids!=pre.fit_rids_: raise RuntimeError("Preprocessors use different Train RIDs.")
            preprocessor_rows.append({"task":task,"modality":modality,"fit_n":pre.fit_n_,
                                      "effective_features":len(effective),
                                      "fit_RID_exactly_train":set(pre.fit_rids_)==set(frame.loc[train_mask,"RID"].astype(int))})
        if [len(features_by_modality.get("clinical_cognitive",[])),
            len(features_by_modality.get("fluid_biomarker",[]))] != [29,3]:
            raise RuntimeError(f"{task}: expected 29 clinical + 3 biomarker predictors.")
        pack=PreparedTask(task,"S0_verified_asof",frame,spec["target"],spec["n_classes"],
                          active,features_by_modality,matrices,np.column_stack(masks).astype(np.float32),
                          processors,fit_rids or [])
        prepared[(task,"S0_verified_asof")]=pack

    variants=ABLATION_VARIANTS
    plan=pd.concat([
        build_training_plan(prepared,["S0_verified_asof"],variants,SEEDS,tasks=[t])
        for t in TASKS
    ],ignore_index=True)
    if len(plan)!=105: raise RuntimeError(f"Expected 105 jobs; found {len(plan)}.")
    plan_text=plan.to_csv(index=False,lineterminator="\n")
    plan_hash=hashlib.sha256(plan_text.encode("utf-8")).hexdigest()

    model_factory_hash=sha256_file(package/"code/auaf_model_factory.py")
    ablation_factory_hash=sha256_file(package/"code/auaf_ablation_factory.py")
    engine_hash=sha256_file(package/"code/auaf_safe_training_engine.py")
    combined_model_hash=hashlib.sha256(
        (model_factory_hash+ablation_factory_hash).encode("ascii")
    ).hexdigest()

    config=TrainingConfig(
        max_epochs=20,patience=6,batch_size=64,learning_rate=8e-4,
        weight_decay=1e-4,scheduler_patience=3,scheduler_factor=0.5,
        min_delta=1e-5,gradient_clip_norm=5.0,num_workers=0,num_threads=2,device=None
    )
    immutable={
        "schema_version":1,
        "run_id":"stage4_6_ablation_finalcohort_20261116_v1",
        "stage":"4.6_component_ablation_final_temporal_clean_cohort",
        "tasks":list(TASKS),
        "feature_set":"S0_verified_asof",
        "candidate_predictors":34,
        "verified_predictors":32,
        "clinical_cognitive_predictors":29,
        "fluid_biomarker_predictors":3,
        "excluded_unverified_fields":sorted(UNVERIFIED_FIELDS),
        "seeds":SEEDS,
        "variants":variants,
        "variant_definitions":VARIANT_DEFINITIONS,
        "planned_jobs":105,
        "training_config":config.to_dict(),
        "model_selection_metric":"validation_macro_f1",
        "test_evaluation_policy":"Best checkpoint selected by validation Macro-F1; test evaluated once after selection.",
        "preprocessing_policy":"Train-only median imputation and standardization, exactly as Stage 4.5 S0.",
        "split_policy":"Frozen Stage 4.3 participant-level split manifests; no re-splitting.",
        "input_sha256":input_hashes,
        "model_factory_sha256":model_factory_hash,
        "ablation_factory_sha256":ablation_factory_hash,
        "combined_model_code_sha256":combined_model_hash,
        "training_engine_sha256":engine_hash,
        "training_plan_sha256":plan_hash,
        "stage4_5_final_manifest_sha256":hashlib.sha256(
            (package/"STAGE4_5_MANIFEST.json").read_bytes()
        ).hexdigest(),
        "architecture_preflight":{
            "mask_token_perturbation_max_logit_difference":0.0,
            "no_mask_tokens_shared_weight_max_logit_difference":0.0,
            "interpretation":"In the current AUAF forward pass, absent-modality tokens are key-padded in Transformer attention and receive zero gate weight; their learned mask-token values therefore have no path to the prediction output."
        }
    }
    run_hash=initialize_run_directory(out,immutable,resume=True)
    preflight=out/"preflight"; preflight.mkdir(parents=True,exist_ok=True)
    atomic_write_csv(preflight/"training_plan.csv",plan)
    atomic_write_csv(preflight/"split_checks.csv",pd.DataFrame(split_checks))
    atomic_write_csv(preflight/"preprocessor_fit_audit.csv",pd.DataFrame(preprocessor_rows))
    atomic_write_csv(preflight/"variant_definitions.csv",pd.DataFrame(
        [{"variant":k,"definition":v} for k,v in VARIANT_DEFINITIONS.items()]
    ))
    results=run_training_plan(
        prepared=prepared,plan=plan,run_dir=out,run_manifest_sha256=run_hash,
        model_factory=build_ablation_model,config=config,input_hashes=input_hashes,
        model_code_sha256=combined_model_hash,resume=True,max_jobs=args.max_jobs
    )
    atomic_write_csv(out/"execution_results.csv",results)
    print(json.dumps({"status":"completed","planned_jobs":len(plan),
        "executed_jobs":len(results),
        "completed":int(results.status.isin(["COMPLETED","SKIPPED_COMPLETE"]).sum()),
        "plan_sha256":plan_hash,"run_manifest_sha256":run_hash},indent=2))

if __name__=="__main__":
    main()
