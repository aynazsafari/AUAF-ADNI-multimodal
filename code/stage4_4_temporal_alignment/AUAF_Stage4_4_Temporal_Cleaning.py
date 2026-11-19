#!/usr/bin/env python3
"""
AUAF Stage 4.4 — Temporal-safe dataset assembly.
This script preserves the frozen Stage 4.3 cohorts and supplied subject splits.
It does not retrain models, impute missing values, or scale features.
"""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
import pandas as pd

CLINICAL_FEATURES = [
    "PTGENDER","PTEDUCAT","PTENGSPKAGE","PTIMMAGE","MMSEASON","TRAILS",
    "CLOCKCON","CLOCKNO","CLOCKHAN","FFLUENCY","MOCA","FAQFINAN","FAQFORM",
    "FAQSHOP","FAQGAME","FAQBEVG","FAQMEAL","FAQEVENT","FAQTV","FAQREM",
    "FAQTRAVL","FAQTOTAL","CDMEMORY","CDRSB","CLOCKCIRC","CLOCKSYM",
    "CLOCKNUM","CLOCKHAND","CLOCKTIME","CLOCKSCOR","DIGITSCOR"
]
BIOMARKER_FEATURES = ["ABETA40","ABETA42","PTAU"]
ALL_FEATURES = CLINICAL_FEATURES + BIOMARKER_FEATURES

def first_nonmissing(s):
    s = s.dropna()
    return s.iloc[0] if len(s) else np.nan

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def validate_split(split, frozen, target, frozen_target, task):
    if split["RID"].duplicated().any():
        raise RuntimeError(f"{task}: duplicate RID in split manifest")
    joined = split.merge(frozen[["RID", frozen_target]], on="RID",
                         how="outer", validate="one_to_one", indicator=True)
    if not (joined["_merge"] == "both").all():
        raise RuntimeError(f"{task}: split RID set differs from frozen cohort")
    if not (joined[target] == joined[frozen_target]).all():
        raise RuntimeError(f"{task}: split labels differ from frozen cohort")
    sets = {s: set(split.loc[split["_split"] == s, "RID"])
            for s in ["train", "validation", "test"]}
    if sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"]:
        raise RuntimeError(f"{task}: split sets overlap")
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--diagnostic-manifest", required=True)
    ap.add_argument("--conversion-manifest", required=True)
    ap.add_argument("--split-multiclass", required=True)
    ap.add_argument("--split-binary", required=True)
    ap.add_argument("--split-conversion", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.data, low_memory=False)
    diag = pd.read_csv(args.diagnostic_manifest, low_memory=False)
    conv = pd.read_csv(args.conversion_manifest, low_memory=False)
    smc = pd.read_csv(args.split_multiclass, low_memory=False)
    sbin = pd.read_csv(args.split_binary, low_memory=False)
    scon = pd.read_csv(args.split_conversion, low_memory=False)

    data["RID"] = pd.to_numeric(data["RID"], errors="raise").astype("int64")
    data["DIAGNOSIS"] = pd.to_numeric(data["DIAGNOSIS"], errors="coerce")
    data["EXAMDATE_dt"] = pd.to_datetime(data["EXAMDATE"], errors="coerce")
    for m in (diag, conv):
        m["RID"] = pd.to_numeric(m["RID"], errors="raise").astype("int64")
        m["INDEX_DATE_dt"] = pd.to_datetime(m["INDEX_DATE"], errors="coerce")
    for s in (smc, sbin, scon):
        s["RID"] = pd.to_numeric(s["RID"], errors="raise").astype("int64")
        s["_split"] = s["_split"].astype(str).str.lower()

    missing = [f for f in CLINICAL_FEATURES if f not in data.columns]
    if missing:
        raise KeyError(f"Missing clinical features: {missing}")

    dx = data[data["DIAGNOSIS"].isin([1.0, 2.0, 3.0])].copy()
    dx = dx.sort_values(["RID","VISCODE","EXAMDATE_dt","EXAMDATE_BIO"], na_position="last")
    visits = dx.groupby(
        ["RID","VISCODE","EXAMDATE_dt","DIAGNOSIS"], dropna=False, as_index=False
    ).agg({f:first_nonmissing for f in CLINICAL_FEATURES})

    dj = diag.merge(
        visits,
        left_on=["RID","VISCODE","INDEX_DATE_dt","DIAGNOSIS_CODE"],
        right_on=["RID","VISCODE","EXAMDATE_dt","DIAGNOSIS"],
        how="left", validate="one_to_one", indicator=True
    )
    cj = conv.merge(
        visits,
        left_on=["RID","INDEX_VISCODE","INDEX_DATE_dt","DXSUM_INDEX_DIAGNOSIS"],
        right_on=["RID","VISCODE","EXAMDATE_dt","DIAGNOSIS"],
        how="left", validate="one_to_one", indicator=True
    )
    if not (dj["_merge"] == "both").all() or len(dj) != 3700:
        raise RuntimeError("Diagnostic index rows did not match the clinical source exactly.")
    if not (cj["_merge"] == "both").all() or len(cj) != 911:
        raise RuntimeError("Conversion index rows did not match the clinical source exactly.")

    validate_split(smc, diag, "target_multiclass", "TARGET_MULTICLASS", "cn_mci_ad")
    validate_split(sbin, diag, "target_binary", "TARGET_BINARY", "ad_vs_nonad")
    validate_split(scon, conv, "target_decline", "TARGET_CONVERSION", "early_decline")

    def build(manifest, clinical_join, split, target):
        out_df = manifest.merge(split[["RID","_split",target]], on="RID",
                                how="left", validate="one_to_one")
        out_df = out_df.merge(clinical_join[["RID"] + CLINICAL_FEATURES],
                              on="RID", how="left", validate="one_to_one")
        for f in CLINICAL_FEATURES:
            out_df[f] = pd.to_numeric(out_df[f], errors="coerce").replace([-1,-4], np.nan)
        asof = out_df["BIOMARKER_AVAILABLE_AS_OF_INDEX"].astype(bool) & out_df["INDEX_DATE_dt"].notna()
        for f in BIOMARKER_FEATURES:
            out_df[f] = pd.to_numeric(out_df[f], errors="coerce")
            out_df.loc[~asof, f] = np.nan
        model_cols = ["RID","_split",target] + ALL_FEATURES
        model_df = out_df[model_cols].copy()
        if out_df.loc[out_df["BIOMARKER_AFTER_INDEX"].astype(bool), BIOMARKER_FEATURES].notna().any(axis=None):
            raise RuntimeError(f"{target}: post-index biomarkers survived filtering")
        return out_df, model_df

    tasks = {
        "cn_mci_ad": build(diag, dj, smc, "target_multiclass"),
        "ad_vs_nonad": build(diag, dj, sbin, "target_binary"),
        "early_decline": build(conv, cj, scon, "target_decline")
    }

    for name, (_, model_df) in tasks.items():
        model_df.to_csv(out / f"{name}_asof.csv", index=False)

    summary = {}
    for name, (full, model_df) in tasks.items():
        summary[name] = {
            "n": int(len(model_df)),
            "unique_RID": int(model_df["RID"].nunique()),
            "n_features": int(len(ALL_FEATURES)),
            "clinical_features": int(len(CLINICAL_FEATURES)),
            "biomarker_features": int(len(BIOMARKER_FEATURES)),
            "index_date_missing": int(full["INDEX_DATE"].isna().sum()),
            "biomarker_exact_visit_source": int(full["BIOMARKER_SOURCE_ROW_PRESENT"].astype(bool).sum()),
            "biomarker_asof": int(full["BIOMARKER_AVAILABLE_AS_OF_INDEX"].astype(bool).sum()),
            "biomarker_post_index_excluded": int(full["BIOMARKER_AFTER_INDEX"].astype(bool).sum()),
            "biomarker_available_in_model": int(model_df[BIOMARKER_FEATURES].notna().any(axis=1).sum())
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    missing_rows = []
    for name, (_, model_df) in tasks.items():
        target = model_df.columns[2]
        for split_name in ["all","train","validation","test"]:
            part = model_df if split_name == "all" else model_df[model_df["_split"] == split_name]
            for f in ALL_FEATURES:
                n_non = int(part[f].notna().sum())
                missing_rows.append({
                    "task":name,"split":split_name,"feature":f,
                    "modality":"clinical" if f in CLINICAL_FEATURES else "biomarker",
                    "n":len(part),"n_nonmissing":n_non,
                    "n_missing":len(part)-n_non,
                    "missing_pct":round(100*(len(part)-n_non)/len(part),2)
                })
    pd.DataFrame(missing_rows).to_csv(out / "feature_missingness.csv", index=False)

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
