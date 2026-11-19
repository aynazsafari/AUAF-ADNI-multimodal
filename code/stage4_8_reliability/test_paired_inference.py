import numpy as np
import pandas as pd
from paired_inference import (
    temperature_scale, fit_temperature, paired_bootstrap_difference,
    bootstrap_metric_ci, finite_sample_conformal_threshold,
    raps_nonconformity, raps_prediction_sets, conformal_test_summary,
    benjamini_hochberg,
)


def _toy():
    rng = np.random.default_rng(5)
    y = np.array([0,1,2] * 40)
    logits = rng.normal(size=(len(y),3))
    logits[np.arange(len(y)), y] += 1.5
    e = np.exp(logits - logits.max(1,keepdims=True))
    p = e/e.sum(1,keepdims=True)
    return y,p


def test_temperature_scaling_is_valid_and_positive():
    y,p = _toy()
    t = fit_temperature(y,p)
    calibrated = temperature_scale(p,t)
    assert t > 0
    assert calibrated.shape == p.shape
    assert np.allclose(calibrated.sum(1),1.0)


def test_paired_bootstrap_uses_all_seeds_and_same_cohort():
    y,p = _toy()
    a = [p, p.copy(), p.copy(), p.copy(), p.copy()]
    b = [np.roll(p,1,axis=1) for _ in range(5)]
    out = paired_bootstrap_difference(y,a,b,metric="f1_macro",n_boot=80,random_seed=7)
    assert out["n_seeds"] == 5
    assert out["n_participants"] == len(y)
    assert out["n_boot"] == 80
    assert out["ci_low"] <= out["ci_high"]


def test_bootstrap_metric_ci_reports_seed_and_participant_uncertainty():
    y,p = _toy()
    out = bootstrap_metric_ci(y,[p]*5,metric="roc_auc",n_boot=80,random_seed=8)
    assert out["n_seeds"] == 5
    assert out["participant_bootstrap_ci_low"] <= out["participant_bootstrap_ci_high"]


def test_finite_sample_quantile_uses_order_statistic_not_interpolation():
    scores = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    q = finite_sample_conformal_threshold(scores,alpha=0.2)
    k = int(np.ceil((len(scores)+1)*(1-0.2)))
    assert q == np.partition(scores,k-1)[k-1]


def test_raps_sets_and_summary_definitions():
    y,p = _toy()
    scores = raps_nonconformity(p,y)
    q = finite_sample_conformal_threshold(scores,alpha=0.1)
    sets = raps_prediction_sets(p,q)
    assert len(sets)==len(y)
    result, sets2 = conformal_test_summary(y,p,y,p,alpha=0.1)
    assert sets2 == sets
    assert 0 <= result["singleton_rate"] <= 1
    assert 0 <= result["full_set_rate"] <= 1
    assert 0 <= result["empty_set_rate"] <= 1
    assert result["formal_guarantee_claimed"] is False


def test_bh_adjustment_is_monotonic_in_rank():
    q = benjamini_hochberg([0.01,0.04,0.03,0.2])
    assert np.all((q>=0)&(q<=1))
    assert q[0] <= q[2] <= q[1] <= q[3]



def test_bh_adjustment_with_unsorted_pvalues():
    q = benjamini_hochberg([0.2, 0.01, 0.04, 0.03])
    assert q[1] <= q[3] <= q[2] <= q[0]


def test_conformal_bootstrap_intervals_are_finite():
    from paired_inference import bootstrap_conformal_intervals
    y,p = _toy()
    sets = [[int(p[i].argmax())] for i in range(len(y))]
    out = bootstrap_conformal_intervals(y, sets, n_classes=3, n_boot=80, random_seed=9)
    assert 0 <= out["coverage_ci_low"] <= out["coverage_ci_high"] <= 1



def test_abstention_definition_is_non_singleton_not_full_set():
    from paired_inference import conformal_test_summary
    y = np.array([0, 1, 2, 0])
    p = np.array([
        [0.90, 0.08, 0.02],
        [0.45, 0.40, 0.15],
        [0.34, 0.33, 0.33],
        [0.70, 0.20, 0.10],
    ])
    result, sets = conformal_test_summary(
        y, p, y, p, alpha=0.10, lam=0.01, k_reg=1, force_nonempty=False
    )
    assert np.isclose(
        result["singleton_rate"] + result["multi_label_rate"] + result["empty_set_rate"], 1.0
    )
    assert np.isclose(
        result["abstention_rate"], result["multi_label_rate"] + result["empty_set_rate"]
    )
    assert np.isclose(
        result["full_set_rate"] <= result["multi_label_rate"], True
    )


def test_pooled_conformal_bootstrap_keeps_seed_cohort_pairing():
    from paired_inference import bootstrap_conformal_intervals_across_seeds
    y = np.array([0, 1, 0, 1, 0, 1] * 10)
    sets_a = [[int(v)] for v in y]
    sets_b = [[0, 1] for _ in y]
    out = bootstrap_conformal_intervals_across_seeds(
        y, [sets_a, sets_b, sets_a, sets_b, sets_a],
        n_classes=2, n_boot=100, random_seed=12,
    )
    assert out["n_seeds"] == 5
    assert out["n_participants"] == len(y)
    assert out["coverage_mean_over_seeds"] == 1.0
    assert out["singleton_rate_mean_over_seeds"] == 0.6
    assert out["coverage_participant_bootstrap_ci_low"] <= out["coverage_participant_bootstrap_ci_high"]



def test_legacy_threshold_matches_original_interpolated_quantile_rule():
    from paired_inference import legacy_interpolated_conformal_threshold
    scores = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
    alpha = 0.10
    level = min(np.ceil((len(scores) + 1) * (1-alpha)) / len(scores), 1.0)
    expected = np.quantile(scores, level)
    assert np.isclose(legacy_interpolated_conformal_threshold(scores, alpha), expected)


def test_replace_auaf_mc_probabilities_is_strictly_paired():
    from stage48_reliability import replace_auaf_with_mc_probabilities
    base = pd.DataFrame([
        {"task":"t","model":"AUAF","seed":42,"split":split,"row_index":i,
         "y_true":i%2,"p_class_0":0.6,"p_class_1":0.4}
        for split in ["validation", "test"] for i in range(4)
    ])
    mc = pd.DataFrame([
        {"task":"t","model":"AUAF","seed":42,"split":split,"row_index":i,
         "y_true":i%2,"p_class_0":0.2,"p_class_1":0.8}
        for split in ["validation", "test"] for i in range(4)
    ])
    out = replace_auaf_with_mc_probabilities(base, mc, {"t":2})
    assert np.allclose(out["p_class_0"], 0.2)
    assert np.allclose(out["p_class_1"], 0.8)
    mc.loc[0,"y_true"] = 1
    try:
        replace_auaf_with_mc_probabilities(base, mc, {"t":2})
    except ValueError:
        pass
    else:
        raise AssertionError("Expected a strict pairing failure.")


def test_paired_s0_tests_uses_final_six_comparators_and_126_tests():
    from paired_inference import paired_s0_tests
    from types import SimpleNamespace
    tasks = ["diagnosis_multiclass","diagnosis_binary","mci_conversion"]
    models = ["AUAF","EarlyFusion","LateFusion","AttentionFusion","XGBoost","CatBoost","ExtraTrees"]
    seeds = [42,123,456,789,1024]
    n = 60
    y = np.array([0,1]*30)
    rows = []
    cal_rows = []
    rng = np.random.default_rng(123)
    for task in tasks:
        for model in models:
            for seed in seeds:
                p = np.column_stack([0.25+0.5*(1-y),0.25+0.5*y])
                p += rng.normal(0,0.015,p.shape)
                p = np.clip(p,0.01,None); p /= p.sum(axis=1,keepdims=True)
                for split in ["validation","test"]:
                    for i in range(n):
                        rows.append({
                            "task":task,"model":model,"seed":seed,"split":split,
                            "row_index":i,"y_true":int(y[i]),
                            "p_class_0":float(p[i,0]),"p_class_1":float(p[i,1]),
                        })
                pc = p.copy()
                for i in range(n):
                    cal_rows.append({
                        "task":task,"model":model,"seed":seed,
                        "row_index":i,"y_true":int(y[i]),
                        "p_raw_class_0":float(p[i,0]),"p_raw_class_1":float(p[i,1]),
                        "p_cal_class_0":float(pc[i,0]),"p_cal_class_1":float(pc[i,1]),
                    })
    pred = pd.DataFrame(rows)
    cal = pd.DataFrame(cal_rows)
    prepared = {task: SimpleNamespace(n_classes=2) for task in tasks}
    prepared = {(task,"S0_verified_asof"): SimpleNamespace(n_classes=2) for task in tasks}
    result = paired_s0_tests(pred, cal, prepared, seeds, n_boot=50, random_seed=10)
    assert len(result) == 126
    assert set(result["comparator"].unique()) == set(models)-{"AUAF"}
    assert result["q_value_BH_global"].between(0,1).all()



def test_conversion_followup_selection_audit_reports_included_vs_excluded(tmp_path):
    from stage48_reliability import run_conversion_followup_selection_audit
    all_mci = pd.DataFrame({
        "RID": list(range(1,11)),
        "AGE": [60,61,62,63,64,65,66,67,68,69],
        "MMSE": [29,28,27,26,25,24,23,22,21,20],
    })
    eligible = pd.DataFrame({"RID": [1,2,3,4,5,6]})
    all_path = tmp_path / "all_mci.csv"
    eligible_path = tmp_path / "eligible.csv"
    all_mci.to_csv(all_path,index=False)
    eligible.to_csv(eligible_path,index=False)
    flow, features = run_conversion_followup_selection_audit(
        all_path, eligible_path, ["AGE","MMSE"],
        expected_all_n=10, expected_eligible_n=6,
    )
    assert flow.loc[0,"included_36m_eligible_n"] == 6
    assert flow.loc[0,"excluded_insufficient_followup_n"] == 4
    assert set(features["feature"]) == {"AGE","MMSE"}



def test_conformal_s0_results_primary_and_legacy_variants():
    from paired_inference import conformal_s0_results
    from types import SimpleNamespace
    y_val = np.array([0,1]*10)
    y_test = np.array([0,1]*8)
    p_val = np.column_stack([0.75-0.3*y_val, 0.25+0.3*y_val])
    p_test = np.column_stack([0.70-0.25*y_test, 0.30+0.25*y_test])
    rows=[]
    for split, y, p in [("validation",y_val,p_val),("test",y_test,p_test)]:
        for i in range(len(y)):
            rows.append({
                "task":"diagnosis_binary","model":"AUAF","seed":42,
                "split":split,"row_index":i,"y_true":int(y[i]),
                "p_class_0":float(p[i,0]),"p_class_1":float(p[i,1]),
            })
    pred=pd.DataFrame(rows)
    params=pd.DataFrame([{
        "task":"diagnosis_binary","model":"AUAF","seed":42,"temperature":1.0
    }])
    prepared={("diagnosis_binary","S0_verified_asof"):SimpleNamespace(n_classes=2)}
    primary, primary_sets = conformal_s0_results(
        pred,params,prepared,force_nonempty=False,
        threshold_method="finite_sample",use_temperature=True
    )
    legacy, legacy_sets = conformal_s0_results(
        pred,params,prepared,force_nonempty=True,
        threshold_method="legacy_interpolated",use_temperature=False
    )
    assert len(primary)==1 and len(primary_sets)==len(y_test)
    assert bool(primary.iloc[0]["temperature_applied_to_RAPS_probabilities"])
    assert primary.iloc[0]["threshold_method"]=="finite_sample"
    assert not bool(legacy.iloc[0]["temperature_applied_to_RAPS_probabilities"])
    assert legacy.iloc[0]["threshold_method"]=="legacy_interpolated"
    assert bool(legacy.iloc[0]["argmax_fallback"])
