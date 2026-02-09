#!/usr/bin/env python3
"""
SHAP (SHapley Additive exPlanations) analysis for XGBoost call outcome classifier.
Global importance, class-specific importance, sample explanations, partial dependence summaries.
"""

import json
from pathlib import Path

import numpy as np

# Must match train_xgb.py
NUMERIC_FEATURES = [
    "total_duration_sec",
    "time_to_first_user_speech_sec",
    "time_to_first_tool_call_sec",
    "avg_response_latency_sec",
    "agent_response_latency_p75",
    "avg_silence_duration_sec",
    "agent_talk_ratio",
    "user_talk_ratio",
    "silence_ratio",
    "silence_count",
    "user_speech_trend",
    "user_words_trend",
    "speech_entropy",
    "agent_flexibility",
    "turn_count",
    "words_per_turn_user",
    "words_per_turn_agent",
    "user_engagement_slope",
    "interruption_count",
    "cumulative_user_words",
    "tools_called_count",
    "tools_per_minute",
    "survey_completion_rate",
]
CATEGORICAL_FEATURES = ["agent_id", "org_id", "call_purpose", "time_of_day", "day_of_week"]
FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES
OUTCOME_CLASSES = ["completed", "abandoned", "transferred", "error"]


def load_and_encode(data_path: Path):
    """Same as train_xgb: load rows, encode to X, y; return rows, X, y."""
    with open(data_path) as f:
        rows = json.load(f)
    X_list = []
    y_list = []
    for r in rows:
        feats = r.get("features", {})
        row = []
        for k in FEATURE_ORDER:
            v = feats.get(k)
            row.append(v)
        X_list.append(row)
        y_list.append(r.get("outcome", ""))
    X_raw = np.array(X_list, dtype=object)
    y_raw = np.array(y_list)
    n_num = len(NUMERIC_FEATURES)
    X_enc = np.zeros((len(rows), len(FEATURE_ORDER)))
    encoders = {}
    for j, name in enumerate(FEATURE_ORDER):
        col = X_raw[:, j]
        if j < n_num:
            X_enc[:, j] = np.asarray([float(x) if x is not None and str(x).strip() != "" else 0.0 for x in col])
        else:
            uniques = sorted(set(str(x) if x is not None else "" for x in col))
            encoders[name] = {v: i for i, v in enumerate(uniques)}
            X_enc[:, j] = np.array([encoders[name].get(str(x), 0) for x in col])
    class_to_idx = {c: i for i, c in enumerate(OUTCOME_CLASSES)}
    y = np.array([class_to_idx.get(str(o), 0) for o in y_raw])
    return rows, X_enc, y, encoders, class_to_idx


def main():
    base = Path(__file__).parent
    data_path = base / "call_features.json"
    model_path = base / "xgb_model.json"
    out_path = base / "shap_analysis.json"
    xgb_results_path = base / "xgb_training_results.json"

    try:
        import xgboost as xgb
    except ImportError:
        raise SystemExit("Install xgboost: pip install xgboost")
    from sklearn.model_selection import train_test_split

    rows, X, y, encoders, class_to_idx = load_and_encode(data_path)
    n = len(X)

    # Same stratified split as train_xgb (by indices to retain row mapping)
    indices = np.arange(n)
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=y[temp_idx], random_state=42)

    X_test = X[test_idx]
    y_test = y[test_idx]
    test_rows = [rows[i] for i in test_idx]
    n_test = len(X_test)

    # Load trained model (must exist; run train_xgb.py first)
    if not model_path.exists():
        raise SystemExit("Run train_xgb.py first to create xgb_model.json")
    booster = xgb.Booster()
    booster.load_model(str(model_path))

    # TreeSHAP via XGBoost native pred_contribs (same as SHAP TreeExplainer for trees)
    # Shape with strict_shape=True: (n_test, n_classes, n_features+1); last column is bias
    dtest = xgb.DMatrix(X_test, feature_names=FEATURE_ORDER)
    contribs = booster.predict(dtest, pred_contribs=True, strict_shape=True)
    contribs = np.asarray(contribs)
    # contribs: (n_test, n_classes, n_features+1) -> drop bias -> (n_test, n_classes, n_features)
    if contribs.shape[-1] == len(FEATURE_ORDER) + 1:
        contribs = contribs[:, :, :-1]
    # We want (n_test, n_features, n_classes) for downstream
    shap_arr = np.transpose(contribs, (0, 2, 1))  # (n_test, n_features, n_classes)
    n_features = len(FEATURE_ORDER)

    # 1. GLOBAL FEATURE IMPORTANCE: mean(|SHAP|) per feature across test samples and classes
    mean_abs_shap = np.mean(np.abs(shap_arr), axis=(0, 2))  # (n_features,)
    rank_order = np.argsort(-mean_abs_shap)
    global_shap_importance = [
        {"feature": FEATURE_ORDER[j], "mean_abs_shap": round(float(mean_abs_shap[j]), 4), "rank": r + 1}
        for r, j in enumerate(rank_order)
    ]

    # Correlation with XGBoost importance (Spearman)
    spearman_correlation = None
    if xgb_results_path.exists():
        with open(xgb_results_path) as f:
            xgb_res = json.load(f)
        xgb_rank = {x["feature"]: x["rank"] for x in xgb_res.get("feature_importance", [])}
        shap_rank = {x["feature"]: x["rank"] for x in global_shap_importance}
        try:
            from scipy.stats import spearmanr
            xgb_r = [xgb_rank.get(f, 99) for f in FEATURE_ORDER]
            shap_r = [shap_rank.get(f, 99) for f in FEATURE_ORDER]
            rho, _ = spearmanr(xgb_r, shap_r)
            spearman_correlation = round(float(rho), 4)
        except Exception:
            pass

    # 2. CLASS-SPECIFIC IMPORTANCE: mean SHAP (not absolute) per feature per class
    class_specific_importance = {}
    for c, class_name in enumerate(OUTCOME_CLASSES):
        mean_shap_c = np.mean(shap_arr[:, :, c], axis=0)  # (n_features,)
        sorted_idx = np.argsort(-np.abs(mean_shap_c))
        class_specific_importance[class_name] = [
            {"feature": FEATURE_ORDER[j], "mean_shap": round(float(mean_shap_c[j]), 4)}
            for j in sorted_idx
        ]

    # 3. SAMPLE EXPLANATIONS: 5 test records, 1-2 per class if possible
    pred_proba = booster.predict(xgb.DMatrix(X_test))
    pred_class = np.argmax(pred_proba, axis=1)
    confidence = np.max(pred_proba, axis=1)

    # Select 5 test indices: try to get 1-2 per predicted class
    selected = []
    for c in range(4):
        idx_c = np.where(pred_class == c)[0]
        selected.extend(idx_c[:2].tolist())
    selected = selected[:5]
    if len(selected) < 5:
        for i in range(n_test):
            if i not in selected:
                selected.append(i)
                if len(selected) >= 5:
                    break

    sample_explanations = []
    for idx in selected:
        row = test_rows[idx]
        call_id = row.get("call_id", "")
        actual = row.get("outcome", "")
        pred = OUTCOME_CLASSES[pred_class[idx]]
        conf = round(float(confidence[idx]), 4)
        # SHAP for predicted class (how much each feature pushed toward this class)
        shaps_pred = shap_arr[idx, :, pred_class[idx]]  # (n_features,)
        feat_values = row.get("features", {})
        # Top 5 supporting (positive SHAP for predicted class)
        order_supp = np.argsort(-shaps_pred)
        top_5_supporting = []
        for j in order_supp[:5]:
            fname = FEATURE_ORDER[j]
            val = feat_values.get(fname)
            if isinstance(val, (int, float)):
                val = round(float(val), 4) if isinstance(val, float) else int(val)
            top_5_supporting.append({
                "feature": fname,
                "value": val,
                "shap_contribution": round(float(shaps_pred[j]), 4),
                "direction": "supports prediction",
            })
        # Top 3 opposing (negative SHAP for predicted class)
        order_opp = np.argsort(shaps_pred)
        top_3_opposing = []
        for j in order_opp[:3]:
            if shaps_pred[j] >= 0:
                continue
            fname = FEATURE_ORDER[j]
            val = feat_values.get(fname)
            if isinstance(val, (int, float)):
                val = round(float(val), 4) if isinstance(val, float) else int(val)
            top_3_opposing.append({
                "feature": fname,
                "value": val,
                "shap_contribution": round(float(shaps_pred[j]), 4),
                "direction": "opposes prediction",
            })
        sample_explanations.append({
            "call_id": call_id,
            "actual_outcome": actual,
            "predicted_outcome": pred,
            "confidence": conf,
            "top_5_supporting_factors": top_5_supporting,
            "top_3_opposing_factors": top_3_opposing,
        })

    # 4. PARTIAL DEPENDENCE SUMMARIES: top 5 features, describe SHAP vs feature value (data-driven)
    top_5_feature_names = [global_shap_importance[i]["feature"] for i in range(5)]
    partial_dependence_summaries = {}
    for fname in top_5_feature_names:
        j = FEATURE_ORDER.index(fname)
        f_vals = X_test[:, j]
        # SHAP for "completed" class (index 0)
        shap_completed = shap_arr[:, j, 0]
        low_p, mid_p, high_p = np.percentile(f_vals, [33, 50, 67])
        mask_low = f_vals <= low_p
        mask_mid = (f_vals > low_p) & (f_vals <= high_p)
        mask_high = f_vals > high_p
        mean_shap_low = float(np.mean(shap_completed[mask_low])) if np.any(mask_low) else 0.0
        mean_shap_mid = float(np.mean(shap_completed[mask_mid])) if np.any(mask_mid) else 0.0
        mean_shap_high = float(np.mean(shap_completed[mask_high])) if np.any(mask_high) else 0.0
        # Describe in business-friendly text
        if mean_shap_high < mean_shap_low - 0.05:
            trend = f"SHAP values are highly negative (prevent completion) when {fname} is high (>{high_p:.2f}); become positive or near-zero when low (≤{low_p:.2f}). Clear negative relationship."
        elif mean_shap_high > mean_shap_low + 0.05:
            trend = f"SHAP values are positive (support completion) when {fname} is high; negative or near-zero when low. Positive correlation—more of this feature helps completion."
        else:
            trend = f"SHAP values vary across the range (low/mid/high mean SHAP for completed: {mean_shap_low:.3f}, {mean_shap_mid:.3f}, {mean_shap_high:.3f}). Moderate or non-linear effect."
        partial_dependence_summaries[fname] = trend

    report = {
        "global_shap_importance": global_shap_importance,
        "spearman_with_xgb_importance": spearman_correlation,
        "class_specific_importance": class_specific_importance,
        "sample_explanations": sample_explanations,
        "partial_dependence_summaries": partial_dependence_summaries,
    }

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"SHAP analysis written to {out_path}")
    print(f"Top 3 global SHAP features: {[x['feature'] for x in global_shap_importance[:3]]}")
    if spearman_correlation is not None:
        print(f"Spearman correlation with XGBoost importance: {spearman_correlation}")


if __name__ == "__main__":
    main()
