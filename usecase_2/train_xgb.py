#!/usr/bin/env python3
"""
XGBoost 4-class classifier for call outcome prediction.
Label-encodes categoricals, stratified train/val/test, 5-fold CV, early stopping, evaluation.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Feature order (must match call_features.json)
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

    # Label-encode categoricals (columns 23-27)
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

    # Outcome labels: completed=0, abandoned=1, transferred=2, error=3
    class_to_idx = {c: i for i, c in enumerate(OUTCOME_CLASSES)}
    y = np.array([class_to_idx.get(str(o), 0) for o in y_raw])
    return X_enc, y, encoders, class_to_idx


def main():
    data_path = Path(__file__).parent / "call_features.json"
    out_path = Path(__file__).parent / "xgb_training_results.json"

    try:
        import xgboost as xgb
    except ImportError:
        raise SystemExit("Install xgboost: pip install xgboost")

    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    X, y, encoders, class_to_idx = load_and_encode(data_path)
    n_total = len(X)

    # Stratified 80% train, 10% val, 10% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    n_train, n_val, n_test = len(X_train), len(X_val), len(X_test)

    # Class distribution in train
    train_dist = {}
    for c in OUTCOME_CLASSES:
        train_dist[c] = int(np.sum(y_train == class_to_idx[c]))

    # 5-fold stratified CV on train
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for tr_idx, va_idx in cv.split(X_train, y_train):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_va, label=y_va)
        params = {
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "multi:softprob",
            "num_class": 4,
            "eval_metric": "mlogloss",
            "random_state": 42,
            "verbosity": 1,
        }
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dval, "val")],
            early_stopping_rounds=10,
            verbose_eval=False,
        )
        pred_proba = model.predict(dval)
        pred = np.argmax(pred_proba, axis=1)
        acc = accuracy_score(y_va, pred)
        cv_scores.append(round(float(acc), 4))
    cv_mean = round(float(np.mean(cv_scores)), 4)
    cv_stdev = round(float(np.std(cv_scores)), 4)

    # Final model: train on full train, validate on val, early stop
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    params = {
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "multi:softprob",
        "num_class": 4,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "verbosity": 1,
    }
    evals_result = {}
    final_model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=10,
        verbose_eval=False,
        evals_result=evals_result,
    )
    best_iteration = final_model.best_iteration if hasattr(final_model, "best_iteration") else 100

    # Save model for SHAP / inference
    model_path = Path(__file__).parent / "xgb_model.json"
    final_model.save_model(str(model_path))

    # Test evaluation
    dtest = xgb.DMatrix(X_test)
    test_proba = final_model.predict(dtest)
    test_pred = np.argmax(test_proba, axis=1)
    test_accuracy = round(float(accuracy_score(y_test, test_pred)), 4)
    test_precision_macro = round(float(precision_score(y_test, test_pred, average="macro", zero_division=0)), 4)
    test_recall_macro = round(float(recall_score(y_test, test_pred, average="macro", zero_division=0)), 4)
    test_f1_macro = round(float(f1_score(y_test, test_pred, average="macro", zero_division=0)), 4)

    # Confusion matrix (rows = true, cols = pred)
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1, 2, 3])
    confusion_dict = {}
    for i, true_label in enumerate(OUTCOME_CLASSES):
        confusion_dict[true_label] = {
            OUTCOME_CLASSES[j]: int(cm[i, j]) for j in range(4)
        }

    # Per-class precision, recall, F1
    per_class = {}
    for i, c in enumerate(OUTCOME_CLASSES):
        p = precision_score(y_test, test_pred, labels=[i], average="macro", zero_division=0)
        r = recall_score(y_test, test_pred, labels=[i], average="macro", zero_division=0)
        f = f1_score(y_test, test_pred, labels=[i], average="macro", zero_division=0)
        # Use binary for this class
        mask_true = y_test == i
        mask_pred = test_pred == i
        tp = np.sum((y_test == i) & (test_pred == i))
        p = tp / np.sum(test_pred == i) if np.sum(test_pred == i) > 0 else 0.0
        r = tp / np.sum(y_test == i) if np.sum(y_test == i) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class[c] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
        }

    # Feature importance (gain)
    imp = final_model.get_score(importance_type="gain")
    # imp keys may be "f0", "f1", ... in some xgboost versions
    importance_by_idx = {}
    for k, v in imp.items():
        if k.startswith("f"):
            idx = int(k[1:])
            importance_by_idx[idx] = float(v)
    # Build list (feature name, importance), sort by importance desc
    feat_imp_list = [
        (FEATURE_ORDER[j], importance_by_idx.get(j, 0.0)) for j in range(len(FEATURE_ORDER))
    ]
    feat_imp_list.sort(key=lambda x: -x[1])
    feature_importance = [
        {"feature": name, "importance": round(imp_val, 4), "rank": r + 1}
        for r, (name, imp_val) in enumerate(feat_imp_list[:15])
    ]

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_id = f"xgb_v1_{ts}"

    report = {
        "model_id": model_id,
        "training_metadata": {
            "total_samples": n_total,
            "train_samples": n_train,
            "val_samples": n_val,
            "test_samples": n_test,
            "class_distribution_train": train_dist,
        },
        "hyperparameters": {
            "max_depth": 5,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "multi:softprob",
            "num_class": 4,
            "early_stopping_rounds": 10,
            "best_iteration": int(best_iteration),
        },
        "training_results": {
            "cv_scores": cv_scores,
            "cv_mean": cv_mean,
            "cv_stdev": cv_stdev,
            "final_n_estimators_used": int(best_iteration),
            "test_accuracy": test_accuracy,
            "test_precision_macro": test_precision_macro,
            "test_recall_macro": test_recall_macro,
            "test_f1_macro": test_f1_macro,
        },
        "confusion_matrix": confusion_dict,
        "feature_importance": feature_importance,
        "per_class_metrics": per_class,
    }

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Training complete. Results: {out_path}")
    print(f"CV mean accuracy: {cv_mean} (±{cv_stdev})")
    print(f"Test accuracy: {test_accuracy}, F1 macro: {test_f1_macro}")
    print(f"Best iteration: {best_iteration}")


if __name__ == "__main__":
    main()
