#!/usr/bin/env python3
"""
Complete ensemble pipeline for call outcome prediction.

Prerequisites (run in order):
  1. train_xgb.py  -> xgb_model.json
  2. train_lstm.py -> lstm_model.keras (and partial_sequences.json must exist)
  3. ensemble_strategy.json (weights and config)

Pipeline:
  - Load partial_sequences.json; use same stratified 80/20 split as LSTM (random_state=42).
  - Build XGBoost features from features_so_far (fit encoders on train, encode test).
  - Build LSTM inputs (X_type, X_cont) for test set.
  - Load XGBoost and LSTM; get probability distributions on test set.
  - Combine: P_ensemble = w_xgb * P_xgb + w_lstm * P_lstm (from ensemble_strategy.json).
  - Compute accuracy, precision, recall, F1 (macro); compare to XGB-only and LSTM-only.
  - Update ensemble_strategy.json with ensemble_test_performance and a real prediction_example.
  - Write ensemble_results.json with full metrics.
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter

# Align with train_xgb.py and train_lstm.py
FEATURE_ORDER = [
    "total_duration_sec", "time_to_first_user_speech_sec", "time_to_first_tool_call_sec",
    "avg_response_latency_sec", "agent_response_latency_p75", "avg_silence_duration_sec",
    "agent_talk_ratio", "user_talk_ratio", "silence_ratio", "silence_count",
    "user_speech_trend", "user_words_trend", "speech_entropy", "agent_flexibility",
    "turn_count", "words_per_turn_user", "words_per_turn_agent", "user_engagement_slope",
    "interruption_count", "cumulative_user_words", "tools_called_count", "tools_per_minute",
    "survey_completion_rate",
    "agent_id", "org_id", "call_purpose", "time_of_day", "day_of_week",
]
NUMERIC_FEATURES = FEATURE_ORDER[:23]
CATEGORICAL_FEATURES = FEATURE_ORDER[23:]
OUTCOME_CLASSES = ["completed", "abandoned", "transferred", "error"]
MAX_SEQ_LEN = 50
EVENT_TYPE_MAP = {
    "call_start": 0, "agent_speech": 1, "user_speech": 2, "silence": 3,
    "tool_call": 4, "call_end": 5, "padding": 6, "unknown": 7,
}


def load_partial_sequences(data_path: Path):
    with open(data_path) as f:
        data = json.load(f)
    return data.get("partial_sequences", [])


def build_xgb_features(sequences: list[dict], encoders: dict | None = None):
    """Build X (n, 28) from features_so_far; fit encoders if None (on train), else use provided."""
    n_num = len(NUMERIC_FEATURES)
    X_raw = []
    for rec in sequences:
        feats = rec.get("features_so_far", {})
        row = [feats.get(k) for k in FEATURE_ORDER]
        X_raw.append(row)
    X_raw = np.array(X_raw, dtype=object)
    if encoders is None:
        encoders = {}
        for j, name in enumerate(FEATURE_ORDER):
            col = X_raw[:, j]
            if j < n_num:
                continue
            uniques = sorted(set(str(x) if x is not None else "" for x in col))
            encoders[name] = {v: i for i, v in enumerate(uniques)}
    X_enc = np.zeros((len(sequences), len(FEATURE_ORDER)), dtype=np.float32)
    for j, name in enumerate(FEATURE_ORDER):
        col = X_raw[:, j]
        if j < n_num:
            X_enc[:, j] = [float(x) if x is not None and str(x).strip() != "" else 0.0 for x in col]
        else:
            enc = encoders.get(name, {})
            X_enc[:, j] = [enc.get(str(x), 0) for x in col]
    return X_enc, encoders


def build_lstm_inputs(sequences: list[dict]):
    """Build X_type (n, 50), X_cont (n, 50, 2) from sequence."""
    X_type_list = []
    X_cont_list = []
    y_list = []
    for rec in sequences:
        events = rec.get("sequence", [])
        type_seq = []
        cont_seq = []
        for ev in events:
            t = ev.get("type", "")
            type_idx = EVENT_TYPE_MAP.get(t, EVENT_TYPE_MAP["unknown"])
            dur = ev.get("duration_ms", 0) or 0
            words = ev.get("words", 0) or 0
            type_seq.append(type_idx)
            cont_seq.append([float(dur), float(words)])
        while len(type_seq) < MAX_SEQ_LEN:
            type_seq.append(EVENT_TYPE_MAP["padding"])
            cont_seq.append([0.0, 0.0])
        type_seq = type_seq[:MAX_SEQ_LEN]
        cont_seq = cont_seq[:MAX_SEQ_LEN]
        X_type_list.append(type_seq)
        X_cont_list.append(cont_seq)
        outcome = rec.get("outcome", "")
        class_idx = OUTCOME_CLASSES.index(outcome) if outcome in OUTCOME_CLASSES else 0
        y_list.append(class_idx)
    X_type = np.array(X_type_list, dtype=np.int32)
    X_cont = np.array(X_cont_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X_type, X_cont, y


def main():
    base = Path(__file__).parent
    partial_path = base / "partial_sequences.json"
    strategy_path = base / "ensemble_strategy.json"
    xgb_model_path = base / "xgb_model.json"
    lstm_model_path = base / "lstm_model.keras"
    results_path = base / "ensemble_results.json"

    if not xgb_model_path.exists():
        raise SystemExit("Run train_xgb.py first to create xgb_model.json")
    if not lstm_model_path.exists():
        raise SystemExit("Run train_lstm.py first to create lstm_model.keras")

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    sequences = load_partial_sequences(partial_path)
    if len(sequences) != 2500:
        raise SystemExit(f"Expected 2500 partial sequences, got {len(sequences)}")

    # Same stratified 80/20 split as train_lstm (random_state=42)
    y = np.array([OUTCOME_CLASSES.index(s.get("outcome", "")) if s.get("outcome") in OUTCOME_CLASSES else 0 for s in sequences], dtype=np.int32)
    train_idx, test_idx = train_test_split(
        np.arange(len(sequences)), test_size=0.2, stratify=y, random_state=42, shuffle=True
    )
    train_seqs = [sequences[i] for i in train_idx]
    test_seqs = [sequences[i] for i in test_idx]
    y_test_arr = y[test_idx]
    n_test = len(y_test_arr)

    # LSTM inputs for test set
    X_type, X_cont, _ = build_lstm_inputs(sequences)
    X_type_test = X_type[test_idx]
    X_cont_test = X_cont[test_idx]

    # XGBoost features: fit encoders on train, encode test
    X_train_xgb, encoders = build_xgb_features(train_seqs, encoders=None)
    X_test_xgb, _ = build_xgb_features(test_seqs, encoders=encoders)

    # Load XGBoost
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(str(xgb_model_path))
    dtest_xgb = xgb.DMatrix(X_test_xgb)
    P_xgb = booster.predict(dtest_xgb)
    pred_xgb = np.argmax(P_xgb, axis=1)
    acc_xgb = float(accuracy_score(y_test_arr, pred_xgb))
    f1_xgb = float(f1_score(y_test_arr, pred_xgb, average="macro", zero_division=0))

    # Load LSTM
    from tensorflow import keras
    lstm_model = keras.models.load_model(str(lstm_model_path))
    P_lstm = lstm_model.predict([X_type_test, X_cont_test], verbose=0)
    pred_lstm = np.argmax(P_lstm, axis=1)
    acc_lstm = float(accuracy_score(y_test_arr, pred_lstm))
    f1_lstm = float(f1_score(y_test_arr, pred_lstm, average="macro", zero_division=0))

    # Load ensemble config
    with open(strategy_path) as f:
        strategy = json.load(f)
    w_xgb = strategy["ensemble_config"]["weights"]["xgboost"]
    w_lstm = strategy["ensemble_config"]["weights"]["lstm"]

    # Ensemble probabilities
    P_ensemble = w_xgb * P_xgb + w_lstm * P_lstm
    pred_ensemble = np.argmax(P_ensemble, axis=1)
    acc_ensemble = float(accuracy_score(y_test_arr, pred_ensemble))
    prec_ensemble = float(precision_score(y_test_arr, pred_ensemble, average="macro", zero_division=0))
    rec_ensemble = float(recall_score(y_test_arr, pred_ensemble, average="macro", zero_division=0))
    f1_ensemble = float(f1_score(y_test_arr, pred_ensemble, average="macro", zero_division=0))

    # Improvement over single models (on this test set)
    imp_xgb = acc_ensemble - acc_xgb
    imp_lstm = acc_ensemble - acc_lstm
    imp_xgb_pct = f"+{round(imp_xgb * 100)}%" if imp_xgb >= 0 else f"{round(imp_xgb * 100)}%"
    imp_lstm_pct = f"+{round(imp_lstm * 100)}%" if imp_lstm >= 0 else f"{round(imp_lstm * 100)}%"

    ensemble_test_performance = {
        "test_accuracy": round(acc_ensemble, 4),
        "test_precision_macro": round(prec_ensemble, 4),
        "test_recall_macro": round(rec_ensemble, 4),
        "test_f1_macro": round(f1_ensemble, 4),
        "improvement_over_xgb": imp_xgb_pct,
        "improvement_over_lstm": imp_lstm_pct,
        "xgb_only_on_this_test": {"accuracy": round(acc_xgb, 4), "f1_macro": round(f1_xgb, 4)},
        "lstm_only_on_this_test": {"accuracy": round(acc_lstm, 4), "f1_macro": round(f1_lstm, 4)},
        "test_samples": n_test,
    }

    # Prediction example: pick one test sample (e.g. index 0)
    idx = 0
    rec = test_seqs[idx]
    pred_class_ens = pred_ensemble[idx]
    conf_ens = float(P_ensemble[idx].max())
    risk = 1.0 - conf_ens
    pred_class_xgb = pred_xgb[idx]
    conf_xgb = float(P_xgb[idx].max())
    pred_class_lstm = pred_lstm[idx]
    conf_lstm = float(P_lstm[idx].max())
    prediction_example = {
        "call_id": rec.get("call_id", ""),
        "completion_percent": rec.get("completion_percent", 0),
        "actual_outcome": rec.get("outcome", ""),
        "xgb_prediction": {"outcome": OUTCOME_CLASSES[pred_class_xgb], "confidence": round(conf_xgb, 4)},
        "lstm_prediction": {"outcome": OUTCOME_CLASSES[pred_class_lstm], "confidence": round(conf_lstm, 4)},
        "ensemble_prediction": {
            "outcome": OUTCOME_CLASSES[pred_class_ens],
            "confidence": round(conf_ens, 4),
            "risk_score": round(risk, 4),
        },
        "explanation": "Consensus" if pred_class_xgb == pred_class_lstm == pred_class_ens else "Models disagree; ensemble voted.",
    }

    # Update strategy JSON with real ensemble_test_performance
    strategy["ensemble_test_performance"] = ensemble_test_performance
    strategy["prediction_example"] = prediction_example
    strategy["ensemble_test_performance"]["note"] = "Computed on common test set (500 partial sequences, 80/20 split from 2500)."
    with open(strategy_path, "w") as f:
        json.dump(strategy, f, indent=2)

    # Full results file
    results = {
        "ensemble_test_performance": ensemble_test_performance,
        "prediction_example": prediction_example,
        "config": {
            "weights": {"xgboost": w_xgb, "lstm": w_lstm},
            "test_samples": n_test,
        },
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Ensemble pipeline complete. Test accuracy: {acc_ensemble:.4f}, F1 macro: {f1_ensemble:.4f}")
    print(f"XGB only (this test): {acc_xgb:.4f}; LSTM only: {acc_lstm:.4f}")
    print(f"Improvement over XGB: {imp_xgb_pct}, over LSTM: {imp_lstm_pct}")
    print(f"Updated {strategy_path}; wrote {results_path}")


if __name__ == "__main__":
    main()
