"""
REST API for Call Outcome Prediction.
Endpoints: POST /api/model/train, POST /api/predict, GET /api/model/{model_id}/importance, GET /api/models.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

# Ensure usecase_2 is on path when running as api.main
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

MODELS_DIR = BASE_DIR / "models"
REGISTRY_PATH = MODELS_DIR / "registry.json"
MAX_VERSIONS_PER_TYPE = 3
OUTCOME_CLASSES = ["completed", "abandoned", "transferred", "error"]
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
EVENT_TYPE_MAP = {
    "call_start": 0, "agent_speech": 1, "user_speech": 2, "silence": 3,
    "tool_call": 4, "call_end": 5, "padding": 6, "unknown": 7,
}
MAX_SEQ_LEN = 50

# In-memory model cache: model_id -> { type, booster?, lstm_model?, encoders?, weights?, xgb_path?, lstm_path? }
_model_cache = {}
_registry = None


def _ensure_models_dir():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _load_registry():
    global _registry
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            _registry = json.load(f)
    else:
        _registry = {"models": [], "default_model_id": None}
    return _registry


def _save_registry():
    _ensure_models_dir()
    with open(REGISTRY_PATH, "w") as f:
        json.dump(_registry, f, indent=2)


def _prune_versions():
    by_type = {}
    for m in _registry["models"]:
        t = m.get("type", "xgboost")
        by_type.setdefault(t, []).append(m)
    for t, list_m in by_type.items():
        list_m.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        for m in list_m[MAX_VERSIONS_PER_TYPE:]:
            _registry["models"].remove(m)
            pid = m.get("model_id")
            for suffix in ["", "_encoders.json", "_meta.json"]:
                p = MODELS_DIR / f"{pid}{suffix}"
                if p.suffix == ".json":
                    p = MODELS_DIR / f"{pid}{suffix}"
                else:
                    p = MODELS_DIR / f"{pid}.json" if not suffix else MODELS_DIR / f"{pid}{suffix}"
                if p.exists():
                    try:
                        p.unlink()
                    except Exception:
                        pass
            if t == "xgboost":
                for ext in [".json"]:
                    (MODELS_DIR / f"{pid}{ext}").unlink(missing_ok=True)
            elif t == "lstm":
                (MODELS_DIR / f"{pid}.keras").unlink(missing_ok=True)
    _save_registry()


def _events_to_features(call_id: str, events: list, metadata: dict):
    """Compute 28 features from events_so_far + metadata (partial sequence; survey=0)."""
    from feature_engineering import compute_features
    events = sorted(events, key=lambda e: (e.get("ts", 0), e.get("type", "")))
    call = {
        "call_id": call_id,
        "metadata": metadata or {},
        "events": events,
        "outcome": "",
        "survey_completion_rate": 0.0,
    }
    try:
        out = compute_features(call)
        return out.get("features", {})
    except Exception as e:
        raise RuntimeError(f"Feature computation failed: {e}") from e


def _features_to_xgb_row(features: dict, encoders: dict):
    """Build single row (28,) for XGBoost from features dict + encoders."""
    import numpy as np
    n_num = len(NUMERIC_FEATURES)
    row = np.zeros((len(FEATURE_ORDER),), dtype=np.float32)
    for j, name in enumerate(FEATURE_ORDER):
        v = features.get(name)
        if j < n_num:
            if v is None or (isinstance(v, str) and not v.strip()):
                row[j] = 0.0
            else:
                try:
                    row[j] = float(v)
                except (TypeError, ValueError):
                    row[j] = 0.0
        else:
            enc = encoders.get(name, {})
            row[j] = enc.get(str(v), 0)
    return row.reshape(1, -1)


def _events_to_lstm_inputs(events: list):
    """Build X_type (1, 50), X_cont (1, 50, 2) from events."""
    import numpy as np
    events = sorted(events, key=lambda e: (e.get("ts", 0), e.get("type", "")))
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
    X_type = np.array([type_seq], dtype=np.int32)
    X_cont = np.array([cont_seq], dtype=np.float32)
    return X_type, X_cont


def _get_completion_percent(events: list):
    if not events:
        return 0
    ts_list = [e.get("ts", 0) for e in events if isinstance(e.get("ts"), (int, float))]
    if not ts_list:
        return 0
    duration_so_far = max(ts_list) - min(ts_list)
    # Assume max call 600s (10 min)
    return min(100, int(100 * duration_so_far / 600.0))


def _load_model_into_cache(model_id: str):
    """Load model from disk into _model_cache."""
    _load_registry()
    entry = next((m for m in _registry["models"] if m.get("model_id") == model_id), None)
    if not entry:
        return None
    mtype = entry.get("type", "xgboost")
    pid = entry.get("model_id")
    if mtype == "xgboost":
        import xgboost as xgb
        path = MODELS_DIR / f"{pid}.json"
        if not path.exists():
            path = BASE_DIR / "xgb_model.json"
        if not path.exists():
            return None
        encoders_path = MODELS_DIR / f"{pid}_encoders.json"
        encoders = {}
        if encoders_path.exists():
            with open(encoders_path) as f:
                encoders = json.load(f)
        else:
            try:
                with open(BASE_DIR / "call_features.json") as f:
                    rows = json.load(f)
                from ensemble_pipeline import build_xgb_features
                seqs = [{"features_so_far": r.get("features", {})} for r in rows]
                _, encoders = build_xgb_features(seqs, encoders=None)
            except Exception:
                pass
        booster = xgb.Booster()
        booster.load_model(str(path))
        _model_cache[model_id] = {"type": "xgboost", "booster": booster, "encoders": encoders}
    elif mtype == "lstm":
        from tensorflow import keras
        path = MODELS_DIR / f"{pid}.keras"
        if not path.exists():
            path = BASE_DIR / "lstm_model.keras"
        if not path.exists():
            return None
        model = keras.models.load_model(str(path))
        _model_cache[model_id] = {"type": "lstm", "lstm_model": model}
    elif mtype == "ensemble":
        xgb_id = entry.get("xgb_model_id")
        lstm_id = entry.get("lstm_model_id")
        weights = entry.get("weights", {"xgboost": 0.65, "lstm": 0.35})
        if not xgb_id or not lstm_id:
            return None
        _load_model_into_cache(xgb_id)
        _load_model_into_cache(lstm_id)
        _model_cache[model_id] = {
            "type": "ensemble",
            "xgb_model_id": xgb_id,
            "lstm_model_id": lstm_id,
            "weights": weights,
        }
    return _model_cache.get(model_id)


def _predict_xgb(features: dict, encoders: dict, booster):
    import xgboost as xgb
    import numpy as np
    row = _features_to_xgb_row(features, encoders)
    d = xgb.DMatrix(row)
    proba = booster.predict(d)
    return np.squeeze(proba).tolist()


def _predict_lstm(events: list, lstm_model):
    X_type, X_cont = _events_to_lstm_inputs(events)
    proba = lstm_model.predict([X_type, X_cont], verbose=0)
    return proba[0].tolist()


def _predict_ensemble(features: dict, events: list, cache_entry: dict):
    xgb_id = cache_entry["xgb_model_id"]
    lstm_id = cache_entry["lstm_model_id"]
    w = cache_entry["weights"]
    xgb_entry = _model_cache.get(xgb_id)
    lstm_entry = _model_cache.get(lstm_id)
    if not xgb_entry or not lstm_entry:
        return None
    p_xgb = _predict_xgb(features, xgb_entry["encoders"], xgb_entry["booster"])
    p_lstm = _predict_lstm(events, lstm_entry["lstm_model"])
    import numpy as np
    p_ens = np.array(p_xgb) * w.get("xgboost", 0.65) + np.array(p_lstm) * w.get("lstm", 0.35)
    return p_ens.tolist()


def _xgb_pred_contrib(features: dict, encoders: dict, booster):
    """Single-row pred_contribs for top_factors (impact per feature)."""
    import xgboost as xgb
    import numpy as np
    row = _features_to_xgb_row(features, encoders)
    d = xgb.DMatrix(row)
    contrib = booster.predict(d, pred_contribs=True, strict_shape=True)
    contrib = np.squeeze(contrib)
    if contrib.ndim == 2:
        # (1, n_classes, n_features+1) -> drop bias
        contrib = contrib[0, :, :-1]
        impact_by_feature = np.mean(np.abs(contrib), axis=0)
    else:
        impact_by_feature = np.abs(contrib[:-1])
    top_idx = np.argsort(-impact_by_feature)[:5]
    return [(FEATURE_ORDER[i], float(features.get(FEATURE_ORDER[i], 0)), float(impact_by_feature[i])) for i in top_idx if i < len(FEATURE_ORDER)]


# --- FastAPI app ---
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError("Install fastapi and uvicorn: pip install fastapi uvicorn")

app = FastAPI(title="Call Outcome Prediction API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def startup():
    _ensure_models_dir()
    _load_registry()
    if not _registry["models"] and (BASE_DIR / "xgb_model.json").exists() and (BASE_DIR / "lstm_model.keras").exists():
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        _registry["models"] = [
            {"model_id": "xgb_v1_latest", "type": "xgboost", "accuracy": 0.0, "status": "active", "created_at": ts},
            {"model_id": "lstm_v1_latest", "type": "lstm", "accuracy": 0.0, "status": "active", "created_at": ts},
            {"model_id": "ensemble_v1_latest", "type": "ensemble", "xgb_model_id": "xgb_v1_latest", "lstm_model_id": "lstm_v1_latest", "weights": {"xgboost": 0.65, "lstm": 0.35}, "accuracy": 0.0, "status": "active", "created_at": ts},
        ]
        _registry["default_model_id"] = "ensemble_v1_latest"
        _save_registry()


class TrainRequest(BaseModel):
    training_data_path: str = Field(description="Path to calls or partial_sequences JSON")
    model_type: str = Field(default="ensemble", description="xgboost | lstm | ensemble")


class PredictEvent(BaseModel):
    ts: int | float
    type: str
    duration_ms: int = 0
    words: int = 0
    tool: str | None = None


class PredictRequest(BaseModel):
    call_id: str
    events_so_far: list[dict]
    metadata: dict | None = Field(default_factory=dict)


@app.post("/api/model/train", response_model=dict)
def api_train(req: TrainRequest):
    """Train XGBoost, LSTM, or ensemble. Saves to ./models/ and keeps last 3 versions per type."""
    _ensure_models_dir()
    _load_registry()
    t0 = time.time()
    model_type = (req.model_type or "ensemble").lower()
    if model_type not in ("xgboost", "lstm", "ensemble"):
        raise HTTPException(status_code=400, detail="model_type must be xgboost, lstm, or ensemble")
    data_path = BASE_DIR / req.training_data_path if not os.path.isabs(req.training_data_path) else Path(req.training_data_path)
    if not data_path.exists():
        raise HTTPException(status_code=400, detail=f"training_data_path not found: {req.training_data_path}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_id = f"{model_type}_v1_{ts}"
    metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    try:
        if model_type == "xgboost":
            out = subprocess.run(
                [sys.executable, str(BASE_DIR / "train_xgb.py")],
                cwd=str(BASE_DIR), capture_output=True, text=True, timeout=300
            )
            if out.returncode != 0:
                raise HTTPException(status_code=500, detail=f"XGBoost training failed: {out.stderr or out.stdout}")
            # Copy xgb_model.json to models/
            import shutil
            shutil.copy(BASE_DIR / "xgb_model.json", MODELS_DIR / f"{model_id}.json")
            # Build and save encoders from call_features.json
            with open(BASE_DIR / "call_features.json") as f:
                rows = json.load(f)
            from ensemble_pipeline import build_xgb_features
            from ensemble_pipeline import load_partial_sequences
            seqs = [{"features_so_far": r.get("features", {})} for r in rows]
            _, encoders = build_xgb_features(seqs, encoders=None)
            with open(MODELS_DIR / f"{model_id}_encoders.json", "w") as f:
                json.dump(encoders, f)
            res_path = BASE_DIR / "xgb_training_results.json"
            if res_path.exists():
                with open(res_path) as f:
                    res = json.load(f)
                tr = res.get("training_results", {})
                metrics = {"accuracy": tr.get("test_accuracy", 0), "precision": tr.get("test_precision_macro", 0), "recall": tr.get("test_recall_macro", 0), "f1": tr.get("test_f1_macro", 0)}
        elif model_type == "lstm":
            out = subprocess.run(
                [sys.executable, str(BASE_DIR / "train_lstm.py")],
                cwd=str(BASE_DIR), capture_output=True, text=True, timeout=120
            )
            if out.returncode != 0:
                raise HTTPException(status_code=500, detail=f"LSTM training failed: {out.stderr or out.stdout}")
            import shutil
            shutil.copy(BASE_DIR / "lstm_model.keras", MODELS_DIR / f"{model_id}.keras")
            res_path = BASE_DIR / "lstm_training_results.json"
            if res_path.exists():
                with open(res_path) as f:
                    res = json.load(f)
                tr = res.get("training_results", {})
                metrics = {"accuracy": tr.get("test_accuracy", 0), "precision": tr.get("test_precision_macro", 0), "recall": tr.get("test_recall_macro", 0), "f1": tr.get("test_f1_macro", 0)}
        else:
            out = subprocess.run(
                [sys.executable, str(BASE_DIR / "ensemble_pipeline.py")],
                cwd=str(BASE_DIR), capture_output=True, text=True, timeout=60
            )
            if out.returncode != 0:
                raise HTTPException(status_code=500, detail=f"Ensemble pipeline failed: {out.stderr or out.stdout}")
            with open(BASE_DIR / "ensemble_results.json") as f:
                res = json.load(f)
            perf = res.get("ensemble_test_performance", {})
            metrics = {"accuracy": perf.get("test_accuracy", 0), "precision": perf.get("test_precision_macro", 0), "recall": perf.get("test_recall_macro", 0), "f1": perf.get("test_f1_macro", 0)}
            # Register ensemble pointing to latest xgb and lstm
            xgb_entries = [m for m in _registry["models"] if m.get("type") == "xgboost"]
            lstm_entries = [m for m in _registry["models"] if m.get("type") == "lstm"]
            xgb_id = xgb_entries[0]["model_id"] if xgb_entries else None
            lstm_id = lstm_entries[0]["model_id"] if lstm_entries else None
            if not xgb_id or not lstm_id:
                xgb_id = xgb_id or "xgb_v1_latest"
                lstm_id = lstm_id or "lstm_v1_latest"
            with open(BASE_DIR / "ensemble_strategy.json") as f:
                strat = json.load(f)
            w = strat.get("ensemble_config", {}).get("weights", {"xgboost": 0.65, "lstm": 0.35})
            _registry["models"].append({
                "model_id": model_id,
                "type": "ensemble",
                "xgb_model_id": xgb_id,
                "lstm_model_id": lstm_id,
                "weights": w,
                "accuracy": metrics["accuracy"],
                "status": "active",
                "created_at": ts,
            })
            _save_registry()
            training_time_seconds = round(time.time() - t0, 2)
            return {
                "model_id": model_id,
                "status": "trained",
                "metrics": metrics,
                "training_time_seconds": training_time_seconds,
                "model_uri": f"/api/model/{model_id}",
            }
        meta = {"model_id": model_id, "type": model_type, "accuracy": metrics["accuracy"], "precision": metrics["precision"], "recall": metrics["recall"], "f1": metrics["f1"], "status": "active", "created_at": ts}
        _registry["models"].append(meta)
        _registry["default_model_id"] = model_id
        _prune_versions()
        _save_registry()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}") from e

    training_time_seconds = round(time.time() - t0, 2)
    return {
        "model_id": model_id,
        "status": "trained",
        "metrics": metrics,
        "training_time_seconds": training_time_seconds,
        "model_uri": f"/api/model/{model_id}",
    }


@app.post("/api/predict", response_model=dict)
def api_predict(req: PredictRequest, model_id: str | None = None):
    """Predict outcome from events_so_far + metadata. Uses default or specified model_id."""
    if len(req.events_so_far) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 events")
    try:
        features = _events_to_features(req.call_id, req.events_so_far, req.metadata or {})
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    _load_registry()
    mid = model_id or _registry.get("default_model_id")
    if not mid:
        mid = next((m["model_id"] for m in _registry["models"] if m.get("type") == "ensemble"), None)
    if not mid:
        mid = next((m["model_id"] for m in _registry["models"]), None)
    if not mid:
        raise HTTPException(status_code=503, detail="Model not ready. Train a model first.")
    if mid not in _model_cache:
        if _load_model_into_cache(mid) is None:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id or 'default'}")
    entry = _model_cache[mid]
    mtype = entry.get("type", "xgboost")

    try:
        if mtype == "xgboost":
            proba = _predict_xgb(features, entry["encoders"], entry["booster"])
        elif mtype == "lstm":
            proba = _predict_lstm(req.events_so_far, entry["lstm_model"])
        else:
            proba = _predict_ensemble(features, req.events_so_far, entry)
        if proba is None:
            raise HTTPException(status_code=500, detail="Model inference failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}") from e

    pred_idx = int(max(range(len(proba)), key=lambda i: proba[i]))
    predicted_outcome = OUTCOME_CLASSES[pred_idx]
    confidence = round(float(proba[pred_idx]), 4)
    risk_score = round(1.0 - confidence, 4)
    completion_percent = _get_completion_percent(req.events_so_far)

    top_factors = []
    if mtype in ("xgboost", "ensemble"):
        xgb_entry = entry if mtype == "xgboost" else _model_cache.get(entry.get("xgb_model_id"))
        if xgb_entry and "booster" in xgb_entry:
            try:
                top_factors = [
                    {"feature": f, "value": v, "impact": round(imp, 4)}
                    for f, v, imp in _xgb_pred_contrib(features, xgb_entry["encoders"], xgb_entry["booster"])
                ]
            except Exception:
                pass
    if not top_factors:
        for f in FEATURE_ORDER[:5]:
            top_factors.append({"feature": f, "value": features.get(f, 0), "impact": 0.0})

    alternative_predictions = [
        {"outcome": OUTCOME_CLASSES[i], "probability": round(float(proba[i]), 4)}
        for i in range(len(OUTCOME_CLASSES)) if i != pred_idx
    ]

    return {
        "call_id": req.call_id,
        "predicted_outcome": predicted_outcome,
        "confidence": confidence,
        "risk_score": risk_score,
        "completion_percent": completion_percent,
        "model_used": mtype,
        "top_factors": top_factors[:5],
        "alternative_predictions": alternative_predictions,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


@app.get("/api/model/{model_id}/importance", response_model=dict)
def api_model_importance(model_id: str):
    """Return global and class-specific feature importance for the model."""
    _load_registry()
    entry = next((m for m in _registry["models"] if m.get("model_id") == model_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Model not found")
    mtype = entry.get("type", "xgboost")
    if mtype == "lstm":
        return {"model_id": model_id, "importance_type": "none", "global_importance": [], "class_specific_importance": {}}
    load_id = model_id
    if mtype == "ensemble":
        load_id = entry.get("xgb_model_id") or model_id
    path = MODELS_DIR / f"{load_id}.json"
    if not path.exists():
        path = BASE_DIR / "xgb_model.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model file not found")
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(str(path))
    imp = booster.get_score(importance_type="gain")
    importance_by_idx = {}
    for k, v in imp.items():
        if k.startswith("f"):
            importance_by_idx[int(k[1:])] = float(v)
    global_importance = []
    for i, name in enumerate(FEATURE_ORDER):
        if i in importance_by_idx:
            global_importance.append({"rank": len(global_importance) + 1, "feature": name, "importance": round(importance_by_idx[i], 4)})
    global_importance.sort(key=lambda x: -x["importance"])
    for r, item in enumerate(global_importance):
        item["rank"] = r + 1
    class_specific = {}
    shap_path = BASE_DIR / "shap_analysis.json"
    if shap_path.exists():
        with open(shap_path) as f:
            shap = json.load(f)
        class_specific = shap.get("class_specific_importance", {})
    return {"model_id": model_id, "importance_type": "gain", "global_importance": global_importance[:20], "class_specific_importance": class_specific}


@app.get("/api/models", response_model=dict)
def api_models():
    """List all registered models with id, type, accuracy, status."""
    _load_registry()
    models = []
    for m in _registry["models"]:
        models.append({
            "model_id": m.get("model_id"),
            "type": m.get("type", "xgboost"),
            "accuracy": m.get("accuracy", 0),
            "status": m.get("status", "active"),
        })
    return {"models": models}


@app.post("/api/pipeline/run", response_model=dict)
def api_pipeline_run():
    """Run full pipeline: generate → validate → features → validation report → XGB → SHAP → partials → LSTM → ensemble."""
    try:
        from pipeline import run_pipeline
        result = run_pipeline(timeout_per_step=600)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}") from e


@app.get("/api/monitoring/report", response_model=dict)
def api_monitoring_report(date: str | None = None):
    """Generate daily monitoring report. Optional query: date=YYYY-MM-DD (default: latest in logs)."""
    try:
        from monitoring import run_monitoring, _serialize
        log_path = BASE_DIR / "production_logs.json"
        report_path = BASE_DIR / "feature_validation_report.json"
        report_text, metrics = run_monitoring(log_path, report_path, date)
        return {"report_text": report_text, "metrics": _serialize(metrics)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
