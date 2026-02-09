#!/usr/bin/env python3
"""
Production monitoring for Call Outcome Prediction.
Loads production call logs + predictions, computes 6 metrics, checks retraining triggers,
generates daily report, and suggests calls to add to training set.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Baseline from training (60% completed, 25% abandoned, 10% transferred, 5% error)
BASELINE_ACCURACY = 0.84
BASELINE_OUTCOME_PCT = {"completed": 60.0, "abandoned": 25.0, "transferred": 10.0, "error": 5.0}
OUTCOME_CLASSES = ["completed", "abandoned", "transferred", "error"]
DRIFT_STDEV_THRESHOLD = 2.0
CLASS_DRIFT_PCT_THRESHOLD = 10.0
CLASS_DRIFT_RETRAIN_THRESHOLD = 15.0
ACCURACY_DROP_FROM_BASELINE = 0.05
ACCURACY_RETRAIN_THRESHOLD = 0.79
LATENCY_P95_ALERT_MS = 100
LATENCY_RETRAIN_PCT_ABOVE_150MS = 5.0
FEATURE_DRIFT_COUNT_RETRAIN = 3
MODEL_NAMES = ["xgb_v1", "lstm_v1", "ensemble_v1"]


@dataclass
class ProductionRecord:
    call_id: str
    timestamp: str
    date: str
    predicted_outcome: str
    confidence: float
    actual_outcome: str | None
    latency_ms: float
    model_used: str
    features: dict | None = None
    metadata: dict | None = None


def _parse_date(ts: str) -> str:
    """Return YYYY-MM-DD from ISO timestamp."""
    try:
        if "T" in ts:
            return ts.split("T")[0]
        return ts[:10]
    except Exception:
        return ""


def _compute_features_from_events(call_id: str, events: list, metadata: dict) -> dict:
    """Compute 28 features from events + metadata (partial sequence)."""
    from feature_engineering import compute_features
    events = sorted(events, key=lambda e: (e.get("ts", 0), e.get("type", "")))
    call = {
        "call_id": call_id,
        "metadata": metadata or {},
        "events": events,
        "outcome": "",
        "survey_completion_rate": 0.0,
    }
    out = compute_features(call)
    return out.get("features", {})


def load_production_logs(path: str | Path) -> list[ProductionRecord]:
    """Load production log file (JSON array or JSONL). Returns list of ProductionRecord."""
    path = Path(path)
    records = []
    if not path.exists():
        return records
    with open(path) as f:
        raw = f.read().strip()
    if raw.startswith("["):
        rows = json.loads(raw)
    else:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    for r in rows:
        ts = r.get("timestamp", r.get("date", ""))
        date = _parse_date(ts) if isinstance(ts, str) else ""
        if not date and isinstance(ts, (int, float)):
            date = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        features = r.get("features")
        if features is None and r.get("events_so_far") and r.get("metadata"):
            try:
                features = _compute_features_from_events(
                    r.get("call_id", ""), r["events_so_far"], r["metadata"]
                )
            except Exception:
                features = {}
        records.append(
            ProductionRecord(
                call_id=r.get("call_id", ""),
                timestamp=ts if isinstance(ts, str) else str(ts),
                date=date,
                predicted_outcome=r.get("predicted_outcome", ""),
                confidence=float(r.get("confidence", 0)),
                actual_outcome=r.get("actual_outcome"),
                latency_ms=float(r.get("latency_ms", 0)),
                model_used=r.get("model_used", "ensemble"),
                features=features,
                metadata=r.get("metadata"),
            )
        )
    return records


def load_training_reference(report_path: str | Path) -> tuple[dict, set, set]:
    """
    Load feature_validation_report.json for training mean/stdev per feature.
    Returns (training_stats, known_agent_ids, known_org_ids).
    known_agent_ids/org_ids from categorical_distribution.
    """
    path = Path(report_path)
    training_stats = {}
    known_agents = set()
    known_orgs = set()
    if not path.exists():
        return training_stats, known_agents, known_orgs
    with open(path) as f:
        data = json.load(f)
    numeric = data.get("numeric_statistics", {})
    for name, s in numeric.items():
        training_stats[name] = {"mean": s["mean"], "stdev": max(s.get("stdev", 0), 1e-9)}
    cat = data.get("categorical_distribution", {})
    for k in cat.get("agent_id", {}):
        known_agents.add(k)
    for k in cat.get("org_id", {}):
        known_orgs.add(k)
    return training_stats, known_agents, known_orgs


def compute_daily_accuracy(records: list[ProductionRecord], date: str | None = None) -> dict:
    """
    For each day (or given date), compute accuracy on records with actual_outcome.
    Returns {"date": {"accuracy": float, "n": int, "by_model": {model: {"accuracy": float, "n": int}}}}.
    """
    with_gt = [r for r in records if r.actual_outcome]
    by_date = defaultdict(list)
    for r in with_gt:
        if not r.date:
            continue
        by_date[r.date].append(r)
    if date:
        days = [date] if date in by_date else []
    else:
        days = sorted(by_date.keys())
    out = {}
    for d in days:
        rows = by_date[d]
        correct = sum(1 for r in rows if r.predicted_outcome == r.actual_outcome)
        out[d] = {"accuracy": correct / len(rows) if rows else 0.0, "n": len(rows), "by_model": {}}
        by_model = defaultdict(list)
        for r in rows:
            by_model[r.model_used].append(r)
        for model, model_rows in by_model.items():
            c = sum(1 for r in model_rows if r.predicted_outcome == r.actual_outcome)
            out[d]["by_model"][model] = {"accuracy": c / len(model_rows) if model_rows else 0.0, "n": len(model_rows)}
    return out


def rolling_accuracy(records: list[ProductionRecord], window_days: int) -> float | None:
    """Accuracy over last window_days (only records with actual_outcome)."""
    with_gt = [r for r in records if r.actual_outcome]
    if not with_gt:
        return None
    dates = sorted({r.date for r in with_gt if r.date})
    if not dates:
        return None
    cutoff = dates[-1]
    try:
        cut_dt = datetime.strptime(cutoff, "%Y-%m-%d").date()
        start_dt = cut_dt - timedelta(days=window_days - 1)
        start_str = start_dt.strftime("%Y-%m-%d")
    except Exception:
        return None
    in_window = [r for r in with_gt if start_str <= r.date <= cutoff]
    if not in_window:
        return None
    correct = sum(1 for r in in_window if r.predicted_outcome == r.actual_outcome)
    return correct / len(in_window)


def confidence_distribution(records: list[ProductionRecord], date: str | None = None) -> dict:
    """Mean confidence, histogram bins (0-0.2, 0.2-0.4, ...). Optionally filter by date."""
    if date:
        records = [r for r in records if r.date == date]
    if not records:
        return {"mean": 0.0, "histogram": [], "n": 0}
    confs = [r.confidence for r in records]
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    hist = [0] * (len(bins) - 1)
    for c in confs:
        for i in range(len(bins) - 1):
            if bins[i] <= c < bins[i + 1]:
                hist[i] += 1
                break
        else:
            if c >= 1.0:
                hist[-1] += 1
    return {
        "mean": sum(confs) / len(confs),
        "histogram": [{"bin": f"{bins[i]:.1f}-{bins[i+1]:.1f}", "count": hist[i]} for i in range(len(hist))],
        "n": len(records),
    }


def feature_drift(records: list[ProductionRecord], training_stats: dict, date: str | None = None) -> dict:
    """
    For each numeric feature, production mean vs training mean.
    Alert if production mean is >2 stdev from training mean.
    Returns {feature: {"training_mean", "production_mean", "production_stdev", "drift_stdev", "alert"}}.
    """
    if date:
        records = [r for r in records if r.date == date and r.features]
    else:
        records = [r for r in records if r.features]
    if not records:
        return {}
    numeric_features = [k for k in list(records[0].features.keys()) if k in training_stats]
    out = {}
    for f in numeric_features:
        vals = [r.features.get(f) for r in records if r.features.get(f) is not None]
        if isinstance(vals[0], (int, float)):
            pass
        else:
            vals = [v for v in vals if isinstance(v, (int, float))]
        if not vals:
            out[f] = {"training_mean": training_stats[f]["mean"], "production_mean": None, "production_stdev": None, "drift_stdev": None, "alert": False}
            continue
        prod_mean = sum(vals) / len(vals)
        prod_var = sum((x - prod_mean) ** 2 for x in vals) / len(vals)
        prod_stdev = math.sqrt(max(0, prod_var))
        train_mean = training_stats[f]["mean"]
        train_stdev = training_stats[f]["stdev"]
        drift_stdev = (prod_mean - train_mean) / train_stdev if train_stdev else 0
        alert = abs(drift_stdev) > DRIFT_STDEV_THRESHOLD
        out[f] = {
            "training_mean": train_mean,
            "production_mean": prod_mean,
            "production_stdev": prod_stdev,
            "drift_stdev": drift_stdev,
            "alert": alert,
        }
    return out


def class_distribution_drift(records: list[ProductionRecord], date: str | None = None) -> dict:
    """
    Outcome distribution (actual_outcome when available; else predicted_outcome) vs baseline %.
    Alert if any class % changes >10% from baseline.
    """
    if date:
        records = [r for r in records if r.date == date]
    # Prefer actual_outcome for distribution of outcomes; fallback to predicted
    outcomes = []
    for r in records:
        o = r.actual_outcome if r.actual_outcome else r.predicted_outcome
        if o in OUTCOME_CLASSES:
            outcomes.append(o)
    if not outcomes:
        return {"distribution": {}, "alerts": [], "n": 0}
    n = len(outcomes)
    dist = {c: 100.0 * sum(1 for o in outcomes if o == c) / n for c in OUTCOME_CLASSES}
    alerts = []
    for c in OUTCOME_CLASSES:
        diff = dist.get(c, 0) - BASELINE_OUTCOME_PCT.get(c, 0)
        if abs(diff) > CLASS_DRIFT_PCT_THRESHOLD:
            alerts.append({"outcome": c, "production_pct": dist[c], "baseline_pct": BASELINE_OUTCOME_PCT[c], "diff": diff})
    return {"distribution": dist, "alerts": alerts, "n": n}


def latency_p95_by_model(records: list[ProductionRecord], date: str | None = None) -> dict:
    """P95 latency per model (ms)."""
    if date:
        records = [r for r in records if r.date == date]
    by_model = defaultdict(list)
    for r in records:
        if r.latency_ms is not None and r.latency_ms >= 0:
            by_model[r.model_used].append(r.latency_ms)
    out = {}
    for model, latencies in by_model.items():
        if not latencies:
            out[model] = None
            continue
        s = sorted(latencies)
        idx = min(int(0.95 * len(s)), len(s) - 1)
        out[model] = s[idx]
    return out


def pct_above_latency(records: list[ProductionRecord], threshold_ms: float, date: str | None = None) -> float:
    """Percentage of requests with latency > threshold_ms."""
    if date:
        records = [r for r in records if r.date == date]
    with_lat = [r for r in records if r.latency_ms is not None and r.latency_ms >= 0]
    if not with_lat:
        return 0.0
    above = sum(1 for r in with_lat if r.latency_ms > threshold_ms)
    return 100.0 * above / len(with_lat)


def check_retraining_triggers(
    records: list[ProductionRecord],
    training_stats: dict,
    known_agents: set,
    known_orgs: set,
    date: str | None = None,
) -> dict:
    """
    Returns {
        "retrain_needed": bool,
        "reasons": [],
        "accuracy_below": bool,
        "feature_drift_count": int,
        "class_drift_over_15": bool,
        "latency_bad": bool,
        "new_agent_or_org": bool,
    }
    """
    if date:
        recs = [r for r in records if r.date == date]
    else:
        recs = records
    reasons = []
    acc_7 = rolling_accuracy(records, 7)
    accuracy_below = acc_7 is not None and acc_7 < ACCURACY_RETRAIN_THRESHOLD
    if accuracy_below:
        reasons.append(f"Accuracy (7-day) {acc_7:.2%} below {ACCURACY_RETRAIN_THRESHOLD}")

    drift = feature_drift(recs, training_stats, date)
    drifted = sum(1 for v in drift.values() if v.get("alert"))
    feature_drift_count = drifted
    if drifted >= FEATURE_DRIFT_COUNT_RETRAIN:
        reasons.append(f"Feature drift in {drifted} features (>= {FEATURE_DRIFT_COUNT_RETRAIN})")

    class_d = class_distribution_drift(recs, date)
    class_drift_over_15 = any(
        abs(class_d["distribution"].get(c, 0) - BASELINE_OUTCOME_PCT.get(c, 0)) > CLASS_DRIFT_RETRAIN_THRESHOLD
        for c in OUTCOME_CLASSES
    )
    if class_drift_over_15:
        reasons.append("Class distribution shifted >15% from baseline")

    pct_150 = pct_above_latency(recs, 150.0, date)
    latency_bad = pct_150 > LATENCY_RETRAIN_PCT_ABOVE_150MS
    if latency_bad:
        reasons.append(f"Latency >150ms for {pct_150:.1f}% of requests (>{LATENCY_RETRAIN_PCT_ABOVE_150MS}%)")

    new_agent_or_org = False
    for r in recs:
        meta = r.metadata or {}
        a = meta.get("agent_id")
        o = meta.get("org_id")
        if a and a not in known_agents:
            new_agent_or_org = True
            break
        if o and o not in known_orgs:
            new_agent_or_org = True
            break
    if new_agent_or_org:
        reasons.append("New agent_id or org_id not seen in training (need sample data first)")

    retrain_needed = (
        accuracy_below
        or feature_drift_count >= FEATURE_DRIFT_COUNT_RETRAIN
        or class_drift_over_15
        or latency_bad
        or new_agent_or_org
    )
    return {
        "retrain_needed": retrain_needed,
        "reasons": reasons,
        "accuracy_below": accuracy_below,
        "feature_drift_count": feature_drift_count,
        "class_drift_over_15": class_drift_over_15,
        "latency_bad": latency_bad,
        "new_agent_or_org": new_agent_or_org,
    }


def suggest_calls_to_add(records: list[ProductionRecord], limit: int = 20) -> list[dict]:
    """
    Suggest calls to add to training: mispredictions (with reason), new agent/org, high-confidence wrong.
    """
    with_gt = [r for r in records if r.actual_outcome]
    mispreds = [r for r in with_gt if r.predicted_outcome != r.actual_outcome]
    suggestions = []
    for r in mispreds[:limit]:
        reason = "prediction mismatch"
        if r.features:
            low_user = r.features.get("user_talk_ratio", 0) is not None and (r.features.get("user_talk_ratio") or 0) < 0.25
            high_silence = r.features.get("silence_ratio", 0) is not None and (r.features.get("silence_ratio") or 0) > 0.4
            if low_user and high_silence and r.actual_outcome == "abandoned":
                reason = "low user_talk_ratio + high silence_ratio = abandonment signals; model missed (e.g. early in call)"
            elif r.actual_outcome == "completed" and r.predicted_outcome == "transferred":
                reason = "long agent speeches = transfer signal, but agent handling led to completion; need more 'long_agent_speech_but_still_completed' examples"
        suggestions.append({
            "call_id": r.call_id,
            "predicted": r.predicted_outcome,
            "actual": r.actual_outcome,
            "confidence": round(r.confidence, 2),
            "reason": reason,
        })
    return suggestions


def generate_daily_report(
    records: list[ProductionRecord],
    training_stats: dict,
    known_agents: set,
    known_orgs: set,
    report_date: str,
    yesterday_confidence_mean: float | None = None,
) -> str:
    """Generate text daily report."""
    lines = [
        "=== Daily Model Performance Report ===",
        f"Date: {report_date}",
        f"Models: {', '.join(MODEL_NAMES)}",
        "",
        "Accuracy (last 24h):",
        "",
    ]
    daily_acc = compute_daily_accuracy(records, report_date)
    day_data = daily_acc.get(report_date, {})
    by_model = day_data.get("by_model", {})
    for model in MODEL_NAMES:
        mdata = by_model.get(model, {})
        acc = mdata.get("accuracy", 0.0)
        n = mdata.get("n", 0)
        if model.lower().startswith("xgb"):
            base = BASELINE_ACCURACY
            diff = (acc - base) * 100
            arrow = "↓" if diff < 0 else "↑"
            lines.append(f"XGBoost: {acc:.2f} ({arrow} {diff:+.0f}% from baseline {base:.2f})")
        elif model.lower().startswith("lstm"):
            lines.append(f"LSTM: {acc:.2f}")
        else:
            base = BASELINE_ACCURACY
            diff = (acc - base) * 100
            arrow = "↓" if diff < 0 else "↑"
            lines.append(f"Ensemble: {acc:.2f} ({arrow} {diff:+.0f}% from baseline {base:.2f})")
    acc_7 = rolling_accuracy(records, 7)
    acc_30 = rolling_accuracy(records, 30)
    lines.append("")
    if acc_7 is not None:
        lines.append(f"Rolling 7-day accuracy: {acc_7:.2%}")
    if acc_30 is not None:
        lines.append(f"Rolling 30-day accuracy: {acc_30:.2%}")
    lines.append("")
    lines.append("Confidence Calibration:")
    lines.append("")
    conf_dist = confidence_distribution(records, report_date)
    mean_conf = conf_dist["mean"]
    lines.append(f"Mean predicted confidence: {mean_conf:.2f}")
    if yesterday_confidence_mean is not None:
        lines.append(f"  (↓ from {yesterday_confidence_mean:.2f} yesterday)" if mean_conf < yesterday_confidence_mean else f"  (↑ from {yesterday_confidence_mean:.2f} yesterday)")
    day_acc_val = day_data.get("accuracy", 0.0)
    gap = mean_conf - day_acc_val
    lines.append(f"Confidence vs Accuracy gap: {gap:.2f} (good)" if abs(gap) <= 0.05 else f"Confidence vs Accuracy gap: {gap:.2f} (miscalibration risk)")
    lines.append("")
    lines.append("Feature Drift Alerts:")
    lines.append("")
    drift = feature_drift(records, training_stats, report_date)
    # Show a few key features
    for f in ["silence_ratio", "user_talk_ratio"]:
        if f in drift and drift[f].get("production_mean") is not None:
            d = drift[f]
            tm = d["training_mean"]
            pm = d["production_mean"]
            diff = pm - tm
            arrow = "↑" if diff > 0 else "↓"
            status = "ALERT" if d.get("alert") else "NORMAL"
            lines.append(f"  {f}: mean {pm:.2f} ({arrow} {diff:+.2f} from training mean {tm:.2f}) - {status}")
    drifted_names = [k for k, v in drift.items() if v.get("alert")]
    if not drifted_names:
        lines.append("  No critical drifts detected")
    else:
        for f in drifted_names[:5]:
            lines.append(f"  {f}: ALERT (>{DRIFT_STDEV_THRESHOLD} stdev from training)")
    lines.append("")
    lines.append("Class Distribution:")
    lines.append("")
    class_d = class_distribution_drift(records, report_date)
    for c in OUTCOME_CLASSES:
        pct = class_d["distribution"].get(c, 0)
        base = BASELINE_OUTCOME_PCT.get(c, 0)
        diff = pct - base
        if abs(diff) > CLASS_DRIFT_PCT_THRESHOLD:
            status = "MINOR (within 10%)" if abs(diff) <= CLASS_DRIFT_PCT_THRESHOLD else "ALERT"
        else:
            status = "NORMAL"
        lines.append(f"  {c.capitalize()}: {pct:.0f}% (baseline {base:.0f}%) - {status}")
    lines.append("")
    lines.append("Latency (p95):")
    lines.append("")
    lat = latency_p95_by_model(records, report_date)
    all_good = True
    for model, ms in lat.items():
        if ms is not None:
            if ms >= LATENCY_P95_ALERT_MS:
                all_good = False
            lines.append(f"  {model}: {ms:.0f}ms")
    if not lat:
        lines.append("  (no latency data)")
    else:
        lines.append("  (all <100ms, GOOD)" if all_good else "  (some >=100ms, check model loading)")
    lines.append("")
    lines.append("Retraining Recommendation:")
    lines.append("")
    triggers = check_retraining_triggers(records, training_stats, known_agents, known_orgs, report_date)
    if triggers["retrain_needed"]:
        lines.append("  Status: RECOMMENDED")
        for reason in triggers["reasons"]:
            lines.append(f"  - {reason}")
    else:
        next_check = (datetime.strptime(report_date, "%Y-%m-%d") + timedelta(days=7)).strftime("%Y-%m-%d")
        lines.append("  Status: NOT NEEDED (all metrics healthy)")
        lines.append(f"  Next check: {next_check}")
    lines.append("")
    lines.append("Recent Mispredictions (analysis):")
    lines.append("")
    suggestions = suggest_calls_to_add(records, limit=5)
    if not suggestions:
        lines.append("  None in this period.")
    else:
        for s in suggestions:
            lines.append(f"  {s['call_id']}: predicted \"{s['predicted']}\" ({s['confidence']}), actual \"{s['actual']}\"")
            lines.append(f"    Reason: {s['reason']}")
    return "\n".join(lines)


def run_monitoring(
    production_log_path: str | Path,
    validation_report_path: str | Path,
    report_date: str | None = None,
) -> tuple[str, dict]:
    """
    Load logs and training reference, compute metrics, generate report.
    report_date: YYYY-MM-DD or None for latest date in logs.
    Returns (report_text, metrics_dict).
    """
    records = load_production_logs(production_log_path)
    training_stats, known_agents, known_orgs = load_training_reference(validation_report_path)
    if not records:
        return "No production records found.\n", {}
    if report_date is None:
        report_date = max(r.date for r in records if r.date) if records else datetime.now(timezone.utc).strftime("%Y-%m-%d")
    yesterday = (datetime.strptime(report_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    yesterday_conf = confidence_distribution(records, yesterday)
    yesterday_mean = yesterday_conf["mean"] if yesterday_conf["n"] else None
    report_text = generate_daily_report(
        records, training_stats, known_agents, known_orgs, report_date, yesterday_mean
    )
    metrics = {
        "daily_accuracy": compute_daily_accuracy(records, report_date),
        "rolling_7d_accuracy": rolling_accuracy(records, 7),
        "rolling_30d_accuracy": rolling_accuracy(records, 30),
        "confidence_distribution": confidence_distribution(records, report_date),
        "feature_drift": feature_drift(records, training_stats, report_date),
        "class_distribution": class_distribution_drift(records, report_date),
        "latency_p95_by_model": latency_p95_by_model(records, report_date),
        "retraining_triggers": check_retraining_triggers(records, training_stats, known_agents, known_orgs, report_date),
        "suggested_calls_to_add": suggest_calls_to_add(records, 20),
    }
    return report_text, metrics


def _serialize(obj):
    """JSON-serialize metrics (sets -> lists, etc.)."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(x) for x in obj]
    if isinstance(obj, set):
        return list(obj)
    return str(obj)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Daily model performance monitoring")
    p.add_argument("--date", default=None, help="Report date YYYY-MM-DD (default: latest in logs)")
    p.add_argument("--logs", default=None, help="Production log JSON path (default: production_logs.json)")
    p.add_argument("--report", default=None, help="Feature validation report path (default: feature_validation_report.json)")
    args = p.parse_args()
    base = Path(__file__).parent
    log_path = Path(args.logs) if args.logs else base / "production_logs.json"
    report_path = Path(args.report) if args.report else base / "feature_validation_report.json"
    report_text, metrics = run_monitoring(log_path, report_path, args.date)
    print(report_text)
    out_metrics = base / "monitoring_metrics.json"
    with open(out_metrics, "w") as f:
        json.dump(_serialize(metrics), f, indent=2)
    print(f"\nMetrics saved to {out_metrics}")
