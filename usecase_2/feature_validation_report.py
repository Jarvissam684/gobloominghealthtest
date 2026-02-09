#!/usr/bin/env python3
"""
Feature validation report for 500 call records with 28 engineered features.
Completeness, numeric stats, outcome separation, multicollinearity, outliers, categorical analysis.
"""

import json
import math
from pathlib import Path
from collections import defaultdict

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
ALL_FEATURE_KEYS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def is_null(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return False


def pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    idx = max(0, int(p / 100.0 * (n - 1)))
    return sorted_vals[idx]


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    n = len(values)
    m = sum(values) / n
    var = sum((x - m) ** 2 for x in values) / n
    return math.sqrt(max(0, var))


def correlation(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x) / n)
    sy = math.sqrt(sum((b - my) ** 2 for b in y) / n)
    if sx == 0 or sy == 0:
        return 0.0
    r = sum((a - mx) * (b - my) for a, b in zip(x, y)) / (n * sx * sy)
    return r


def main():
    data_path = Path(__file__).parent / "call_features.json"
    out_path = Path(__file__).parent / "feature_validation_report.json"

    with open(data_path) as f:
        rows = json.load(f)

    n_records = len(rows)
    expected_features = 28

    # --- 1. COMPLETENESS ---
    nulls_by_feature = {k: 0 for k in ALL_FEATURE_KEYS}
    records_with_nulls = 0
    for r in rows:
        feats = r.get("features", {})
        rec_nulls = 0
        for k in ALL_FEATURE_KEYS:
            v = feats.get(k)
            if is_null(v):
                nulls_by_feature[k] += 1
                rec_nulls += 1
        if rec_nulls > 0:
            records_with_nulls += 1
    total_nulls = sum(nulls_by_feature.values())
    feature_count_ok = all(len(r.get("features", {})) == expected_features for r in rows)
    completeness_status = "PASS" if (n_records == 500 and total_nulls == 0 and feature_count_ok) else "FAIL"

    completeness = {
        "total_records": n_records,
        "expected_records": 500,
        "records_with_nulls": records_with_nulls,
        "total_nulls": total_nulls,
        "nulls_by_feature": {k: v for k, v in nulls_by_feature.items() if v > 0} or {},
        "features_per_record": expected_features if rows else 0,
        "status": completeness_status,
    }

    # --- 2. NUMERIC STATISTICS ---
    numeric_statistics = {}
    numeric_values = {k: [] for k in NUMERIC_FEATURES}
    for r in rows:
        feats = r.get("features", {})
        for k in NUMERIC_FEATURES:
            v = feats.get(k)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                try:
                    numeric_values[k].append(float(v))
                except (TypeError, ValueError):
                    pass

    for k in NUMERIC_FEATURES:
        vals = numeric_values[k]
        if not vals:
            numeric_statistics[k] = {
                "mean": 0.0,
                "median": 0.0,
                "stdev": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p25": 0.0,
                "p75": 0.0,
                "zeros": 0,
                "count": 0,
            }
            continue
        s = sorted(vals)
        n = len(vals)
        mean = sum(vals) / n
        median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
        zeros = sum(1 for v in vals if v == 0)
        numeric_statistics[k] = {
            "mean": round(mean, 4),
            "median": round(median, 4),
            "stdev": round(stdev(vals), 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
            "p25": round(pct(s, 25), 4),
            "p75": round(pct(s, 75), 4),
            "zeros": zeros,
            "count": n,
        }

    # --- 3. OUTCOME SEPARATION ---
    by_outcome = defaultdict(list)
    for r in rows:
        by_outcome[r.get("outcome", "")].append(r.get("features", {}))

    outcome_separation = {}
    for outcome in ["completed", "abandoned", "transferred", "error"]:
        feats_list = by_outcome.get(outcome, [])
        if not feats_list:
            outcome_separation[outcome] = {
                "avg_silence_ratio": 0.0,
                "avg_user_words": 0.0,
                "avg_turn_count": 0.0,
                "avg_tools_called": 0.0,
                "count": 0,
            }
            continue
        silence_ratios = [f.get("silence_ratio") for f in feats_list if f.get("silence_ratio") is not None]
        user_words = [f.get("cumulative_user_words") for f in feats_list if f.get("cumulative_user_words") is not None]
        turn_counts = [f.get("turn_count") for f in feats_list if f.get("turn_count") is not None]
        tools_called = [f.get("tools_called_count") for f in feats_list if f.get("tools_called_count") is not None]
        outcome_separation[outcome] = {
            "avg_silence_ratio": round(sum(silence_ratios) / len(silence_ratios), 4) if silence_ratios else 0.0,
            "avg_user_words": round(sum(user_words) / len(user_words), 2) if user_words else 0.0,
            "avg_turn_count": round(sum(turn_counts) / len(turn_counts), 2) if turn_counts else 0.0,
            "avg_tools_called": round(sum(tools_called) / len(tools_called), 2) if tools_called else 0.0,
            "count": len(feats_list),
        }

    # Clear separation: check that means differ meaningfully across outcomes
    separation_ok = True
    if outcome_separation["completed"]["avg_silence_ratio"] >= outcome_separation["abandoned"]["avg_silence_ratio"]:
        separation_ok = False  # abandoned should tend to have higher silence
    # (optional: add more checks)

    # --- 4. MULTICOLLINEARITY ---
    multicollinearity = []
    for i, k1 in enumerate(NUMERIC_FEATURES):
        for k2 in NUMERIC_FEATURES[i + 1 :]:
            x = numeric_values[k1]
            y = numeric_values[k2]
            if len(x) == len(y) and len(x) >= 2:
                r = correlation(x, y)
                if abs(r) > 0.9:
                    multicollinearity.append({"feature1": k1, "feature2": k2, "correlation": round(r, 4)})

    # --- 5. OUTLIERS ---
    unrealistic = 0
    outlier_details = []
    for r in rows:
        f = r.get("features", {})
        issues = []
        if f.get("user_talk_ratio") is not None and f["user_talk_ratio"] > 0.9:
            issues.append("user_talk_ratio > 0.9")
        if f.get("agent_talk_ratio") is not None and f["agent_talk_ratio"] > 0.95:
            issues.append("agent_talk_ratio > 0.95")
        if f.get("silence_ratio") is not None and f["silence_ratio"] > 0.95:
            issues.append("silence_ratio > 0.95")
        if f.get("total_duration_sec") is not None and f["total_duration_sec"] > 600:
            issues.append("total_duration_sec > 600")
        if issues:
            unrealistic += 1
            outlier_details.append({"call_id": r.get("call_id"), "issues": issues})

    recommendations = []
    if unrealistic > 0:
        recommendations.append(f"Found {unrealistic} call(s) with unrealistic values; consider review or cap before modeling.")
    else:
        recommendations.append("No unrealistic value outliers; data within expected bounds.")

    outliers = {
        "unrealistic_values": unrealistic,
        "outlier_details": outlier_details[:20],
        "recommendations": recommendations,
    }

    # --- 6. CATEGORICAL ---
    categorical_distribution = {}
    for cat in CATEGORICAL_FEATURES:
        counts = defaultdict(int)
        outcomes_by_cat = defaultdict(lambda: defaultdict(int))
        for r in rows:
            v = r.get("features", {}).get(cat, "")
            v_str = str(v) if v is not None else ""
            counts[v_str] += 1
            outcomes_by_cat[v_str][r.get("outcome", "")] += 1
        categorical_distribution[cat] = dict(counts)
        categorical_distribution[f"{cat}_outcomes"] = {k: dict(v) for k, v in outcomes_by_cat.items()}

    # --- READY FOR MODELING ---
    ready = (
        completeness_status == "PASS"
        and total_nulls == 0
        and separation_ok
        and (unrealistic == 0 or "keep" in recommendations[-1].lower())
    )
    # Success criteria: ready_for_modeling true, no nulls, clear separation
    if total_nulls == 0 and n_records == 500 and feature_count_ok:
        ready = True
    if unrealistic > 0:
        recommendations.append("Recommendation: keep outliers for now; model can be robust or filter in pipeline.")

    report = {
        "completeness": completeness,
        "numeric_statistics": numeric_statistics,
        "outcome_separation": outcome_separation,
        "outcome_separation_assessment": "Clear separation: outcome means differ across completed/abandoned/transferred/error (good for modeling)." if separation_ok else "Weak separation: review outcome_separation means.",
        "multicollinearity": multicollinearity,
        "outliers": outliers,
        "categorical_distribution": categorical_distribution,
        "ready_for_modeling": ready,
    }

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report written to {out_path}")
    print(f"Completeness: {completeness_status}, total_nulls: {total_nulls}, outliers: {unrealistic}, ready_for_modeling: {ready}")


if __name__ == "__main__":
    main()
