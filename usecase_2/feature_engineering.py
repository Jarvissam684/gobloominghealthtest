#!/usr/bin/env python3
"""
Feature engineering module for call outcome prediction.
Computes 28 features from validated call records (timing, speech dynamics, engagement, progress, context).
"""

import json
import math
from pathlib import Path
from typing import Any

MISSING_TIMING = 999.0  # sentinel when event never occurs


def _linear_slope(x_list: list[float], y_list: list[float]) -> float:
    """Linear regression slope: (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x²) - sum(x)²). Returns 0 if undefined."""
    n = len(x_list)
    if n < 2:
        return 0.0
    sx = sum(x_list)
    sy = sum(y_list)
    sxy = sum(x * y for x, y in zip(x_list, y_list))
    sxx = sum(x * x for x in x_list)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return 0.0
    return (n * sxy - sx * sy) / denom


def _std(values: list[float]) -> float:
    """Standard deviation; 0 if len < 2."""
    if len(values) < 2:
        return 0.0
    n = len(values)
    m = sum(values) / n
    var = sum((x - m) ** 2 for x in values) / n
    return math.sqrt(max(0, var))


def _p75(values: list[float]) -> float:
    """75th percentile; 0 if < 4 values."""
    if len(values) < 4:
        return 0.0
    s = sorted(values)
    idx = int(0.75 * (len(s) - 1))
    return s[idx]


def compute_features(call: dict) -> dict:
    """
    Compute 28 features for one call.
    Returns {"call_id", "outcome", "features"} with no null values; uses 0 or 999 where specified.
    """
    call_id = call.get("call_id", "")
    outcome = call.get("outcome", "")
    meta = call.get("metadata", {})
    events = call.get("events", [])
    survey_completion_rate = call.get("survey_completion_rate", 0.0)
    if not isinstance(survey_completion_rate, (int, float)):
        survey_completion_rate = 0.0
    survey_completion_rate = max(0.0, min(1.0, float(survey_completion_rate)))

    if not events:
        return _empty_features(call_id, outcome, meta, survey_completion_rate)

    first_ts = events[0].get("ts", 0)
    last_ts = events[-1].get("ts", 0)
    total_duration_sec = float(last_ts - first_ts) if last_ts >= first_ts else 0.0
    total_duration_ms = total_duration_sec * 1000.0
    if total_duration_ms <= 0:
        total_duration_ms = 1.0  # avoid div by zero

    # --- TIMING ---
    time_to_first_user_speech_sec = MISSING_TIMING
    time_to_first_tool_call_sec = MISSING_TIMING
    for ev in events:
        t = ev.get("type", "")
        ts = ev.get("ts", 0)
        if t == "user_speech" and time_to_first_user_speech_sec == MISSING_TIMING:
            time_to_first_user_speech_sec = float(ts)
        if t == "tool_call" and time_to_first_tool_call_sec == MISSING_TIMING:
            time_to_first_tool_call_sec = float(ts)

    # Response latencies: for each agent_speech, latency = agent_ts - preceding_user_speech.ts
    response_latencies_sec: list[float] = []
    last_user_speech_ts: float | None = None
    for ev in events:
        t = ev.get("type", "")
        ts = ev.get("ts", 0)
        if t == "user_speech":
            last_user_speech_ts = float(ts)
        elif t == "agent_speech" and last_user_speech_ts is not None:
            response_latencies_sec.append(float(ts) - last_user_speech_ts)

    avg_response_latency_sec = float(sum(response_latencies_sec) / len(response_latencies_sec)) if response_latencies_sec else 0.0
    agent_response_latency_p75 = _p75(response_latencies_sec) if response_latencies_sec else 0.0

    silence_durations_ms = [ev.get("duration_ms", 0) for ev in events if ev.get("type") == "silence"]
    avg_silence_duration_sec = (sum(silence_durations_ms) / len(silence_durations_ms)) / 1000.0 if silence_durations_ms else 0.0

    # --- SPEECH DYNAMICS ---
    agent_speech_ms = sum(ev.get("duration_ms", 0) for ev in events if ev.get("type") == "agent_speech")
    user_speech_ms = sum(ev.get("duration_ms", 0) for ev in events if ev.get("type") == "user_speech")
    silence_ms = sum(ev.get("duration_ms", 0) for ev in events if ev.get("type") == "silence")

    agent_talk_ratio = float(agent_speech_ms) / total_duration_ms
    user_talk_ratio = float(user_speech_ms) / total_duration_ms
    silence_ratio = float(silence_ms) / total_duration_ms
    silence_count = len(silence_durations_ms)

    # user_speech_trend: slope of (event_index among user_speech, duration_ms)
    user_speech_events = [(i, ev) for i, ev in enumerate(events) if ev.get("type") == "user_speech"]
    user_speech_indices = [float(x[0]) for x in user_speech_events]
    user_speech_durations = [float(x[1].get("duration_ms", 0)) for x in user_speech_events]
    user_speech_trend = _linear_slope(user_speech_indices, user_speech_durations) if len(user_speech_events) >= 2 else 0.0

    # speech_entropy = std of all speech event durations (agent + user)
    speech_durations = (
        [ev.get("duration_ms", 0) for ev in events if ev.get("type") in ("agent_speech", "user_speech")]
    )
    speech_entropy = _std([float(x) for x in speech_durations]) if len(speech_durations) >= 2 else 0.0

    agent_durations = [ev.get("duration_ms", 0) for ev in events if ev.get("type") == "agent_speech"]
    agent_flexibility = _std([float(x) for x in agent_durations]) if len(agent_durations) >= 2 else 0.0

    # user_engagement_slope = slope of (event_index, cumulative_user_words); 0 if duration < 10s
    cum_user_words = 0
    event_index_list: list[float] = []
    cum_user_words_list: list[float] = []
    for i, ev in enumerate(events):
        event_index_list.append(float(i))
        if ev.get("type") == "user_speech":
            cum_user_words += ev.get("words", 0)
        cum_user_words_list.append(float(cum_user_words))
    user_engagement_slope = (
        _linear_slope(event_index_list, cum_user_words_list) if total_duration_sec >= 10.0 and len(events) >= 2 else 0.0
    )

    # --- ENGAGEMENT ---
    turn_count = 0
    prev_speaker: str | None = None
    for ev in events:
        t = ev.get("type", "")
        if t not in ("agent_speech", "user_speech"):
            continue
        if prev_speaker is not None and t != prev_speaker:
            turn_count += 1
        prev_speaker = t

    user_speech_evs = [ev for ev in events if ev.get("type") == "user_speech"]
    agent_speech_evs = [ev for ev in events if ev.get("type") == "agent_speech"]
    total_user_words = sum(ev.get("words", 0) for ev in user_speech_evs)
    total_agent_words = sum(ev.get("words", 0) for ev in agent_speech_evs)
    words_per_turn_user = (total_user_words / len(user_speech_evs)) if user_speech_evs else 0.0
    words_per_turn_agent = (total_agent_words / len(agent_speech_evs)) if agent_speech_evs else 0.0

    # interruption_count: agent_speech starting <0.5s after user_speech end
    last_user_end_sec: float | None = None
    interruption_count = 0
    for ev in events:
        t = ev.get("type", "")
        ts = float(ev.get("ts", 0))
        dur_ms = ev.get("duration_ms", 0)
        if t == "user_speech":
            last_user_end_sec = ts + dur_ms / 1000.0
        elif t == "agent_speech" and last_user_end_sec is not None:
            if last_user_end_sec <= ts < last_user_end_sec + 0.5:
                interruption_count += 1

    cumulative_user_words = total_user_words

    # --- PROGRESS ---
    tools_called_count = sum(1 for ev in events if ev.get("type") == "tool_call")
    tools_per_minute = (tools_called_count / (total_duration_sec / 60.0)) if total_duration_sec >= 60.0 else 0.0

    # --- CONTEXT ---
    agent_id = meta.get("agent_id", "")
    org_id = meta.get("org_id", "")
    call_purpose = meta.get("call_purpose", "")
    time_of_day = meta.get("time_of_day", "")
    day_of_week = meta.get("day_of_week", "")
    if not isinstance(agent_id, str):
        agent_id = str(agent_id) if agent_id is not None else ""
    if not isinstance(org_id, str):
        org_id = str(org_id) if org_id is not None else ""
    if not isinstance(call_purpose, str):
        call_purpose = str(call_purpose) if call_purpose is not None else ""
    if not isinstance(time_of_day, str):
        time_of_day = str(time_of_day) if time_of_day is not None else ""
    if not isinstance(day_of_week, str):
        day_of_week = str(day_of_week) if day_of_week is not None else ""

    # Clamp ratios to [0,1] where specified
    agent_talk_ratio = max(0.0, min(1.0, agent_talk_ratio))
    user_talk_ratio = max(0.0, min(1.0, user_talk_ratio))
    silence_ratio = max(0.0, min(1.0, silence_ratio))

    features = {
        "total_duration_sec": round(total_duration_sec, 4),
        "time_to_first_user_speech_sec": time_to_first_user_speech_sec,
        "time_to_first_tool_call_sec": time_to_first_tool_call_sec,
        "avg_response_latency_sec": round(avg_response_latency_sec, 4),
        "agent_response_latency_p75": round(agent_response_latency_p75, 4),
        "avg_silence_duration_sec": round(avg_silence_duration_sec, 4),
        "agent_talk_ratio": round(agent_talk_ratio, 4),
        "user_talk_ratio": round(user_talk_ratio, 4),
        "silence_ratio": round(silence_ratio, 4),
        "silence_count": int(silence_count),
        "user_speech_trend": round(user_speech_trend, 4),
        "user_words_trend": round(user_engagement_slope, 4),
        "speech_entropy": round(speech_entropy, 4),
        "agent_flexibility": round(agent_flexibility, 4),
        "turn_count": int(turn_count),
        "words_per_turn_user": round(words_per_turn_user, 4),
        "words_per_turn_agent": round(words_per_turn_agent, 4),
        "user_engagement_slope": round(user_engagement_slope, 4),
        "interruption_count": int(interruption_count),
        "cumulative_user_words": int(cumulative_user_words),
        "tools_called_count": int(tools_called_count),
        "tools_per_minute": round(tools_per_minute, 4),
        "survey_completion_rate": round(survey_completion_rate, 4),
        "agent_id": agent_id,
        "org_id": org_id,
        "call_purpose": call_purpose,
        "time_of_day": time_of_day,
        "day_of_week": day_of_week,
    }
    return {"call_id": call_id, "outcome": outcome, "features": features}


def _empty_features(call_id: str, outcome: str, meta: dict, survey_completion_rate: float) -> dict:
    """Return feature dict for call with no events; all numeric features 0 or 999, context from meta."""
    return {
        "call_id": call_id,
        "outcome": outcome,
        "features": {
            "total_duration_sec": 0.0,
            "time_to_first_user_speech_sec": MISSING_TIMING,
            "time_to_first_tool_call_sec": MISSING_TIMING,
            "avg_response_latency_sec": 0.0,
            "agent_response_latency_p75": 0.0,
            "avg_silence_duration_sec": 0.0,
            "agent_talk_ratio": 0.0,
            "user_talk_ratio": 0.0,
            "silence_ratio": 0.0,
            "silence_count": 0,
            "user_speech_trend": 0.0,
            "user_words_trend": 0.0,
            "speech_entropy": 0.0,
            "agent_flexibility": 0.0,
            "turn_count": 0,
            "words_per_turn_user": 0.0,
            "words_per_turn_agent": 0.0,
            "user_engagement_slope": 0.0,
            "interruption_count": 0,
            "cumulative_user_words": 0,
            "tools_called_count": 0,
            "tools_per_minute": 0.0,
            "survey_completion_rate": round(max(0.0, min(1.0, survey_completion_rate)), 4),
            "agent_id": meta.get("agent_id") or "",
            "org_id": meta.get("org_id") or "",
            "call_purpose": meta.get("call_purpose") or "",
            "time_of_day": meta.get("time_of_day") or "",
            "day_of_week": meta.get("day_of_week") or "",
        },
    }


def impute_missing(feature_rows: list[dict]) -> list[dict]:
    """Replace any None with 0 (numeric) or '' (string). Leave 999 as-is for time_to_first_* (never occurred)."""
    if not feature_rows:
        return feature_rows
    str_keys = {"agent_id", "org_id", "call_purpose", "time_of_day", "day_of_week"}
    for r in feature_rows:
        for key, v in list(r["features"].items()):
            if v is None:
                r["features"][key] = "" if key in str_keys else 0.0
    return feature_rows


def main():
    data_path = Path(__file__).parent / "calls_500.json"
    out_path = Path(__file__).parent / "call_features.json"

    with open(data_path) as f:
        data = json.load(f)
    calls = data.get("calls", [])

    rows = [compute_features(c) for c in calls]
    rows = impute_missing(rows)

    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"Computed features for {len(rows)} calls. Output: {out_path}")
    sample = rows[0]
    print(f"Sample call_id={sample['call_id']}, outcome={sample['outcome']}, feature keys={len(sample['features'])}")


if __name__ == "__main__":
    main()
