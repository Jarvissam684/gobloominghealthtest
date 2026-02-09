#!/usr/bin/env python3
"""
Data quality validator for synthetic voice AI call records.
Performs schema, sequence, and outcome correlation validation.
"""

import json
import sys
from pathlib import Path

REQUIRED_CALL_KEYS = {"call_id", "metadata", "events", "outcome", "survey_completion_rate"}
REQUIRED_METADATA_KEYS = {"agent_id", "org_id", "call_purpose", "caller_phone_type", "time_of_day", "day_of_week"}
VALID_OUTCOMES = {"completed", "abandoned", "transferred", "error"}
VALID_EVENT_TYPES = {"call_start", "agent_speech", "user_speech", "silence", "tool_call", "call_end"}
REQUIRED_EVENT_KEYS = {"ts", "type", "duration_ms", "words", "tool"}


def compute_metrics(events: list[dict]) -> dict | None:
    """Compute derived metrics from events. Returns None if events invalid."""
    call_duration_sec = 0
    total_user_words = 0
    agent_speech_ms = 0
    user_speech_ms = 0
    silence_ms = 0
    tools_called = 0
    time_to_first_user_speech = None

    for ev in events:
        t = ev.get("type")
        dur = ev.get("duration_ms", 0)
        words = ev.get("words", 0)
        ts = ev.get("ts", 0)
        if t == "call_end":
            call_duration_sec = ts
        elif t == "agent_speech":
            agent_speech_ms += dur
        elif t == "user_speech":
            user_speech_ms += dur
            total_user_words += words
            if time_to_first_user_speech is None:
                time_to_first_user_speech = ts
        elif t == "silence":
            silence_ms += dur
        elif t == "tool_call":
            tools_called += 1

    total_speech_ms = agent_speech_ms + user_speech_ms
    total_ms = call_duration_sec * 1000 if call_duration_sec else 1
    silence_ratio = silence_ms / total_ms if total_ms else 0
    agent_speech_ratio = agent_speech_ms / total_speech_ms if total_speech_ms else 0

    return {
        "call_duration": call_duration_sec,
        "total_user_words": total_user_words,
        "silence_ratio": silence_ratio,
        "agent_speech_ratio": agent_speech_ratio,
        "tools_called": tools_called,
        "time_to_first_user_speech": time_to_first_user_speech,
        "has_user_speech": time_to_first_user_speech is not None,
    }


def check_schema(calls: list[dict]) -> tuple[bool, list[str], int]:
    """Validate schema. Returns (valid, list of issues, count of records with missing fields)."""
    issues = []
    records_missing_fields = 0

    for i, c in enumerate(calls):
        rec_issues = []
        if not isinstance(c, dict):
            rec_issues.append(f"call {i} is not a dict")
            records_missing_fields += 1
            continue
        missing = REQUIRED_CALL_KEYS - set(c.keys())
        if missing:
            rec_issues.append(f"missing keys: {missing}")
        if "metadata" in c:
            if not isinstance(c["metadata"], dict):
                rec_issues.append("metadata is not a dict")
            else:
                meta_missing = REQUIRED_METADATA_KEYS - set(c["metadata"].keys())
                if meta_missing:
                    rec_issues.append(f"metadata missing: {meta_missing}")
                for k in REQUIRED_METADATA_KEYS:
                    if k in c["metadata"] and not isinstance(c["metadata"][k], str):
                        rec_issues.append(f"metadata.{k} must be string")
        if "outcome" in c and c["outcome"] not in VALID_OUTCOMES:
            rec_issues.append(f"invalid outcome: {c.get('outcome')}")
        if "survey_completion_rate" in c:
            v = c["survey_completion_rate"]
            if not isinstance(v, (int, float)) or v < 0 or v > 1:
                rec_issues.append(f"survey_completion_rate must be float [0,1]: {v}")
        if "events" in c:
            if not isinstance(c["events"], list):
                rec_issues.append("events must be list")
            else:
                for j, ev in enumerate(c["events"]):
                    if not isinstance(ev, dict):
                        rec_issues.append(f"events[{j}] not dict")
                        continue
                    ev_missing = REQUIRED_EVENT_KEYS - set(ev.keys())
                    if ev_missing:
                        rec_issues.append(f"events[{j}] missing: {ev_missing}")
                    if "ts" in ev and not isinstance(ev["ts"], int):
                        rec_issues.append(f"events[{j}].ts must be int")
                    if "type" in ev and ev["type"] not in VALID_EVENT_TYPES:
                        rec_issues.append(f"events[{j}].type invalid: {ev['type']}")
                    if "duration_ms" in ev:
                        d = ev["duration_ms"]
                        if not isinstance(d, int) or d < 0:
                            rec_issues.append(f"events[{j}].duration_ms must be int >= 0")
                    if "words" in ev:
                        w = ev["words"]
                        if not isinstance(w, int) or w < 0:
                            rec_issues.append(f"events[{j}].words must be int >= 0")

        if rec_issues:
            records_missing_fields += 1
            issues.append(f"call_id={c.get('call_id', i)}: {'; '.join(rec_issues)}")

    return (len(issues) == 0, issues, records_missing_fields)


def check_sequence(calls: list[dict]) -> tuple[bool, list[str], int]:
    """Validate event sequence: strictly increasing ts, call_start first, call_end last."""
    issues = []
    violation_count = 0

    for c in calls:
        if not isinstance(c, dict) or "events" not in c or not isinstance(c["events"], list):
            continue
        events = c["events"]
        if len(events) < 2:
            if len(events) == 1 and events[0].get("type") not in ("call_start", "call_end"):
                issues.append(f"{c.get('call_id')}: single event must be call_start or call_end")
                violation_count += 1
            continue
        if events[0].get("type") != "call_start":
            issues.append(f"{c.get('call_id')}: first event must be call_start, got {events[0].get('type')}")
            violation_count += 1
        if events[-1].get("type") != "call_end":
            issues.append(f"{c.get('call_id')}: last event must be call_end, got {events[-1].get('type')}")
            violation_count += 1
        for i in range(len(events) - 1):
            ts_cur = events[i].get("ts")
            ts_next = events[i + 1].get("ts")
            if isinstance(ts_cur, int) and isinstance(ts_next, int) and ts_cur >= ts_next:
                issues.append(f"{c.get('call_id')}: timestamps not strictly increasing at index {i} (ts[{i}]={ts_cur} >= ts[{i+1}]={ts_next})")
                violation_count += 1
                break

    return (violation_count == 0, issues, violation_count)


def check_correlation(calls: list[dict]) -> tuple[bool, list[str], dict]:
    """Check outcome correlation rules and return per-outcome stats + violations."""
    # Correlation rules
    # Completed: tools_called >= 1 AND total_user_words > 100 AND silence_ratio < 0.4
    # Abandoned: silence_ratio > 0.5 OR time_to_first_user_speech > 15 OR total_user_words < 30
    # Transferred: agent_speech_ratio > 0.55
    # Error: call_duration < 30 OR no user_speech

    outcome_stats = {
        "completed": {"silence_ratio": [], "user_words": [], "tools_called": [], "count": 0},
        "abandoned": {"silence_ratio": [], "time_to_first_user_speech": [], "user_words": [], "count": 0, "pct_silence_gt_05": 0},
        "transferred": {"agent_speech_ratio": [], "call_duration": [], "count": 0},
        "error": {"call_duration": [], "count": 0, "pct_no_user_speech": 0},
    }
    violations = []

    for c in calls:
        if not isinstance(c, dict) or "events" not in c or "outcome" not in c:
            continue
        events = c["events"]
        outcome = c["outcome"]
        m = compute_metrics(events)
        if m is None:
            continue

        if outcome == "completed":
            outcome_stats["completed"]["silence_ratio"].append(m["silence_ratio"])
            outcome_stats["completed"]["user_words"].append(m["total_user_words"])
            outcome_stats["completed"]["tools_called"].append(m["tools_called"])
            outcome_stats["completed"]["count"] += 1
            if not (m["tools_called"] >= 1 and m["total_user_words"] > 100 and m["silence_ratio"] < 0.4):
                violations.append(f"{c.get('call_id')} (completed): tools_called={m['tools_called']}, user_words={m['total_user_words']}, silence_ratio={m['silence_ratio']:.3f}")

        elif outcome == "abandoned":
            outcome_stats["abandoned"]["silence_ratio"].append(m["silence_ratio"])
            outcome_stats["abandoned"]["user_words"].append(m["total_user_words"])
            outcome_stats["abandoned"]["time_to_first_user_speech"].append(m["time_to_first_user_speech"])
            outcome_stats["abandoned"]["count"] += 1
            if m["silence_ratio"] > 0.5:
                outcome_stats["abandoned"]["pct_silence_gt_05"] += 1
            ok = m["silence_ratio"] > 0.5 or (m["time_to_first_user_speech"] is not None and m["time_to_first_user_speech"] > 15) or m["total_user_words"] < 30
            if not ok:
                violations.append(f"{c.get('call_id')} (abandoned): silence_ratio={m['silence_ratio']:.3f}, first_user_ts={m['time_to_first_user_speech']}, user_words={m['total_user_words']}")

        elif outcome == "transferred":
            outcome_stats["transferred"]["agent_speech_ratio"].append(m["agent_speech_ratio"])
            outcome_stats["transferred"]["call_duration"].append(m["call_duration"])
            outcome_stats["transferred"]["count"] += 1
            if not (m["agent_speech_ratio"] > 0.55):
                violations.append(f"{c.get('call_id')} (transferred): agent_speech_ratio={m['agent_speech_ratio']:.3f}")

        elif outcome == "error":
            outcome_stats["error"]["call_duration"].append(m["call_duration"])
            outcome_stats["error"]["count"] += 1
            if not m["has_user_speech"]:
                outcome_stats["error"]["pct_no_user_speech"] += 1
            ok = m["call_duration"] < 30 or not m["has_user_speech"]
            if not ok:
                violations.append(f"{c.get('call_id')} (error): duration={m['call_duration']}, has_user_speech={m['has_user_speech']}")

    # Build outcome_statistics output
    def avg(x):
        return round(sum(x) / len(x), 4) if x else None

    outcome_statistics = {}
    if outcome_stats["completed"]["count"]:
        outcome_statistics["completed"] = {
            "avg_silence_ratio": avg(outcome_stats["completed"]["silence_ratio"]),
            "avg_user_words": round(avg(outcome_stats["completed"]["user_words"]) or 0, 2),
            "avg_tools_called": round(avg(outcome_stats["completed"]["tools_called"]) or 0, 2),
            "count": outcome_stats["completed"]["count"],
        }
    else:
        outcome_statistics["completed"] = {"count": 0}

    if outcome_stats["abandoned"]["count"]:
        n = outcome_stats["abandoned"]["count"]
        tfus = [x for x in outcome_stats["abandoned"]["time_to_first_user_speech"] if x is not None]
        outcome_statistics["abandoned"] = {
            "avg_silence_ratio": avg(outcome_stats["abandoned"]["silence_ratio"]),
            "avg_user_words": round(avg(outcome_stats["abandoned"]["user_words"]) or 0, 2),
            "avg_time_to_first_user_speech": round(avg(tfus), 2) if tfus else None,
            "pct_silence_ratio_gt_05": round(100 * outcome_stats["abandoned"]["pct_silence_gt_05"] / n, 2),
            "count": n,
        }
    else:
        outcome_statistics["abandoned"] = {"count": 0}

    if outcome_stats["transferred"]["count"]:
        outcome_statistics["transferred"] = {
            "avg_agent_speech_ratio": avg(outcome_stats["transferred"]["agent_speech_ratio"]),
            "avg_call_duration": round(avg(outcome_stats["transferred"]["call_duration"]) or 0, 2),
            "count": outcome_stats["transferred"]["count"],
        }
    else:
        outcome_statistics["transferred"] = {"count": 0}

    if outcome_stats["error"]["count"]:
        n = outcome_stats["error"]["count"]
        outcome_statistics["error"] = {
            "avg_call_duration": round(avg(outcome_stats["error"]["call_duration"]) or 0, 2),
            "pct_no_user_speech": round(100 * outcome_stats["error"]["pct_no_user_speech"] / n, 2),
            "count": n,
        }
    else:
        outcome_statistics["error"] = {"count": 0}

    return (len(violations) == 0, violations, outcome_statistics)


def check_suspicious(calls: list[dict]) -> tuple[list[str], list[str]]:
    """Check for suspicious patterns (e.g. all same duration)."""
    warnings = []
    critical = []

    if not calls:
        return (["No calls to analyze"], [])

    durations = []
    event_counts = []
    for c in calls:
        if isinstance(c, dict) and "events" in c and c["events"]:
            last = c["events"][-1]
            if last.get("type") == "call_end":
                durations.append(last.get("ts", 0))
            event_counts.append(len(c["events"]))

    if len(set(durations)) == 1 and len(durations) > 1:
        warnings.append(f"All {len(durations)} calls have identical call_duration={durations[0]}s")
    if len(set(event_counts)) == 1 and len(event_counts) > 1:
        warnings.append(f"All calls have identical event count={event_counts[0]}")

    return (critical, warnings)


def feature_feasibility(calls: list[dict], schema_ok: bool, sequence_ok: bool) -> dict:
    """Determine if timing, speech dynamics, and engagement features can be computed."""
    issues = []
    if not schema_ok:
        issues.append("Schema validation failed; required fields missing or wrong types")
    if not sequence_ok:
        issues.append("Sequence validation failed; timestamps or call_start/call_end order invalid")

    can_timing = schema_ok and sequence_ok
    can_speech = schema_ok and sequence_ok  # we have ts, type, duration_ms, words
    can_engagement = schema_ok and sequence_ok  # we have events to compute ratios

    return {
        "can_compute_timing_features": can_timing,
        "can_compute_speech_dynamics": can_speech,
        "can_compute_engagement_metrics": can_engagement,
        "issues_preventing_features": issues,
    }


def main():
    data_path = Path(__file__).parent / "calls_500.json"
    out_path = Path(__file__).parent / "validation_report.json"

    with open(data_path) as f:
        data = json.load(f)
    calls = data.get("calls", [])
    total = len(calls)

    if total != 500:
        print(f"CRITICAL: expected 500 records, got {total}", file=sys.stderr)
        sys.exit(1)

    schema_ok, schema_issues, missing_count = check_schema(calls)
    seq_ok, seq_issues, seq_violation_count = check_sequence(calls)
    corr_ok, corr_violations, outcome_statistics = check_correlation(calls)
    critical_suspicious, warnings_suspicious = check_suspicious(calls)

    critical_issues = []
    if not schema_ok:
        critical_issues.append(f"Schema validation failed: {missing_count} record(s) with missing/incorrect required fields")
        critical_issues.extend(schema_issues[:10])
        if len(schema_issues) > 10:
            critical_issues.append(f"... and {len(schema_issues) - 10} more")
    if not seq_ok:
        critical_issues.append(f"Sequence validation failed: {seq_violation_count} timestamp/order violation(s)")
        critical_issues.extend(seq_issues[:10])
        if len(seq_issues) > 10:
            critical_issues.append(f"... and {len(seq_issues) - 10} more")
    if not corr_ok:
        critical_issues.append(f"Outcome correlation failed: {len(corr_violations)} record(s) violate correlation rules")
        critical_issues.extend(corr_violations[:10])
        if len(corr_violations) > 10:
            critical_issues.append(f"... and {len(corr_violations) - 10} more")
    critical_issues.extend(critical_suspicious)

    warnings = list(warnings_suspicious)

    report = {
        "validation_summary": {
            "total_records": total,
            "schema_valid": schema_ok,
            "sequence_valid": seq_ok,
            "correlation_valid": corr_ok,
            "critical_issues": critical_issues,
            "warnings": warnings,
            "record_counts": {
                "missing_required_fields": missing_count,
                "timestamp_or_order_violations": seq_violation_count,
                "correlation_rule_violations": len(corr_violations),
            },
        },
        "outcome_statistics": outcome_statistics,
        "feature_feasibility": feature_feasibility(calls, schema_ok, seq_ok),
    }

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Validation complete. Report written to {out_path}")
    if critical_issues:
        print("CRITICAL ISSUES EXIST. Stopping.")
        for issue in critical_issues[:5]:
            print(f"  - {issue}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
