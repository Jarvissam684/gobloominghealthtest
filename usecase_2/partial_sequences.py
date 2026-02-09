#!/usr/bin/env python3
"""
Stratified partial sequences for LSTM training.
For each of 500 calls, create 5 partial sequences (20%, 40%, 60%, 80%, 100% of duration).
Recompute all 28 features on truncated events; survey_completion_rate = 0 for all partials.
Output: 2500 records.
"""

import json
from pathlib import Path

from feature_engineering import compute_features

EVENT_TYPE_ORDER = ["call_start", "agent_speech", "user_speech", "silence", "tool_call", "call_end"]
PERCENTAGES = [20, 40, 60, 80, 100]


def _sort_events(events: list[dict]) -> list[dict]:
    """Sort events by ts, then by type order (for ties)."""
    return sorted(
        events,
        key=lambda e: (e.get("ts", 0), EVENT_TYPE_ORDER.index(e.get("type", "")) if e.get("type") in EVENT_TYPE_ORDER else 99),
    )


def build_partial_sequences(calls: list[dict]) -> list[dict]:
    """For each call, create 5 partial sequences; recompute features on truncated events."""
    partials = []
    for call in calls:
        original_call_id = call.get("call_id", "")
        outcome = call.get("outcome", "")
        meta = call.get("metadata", {})
        events = call.get("events", [])
        if not events:
            continue
        events_sorted = _sort_events(events)
        first_ts = events_sorted[0].get("ts", 0)
        last_ts_full = events_sorted[-1].get("ts", 0)
        total_duration_full = float(last_ts_full - first_ts) if last_ts_full >= first_ts else 0.0

        for p in PERCENTAGES:
            cutoff = total_duration_full * (p / 100.0)
            # Truncate: keep events with ts <= cutoff, exclude call_end (leave sequence incomplete)
            truncated = [
                dict(e)
                for e in events_sorted
                if e.get("ts", 0) <= cutoff and e.get("type") != "call_end"
            ]
            truncated = _sort_events(truncated)
            if not truncated:
                # Should not happen (at least call_start at 0)
                continue

            actual_duration = (
                truncated[-1].get("ts", 0) - truncated[0].get("ts", 0)
                if len(truncated) >= 2
                else 0.0
            )
            if truncated:
                actual_duration = float(truncated[-1].get("ts", 0) - truncated[0].get("ts", 0))

            partial_call_id = f"{original_call_id}_p{p}"
            partial_call = {
                "call_id": partial_call_id,
                "metadata": meta,
                "events": truncated,
                "outcome": outcome,
                "survey_completion_rate": 0.0,  # survey only complete at end
            }
            feats_result = compute_features(partial_call)
            features_so_far = feats_result.get("features", {})

            record = {
                "call_id": partial_call_id,
                "completion_percent": p,
                "original_call_id": original_call_id,
                "outcome": outcome,
                "sequence": truncated,
                "actual_duration_after_truncation_sec": round(actual_duration, 4),
                "features_so_far": features_so_far,
            }
            partials.append(record)

    return partials


def main():
    base = Path(__file__).parent
    data_path = base / "calls_500.json"
    out_path = base / "partial_sequences.json"

    with open(data_path) as f:
        data = json.load(f)
    calls = data.get("calls", [])

    if len(calls) != 500:
        raise SystemExit(f"Expected 500 calls, got {len(calls)}")

    partials = build_partial_sequences(calls)

    # Metadata
    from collections import Counter
    by_p = Counter(r["completion_percent"] for r in partials)
    by_outcome = Counter(r["outcome"] for r in partials)
    metadata = {
        "total_partial_sequences": len(partials),
        "breakdown": {
            "20_percent": by_p.get(20, 0),
            "40_percent": by_p.get(40, 0),
            "60_percent": by_p.get(60, 0),
            "80_percent": by_p.get(80, 0),
            "100_percent": by_p.get(100, 0),
        },
        "outcome_distribution": dict(by_outcome),
    }

    payload = {
        "partial_sequences": partials,
        "metadata": metadata,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {len(partials)} partial sequences to {out_path}")
    print(f"Breakdown: {metadata['breakdown']}")
    print(f"Outcome distribution: {metadata['outcome_distribution']}")


if __name__ == "__main__":
    main()
