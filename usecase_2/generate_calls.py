#!/usr/bin/env python3
"""
Synthetic voice AI call data generator.
Generates 500 call records with outcome-correlated event patterns.
"""

import json
import random
from datetime import datetime, timezone
from typing import Any

# Outcome distribution (exact counts)
OUTCOME_COUNTS = {
    "completed": 300,   # 60%
    "abandoned": 125,   # 25%
    "transferred": 50,  # 10%
    "error": 25,        # 5%
}

AGENT_IDS = ["agent_a1", "agent_b2", "agent_c3", "agent_d4", "agent_e5", "agent_f6", "agent_g7", "agent_h8"]
ORG_IDS = ["org_1", "org_2", "org_3"]
CALL_PURPOSES = ["sdoh_screening", "appointment_scheduling", "billing", "support"]
PHONE_TYPES = ["mobile", "landline"]
TIME_OF_DAY = ["morning", "afternoon", "evening", "night"]
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
TOOLS = ["lookup_patient", "schedule_appt", "check_eligibility", "transfer_queue", "create_ticket", "verify_insurance"]


def compute_metrics(events: list[dict]) -> dict:
    """Compute derived metrics from events for validation."""
    call_duration_sec = 0
    total_user_words = 0
    total_agent_words = 0
    agent_speech_ms = 0
    user_speech_ms = 0
    silence_ms = 0
    tools_called = 0
    time_to_first_user_speech = None

    for ev in events:
        ts = ev["ts"]
        t = ev["type"]
        dur = ev.get("duration_ms", 0)
        words = ev.get("words", 0)
        if t == "call_end":
            call_duration_sec = ts
        elif t == "agent_speech":
            agent_speech_ms += dur
            total_agent_words += words
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
        "total_agent_words": total_agent_words,
        "silence_ratio": silence_ratio,
        "agent_speech_ratio": agent_speech_ratio,
        "tools_called": tools_called,
        "time_to_first_user_speech": time_to_first_user_speech,
        "has_user_speech": time_to_first_user_speech is not None,
    }


def make_event(ts: int, etype: str, duration_ms: int = 0, words: int = 0, tool: str | None = None) -> dict:
    ev = {"ts": ts, "type": etype, "duration_ms": duration_ms, "words": words}
    if etype == "tool_call":
        ev["tool"] = tool or random.choice(TOOLS)
    else:
        ev["tool"] = None
    return ev


def generate_completed_call(call_idx: int) -> dict:
    """Generate a call that satisfies: tools_called >= 1, total_user_words > 100, silence_ratio < 0.4."""
    num_events = random.randint(10, 24)  # 8-25, leave room for call_end
    duration_sec = random.randint(90, 580)
    events = []
    t = 0

    events.append(make_event(t, "call_start"))
    t += random.randint(1, 5)

    agent_ms_total = 0
    user_ms_total = 0
    silence_ms_total = 0
    user_words = 0
    tool_count = 0

    # Target: silence_ratio < 0.4, so silence < 40% of call
    max_silence_ms = int(duration_sec * 1000 * 0.38)
    target_user_words = random.randint(101, 400)
    # We need at least 1 tool_call
    tool_count_target = random.randint(1, 4)

    while t < duration_sec and len(events) < num_events:
        # Decide event type with bias toward speech
        r = random.random()
        if r < 0.35 and (silence_ms_total < max_silence_ms):
            # silence
            seg_ms = min(random.randint(2000, 8000), (max_silence_ms - silence_ms_total), (duration_sec - t) * 1000)
            if seg_ms > 0:
                seg_sec = seg_ms // 1000 or 1
                t += seg_sec
                silence_ms_total += seg_ms
                events.append(make_event(t, "silence", duration_ms=seg_ms))
        elif r < 0.55:
            # agent_speech
            seg_ms = random.randint(3000, 15000)
            seg_sec = min(seg_ms // 1000, duration_sec - t)
            if seg_sec > 0:
                t += seg_sec
                w = random.randint(15, 60) * seg_sec
                agent_ms_total += seg_ms
                events.append(make_event(t, "agent_speech", duration_ms=seg_ms, words=w))
        elif r < 0.85:
            # user_speech - need to reach > 100 words
            seg_ms = random.randint(2000, 12000)
            seg_sec = min(seg_ms // 1000, duration_sec - t)
            if seg_sec > 0:
                t += seg_sec
                w = random.randint(20, 80) * seg_sec
                user_words += w
                user_ms_total += seg_ms
                events.append(make_event(t, "user_speech", duration_ms=seg_ms, words=w))
        elif tool_count < tool_count_target:
            # tool_call
            t += 1
            tool_count += 1
            events.append(make_event(t, "tool_call", tool=random.choice(TOOLS)))

    # Ensure we have enough user words (pad with one more user_speech if needed)
    while user_words <= 100 and t < duration_sec:
        seg_sec = min(random.randint(2, 5), duration_sec - t)
        if seg_sec <= 0:
            break
        t += seg_sec
        w = random.randint(50, 150)
        user_words += w
        user_ms_total += seg_sec * 1000
        events.append(make_event(t, "user_speech", duration_ms=seg_sec * 1000, words=w))

    # Ensure at least one tool_call
    if tool_count == 0:
        insert_at = random.randint(1, max(1, len(events) - 1))
        events.insert(insert_at, make_event(events[insert_at]["ts"] - 1 if insert_at > 0 else 1, "tool_call", tool=random.choice(TOOLS)))
        events.sort(key=lambda e: (e["ts"], ["call_start", "agent_speech", "user_speech", "silence", "tool_call", "call_end"].index(e["type"])))

    events.append(make_event(duration_sec, "call_end"))

    # Enforce 8-25 events
    if len(events) < 8:
        # Add more silence or short speech segments
        for _ in range(8 - len(events)):
            t_ins = random.randint(1, duration_sec - 1)
            events.insert(-1, make_event(t_ins, "silence", duration_ms=1000))
        events.sort(key=lambda e: e["ts"])
        # Re-set call_end
        events = [e for e in events if e["type"] != "call_end"] + [make_event(duration_sec, "call_end")]

    if len(events) > 25:
        middle = events[1:-1]
        random.shuffle(middle)
        events = [events[0]] + middle[:23] + [events[-1]]
        events.sort(key=lambda e: (e["ts"], ["call_start", "agent_speech", "user_speech", "silence", "tool_call", "call_end"].index(e["type"])))

    metrics = compute_metrics(events)
    assert metrics["tools_called"] >= 1 and metrics["total_user_words"] > 100 and metrics["silence_ratio"] < 0.4, (
        f"Completed validation failed: {metrics}"
    )

    return build_call_record(call_idx, events, "completed", survey_rate=random.uniform(0.3, 0.95))


def generate_abandoned_call(call_idx: int) -> dict:
    """Generate call with: silence_ratio > 0.5 OR time_to_first_user_speech > 15 OR total_user_words < 30."""
    variant = random.choice(["silence", "late_speech", "few_words"])
    num_events = random.randint(8, 25)
    duration_sec = random.randint(45, 400)
    events = [make_event(0, "call_start")]
    t = 0

    if variant == "silence":
        # Long silence upfront, then maybe brief interaction, then abandon
        silence_ms = int(duration_sec * 1000 * random.uniform(0.55, 0.85))
        t = silence_ms // 1000
        events.append(make_event(t, "silence", duration_ms=silence_ms))
        if random.random() < 0.5 and t < duration_sec - 5:
            t += random.randint(1, 3)
            events.append(make_event(t, "agent_speech", duration_ms=2000, words=30))
        events.append(make_event(duration_sec, "call_end"))
    elif variant == "late_speech":
        # No user speech until after 15s (or very little user speech)
        t = random.randint(16, 25)
        events.append(make_event(t, "agent_speech", duration_ms=5000, words=100))
        t += 2
        events.append(make_event(t, "user_speech", duration_ms=1000, words=10))  # first user at t>=16
        t = duration_sec
        events.append(make_event(t, "call_end"))
    else:
        # few_words: total_user_words < 30
        t = 2
        events.append(make_event(t, "agent_speech", duration_ms=4000, words=50))
        t += 1
        events.append(make_event(t, "user_speech", duration_ms=500, words=10))
        t += 3
        events.append(make_event(t, "agent_speech", duration_ms=3000, words=40))
        t += 1
        events.append(make_event(t, "user_speech", duration_ms=500, words=8))  # total 18
        events.append(make_event(duration_sec, "call_end"))

    if len(events) < 8:
        for _ in range(8 - len(events)):
            t_ins = random.randint(1, duration_sec - 1)
            events.insert(-1, make_event(t_ins, "silence", duration_ms=1000))
        events.sort(key=lambda e: e["ts"])
        events = [e for e in events if e["type"] != "call_end"] + [make_event(duration_sec, "call_end")]
    if len(events) > 25:
        middle = events[1:-1]
        random.shuffle(middle)
        events = [events[0]] + middle[:23] + [events[-1]]
        events.sort(key=lambda e: (e["ts"], ["call_start", "agent_speech", "user_speech", "silence", "tool_call", "call_end"].index(e["type"])))

    metrics = compute_metrics(events)
    assert (
        metrics["silence_ratio"] > 0.5
        or (metrics["time_to_first_user_speech"] is not None and metrics["time_to_first_user_speech"] > 15)
        or metrics["total_user_words"] < 30
    ), f"Abandoned validation failed: {metrics}"

    return build_call_record(call_idx, events, "abandoned", survey_rate=0.0)


def generate_transferred_call(call_idx: int) -> dict:
    """Generate call with: agent_speech_ratio > 0.55."""
    duration_sec = random.randint(60, 450)
    events = [make_event(0, "call_start")]
    t = 0

    # Guarantee agent_speech_ratio > 0.55: fill agent first, then user
    target_agent_ratio = random.uniform(0.56, 0.85)
    total_speech_budget_ms = int(duration_sec * 1000 * 0.65)
    agent_budget = int(total_speech_budget_ms * target_agent_ratio)
    user_budget = total_speech_budget_ms - agent_budget
    agent_ms = 0
    user_ms = 0

    # Add agent segments until budget met
    while agent_ms < agent_budget and t < duration_sec - 5 and len(events) < 23:
        seg_ms = min(random.randint(4000, 14000), agent_budget - agent_ms, (duration_sec - t) * 1000)
        if seg_ms > 0:
            seg_sec = seg_ms // 1000 or 1
            t += seg_sec
            agent_ms += seg_ms
            events.append(make_event(t, "agent_speech", duration_ms=seg_ms, words=seg_ms // 50))
        else:
            break
    # Add user segments
    while user_ms < user_budget and t < duration_sec - 5 and len(events) < 23:
        seg_ms = min(random.randint(2000, 8000), user_budget - user_ms, (duration_sec - t) * 1000)
        if seg_ms > 0:
            seg_sec = seg_ms // 1000 or 1
            t += seg_sec
            user_ms += seg_ms
            events.append(make_event(t, "user_speech", duration_ms=seg_ms, words=seg_ms // 60))
        else:
            break
    # Optional tool_call
    if random.random() < 0.4 and len(events) < 24:
        t = min(t + 1, duration_sec - 1)
        events.append(make_event(t, "tool_call", tool="transfer_queue"))

    events.append(make_event(duration_sec, "call_end"))

    if len(events) < 8:
        for _ in range(8 - len(events)):
            t_ins = random.randint(1, duration_sec - 1)
            events.insert(-1, make_event(t_ins, "silence", duration_ms=1000))
        events.sort(key=lambda e: e["ts"])
        events = [e for e in events if e["type"] != "call_end"] + [make_event(duration_sec, "call_end")]
    if len(events) > 25:
        middle = events[1:-1]
        # Keep non-user_speech first to preserve agent_speech_ratio > 0.55
        user_ev = [e for e in middle if e["type"] == "user_speech"]
        other_ev = [e for e in middle if e["type"] != "user_speech"]
        to_keep = 23
        kept_other = min(len(other_ev), to_keep)
        kept_user = to_keep - kept_other
        middle = other_ev[:kept_other] + user_ev[:kept_user]
        middle.sort(key=lambda e: (e["ts"], ["call_start", "agent_speech", "user_speech", "silence", "tool_call", "call_end"].index(e["type"])))
        events = [events[0]] + middle + [events[-1]]

    metrics = compute_metrics(events)
    assert metrics["agent_speech_ratio"] > 0.55, f"Transferred validation failed: {metrics}"

    return build_call_record(call_idx, events, "transferred", survey_rate=0.0)


def generate_error_call(call_idx: int) -> dict:
    """Generate call with: call_duration < 30 OR no user_speech after call_start."""
    variant = random.choice(["short", "no_user"])
    if variant == "short":
        duration_sec = random.randint(5, 29)
    else:
        duration_sec = random.randint(30, 120)

    events = [make_event(0, "call_start")]
    t = 0

    if variant == "short":
        t = random.randint(1, duration_sec - 1)
        events.append(make_event(t, "agent_speech", duration_ms=2000, words=20))
        events.append(make_event(duration_sec, "call_end"))
    else:
        # No user_speech - only agent and maybe silence
        t = 2
        events.append(make_event(t, "agent_speech", duration_ms=5000, words=50))
        t += 5
        events.append(make_event(t, "silence", duration_ms=10000))
        t = duration_sec
        events.append(make_event(t, "call_end"))

    if len(events) < 8:
        for _ in range(8 - len(events)):
            t_ins = random.randint(1, max(1, duration_sec - 1))
            events.insert(-1, make_event(t_ins, "silence", duration_ms=500))
        events.sort(key=lambda e: e["ts"])
        events = [e for e in events if e["type"] != "call_end"] + [make_event(duration_sec, "call_end")]
    if len(events) > 25:
        middle = events[1:-1]
        random.shuffle(middle)
        events = [events[0]] + middle[:23] + [events[-1]]
        events.sort(key=lambda e: (e["ts"], ["call_start", "agent_speech", "user_speech", "silence", "tool_call", "call_end"].index(e["type"])))

    metrics = compute_metrics(events)
    assert metrics["call_duration"] < 30 or not metrics["has_user_speech"], f"Error validation failed: {metrics}"

    return build_call_record(call_idx, events, "error", survey_rate=0.0)


def normalize_strictly_increasing(events: list[dict], max_duration_sec: int = 600) -> list[dict]:
    """Ensure ts is strictly increasing; cap call_end at max_duration_sec."""
    if len(events) < 2:
        return events
    out = []
    for i, e in enumerate(events):
        e = dict(e)
        if i == 0:
            out.append(e)
            continue
        prev_ts = out[-1]["ts"]
        if e["ts"] <= prev_ts:
            e["ts"] = prev_ts + 1
        if i == len(events) - 1 and e.get("type") == "call_end" and e["ts"] > max_duration_sec:
            e["ts"] = max_duration_sec
        out.append(e)
    return out


def build_call_record(call_idx: int, events: list[dict], outcome: str, survey_rate: float = 0.0) -> dict:
    """Build full call object with metadata."""
    # Sort events by ts then type order
    events_sorted = sorted(events, key=lambda e: (e["ts"], ["call_start", "agent_speech", "user_speech", "silence", "tool_call", "call_end"].index(e["type"])))
    events_sorted = normalize_strictly_increasing(events_sorted)
    return {
        "call_id": f"call_{call_idx:05d}",
        "metadata": {
            "agent_id": random.choice(AGENT_IDS),
            "org_id": random.choice(ORG_IDS),
            "call_purpose": random.choice(CALL_PURPOSES),
            "caller_phone_type": random.choice(PHONE_TYPES),
            "time_of_day": random.choice(TIME_OF_DAY),
            "day_of_week": random.choice(DAYS),
        },
        "events": events_sorted,
        "outcome": outcome,
        "survey_completion_rate": round(survey_rate, 2) if outcome == "completed" else 0.0,
    }


def main():
    random.seed(42)
    calls: list[dict] = []
    call_idx = 0

    for outcome, count in OUTCOME_COUNTS.items():
        for _ in range(count):
            call_idx += 1
            if outcome == "completed":
                calls.append(generate_completed_call(call_idx))
            elif outcome == "abandoned":
                calls.append(generate_abandoned_call(call_idx))
            elif outcome == "transferred":
                calls.append(generate_transferred_call(call_idx))
            else:
                calls.append(generate_error_call(call_idx))

    # Verify counts
    from collections import Counter
    dist = Counter(c["outcome"] for c in calls)
    assert dist == Counter(OUTCOME_COUNTS), f"Distribution mismatch: {dist}"

    payload = {
        "calls": calls,
        "metadata": {
            "total_calls": len(calls),
            "outcome_distribution": dict(dist),
            "generation_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }

    from pathlib import Path
    out_path = Path(__file__).resolve().parent / "calls_500.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {len(calls)} calls to {out_path}")
    print("Outcome distribution:", payload["metadata"]["outcome_distribution"])


if __name__ == "__main__":
    main()
