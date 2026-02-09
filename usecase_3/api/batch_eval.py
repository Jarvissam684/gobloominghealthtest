"""
Batch evaluation: aggregate statistics, anomaly detection, quality distribution.

- Cost: GPT-4o-mini target ~$0.015/response with caching (50%+ hit rate).
- Concurrency: up to 10 in-flight; group by context_type then agent_id for cache efficiency.
- Max batch size: 500 per request; split larger into multiple requests.
"""

from __future__ import annotations

import os
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from .schemas import (
    AnomalyRecord,
    DimensionStats,
    QualityDistribution,
    FlagsSummary,
    AggregateStats,
    BatchMetadata,
)

DIMENSION_NAMES = (
    "task_completion",
    "empathy",
    "conciseness",
    "naturalness",
    "safety",
    "clarity",
)
SCORE_BUCKETS = ("1-2", "3-4", "5-6", "7-8", "9-10")
OUTLIER_STD_THRESHOLD = 3.0
LOW_CONFIDENCE_THRESHOLD = 0.6


def _cost_per_response_live() -> float:
    try:
        return float(os.environ.get("EVAL_COST_PER_RESPONSE_LIVE", "0.015"))
    except (TypeError, ValueError):
        return 0.015


def _cost_per_response_cache() -> float:
    try:
        return float(os.environ.get("EVAL_COST_PER_RESPONSE_CACHE", "0.0"))
    except (TypeError, ValueError):
        return 0.0


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * p / 100.0
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_values) else f
    return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f]) if f != c else sorted_values[f]


def _dimension_stats(values: list[float]) -> DimensionStats:
    if not values:
        return DimensionStats(mean=0.0, std=0.0, min=0.0, p25=0.0, median=0.0, p75=0.0, max=0.0)
    sorted_v = sorted(values)
    n = len(sorted_v)
    mean = statistics.mean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    return DimensionStats(
        mean=round(mean, 3),
        std=round(std, 3),
        min=min(values),
        p25=round(_percentile(sorted_v, 25), 3),
        median=round(statistics.median(values), 3),
        p75=round(_percentile(sorted_v, 75), 3),
        max=max(values),
    )


def _bucket(overall: float) -> str:
    if overall <= 2:
        return "1-2"
    if overall <= 4:
        return "3-4"
    if overall <= 6:
        return "5-6"
    if overall <= 8:
        return "7-8"
    return "9-10"


def compute_aggregate_stats(
    individual_scores: list[dict],
    *,
    agent_id_key: str = "agent_id",
    prompt_version_key: str = "prompt_version",
    context_type_key: str = "context_type",
) -> AggregateStats:
    """Build aggregate_stats from list of evaluation result dicts (with dimensions, overall_score, metadata)."""
    if not individual_scores:
        empty = DimensionStats(mean=0.0, std=0.0, min=0.0, p25=0.0, median=0.0, p75=0.0, max=0.0)
        return AggregateStats(dimensions={d: empty for d in DIMENSION_NAMES}, overall_score=empty)

    by_agent: dict[str, list[dict]] = defaultdict(list)
    by_pv: dict[str, list[dict]] = defaultdict(list)
    by_ctx: dict[str, list[dict]] = defaultdict(list)

    overall_scores = []
    per_dim: dict[str, list[float]] = {d: [] for d in DIMENSION_NAMES}

    for item in individual_scores:
        dims = item.get("dimensions") or {}
        os = item.get("overall_score", 0.0)
        overall_scores.append(os)
        for dim in DIMENSION_NAMES:
            d = dims.get(dim)
            if isinstance(d, dict) and "score" in d:
                per_dim[dim].append(d["score"])
            else:
                per_dim[dim].append(0)
        meta = item.get("metadata") or {}
        aid = meta.get(agent_id_key) or "unknown"
        pv = meta.get(prompt_version_key) or "unknown"
        ctx = meta.get(context_type_key) or "unknown"
        by_agent[aid].append(item)
        by_pv[pv].append(item)
        by_ctx[ctx].append(item)

    dimensions = {d: _dimension_stats(per_dim[d]) for d in DIMENSION_NAMES}
    overall_score = _dimension_stats(overall_scores)

    def slice_stats(items: list[dict]) -> dict[str, Any]:
        os_list = [x.get("overall_score", 0) for x in items]
        dim_slices = {}
        for dim in DIMENSION_NAMES:
            vals = []
            for x in items:
                d = (x.get("dimensions") or {}).get(dim)
                vals.append(d.get("score", 0) if isinstance(d, dict) else 0)
            dim_slices[dim] = _dimension_stats(vals).model_dump()
        return {"overall_score": _dimension_stats(os_list).model_dump(), "dimensions": dim_slices, "n": len(items)}

    by_agent_id = {k: slice_stats(v) for k, v in by_agent.items()}
    by_prompt_version = {k: slice_stats(v) for k, v in by_pv.items()}
    by_context_type = {k: slice_stats(v) for k, v in by_ctx.items()}

    return AggregateStats(
        dimensions=dimensions,
        overall_score=overall_score,
        by_agent_id=by_agent_id,
        by_prompt_version=by_prompt_version,
        by_context_type=by_context_type,
    )


def compute_flags_summary(individual_scores: list[dict]) -> FlagsSummary:
    by_code: dict[str, int] = defaultdict(int)
    by_severity: dict[str, int] = defaultdict(int)
    total_flagged = 0
    for item in individual_scores:
        flags = item.get("flags") or []
        if not flags:
            continue
        total_flagged += 1
        for f in flags:
            if isinstance(f, dict):
                by_code[f.get("code", "unknown")] += 1
                by_severity[f.get("severity", "unknown")] += 1
            else:
                by_code[getattr(f, "code", "unknown")] += 1
                by_severity[getattr(f, "severity", "unknown")] += 1
    return FlagsSummary(
        total_flagged=total_flagged,
        by_code=dict(by_code),
        by_severity=dict(by_severity),
    )


def compute_quality_distribution(individual_scores: list[dict]) -> QualityDistribution:
    histogram: dict[str, int] = {b: 0 for b in SCORE_BUCKETS}
    red = 0
    green = 0
    for item in individual_scores:
        os = item.get("overall_score", 0)
        histogram[_bucket(os)] = histogram.get(_bucket(os), 0) + 1
        if os < 5:
            red += 1
        if os >= 8:
            green += 1
    n = len(individual_scores)
    return QualityDistribution(
        histogram=histogram,
        red_zone_count=red,
        green_zone_count=green,
        red_zone_pct=round(100.0 * red / n, 2) if n else 0.0,
        green_zone_pct=round(100.0 * green / n, 2) if n else 0.0,
    )


def compute_anomalies(
    individual_scores: list[dict],
    *,
    std_threshold: float = OUTLIER_STD_THRESHOLD,
    low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
) -> list[AnomalyRecord]:
    anomalies: list[AnomalyRecord] = []
    if not individual_scores:
        return anomalies

    # Per-dimension mean/std for outlier detection
    per_dim_values: dict[str, list[float]] = {d: [] for d in DIMENSION_NAMES}
    overall_list: list[float] = []
    for item in individual_scores:
        overall_list.append(item.get("overall_score", 0))
        for dim in DIMENSION_NAMES:
            d = (item.get("dimensions") or {}).get(dim)
            per_dim_values[dim].append(d.get("score", 0) if isinstance(d, dict) else 0)

    dim_means = {d: statistics.mean(per_dim_values[d]) if per_dim_values[d] else 0 for d in DIMENSION_NAMES}
    dim_stds = {d: (statistics.stdev(per_dim_values[d]) if len(per_dim_values[d]) > 1 else 0) for d in DIMENSION_NAMES}
    overall_mean = statistics.mean(overall_list)
    overall_std = statistics.stdev(overall_list) if len(overall_list) > 1 else 0

    for idx, item in enumerate(individual_scores):
        eval_id = item.get("eval_id", "")
        os = item.get("overall_score", 0)
        if overall_std > 0 and abs(os - overall_mean) > std_threshold * overall_std:
            anomalies.append(
                AnomalyRecord(
                    index=idx,
                    eval_id=eval_id,
                    anomaly_type="outlier_score",
                    dimension=None,
                    value=os,
                    message=f"Overall score {os} is >{std_threshold} std from mean {overall_mean:.2f}.",
                )
            )
        dims = item.get("dimensions") or {}
        for dim in DIMENSION_NAMES:
            d = dims.get(dim)
            if not isinstance(d, dict):
                continue
            score = d.get("score", 0)
            conf = d.get("confidence", 1.0)
            if conf < low_confidence_threshold:
                anomalies.append(
                    AnomalyRecord(
                        index=idx,
                        eval_id=eval_id,
                        anomaly_type="low_confidence",
                        dimension=dim,
                        value=conf,
                        message=f"Dimension {dim} confidence {conf} < {low_confidence_threshold}.",
                    )
                )
            std_d = dim_stds.get(dim, 0)
            mean_d = dim_means.get(dim, 0)
            if std_d > 0 and abs(score - mean_d) > std_threshold * std_d:
                anomalies.append(
                    AnomalyRecord(
                        index=idx,
                        eval_id=eval_id,
                        anomaly_type="outlier_dimension",
                        dimension=dim,
                        value=float(score),
                        message=f"Dimension {dim} score {score} is >{std_threshold} std from mean {mean_d:.2f}.",
                    )
                )

    return anomalies


def build_batch_metadata(
    batch_id: str | None,
    total_evaluated: int,
    cache_hits: int,
    cache_misses: int,
    latencies_ms: list[float],
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
) -> BatchMetadata:
    total = cache_hits + cache_misses
    hit_rate = cache_hits / total if total else 0.0
    cost = cache_misses * _cost_per_response_live() + cache_hits * _cost_per_response_cache()
    avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    return BatchMetadata(
        batch_id=batch_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_evaluated=total_evaluated,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        cache_hit_rate=round(hit_rate, 3),
        total_cost_usd=round(cost, 4),
        avg_latency_ms=round(avg_latency, 1),
        warnings=warnings or [],
        errors=errors or [],
    )
