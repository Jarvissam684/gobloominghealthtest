"""
Implementation of A/B compare logic per api/COMPARE_LOGIC.md.

- Dimension-wise winner: |score_a - score_b| > threshold => winner; else tie.
- Context-weighted overall winner: weighted_diff vs threshold.
- Confidence in winner: (1 - normalized_variance_of_diffs) * avg_confidence.
- Recommendation text by winner/tie.
- Bias detection: alert if a or b wins > 65%.
- Batch: aggregate + optional binomial p-value for "statistically better".
"""

from __future__ import annotations

import math
from typing import Literal

# Dimension names (must match schema)
DIMENSION_NAMES = (
    "task_completion",
    "empathy",
    "conciseness",
    "naturalness",
    "safety",
    "clarity",
)

# Config (per COMPARE_LOGIC.md)
DIMENSION_WIN_THRESHOLD = 1  # |score_a - score_b| > 1 => winner; else tie
OVERALL_WIN_THRESHOLD = 0.5  # weighted_diff > 0.5 => A; < -0.5 => B; else tie
CONFIDENCE_HUMAN_REVIEW_THRESHOLD = 0.6  # if confidence_in_winner < this => recommend_human_review
BIAS_ALERT_THRESHOLD = 0.65  # alert if a_win_pct or b_win_pct > this
VARIANCE_SCALE = 25.0  # max variance for normalizing (1-10 scale diffs)


def dimension_winner(
    score_a: int,
    score_b: int,
    *,
    threshold: int = DIMENSION_WIN_THRESHOLD,
) -> Literal["a", "b", "tie"]:
    """
    If |score_a - score_b| > threshold: winner is higher; else tie (too close to call).
    Unbiased: no positional preference.
    """
    diff = abs(score_a - score_b)
    if diff <= threshold:
        return "tie"
    return "a" if score_a > score_b else "b"


def weighted_diff(
    dimensions_a: dict[str, dict],
    dimensions_b: dict[str, dict],
    weights: dict[str, float],
) -> float:
    """
    weighted_diff = Σ(weight[dim] * (score_a[dim] - score_b[dim])).
    Positive => A ahead; negative => B ahead.
    """
    total = 0.0
    for dim, w in weights.items():
        da = dimensions_a.get(dim) or {}
        db = dimensions_b.get(dim) or {}
        sa = da.get("score", 0) if isinstance(da, dict) else 0
        sb = db.get("score", 0) if isinstance(db, dict) else 0
        total += w * (sa - sb)
    return total


def overall_winner(
    weighted_diff_val: float,
    *,
    threshold: float = OVERALL_WIN_THRESHOLD,
) -> Literal["a", "b", "tie"]:
    """If weighted_diff > threshold: A; < -threshold: B; else tie."""
    if weighted_diff_val > threshold:
        return "a"
    if weighted_diff_val < -threshold:
        return "b"
    return "tie"


def confidence_in_winner(
    dimensions_a: dict[str, dict],
    dimensions_b: dict[str, dict],
    dimensions: tuple[str, ...] | list[str] | None = None,
    *,
    variance_scale: float = VARIANCE_SCALE,
) -> float:
    """
    confidence = (1 - normalized_variance_of_diffs) * avg_confidence.
    - High agreement across dimensions (low variance of (score_a - score_b)) => higher confidence.
    - Low LLM confidence in evals => lower confidence.
    """
    dims = dimensions or DIMENSION_NAMES
    diffs: list[float] = []
    confs: list[float] = []
    for d in dims:
        da = dimensions_a.get(d) or {}
        db = dimensions_b.get(d) or {}
        sa = da.get("score", 0) if isinstance(da, dict) else 0
        sb = db.get("score", 0) if isinstance(db, dict) else 0
        diffs.append(sa - sb)
        if isinstance(da, dict) and "confidence" in da:
            confs.append(float(da["confidence"]))
        if isinstance(db, dict) and "confidence" in db:
            confs.append(float(db["confidence"]))
    if not diffs:
        return 0.5
    n = len(diffs)
    mean_d = sum(diffs) / n
    variance = sum((x - mean_d) ** 2 for x in diffs) / n
    normalized_var = min(1.0, variance / variance_scale)
    agreement = 1.0 - normalized_var
    avg_conf = sum(confs) / len(confs) if confs else 0.5
    return round(agreement * avg_conf, 3)


def build_recommendation(
    winner: Literal["a", "b", "tie"],
    context_type: str,
    a_better_dims: list[str],
    b_better_dims: list[str],
) -> str:
    """
    Actionable recommendation per COMPARE_LOGIC.md §4.
    a_better_dims: dimensions where winner was "a"; b_better_dims: where "b".
    """
    if winner == "tie":
        if a_better_dims or b_better_dims:
            return (
                f"Both responses are equivalent overall. Response A is slightly better on: {', '.join(a_better_dims) or 'none'}. "
                f"Response B is slightly better on: {', '.join(b_better_dims) or 'none'}. "
                f"Consider human review for {context_type} contexts."
            )
        return f"Both responses are equivalent across dimensions for {context_type} contexts. Either variant is acceptable."
    if winner == "a":
        return (
            f"Response A is recommended for {context_type} scenarios because it scores higher on the dimensions "
            f"that matter most for this context. Consider using variant A as the default."
        )
    return (
        f"Response B is recommended for {context_type} scenarios because it scores higher on the dimensions "
        f"that matter most for this context. Consider using variant B for empathy-sensitive or screening contexts."
    )


def check_bias(
    a_wins: int,
    b_wins: int,
    ties: int,
    *,
    threshold: float = BIAS_ALERT_THRESHOLD,
) -> tuple[bool, float, float]:
    """
    Returns (alert, a_win_pct, b_win_pct).
    Alert if response_a wins > threshold or response_b wins > threshold (potential positional bias).
    """
    total = a_wins + b_wins + ties
    if total == 0:
        return (False, 0.0, 0.0)
    a_pct = a_wins / total
    b_pct = b_wins / total
    alert = a_pct > threshold or b_pct > threshold
    return (alert, a_pct, b_pct)


def aggregate_batch(
    results: list[dict],
    *,
    winner_key: str = "winner",
) -> dict:
    """
    Aggregate batch comparison results per COMPARE_LOGIC.md §6.
    results: list of dicts with winner_key in ["a", "b", "tie"].
    Returns dict with a_wins, b_wins, ties, total, a_win_pct, b_win_pct, overall_recommendation.
    """
    a_wins = sum(1 for r in results if r.get(winner_key) == "a")
    b_wins = sum(1 for r in results if r.get(winner_key) == "b")
    ties = sum(1 for r in results if r.get(winner_key) == "tie")
    total = len(results)
    a_pct = a_wins / total if total else 0.0
    b_pct = b_wins / total if total else 0.0
    if total == 0:
        rec = "No comparisons to aggregate."
    elif a_pct > b_pct and a_pct > 0.5:
        rec = (
            f"Response A wins on {a_wins}/{total} contexts, Response B on {b_wins}/{total}, tie on {ties}/{total}. "
            "Response A is recommended overall."
        )
    elif b_pct > a_pct and b_pct > 0.5:
        rec = (
            f"Response A wins on {a_wins}/{total} contexts, Response B on {b_wins}/{total}, tie on {ties}/{total}. "
            "Response B is recommended overall."
        )
    else:
        rec = (
            f"Response A wins on {a_wins}/{total} contexts, Response B on {b_wins}/{total}, tie on {ties}/{total}. "
            "No clear overall winner; consider context-specific use."
        )
    return {
        "a_wins": a_wins,
        "b_wins": b_wins,
        "ties": ties,
        "total": total,
        "a_win_pct": round(a_pct, 3),
        "b_win_pct": round(b_pct, 3),
        "overall_recommendation": rec,
    }


def binomial_p_value(wins: int, total: int, *, p0: float = 0.5) -> float:
    """
    Two-tailed binomial test: H0 = true proportion is p0 (e.g. 0.5).
    Returns p-value; if p < 0.05 we can say "statistically better" at α=0.05.
    Uses normal approximation when n*p0*(1-p0) >= 5.
    """
    if total == 0:
        return 1.0
    p_hat = wins / total
    # Normal approximation: z = (p_hat - p0) / sqrt(p0*(1-p0)/n)
    se = math.sqrt(p0 * (1 - p0) / total)
    if se <= 0:
        return 1.0
    z = (p_hat - p0) / se
    # Two-tailed: 2 * (1 - Φ(|z|)); approximate Φ with erf
    from math import erf
    p_one_tail = 0.5 * (1 + erf(abs(z) / math.sqrt(2)))
    p_value = 2 * (1 - p_one_tail)
    return max(0.0, min(1.0, p_value))


def aggregate_batch_with_stats(
    results: list[dict],
    *,
    winner_key: str = "winner",
    alpha: float = 0.05,
) -> dict:
    """
    Like aggregate_batch but adds statistical significance for "A statistically better" / "B statistically better".
    Adds: a_better_p_value, b_better_p_value, a_statistically_better (True if p < alpha), b_statistically_better.
    """
    agg = aggregate_batch(results, winner_key=winner_key)
    total = agg["total"]
    a_wins = agg["a_wins"]
    b_wins = agg["b_wins"]
    agg["a_better_p_value"] = round(binomial_p_value(a_wins, total), 4) if total else 1.0
    agg["b_better_p_value"] = round(binomial_p_value(b_wins, total), 4) if total else 1.0
    agg["a_statistically_better"] = total > 0 and agg["a_better_p_value"] < alpha
    agg["b_statistically_better"] = total > 0 and agg["b_better_p_value"] < alpha
    return agg
