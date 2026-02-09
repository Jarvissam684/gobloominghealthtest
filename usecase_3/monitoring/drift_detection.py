"""
Drift detection and retuning: weekly human samples, SCC, alerting, root cause, retuning validation.

Run from project root (usecase_3): python -m monitoring.drift_detection
Or ensure usecase_3 is on PYTHONPATH when importing.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Project root for calibration import
_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from calibration.validation import (
    DIMENSIONS,
    consensus_overall_from_dimensions,
    drift_check as calibration_drift_check,
    spearman_overall,
    spearman_per_dimension,
)

# -----------------------------------------------------------------------------
# Constants (align with monitoring_protocol.md)
# -----------------------------------------------------------------------------

SCC_THRESHOLD_ALERT = 0.75   # SCC < 0.75 → alert, prepare retuning
SCC_THRESHOLD_HOLD = 0.65    # SCC < 0.65 → halt evaluations, manual review
SCC_THRESHOLD_DEPLOY = 0.80  # Require SCC >= 0.80 to approve deployment
SCC_THRESHOLD_VALIDATE = 0.78 # Retuning pass: all dimensions >= 0.78
REGRESSION_TOLERANCE = 0.02  # No dimension may drop by more than this vs baseline
TREND_DECLINE_THRESHOLD = 0.05  # Week-over-week drop to trigger trend alert
CONFIDENCE_ESCALATION_THRESHOLD = 0.6  # If retuning fails, route eval to human when confidence < this
DIVERGENCE_ABS_THRESHOLD = 1.0  # |LLM - human| > 1 → divergent case

# -----------------------------------------------------------------------------
# Spearman (re-export and pseudocode reference)
# -----------------------------------------------------------------------------


def spearman_scc_reference(x: list[float], y: list[float]) -> tuple[float, float]:
    """
    Spearman rank correlation (reference implementation).
    Formula: ρ = 1 - (6 Σ d_i²) / (n(n²-1)), d_i = rank(x_i) - rank(y_i).
    Use calibration.validation.spearman_scc for production (handles ties).
    """
    n = len(x)
    if n != len(y) or n < 2:
        return (float("nan"), 1.0)

    def rank_simple(v: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: v[i])
        ranks = [0.0] * n
        for r, i in enumerate(order):
            ranks[i] = r + 1
        # Tie handling: same value → average rank
        seen: dict[tuple, list[int]] = defaultdict(list)
        for i, val in enumerate(v):
            seen[(val,)].append(i)
        for indices in seen.values():
            if len(indices) > 1:
                avg = sum(ranks[i] for i in indices) / len(indices)
                for i in indices:
                    ranks[i] = avg
        return ranks

    rx = rank_simple(x)
    ry = rank_simple(y)
    d_sq = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    rho = 1.0 - (6.0 * d_sq) / (n * (n * n - 1))
    return (rho, 0.0)


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass
class EvaluationRecord:
    """One evaluation from production (e.g. from DB or export)."""
    eval_id: str
    case_id: str
    context_type: str
    agent_id: str
    overall_score: float
    dimensions: dict[str, float]
    evaluated_at: str = ""


@dataclass
class HumanRatingRecord:
    """One human rating (single rater) in calibration format."""
    case_id: str
    rater_id: str
    context_type: str
    dimensions: dict[str, float]
    overall_score: float | None = None


@dataclass
class DriftResult:
    """Result of weekly drift check."""
    scc_per_dimension: dict[str, float]
    scc_overall: float
    average_scc: float
    status: str  # "ok" | "alert" | "hold"
    message: str
    alerts: list[str] = field(default_factory=list)
    trend_declining: bool = False
    previous_week_min_scc: float | None = None


@dataclass
class RootCauseSummary:
    """Output of root cause analysis."""
    worst_dimensions: list[tuple[str, float]]  # (dim, scc)
    systematic_bias: list[tuple[str, float]]  # (dim, mean_llm_minus_human)
    by_context_type: list[tuple[str, float, int]]  # (context_type, scc, n)
    summary_sentence: str


@dataclass
class DivergentCase:
    """One case where LLM and human disagree by > 1."""
    case_id: str
    dimension: str
    llm_score: float
    human_score: float
    context_type: str
    response_snippet: str = ""


@dataclass
class RetuningProposal:
    """Proposed prompt/weight change for retuning."""
    finding: str
    modification_type: str  # "add_specificity" | "adjust_rubric" | "weight_correction" | "context_instruction"
    proposed_text: str


# -----------------------------------------------------------------------------
# Stratified sampling
# -----------------------------------------------------------------------------


def stratified_sample(
    evaluations: list[EvaluationRecord],
    target_n: int = 10,
    min_per_context: int = 2,
    context_types: tuple[str, ...] = ("verification", "screening", "clarification", "follow_up"),
) -> list[EvaluationRecord]:
    """
    Stratified sample: by context_type (ensure representation), by score range (low/med/high).
    Optionally by agent_id proportion is approximated by taking from shuffled groups.
    """
    if len(evaluations) <= target_n:
        return list(evaluations)

    def score_band(s: float) -> str:
        if s < 5:
            return "low"
        if s < 8:
            return "medium"
        return "high"

    by_context: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for e in evaluations:
        by_context[e.context_type].append(e)

    chosen: list[EvaluationRecord] = []
    used_ids: set[str] = set()

    # 1) Ensure min_per_context per context_type (up to 2 per if we have 4 context types → 8, then fill 2)
    for ctx in context_types:
        pool = [e for e in by_context.get(ctx, []) if e.eval_id not in used_ids]
        if not pool:
            continue
        # Take up to min_per_context; prefer spread across score bands
        by_band: dict[str, list[EvaluationRecord]] = defaultdict(list)
        for e in pool:
            by_band[score_band(e.overall_score)].append(e)
        taken = 0
        for band in ("low", "medium", "high"):
            for e in by_band.get(band, [])[:1]:  # at most 1 per band per context first
                if taken >= min_per_context:
                    break
                chosen.append(e)
                used_ids.add(e.eval_id)
                taken += 1
        for e in pool:
            if taken >= min_per_context:
                break
            if e.eval_id not in used_ids:
                chosen.append(e)
                used_ids.add(e.eval_id)
                taken += 1

    # 2) Fill to target_n with stratified by score band
    remaining = [e for e in evaluations if e.eval_id not in used_ids]
    by_band_rem: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for e in remaining:
        by_band_rem[score_band(e.overall_score)].append(e)
    need = target_n - len(chosen)
    for band in ("low", "medium", "high"):
        for e in by_band_rem.get(band, [])[: max(1, need // 3)]:
            if len(chosen) >= target_n:
                break
            chosen.append(e)
            used_ids.add(e.eval_id)
    for e in remaining:
        if len(chosen) >= target_n:
            break
        if e.eval_id not in used_ids:
            chosen.append(e)
    return chosen[:target_n]


# -----------------------------------------------------------------------------
# Drift measurement and alerting
# -----------------------------------------------------------------------------


def compute_drift(
    llm_dimensions: dict[str, dict[str, float]],
    human_dimensions: dict[str, dict[str, float]],
    overall_llm: dict[str, float],
    overall_human: dict[str, float],
    previous_week_scc: float | None = None,
) -> DriftResult:
    """
    Compute SCC per dimension and overall; determine status (ok / alert / hold);
    add trend alert if previous_week_scc is provided and current min dropped > TREND_DECLINE_THRESHOLD.
    """
    raw = calibration_drift_check(
        llm_dimensions,
        human_dimensions,
        overall_llm,
        overall_human,
        threshold_alert=SCC_THRESHOLD_ALERT,
        threshold_hold=SCC_THRESHOLD_HOLD,
    )
    scc_dim = raw["scc_per_dimension"]
    scc_overall = raw["scc_overall"]
    rhos = [scc_dim[d] for d in DIMENSIONS if d in scc_dim and not math.isnan(scc_dim[d])]
    avg = sum(rhos) / len(rhos) if rhos else float("nan")
    min_rho = raw["min_rho"]
    alerts: list[str] = []

    # Per-dimension alerts
    for d in DIMENSIONS:
        r = scc_dim.get(d, float("nan"))
        if math.isnan(r):
            continue
        if r < SCC_THRESHOLD_HOLD:
            alerts.append(f"Severe drift—{d} SCC={r:.2f} (< 0.65). Revert to manual review immediately.")
        elif r < SCC_THRESHOLD_ALERT:
            alerts.append(f"Drift detected—{d} SCC={r:.2f}. Consider retuning.")

    if not math.isnan(scc_overall):
        if scc_overall < SCC_THRESHOLD_HOLD:
            alerts.append("Significant drift—overall SCC < 0.65. Revert to manual review immediately.")
        elif scc_overall < SCC_THRESHOLD_ALERT:
            alerts.append("Significant drift—overall SCC < 0.75. Retuning recommended.")

    trend_declining = False
    if previous_week_scc is not None and not math.isnan(min_rho) and not math.isnan(previous_week_scc):
        if previous_week_scc - min_rho >= TREND_DECLINE_THRESHOLD:
            trend_declining = True
            alerts.append("SCC declining week-over-week—investigate root cause.")

    return DriftResult(
        scc_per_dimension=scc_dim,
        scc_overall=scc_overall,
        average_scc=avg,
        status=raw["status"],
        message=raw["message"],
        alerts=alerts,
        trend_declining=trend_declining,
        previous_week_min_scc=previous_week_scc,
    )


# -----------------------------------------------------------------------------
# Root cause analysis
# -----------------------------------------------------------------------------


def root_cause_analysis(
    llm_dimensions: dict[str, dict[str, float]],
    human_dimensions: dict[str, dict[str, float]],
    overall_llm: dict[str, float],
    overall_human: dict[str, float],
    case_context_type: dict[str, str],
) -> RootCauseSummary:
    """
    Identify worst dimensions, systematic bias (mean LLM - human), and SCC by context_type.
    case_context_type: { case_id: context_type } for splitting.
    """
    case_ids = sorted(set(llm_dimensions) & set(human_dimensions))
    if not case_ids:
        return RootCauseSummary(
            worst_dimensions=[],
            systematic_bias=[],
            by_context_type=[],
            summary_sentence="No overlapping cases.",
        )

    scc_per_dim = spearman_per_dimension(llm_dimensions, human_dimensions)
    worst_dimensions = [(d, scc_per_dim[d][0]) for d in DIMENSIONS if d in scc_per_dim]
    worst_dimensions.sort(key=lambda x: (x[1], 0))
    worst_dimensions = [(d, r) for d, r in worst_dimensions if not math.isnan(r)]

    systematic_bias: list[tuple[str, float]] = []
    for dim in DIMENSIONS:
        diffs = [
            llm_dimensions[c].get(dim, float("nan")) - human_dimensions[c].get(dim, float("nan"))
            for c in case_ids
        ]
        valid = [x for x in diffs if not math.isnan(x)]
        if len(valid) >= 2:
            mean_d = sum(valid) / len(valid)
            if abs(mean_d) >= 0.5:
                systematic_bias.append((dim, mean_d))

    by_context_type: list[tuple[str, float, int]] = []
    by_ctx: dict[str, list[str]] = defaultdict(list)
    for c in case_ids:
        ctx = case_context_type.get(c, "unknown")
        by_ctx[ctx].append(c)
    for ctx, ids in by_ctx.items():
        if len(ids) < 3:
            continue
        llm_ctx = {c: llm_dimensions[c] for c in ids}
        human_ctx = {c: human_dimensions[c] for c in ids}
        o_llm = {c: overall_llm.get(c, 0) for c in ids}
        o_human = {c: overall_human.get(c, 0) for c in ids}
        scc_d = spearman_per_dimension(llm_ctx, human_ctx)
        rho_o, _ = spearman_overall(o_llm, o_human)
        rhos = [scc_d[d][0] for d in DIMENSIONS if d in scc_d and not math.isnan(scc_d[d][0])]
        rhos.append(rho_o)
        min_ctx = min(rhos) if rhos else float("nan")
        by_context_type.append((ctx, min_ctx, len(ids)))
    by_context_type.sort(key=lambda x: x[1])

    parts = []
    if worst_dimensions:
        low = [f"{d} (SCC={r:.2f})" for d, r in worst_dimensions if r < SCC_THRESHOLD_ALERT]
        if low:
            parts.append("Drift concentrated in " + ", ".join(low) + ".")
    if systematic_bias:
        for dim, mean_d in systematic_bias:
            direction = "underscoring" if mean_d < 0 else "overscoring"
            parts.append(f"LLM is systematically {direction} {dim} by ~{abs(mean_d):.1f} point(s).")
    if by_context_type:
        low_ctx = [f"{ctx} (SCC={r:.2f}, n={n})" for ctx, r, n in by_context_type if r < SCC_THRESHOLD_ALERT]
        if low_ctx:
            parts.append("Drift concentrated in contexts: " + "; ".join(low_ctx) + ".")
    summary_sentence = " ".join(parts) if parts else "No clear concentration identified."

    return RootCauseSummary(
        worst_dimensions=worst_dimensions,
        systematic_bias=systematic_bias,
        by_context_type=by_context_type,
        summary_sentence=summary_sentence,
    )


# -----------------------------------------------------------------------------
# Divergent cases and retuning proposals
# -----------------------------------------------------------------------------


def identify_divergent_cases(
    llm_dimensions: dict[str, dict[str, float]],
    human_dimensions: dict[str, dict[str, float]],
    case_context_type: dict[str, str],
    response_snippets: dict[str, str] | None = None,
    threshold: float = DIVERGENCE_ABS_THRESHOLD,
) -> list[DivergentCase]:
    """Cases where |LLM - human| > threshold for any dimension."""
    out: list[DivergentCase] = []
    response_snippets = response_snippets or {}
    for c in set(llm_dimensions) & set(human_dimensions):
        for dim in DIMENSIONS:
            llm_s = llm_dimensions[c].get(dim, float("nan"))
            hum_s = human_dimensions[c].get(dim, float("nan"))
            if math.isnan(llm_s) or math.isnan(hum_s):
                continue
            if abs(llm_s - hum_s) > threshold:
                out.append(
                    DivergentCase(
                        case_id=c,
                        dimension=dim,
                        llm_score=llm_s,
                        human_score=hum_s,
                        context_type=case_context_type.get(c, ""),
                        response_snippet=response_snippets.get(c, "")[:200],
                    )
                )
    return out


def propose_retuning(
    systematic_bias: list[tuple[str, float]],
    worst_dimensions: list[tuple[str, float]],
    by_context_type: list[tuple[str, float, int]],
) -> list[RetuningProposal]:
    """
    Propose prompt/weight modifications from RCA.
    Returns list of proposed changes (text); no actual file edits.
    """
    proposals: list[RetuningProposal] = []
    for dim, mean_d in systematic_bias:
        if mean_d < 0:
            proposals.append(
                RetuningProposal(
                    finding=f"LLM systematically underscoring {dim} by ~{abs(mean_d):.1f}",
                    modification_type="add_specificity",
                    proposed_text=(
                        f"{dim} dimension: Look for explicit cues (e.g. 'I understand', 'that sounds difficult'). "
                        "If present, do not score below 6."
                    ),
                )
            )
            proposals.append(
                RetuningProposal(
                    finding=f"LLM underscoring {dim}",
                    modification_type="adjust_rubric",
                    proposed_text=(
                        f"Lower bound for {dim} 5-6 band is now: Must have at least one empathetic or acknowledging phrase."
                    ),
                )
            )
        else:
            proposals.append(
                RetuningProposal(
                    finding=f"LLM systematically overscoring {dim} by ~{mean_d:.1f}",
                    modification_type="adjust_rubric",
                    proposed_text=f"Tighten upper bound for {dim} 7-8 band; require clear evidence.",
                )
            )
    for ctx, scc, n in by_context_type:
        if scc < SCC_THRESHOLD_ALERT and n >= 3:
            proposals.append(
                RetuningProposal(
                    finding=f"Drift concentrated in {ctx} (SCC={scc:.2f})",
                    modification_type="context_instruction",
                    proposed_text=f"For {ctx} contexts, add explicit instruction: weight and rubric follow context_weights for {ctx}; do not penalize dimensions with weight 0.",
                )
            )
    return proposals


# -----------------------------------------------------------------------------
# Retuning validation gate
# -----------------------------------------------------------------------------


def retuning_validation_gate(
    new_scc_per_dimension: dict[str, float],
    new_scc_overall: float,
    baseline_scc_per_dimension: dict[str, float],
    baseline_scc_overall: float,
    min_scc_deploy: float = SCC_THRESHOLD_VALIDATE,
    regression_tolerance: float = REGRESSION_TOLERANCE,
) -> tuple[bool, list[str]]:
    """
    Returns (pass: bool, messages: list).
    Pass if: all dimension SCC >= min_scc_deploy, overall >= min_scc_deploy,
    and no dimension regresses by more than regression_tolerance vs baseline.
    """
    messages: list[str] = []
    for d in DIMENSIONS:
        r = new_scc_per_dimension.get(d, float("nan"))
        if math.isnan(r) or r < min_scc_deploy:
            messages.append(f"Dimension {d} SCC={r:.2f} (required >={min_scc_deploy}).")
        else:
            base = baseline_scc_per_dimension.get(d)
            if base is not None and r < base - regression_tolerance:
                messages.append(f"Regression: {d} new={r:.2f} vs baseline={base:.2f}.")
    if math.isnan(new_scc_overall) or new_scc_overall < min_scc_deploy:
        messages.append(f"Overall SCC={new_scc_overall:.2f} (required >={min_scc_deploy}).")
    else:
        if baseline_scc_overall is not None and new_scc_overall < baseline_scc_overall - regression_tolerance:
            messages.append(f"Overall regression: new={new_scc_overall:.2f} vs baseline={baseline_scc_overall:.2f}.")
    return (len(messages) == 0, messages)


# -----------------------------------------------------------------------------
# Deployment approval checks
# -----------------------------------------------------------------------------


def deployment_checks(
    scc_per_dimension: dict[str, float],
    scc_overall: float,
    min_scc: float = SCC_THRESHOLD_DEPLOY,
) -> tuple[bool, list[str]]:
    """Pre-deploy: require SCC >= 0.80 on calibration set."""
    messages: list[str] = []
    for d in DIMENSIONS:
        r = scc_per_dimension.get(d, float("nan"))
        if math.isnan(r) or r < min_scc:
            messages.append(f"{d} SCC={r:.2f} (required >={min_scc}).")
    if math.isnan(scc_overall) or scc_overall < min_scc:
        messages.append(f"Overall SCC={scc_overall:.2f} (required >={min_scc}).")
    return (len(messages) == 0, messages)


# -----------------------------------------------------------------------------
# Confidence-based fallback
# -----------------------------------------------------------------------------


def should_escalate_to_human(confidence: float, threshold: float = CONFIDENCE_ESCALATION_THRESHOLD) -> bool:
    """If confidence-based escalation is enabled: route to human when confidence < threshold."""
    return confidence < threshold


# -----------------------------------------------------------------------------
# Weekly job entrypoint (pseudocode / stub)
# -----------------------------------------------------------------------------


def run_weekly_drift_job(
    evaluations: list[EvaluationRecord],
    human_ratings: list[HumanRatingRecord],
    context_weights: dict[str, dict[str, float]],
    previous_week_min_scc: float | None = None,
) -> tuple[DriftResult, RootCauseSummary | None]:
    """
    Stub for weekly job:
    1. Stratified sample already done (evaluations = 10 sampled)
    2. human_ratings = collected from raters for those case_ids
    3. Build llm/human dimension and overall dicts; compute consensus from ratings
    4. compute_drift; if alert/hold, run root_cause_analysis
    """
    from calibration.validation import consensus_per_case, consensus_overall_from_dimensions

    case_ids = [e.case_id for e in evaluations]
    llm_dimensions = {e.case_id: e.dimensions for e in evaluations}
    overall_llm = {e.case_id: e.overall_score for e in evaluations}

    # Human consensus from ratings (median per case)
    flat = []
    for r in human_ratings:
        row = {"case_id": r.case_id, "rater_id": r.rater_id}
        row.update(r.dimensions)
        flat.append(row)
    consensus_dim = consensus_per_case(flat)
    # Overall: use default_weights from context_weights (e.g. docs/context_weights.json)
    if isinstance(context_weights, dict) and "default_weights" in context_weights:
        default_weights = context_weights["default_weights"]
    else:
        default_weights = dict(context_weights) if context_weights else {}
    overall_human = consensus_overall_from_dimensions(consensus_dim, default_weights)
    human_dimensions = {c: consensus_dim[c] for c in case_ids if c in consensus_dim}
    overall_human = {c: overall_human[c] for c in case_ids if c in overall_human}

    drift = compute_drift(
        llm_dimensions,
        human_dimensions,
        overall_llm,
        overall_human,
        previous_week_scc=previous_week_min_scc,
    )
    rca = None
    if drift.status in ("alert", "hold"):
        case_context_type = {e.case_id: e.context_type for e in evaluations}
        rca = root_cause_analysis(
            llm_dimensions,
            human_dimensions,
            overall_llm,
            overall_human,
            case_context_type,
        )
    return (drift, rca)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    # Example: dummy data
    evals = [
        EvaluationRecord("e1", "c1", "verification", "agent_a", 7.5, {"task_completion": 8, "empathy": 6, "conciseness": 7, "naturalness": 7, "safety": 8, "clarity": 7}),
        EvaluationRecord("e2", "c2", "screening", "agent_a", 5.0, {"task_completion": 6, "empathy": 4, "conciseness": 6, "naturalness": 5, "safety": 7, "clarity": 6}),
    ] * 5
    human_ratings = [
        HumanRatingRecord("c1", "r1", "verification", {"task_completion": 8, "empathy": 7, "conciseness": 7, "naturalness": 7, "safety": 8, "clarity": 7}, 7.5),
        HumanRatingRecord("c2", "r1", "screening", {"task_completion": 6, "empathy": 5, "conciseness": 6, "naturalness": 5, "safety": 7, "clarity": 6}, 5.0),
    ] * 5
    weights = {"task_completion": 0.25, "empathy": 0.15, "conciseness": 0.1, "naturalness": 0.15, "safety": 0.2, "clarity": 0.15}
    drift, rca = run_weekly_drift_job(evals[:10], human_ratings[:10], weights, previous_week_min_scc=0.82)
    print("Status:", drift.status)
    print("Message:", drift.message)
    print("Alerts:", drift.alerts)
    if rca:
        print("RCA:", rca.summary_sentence)
