"""
Calibration validation: Spearman correlation, Krippendorff's alpha, human consensus, drift detection.

Use for:
- Comparing LLM judge scores to human consensus (Spearman SCC; target > 0.80).
- Inter-rater agreement among human raters (Krippendorff's alpha; target > 0.70).
- Weekly drift checks (SCC < 0.75 alert, < 0.65 hold).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

# Optional: scipy for Spearman; numpy for arrays
try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None

DIMENSIONS = (
    "task_completion",
    "empathy",
    "conciseness",
    "naturalness",
    "safety",
    "clarity",
)


# -----------------------------------------------------------------------------
# Spearman rank correlation (SCC)
# -----------------------------------------------------------------------------


def spearman_scc(x: list[float], y: list[float]) -> tuple[float, float]:
    """
    Spearman rank correlation coefficient between two score vectors.

    Formula: ρ = 1 - (6 Σ d_i²) / (n(n²-1)), where d_i = rank(x_i) - rank(y_i).
    Returns (rho, p_value). Use rho for calibration and drift; p_value for significance.

    Target: SCC > 0.80 per dimension and overall for LLM vs human consensus.
    """
    n = len(x)
    if n != len(y) or n < 2:
        return (float("nan"), 1.0)

    if spearmanr is not None:
        rho, p = spearmanr(x, y)
        return (float(rho), float(p))

    # Fallback: manual rank correlation (handles ties by average rank)
    def rank_vector(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        ranks = [0.0] * len(v)
        for r, i in enumerate(order):
            ranks[i] = r + 1
        # Simple tie handling: same value => same average rank
        seen: dict[tuple, list[int]] = defaultdict(list)
        for i, val in enumerate(v):
            key = (val,)
            seen[key].append(i)
        for indices in seen.values():
            if len(indices) > 1:
                avg = sum(ranks[i] for i in indices) / len(indices)
                for i in indices:
                    ranks[i] = avg
        return ranks

    rx = rank_vector(x)
    ry = rank_vector(y)
    d_sq = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    rho = 1.0 - (6.0 * d_sq) / (n * (n * n - 1))
    # Approximate p-value not computed here; use 0.0 as placeholder
    return (rho, 0.0)


def spearman_per_dimension(
    llm_scores: dict[str, dict[str, float]],
    human_consensus: dict[str, dict[str, float]],
    dimensions: tuple[str, ...] = DIMENSIONS,
) -> dict[str, tuple[float, float]]:
    """
    SCC per dimension between LLM and human consensus scores.

    llm_scores:   { case_id: { task_completion: 9, empathy: 7, ... } }
    human_consensus: { case_id: { task_completion: 9, empathy: 8, ... } }
    Returns: { dimension: (rho, p_value) }
    """
    case_ids = sorted(set(llm_scores) & set(human_consensus))
    result: dict[str, tuple[float, float]] = {}
    for dim in dimensions:
        x = [llm_scores[c].get(dim, float("nan")) for c in case_ids]
        y = [human_consensus[c].get(dim, float("nan")) for c in case_ids]
        valid = [(a, b) for a, b in zip(x, y) if not (math.isnan(a) or math.isnan(b))]
        if len(valid) < 2:
            result[dim] = (float("nan"), 1.0)
            continue
        x_clean, y_clean = zip(*valid)
        result[dim] = spearman_scc(list(x_clean), list(y_clean))
    return result


def spearman_overall(
    llm_scores: dict[str, float],
    human_consensus: dict[str, float],
) -> tuple[float, float]:
    """SCC for overall_score only. llm_scores / human_consensus: { case_id: overall_score }."""
    case_ids = sorted(set(llm_scores) & set(human_consensus))
    x = [llm_scores[c] for c in case_ids]
    y = [human_consensus[c] for c in case_ids]
    return spearman_scc(x, y)


# -----------------------------------------------------------------------------
# Human consensus (majority / median)
# -----------------------------------------------------------------------------


def consensus_per_case(
    ratings: list[dict[str, Any]],
    case_id_field: str = "case_id",
    dimensions: tuple[str, ...] = DIMENSIONS,
) -> dict[str, dict[str, float]]:
    """
    Compute consensus scores per case from 3 raters.

    ratings: list of { case_id, rater_id, task_completion, empathy, ... }
    Returns: { case_id: { task_completion: 9, empathy: 7, ... } } using median per dimension.
    """
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in ratings:
        by_case[r[case_id_field]].append(r)

    out: dict[str, dict[str, float]] = {}
    for cid, ra in by_case.items():
        out[cid] = {}
        for dim in dimensions:
            vals = [r[dim] for r in ra if dim in r and r[dim] is not None]
            if not vals:
                continue
            vals.sort()
            n = len(vals)
            if n % 2 == 1:
                out[cid][dim] = float(vals[n // 2])
            else:
                out[cid][dim] = (vals[n // 2 - 1] + vals[n // 2]) / 2.0
    return out


def consensus_overall_from_dimensions(
    consensus_dimensions: dict[str, dict[str, float]],
    context_weights: dict[str, float],
) -> dict[str, float]:
    """
    Overall score = weighted sum of dimension scores per case.
    consensus_dimensions: { case_id: { task_completion: 9, ... } }
    context_weights: { task_completion: 0.25, empathy: 0.15, ... } (sum 1.0)
    """
    out: dict[str, float] = {}
    for cid, dims in consensus_dimensions.items():
        total = 0.0
        for dim, w in context_weights.items():
            if dim in dims:
                total += dims[dim] * w
        out[cid] = round(total * 2) / 2.0  # round to 0.5
    return out


# -----------------------------------------------------------------------------
# Krippendorff's alpha (ordinal)
# -----------------------------------------------------------------------------


def krippendorff_alpha_ordinal(
    ratings: list[dict[str, Any]],
    case_id_field: str = "case_id",
    rater_id_field: str = "rater_id",
    dimension: str = "task_completion",
) -> float:
    """
    Krippendorff's alpha for ordinal data (dimension scores 1-10).

    ratings: list of { case_id, rater_id, task_completion, ... }
    Returns alpha in [0, 1]. Target > 0.70 per dimension.

    Formula (ordinal): α = 1 - D_o / D_e, where
    D_o = observed disagreement (average squared distance between pairs of values per unit),
    D_e = expected disagreement by chance.
    Ordinal distance: (x-y)^2 for ranks.
    """
    # Build matrix: units (cases) x observers (raters); value = score
    by_case: dict[str, dict[str, int | float]] = defaultdict(dict)
    for r in ratings:
        cid = r[case_id_field]
        rid = r[rater_id_field]
        val = r.get(dimension)
        if val is not None:
            by_case[cid][rid] = float(val)

    units = list(by_case.keys())
    n_units = len(units)
    if n_units < 2:
        return float("nan")

    # Observed disagreement: for each unit, average pairwise squared difference
    def ordinal_distance(a: float, b: float) -> float:
        return (a - b) ** 2

    D_o = 0.0
    n_pairs_o = 0
    for cid in units:
        vals = list(by_case[cid].values())
        if len(vals) < 2:
            continue
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                D_o += ordinal_distance(vals[i], vals[j])
                n_pairs_o += 1
    if n_pairs_o > 0:
        D_o /= n_pairs_o

    # Expected disagreement: distribution of values across all ratings
    all_vals = []
    for cid in units:
        all_vals.extend(by_case[cid].values())
    if len(all_vals) < 2:
        return float("nan")
    n_v = len(all_vals)
    D_e = 0.0
    for i in range(n_v):
        for j in range(n_v):
            if i != j:
                D_e += ordinal_distance(all_vals[i], all_vals[j])
    D_e /= (n_v * (n_v - 1))

    if D_e <= 0:
        return 1.0
    return float(1.0 - D_o / D_e)


def krippendorff_per_dimension(
    ratings: list[dict[str, Any]],
    dimensions: tuple[str, ...] = DIMENSIONS,
) -> dict[str, float]:
    """Alpha per dimension. ratings: list of rater annotations."""
    return {dim: krippendorff_alpha_ordinal(ratings, dimension=dim) for dim in dimensions}


# -----------------------------------------------------------------------------
# Drift detection
# -----------------------------------------------------------------------------


def drift_check(
    llm_scores: dict[str, dict[str, float]],
    human_scores: dict[str, dict[str, float]],
    overall_llm: dict[str, float],
    overall_human: dict[str, float],
    threshold_alert: float = 0.75,
    threshold_hold: float = 0.65,
) -> dict[str, Any]:
    """
    Compare LLM vs human on a sample (e.g. weekly 10). Return status and SCCs.

    Returns:
      - scc_per_dimension: { dim: rho }
      - scc_overall: rho
      - status: "ok" | "alert" | "hold"
      - message: human-readable
    """
    scc_dim = spearman_per_dimension(llm_scores, human_scores)
    rho_overall, _ = spearman_overall(overall_llm, overall_human)

    rhos = [scc_dim[d][0] for d in DIMENSIONS if not math.isnan(scc_dim[d][0])]
    rhos.append(rho_overall)
    min_rho = min(rhos) if rhos else float("nan")

    if math.isnan(min_rho) or min_rho >= threshold_alert:
        status = "ok"
        message = "SCC above threshold; no action."
    elif min_rho >= threshold_hold:
        status = "alert"
        message = "SCC below 0.75; initiate retuning."
    else:
        status = "hold"
        message = "SCC below 0.65; hold new evaluations; revert to manual review."

    return {
        "scc_per_dimension": {d: scc_dim[d][0] for d in DIMENSIONS},
        "scc_overall": rho_overall,
        "status": status,
        "message": message,
        "min_rho": min_rho,
    }


# -----------------------------------------------------------------------------
# SQL pseudocode (for reference; not executed)
# -----------------------------------------------------------------------------
"""
-- Weekly stratified sample of 10 evaluations for drift
WITH recent AS (
  SELECT id, case_id, context_type, agent_type, evaluated_at,
         llm_task_completion, llm_empathy, llm_conciseness, llm_naturalness, llm_safety, llm_clarity, llm_overall_score
  FROM evaluations
  WHERE evaluated_at >= CURRENT_DATE - INTERVAL '7 days'
),
per_context AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY context_type ORDER BY RANDOM()) AS rn
  FROM recent
)
SELECT * FROM per_context WHERE rn <= 2
UNION ALL
(SELECT * FROM recent ORDER BY RANDOM() LIMIT 2);

-- Low-confidence cases for human review (confidence-based sampling)
SELECT id, case_id, dimension, confidence, evaluated_at
FROM evaluation_dimension_scores
WHERE confidence < 0.6 AND evaluated_at >= CURRENT_DATE - INTERVAL '7 days'
ORDER BY confidence ASC;
"""


# -----------------------------------------------------------------------------
# Example: load calibration dataset and run validation
# -----------------------------------------------------------------------------


def load_calibration_cases(path: Path | None = None) -> list[dict[str, Any]]:
    """Load cases from calibration_dataset.json."""
    if path is None:
        path = Path(__file__).parent / "calibration_dataset.json"
    with open(path) as f:
        data = json.load(f)
    return data.get("cases", [])


if __name__ == "__main__":
    # Example: Spearman on dummy data
    llm = {"c1": {"task_completion": 9, "empathy": 7}, "c2": {"task_completion": 6, "empathy": 4}}
    human = {"c1": {"task_completion": 9, "empathy": 8}, "c2": {"task_completion": 6, "empathy": 5}}
    r = spearman_per_dimension(llm, human, dimensions=("task_completion", "empathy"))
    print("Spearman per dimension:", r)
    r2, p = spearman_overall({"c1": 8.0, "c2": 5.0}, {"c1": 8.5, "c2": 5.5})
    print("Spearman overall:", r2, "p:", p)
    # Krippendorff: 3 raters, 2 cases
    ratings = [
        {"case_id": "c1", "rater_id": "r1", "task_completion": 9, "empathy": 7},
        {"case_id": "c1", "rater_id": "r2", "task_completion": 9, "empathy": 8},
        {"case_id": "c1", "rater_id": "r3", "task_completion": 8, "empathy": 7},
        {"case_id": "c2", "rater_id": "r1", "task_completion": 6, "empathy": 4},
        {"case_id": "c2", "rater_id": "r2", "task_completion": 6, "empathy": 5},
        {"case_id": "c2", "rater_id": "r3", "task_completion": 7, "empathy": 4},
    ]
    alpha = krippendorff_per_dimension(ratings, dimensions=("task_completion", "empathy"))
    print("Krippendorff alpha:", alpha)
