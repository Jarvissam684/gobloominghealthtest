"""
API-specific schemas: request with optional directive, result with eval_id/source/cached_at.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, Field

# Re-use Phase 1.1 request; we extend with optional directive at route level
from schemas import EvaluateRequest

# Re-export for OpenAPI
__all__ = [
    "EvaluateRequest",
    "EvaluateRequestBody",
    "EvaluationResult",
    "DimensionScoreResult",
    "FlagResult",
    "CompareContext",
    "CompareRequest",
    "DimensionComparison",
    "ComparisonResult",
    "CompareBatchRequest",
    "CompareBatchResult",
    "BatchEvaluateRequest",
    "BatchEvaluationResult",
    "DimensionStats",
    "AggregateStats",
    "FlagsSummary",
    "QualityDistribution",
    "AnomalyRecord",
    "BatchMetadata",
    "ImproveRequest",
    "ImprovementResult",
    "ChangeRecord",
    "ImprovementVariant",
]


class EvaluateRequestBody(EvaluateRequest):
    """Evaluation request plus optional directive for the judge prompt."""

    directive: Annotated[
        str | None,
        Field(default=None, max_length=500, description="E.g. 'Collect whether the member is employed (yes/no).'"),
    ] = None


class DimensionScoreResult(BaseModel):
    """One dimension in the API response (no validators that depend on default weights)."""

    score: int
    reasoning: str
    confidence: float


class FlagResult(BaseModel):
    """One flag in the API response."""

    code: str
    message: str
    severity: str


class EvaluationResult(BaseModel):
    """
    API response: scores, flags, suggestions, plus eval_id and optional cache metadata.
    overall_score is computed server-side using context weights.
    """

    eval_id: str
    source: Annotated[str, Field(default="live", description="live | cache")] = "live"
    cached_at: Annotated[str | None, Field(default=None, description="ISO datetime when source=cache")] = None
    dimensions: dict[str, DimensionScoreResult]
    overall_score: float
    flags: list[FlagResult] = []
    suggestions: list[str] = []


# --- Compare (A/B) ---


class CompareContext(BaseModel):
    """Shared context for both responses (same as evaluate minus response)."""

    model_config = {"extra": "forbid"}

    directive: str | None = None
    conversation_history: Any = None  # ConversationHistory | None
    metadata: Any = None  # EvaluationMetadata | None
    user_input: str | None = None


class CompareRequest(BaseModel):
    """Input for POST /api/compare: same context, two responses."""

    model_config = {"extra": "forbid"}

    context: CompareContext | None = None
    response_a: Annotated[str, Field(min_length=1, max_length=5000)]
    response_b: Annotated[str, Field(min_length=1, max_length=5000)]


class DimensionComparison(BaseModel):
    """Per-dimension comparison: winner, scores, reasoning, example quote."""

    dimension: str
    winner: Annotated[str, Field(description="a | b | tie")]
    score_a: int
    score_b: int
    reasoning: str
    example: Annotated[str | None, Field(default=None, description="Quote from winning response")] = None


class ComparisonResult(BaseModel):
    """Output of POST /api/compare: winner, per-dimension comparison, recommendation."""

    compare_id: str
    winner: Annotated[str, Field(description="a | b | tie")]
    overall_winner: Annotated[str, Field(description="a | b | tie (context-weighted)")]
    dimension_comparisons: list[DimensionComparison]
    recommendation: str
    confidence_in_winner: Annotated[float, Field(ge=0, le=1)]
    recommend_human_review: bool = False
    eval_id_a: str = ""
    eval_id_b: str = ""


class CompareBatchRequest(BaseModel):
    """Batch compare: same variant pair across multiple contexts."""

    comparisons: list[CompareRequest]


class CompareBatchResult(BaseModel):
    """Aggregate of multiple comparisons."""

    results: list[ComparisonResult]
    aggregate: dict[str, Any]  # a_wins, b_wins, ties, total, overall_recommendation


# --- Batch evaluate ---

BATCH_MAX_SIZE = 500


class BatchEvaluateRequest(BaseModel):
    """Input for POST /api/evaluate/batch. Max 500 evaluations per request."""

    model_config = {"extra": "forbid"}

    evaluations: Annotated[
        list[EvaluateRequestBody],
        Field(min_length=1, max_length=BATCH_MAX_SIZE),
    ]
    batch_id: Annotated[str | None, Field(default=None, max_length=128)] = None


class DimensionStats(BaseModel):
    """Per-dimension aggregate: mean, std, min, p25, median, p75, max."""

    mean: float
    std: float
    min: float
    p25: float
    median: float
    p75: float
    max: float


class AggregateStats(BaseModel):
    """Aggregate statistics: global + per dimension; optional slices by agent_id, prompt_version, context_type."""

    dimensions: dict[str, DimensionStats]  # task_completion, empathy, ...
    overall_score: DimensionStats
    by_agent_id: dict[str, dict[str, Any]] = {}  # agent_id -> { dimensions, overall_score }
    by_prompt_version: dict[str, dict[str, Any]] = {}
    by_context_type: dict[str, dict[str, Any]] = {}


class FlagsSummary(BaseModel):
    """Count of auto-flagged responses and top flag types."""

    total_flagged: int
    by_code: dict[str, int]
    by_severity: dict[str, int]


class QualityDistribution(BaseModel):
    """Histogram and red/green zones."""

    histogram: dict[str, int]  # "1-2", "3-4", "5-6", "7-8", "9-10" -> count
    red_zone_count: int  # overall_score < 5
    green_zone_count: int  # overall_score >= 8
    red_zone_pct: float
    green_zone_pct: float


class AnomalyRecord(BaseModel):
    """Single anomaly for manual review."""

    index: int  # index in evaluations array
    eval_id: str
    anomaly_type: Annotated[str, Field(description="outlier_score | low_confidence | ...")]
    dimension: str | None = None
    value: float | None = None
    message: str


class BatchMetadata(BaseModel):
    """Batch run metadata."""

    batch_id: str | None = None
    timestamp: str  # ISO
    total_evaluated: int
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    total_cost_usd: float
    avg_latency_ms: float
    warnings: list[str] = []
    errors: list[str] = []


class BatchEvaluationResult(BaseModel):
    """Output of POST /api/evaluate/batch."""

    individual_scores: list[EvaluationResult]
    aggregate_stats: AggregateStats
    flags_summary: FlagsSummary
    quality_distribution: QualityDistribution
    anomalies: list[AnomalyRecord] = []
    metadata: BatchMetadata


# --- Improve ---

VALID_IMPROVE_DIMENSIONS = (
    "task_completion",
    "empathy",
    "conciseness",
    "naturalness",
    "safety",
    "clarity",
)


class ImproveRequest(BaseModel):
    """Input for POST /api/improve: context + original response + optional target dimensions."""

    model_config = {"extra": "forbid"}

    context: CompareContext | None = None
    response: Annotated[str, Field(min_length=1, max_length=5000)]
    target_dimensions: Annotated[
        list[str] | None,
        Field(default=None, max_length=6, description="e.g. ['empathy', 'conciseness']; if omitted, focus on lowest-scoring"),
    ] = None
    num_variants: Annotated[int, Field(default=1, ge=1, le=3)] = 1


class ChangeRecord(BaseModel):
    """One categorized change: original vs improved snippet and how it drives improvement."""

    category: Annotated[str, Field(description="lexical | structural | additive | subtractive")]
    dimension: Annotated[str, Field(description="Which dimension this change improves")]
    original_text: str | None = None
    improved_text: str | None = None
    description: str


class ImprovementVariant(BaseModel):
    """One improvement variant (e.g. empathy-focused vs conciseness-focused)."""

    variant_id: str
    style: Annotated[str, Field(description="e.g. empathy_focused | conciseness_focused | naturalness_focused")]
    improved_response: str
    predicted_scores: dict[str, float] = {}
    changes_made: list[ChangeRecord] = []
    confidence_in_improvement: float = 0.0


class ImprovementResult(BaseModel):
    """Output of POST /api/improve."""

    original_response: str
    original_scores: dict[str, Any] = {}  # dimension -> score and optionally reasoning
    improved_response: str
    predicted_scores: dict[str, Any] = {}
    changes_made: list[ChangeRecord] = []
    confidence_in_improvement: Annotated[float, Field(ge=0, le=1)]
    recommend_human_review: bool = False
    variants: list[ImprovementVariant] = []
    eval_id_original: str = ""
    eval_id_improved: str = ""
