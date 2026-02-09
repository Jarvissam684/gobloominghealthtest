"""
Output schema for LLM response quality scoring.

Dimensions: task_completion, empathy, conciseness, naturalness, safety, clarity.
Each dimension has score (1-10), reasoning (15-200 chars), confidence (0-1).
Includes overall score, flags, and up to 3 suggestions.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator

from .validation import (
    CONFIDENCE_MAX,
    CONFIDENCE_MIN,
    REASONING_MAX_CHARS,
    REASONING_MIN_CHARS,
    SCORE_MAX,
    SCORE_MIN,
    SUGGESTIONS_MAX,
    normalize_string,
    truncate_to_max,
)


class DimensionNameEnum(str, Enum):
    """Evaluation dimensions for voice AI response quality."""

    TASK_COMPLETION = "task_completion"
    EMPATHY = "empathy"
    CONCISENESS = "conciseness"
    NATURALNESS = "naturalness"
    SAFETY = "safety"
    CLARITY = "clarity"


# Default weights for overall score (must sum to 1.0)
DIMENSION_WEIGHTS: dict[DimensionNameEnum, float] = {
    DimensionNameEnum.TASK_COMPLETION: 0.25,
    DimensionNameEnum.EMPATHY: 0.15,
    DimensionNameEnum.CONCISENESS: 0.10,
    DimensionNameEnum.NATURALNESS: 0.15,
    DimensionNameEnum.SAFETY: 0.20,
    DimensionNameEnum.CLARITY: 0.15,
}


class DimensionScore(BaseModel):
    """
    Score for one dimension: integer 1-10, short reasoning, confidence 0-1.
    """

    model_config = {"extra": "forbid", "str_strip_whitespace": True}

    score: Annotated[
        int,
        Field(ge=SCORE_MIN, le=SCORE_MAX, description="Integer 1-10 only; no 0 or 11."),
    ]
    reasoning: Annotated[
        str,
        Field(
            min_length=REASONING_MIN_CHARS,
            max_length=REASONING_MAX_CHARS,
            description="Brief explanation (15-200 chars).",
        ),
    ]
    confidence: Annotated[
        float,
        Field(
            ge=CONFIDENCE_MIN,
            le=CONFIDENCE_MAX,
            description="Confidence in this dimension score (0-1).",
        ),
    ]

    @field_validator("reasoning", mode="before")
    @classmethod
    def normalize_reasoning(cls, v: str | None) -> str:
        if v is None:
            raise ValueError("reasoning is required")
        s = normalize_string(str(v))
        if len(s) < REASONING_MIN_CHARS:
            raise ValueError(f"reasoning must be {REASONING_MIN_CHARS}-{REASONING_MAX_CHARS} chars")
        return truncate_to_max(s, REASONING_MAX_CHARS)


class FlagSeverityEnum(str, Enum):
    """Severity of a quality or safety flag."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EvaluationFlag(BaseModel):
    """
    A single flag: code, message, severity.
    Triggered by safety issues, policy violations, low scores, or anomalies.
    """

    model_config = {"extra": "forbid", "str_strip_whitespace": True}

    code: Annotated[str, Field(min_length=1, max_length=64, description="Machine-readable code")]
    message: Annotated[str, Field(min_length=1, max_length=256, description="Human-readable message")]
    severity: Annotated[
        FlagSeverityEnum,
        Field(description="low | medium | high | critical"),
    ] = FlagSeverityEnum.MEDIUM


class EvaluateResponse(BaseModel):
    """
    Response from /api/evaluate: dimension scores, overall score, flags, suggestions.

    Overall score = weighted sum of dimension scores (weights in DIMENSION_WEIGHTS).
    Flags: triggered by safety, policy, or quality thresholds.
    Suggestions: max 3 per response.
    """

    model_config = {"extra": "forbid", "str_strip_whitespace": True}

    dimensions: Annotated[
        dict[DimensionNameEnum, DimensionScore],
        Field(description="Scores for each dimension (task_completion, empathy, etc.)."),
    ]
    overall_score: Annotated[
        float,
        Field(ge=0.0, le=10.0, description="Weighted aggregate of dimension scores."),
    ]
    flags: Annotated[
        list[EvaluationFlag],
        Field(default_factory=list, description="Quality or safety flags; may be empty."),
    ]
    suggestions: Annotated[
        list[str],
        Field(
            max_length=SUGGESTIONS_MAX,
            description="Up to 3 improvement suggestions per response.",
        ),
    ] = []

    @field_validator("suggestions", mode="before")
    @classmethod
    def truncate_suggestions(cls, v: list[str] | None) -> list[str]:
        if v is None:
            return []
        out = [normalize_string(s) for s in v if s and normalize_string(s)]
        return out[:SUGGESTIONS_MAX]

    @field_validator("suggestions")
    @classmethod
    def limit_suggestion_length(cls, v: list[str]) -> list[str]:
        return [s[:500] if len(s) > 500 else s for s in v]

    @model_validator(mode="after")
    def check_dimensions_complete(self) -> "EvaluateResponse":
        """All six dimensions must be present."""
        required = set(DimensionNameEnum)
        present = set(self.dimensions.keys())
        if present != required:
            missing = required - present
            extra = present - required
            msg = []
            if missing:
                msg.append(f"missing dimensions: {missing}")
            if extra:
                msg.append(f"unknown dimensions: {extra}")
            raise ValueError("; ".join(msg))
        return self

    @model_validator(mode="after")
    def check_overall_matches_weighted_sum(self) -> "EvaluateResponse":
        """Ensure overall_score is consistent with dimension weights and scores."""
        expected = 0.0
        for dim, weight in DIMENSION_WEIGHTS.items():
            expected += self.dimensions[dim].score * weight
        if abs(self.overall_score - expected) > 0.01:
            raise ValueError(
                f"overall_score {self.overall_score} does not match weighted sum ~{expected:.2f}"
            )
        return self

    @model_validator(mode="after")
    def check_confidence_not_all_mid(self) -> "EvaluateResponse":
        """Confidence scores must not all be exactly 0.5 (suggests default/no thought)."""
        confs = [d.confidence for d in self.dimensions.values()]
        if len(confs) >= 2 and all(c == 0.5 for c in confs):
            raise ValueError("Confidence scores should not all be 0.5 (suggests default/no thought)")
        return self
