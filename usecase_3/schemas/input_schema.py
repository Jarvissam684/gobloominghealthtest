"""
Input schema for POST /api/evaluate: LLM response quality evaluation.

Handles edge cases: empty strings, nulls, extremely long text (>5000 chars),
and low-content responses (e.g. "...").

Low-content handling: A response that is just "..." (or similar tokens like "..",
"n/a", "idk") is allowed by this schema (min 1 char, so "..." is valid). Use
is_low_content(response) from validation to detect such cases and flag them
in the output or apply different scoring logic.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, field_validator

from .conversation import ConversationHistory
from .metadata import EvaluationMetadata
from .validation import (
    RESPONSE_MAX_CHARS,
    RESPONSE_MIN_CHARS,
    normalize_string,
    truncate_to_max,
)


class EvaluateRequest(BaseModel):
    """
    Request body for /api/evaluate.

    Required: response (the LLM output to score).
    Optional: conversation_history, metadata, user_input (current user message).
    Data is immutable once stored; validation is defensive.
    """

    model_config = {
        "extra": "forbid",
        "str_strip_whitespace": True,
        "validate_default": True,
    }

    response: Annotated[
        str,
        Field(
            min_length=RESPONSE_MIN_CHARS,
            max_length=RESPONSE_MAX_CHARS,
            description="LLM response to evaluate (1-5000 chars). May be non-English.",
        ),
    ]
    conversation_history: Annotated[
        ConversationHistory | None,
        Field(default=None, description="3-20 turns for context; optional."),
    ] = None
    metadata: Annotated[
        EvaluationMetadata | None,
        Field(default=None, description="Agent, prompt version, model, language, context type."),
    ] = None
    user_input: Annotated[
        str | None,
        Field(default=None, max_length=2000, description="Current user message (for context)."),
    ] = None

    @field_validator("response", mode="before")
    @classmethod
    def normalize_and_bound_response(cls, v: str | None) -> str:
        """Reject null/empty; normalize whitespace; truncate if over limit."""
        if v is None:
            raise ValueError("response is required and cannot be null")
        s = normalize_string(str(v))
        if len(s) < RESPONSE_MIN_CHARS:
            raise ValueError(
                f"response must have at least {RESPONSE_MIN_CHARS} character(s) after normalization"
            )
        return truncate_to_max(s, RESPONSE_MAX_CHARS)

    @field_validator("user_input", mode="before")
    @classmethod
    def normalize_user_input(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = normalize_string(str(v))
        return s if s else None
