"""
Metadata schema for LLM evaluation: agent, prompt version, model, language, context type.

Supports healthcare/social services voice AI (SDOH screening, verification).
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator

from .validation import PROMPT_VERSION_PATTERN


class AgentIdEnum(str, Enum):
    """Known agent identifiers for voice AI in healthcare/social services."""

    SURVEY_AGENT = "survey_agent"
    SUPPORT_AGENT = "support_agent"
    VERIFICATION_AGENT = "verification_agent"
    SCREENING_AGENT = "screening_agent"
    CLARIFICATION_AGENT = "clarification_agent"
    TRIAGE_AGENT = "triage_agent"


class ContextTypeEnum(str, Enum):
    """Type of conversation context for evaluation."""

    VERIFICATION = "verification"
    SCREENING = "screening"
    CLARIFICATION = "clarification"
    TRIAGE = "triage"
    FOLLOW_UP = "follow_up"
    INTAKE = "intake"
    REFERRAL = "referral"


class EvaluationMetadata(BaseModel):
    """
    Metadata for a single evaluation request.
    All fields optional at input; downstream may require agent_id and context_type.
    """

    model_config = {"extra": "forbid", "str_strip_whitespace": True}

    agent_id: Annotated[
        AgentIdEnum | None,
        Field(default=None, description="Agent role: survey_agent, support_agent, etc."),
    ] = None
    prompt_version: Annotated[
        str | None,
        Field(default=None, description="Semantic version e.g. v2.1, v3.0"),
    ] = None
    model: Annotated[
        str | None,
        Field(default=None, max_length=128, description="Model name e.g. gpt-4o-mini, claude-opus"),
    ] = None
    language_detected: Annotated[
        str | None,
        Field(default=None, max_length=32, description="ISO 639-1 or 'auto' for auto-detect"),
    ] = None
    context_type: Annotated[
        ContextTypeEnum | None,
        Field(default=None, description="Conversation type: verification, screening, etc."),
    ] = None

    @field_validator("prompt_version")
    @classmethod
    def validate_prompt_version(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return None
        normalized = v.strip().lower()
        if not normalized:
            return None
        # Allow optional leading 'v'
        if normalized.startswith("v"):
            normalized = normalized[1:]
        if not PROMPT_VERSION_PATTERN.match(normalized):
            raise ValueError("prompt_version must be semantic style: e.g. 2.1, v3.0")
        return f"v{normalized}" if not v.strip().lower().startswith("v") else v.strip()

    @field_validator("model")
    @classmethod
    def validate_model_not_empty(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            return None
        return v

    @field_validator("language_detected")
    @classmethod
    def validate_language(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            return None
        return v
