"""
Conversation context schema: turn-taking (user/assistant), metadata, size limits.

Supports 3-20 turns and context window limits for voice AI evaluation.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator

from .validation import (
    CONTEXT_WINDOW_MAX_CHARS,
    CONVERSATION_TURNS_MAX,
    CONVERSATION_TURNS_MIN,
    TURN_CONTENT_MAX_CHARS,
    normalize_string,
    truncate_to_max,
)


class TurnRoleEnum(str, Enum):
    """Speaker role in a conversation turn."""

    USER = "user"
    ASSISTANT = "assistant"


class ConversationTurn(BaseModel):
    """
    A single turn in the conversation history.
    Immutable once stored; content length bounded for context window safety.
    """

    model_config = {"extra": "forbid", "str_strip_whitespace": True}

    role: Annotated[
        TurnRoleEnum,
        Field(description="Speaker: user or assistant"),
    ]
    content: Annotated[
        str,
        Field(min_length=1, max_length=TURN_CONTENT_MAX_CHARS, description="Turn text"),
    ]
    timestamp: Annotated[
        datetime | str | None,
        Field(default=None, description="ISO timestamp or equivalent"),
    ] = None
    agent_id: Annotated[
        str | None,
        Field(default=None, max_length=64, description="Agent identifier for this turn"),
    ] = None
    prompt_version: Annotated[
        str | None,
        Field(default=None, max_length=32, description="Prompt version for this turn"),
    ] = None

    @field_validator("content", mode="before")
    @classmethod
    def normalize_and_truncate_content(cls, v: str | None) -> str:
        if v is None:
            raise ValueError("Turn content is required")
        s = normalize_string(str(v))
        if not s:
            raise ValueError("Turn content cannot be empty after normalization")
        return truncate_to_max(s, TURN_CONTENT_MAX_CHARS)

    @field_validator("timestamp", mode="before")
    @classmethod
    def coerce_timestamp(cls, v: datetime | str | None) -> datetime | str | None:
        if v is None:
            return None
        if isinstance(v, datetime):
            return v
        s = str(v).strip()
        if not s:
            return None
        return s


class ConversationHistory(BaseModel):
    """
    Ordered list of turns (user/assistant). 3-20 turns, total length bounded.
    """

    model_config = {"extra": "forbid"}

    turns: Annotated[
        list[ConversationTurn],
        Field(
            min_length=CONVERSATION_TURNS_MIN,
            max_length=CONVERSATION_TURNS_MAX,
            description="Ordered conversation turns",
        ),
    ]

    @model_validator(mode="after")
    def check_context_window_size(self) -> "ConversationHistory":
        total = sum(len(t.content) for t in self.turns)
        if total > CONTEXT_WINDOW_MAX_CHARS:
            raise ValueError(
                f"Total conversation context ({total} chars) exceeds limit {CONTEXT_WINDOW_MAX_CHARS}"
            )
        return self
