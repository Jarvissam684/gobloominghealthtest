"""
LLM Response Quality Evaluator – Pydantic v2 validation schemas.

Use for POST /api/evaluate: input validation, conversation context,
metadata, and scoring output. Defensive validation; data immutable once stored.
"""

from .conversation import (
    ConversationHistory,
    ConversationTurn,
    TurnRoleEnum,
)
from .examples import (
    EDGE_AMBIGUOUS_USER_INPUT_REQUEST,
    EDGE_LONG_RESPONSE_REQUEST,
    EDGE_LOW_CONTENT_RESPONSE_REQUEST,
    FULL_EVALUATE_REQUEST,
    MINIMAL_EVALUATE_REQUEST,
    get_example_evaluate_response,
    get_full_request_model,
    get_minimal_request_model,
)
from .input_schema import EvaluateRequest
from .metadata import (
    AgentIdEnum,
    ContextTypeEnum,
    EvaluationMetadata,
)
from .output_schema import (
    DIMENSION_WEIGHTS,
    DimensionNameEnum,
    DimensionScore,
    EvaluateResponse,
    EvaluationFlag,
    FlagSeverityEnum,
)
from .validation import (
    CONFIDENCE_MAX,
    CONFIDENCE_MIN,
    CONTEXT_WINDOW_MAX_CHARS,
    CONVERSATION_TURNS_MAX,
    CONVERSATION_TURNS_MIN,
    REASONING_MAX_CHARS,
    REASONING_MIN_CHARS,
    RESPONSE_MAX_CHARS,
    RESPONSE_MIN_CHARS,
    SCORE_MAX,
    SCORE_MIN,
    SUGGESTIONS_MAX,
    is_low_content,
    normalize_string,
)

__all__ = [
    # Input
    "EvaluateRequest",
    # Output
    "EvaluateResponse",
    "DimensionScore",
    "DimensionNameEnum",
    "DIMENSION_WEIGHTS",
    "EvaluationFlag",
    "FlagSeverityEnum",
    # Conversation
    "ConversationHistory",
    "ConversationTurn",
    "TurnRoleEnum",
    # Metadata
    "EvaluationMetadata",
    "AgentIdEnum",
    "ContextTypeEnum",
    # Validation
    "SCORE_MIN",
    "SCORE_MAX",
    "REASONING_MIN_CHARS",
    "REASONING_MAX_CHARS",
    "CONFIDENCE_MIN",
    "CONFIDENCE_MAX",
    "RESPONSE_MIN_CHARS",
    "RESPONSE_MAX_CHARS",
    "CONVERSATION_TURNS_MIN",
    "CONVERSATION_TURNS_MAX",
    "CONTEXT_WINDOW_MAX_CHARS",
    "SUGGESTIONS_MAX",
    "normalize_string",
    "is_low_content",
    # Examples
    "MINIMAL_EVALUATE_REQUEST",
    "FULL_EVALUATE_REQUEST",
    "EDGE_LONG_RESPONSE_REQUEST",
    "EDGE_AMBIGUOUS_USER_INPUT_REQUEST",
    "EDGE_LOW_CONTENT_RESPONSE_REQUEST",
    "get_minimal_request_model",
    "get_full_request_model",
    "get_example_evaluate_response",
]
