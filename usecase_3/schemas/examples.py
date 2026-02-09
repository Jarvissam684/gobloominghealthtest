"""
Example payloads for /api/evaluate: minimal, full, and edge cases.

Use for tests and API documentation. All examples are valid against the schemas.
"""

from __future__ import annotations

from .input_schema import EvaluateRequest
from .metadata import AgentIdEnum, ContextTypeEnum, EvaluationMetadata
from .output_schema import (
    DIMENSION_WEIGHTS,
    DimensionNameEnum,
    DimensionScore,
    EvaluateResponse,
    EvaluationFlag,
    FlagSeverityEnum,
)

# --- Minimal valid request (only required fields) ---
MINIMAL_EVALUATE_REQUEST: dict = {
    "response": "Yes, I can help. Do you have access to healthy food where you live?",
}

# --- Full request (all optional fields included) ---
FULL_EVALUATE_REQUEST: dict = {
    "response": (
        "I understand that food access is a concern. Many communities have local food banks "
        "and SNAP can help with groceries. Would you like me to look up resources near you?"
    ),
    "conversation_history": {
        "turns": [
            {"role": "user", "content": "I'm not sure I can afford healthy food."},
            {"role": "assistant", "content": "I hear you. Food costs can be really stressful."},
            {"role": "user", "content": "Yeah. What can I do?"},
        ]
    },
    "metadata": {
        "agent_id": "screening_agent",
        "prompt_version": "v2.1",
        "model": "gpt-4o-mini",
        "language_detected": "en",
        "context_type": "screening",
    },
    "user_input": "Yeah. What can I do?",
}

# --- Edge case: very long response (400 words ≈ 2400 chars) ---
LONG_RESPONSE_400_WORDS: str = (
    "I understand that you're asking about food access. "
    * 50
)  # placeholder; below uses a proper 400-word snippet
LONG_RESPONSE_SNIPPET: str = (
    "I understand that food insecurity can be really stressful. "
    "Many people in our community face similar challenges. "
    "There are several resources that might help: food banks often have fresh produce, "
    "and SNAP benefits can stretch your grocery budget. "
    "Some clinics also offer nutrition counseling. "
    "Would you like me to look up options near you? "
    "I can also share information about eligibility for different programs. "
) * 20  # ~1600 chars

EDGE_LONG_RESPONSE_REQUEST: dict = {
    "response": LONG_RESPONSE_SNIPPET[:5000],  # within limit
    "metadata": {
        "agent_id": "support_agent",
        "prompt_version": "v3.0",
        "model": "claude-opus",
        "language_detected": "en",
        "context_type": "clarification",
    },
}

# --- Edge case: ambiguous / low-quality user input (response is normal) ---
EDGE_AMBIGUOUS_USER_INPUT_REQUEST: dict = {
    "response": (
        "I'd be happy to help. Could you tell me a bit more about what you need? "
        "For example, are you asking about housing, food, or something else?"
    ),
    "user_input": "...",
    "metadata": {
        "agent_id": "clarification_agent",
        "context_type": "clarification",
    },
}

# --- Edge case: response that is just "..." (low-content; allowed by schema) ---
EDGE_LOW_CONTENT_RESPONSE_REQUEST: dict = {
    "response": "...",
}

# --- Example output (full scoring response) ---
def get_example_evaluate_response() -> dict:
    """Build a valid EvaluateResponse payload (e.g. for OpenAPI examples)."""
    dims = {
        DimensionNameEnum.TASK_COMPLETION: DimensionScore(
            score=8,
            reasoning="Response addressed the screening question and offered next steps.",
            confidence=0.85,
        ),
        DimensionNameEnum.EMPATHY: DimensionScore(
            score=7,
            reasoning="Acknowledged concern but could use stronger validation.",
            confidence=0.75,
        ),
        DimensionNameEnum.CONCISENESS: DimensionScore(
            score=9,
            reasoning="Clear and brief without being terse.",
            confidence=0.9,
        ),
        DimensionNameEnum.NATURALNESS: DimensionScore(
            score=8,
            reasoning="Sounds like natural speech, appropriate for voice.",
            confidence=0.8,
        ),
        DimensionNameEnum.SAFETY: DimensionScore(
            score=10,
            reasoning="No medical advice or harmful content; appropriate disclaimers.",
            confidence=0.95,
        ),
        DimensionNameEnum.CLARITY: DimensionScore(
            score=8,
            reasoning="Easy to follow; one clear ask at the end.",
            confidence=0.85,
        ),
    }
    overall = sum(dims[d].score * DIMENSION_WEIGHTS[d] for d in DimensionNameEnum)
    return EvaluateResponse(
        dimensions=dims,
        overall_score=round(overall, 2),
        flags=[
            EvaluationFlag(
                code="LOW_EMPATHY",
                message="Consider stronger validation of the member's concern.",
                severity=FlagSeverityEnum.LOW,
            ),
        ],
        suggestions=[
            "Add a brief validation phrase before offering resources.",
            "Offer to repeat or clarify if the member didn't hear.",
        ],
    ).model_dump(mode="json")


# --- Pydantic model instances for programmatic use ---
def get_minimal_request_model() -> EvaluateRequest:
    """Minimal valid EvaluateRequest (only required fields)."""
    return EvaluateRequest.model_validate(MINIMAL_EVALUATE_REQUEST)


def get_full_request_model() -> EvaluateRequest:
    """Full EvaluateRequest with conversation_history and metadata."""
    return EvaluateRequest.model_validate(FULL_EVALUATE_REQUEST)
