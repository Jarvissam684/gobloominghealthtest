"""
Shared validation constants and rules for the LLM Response Quality Evaluator.

Used across input, output, and metadata schemas. Defensive: assume bad data.
"""

from __future__ import annotations

import re
from typing import Any

# --- Score bounds (integers only, no 0 or 11) ---
SCORE_MIN: int = 1
SCORE_MAX: int = 10

# --- Reasoning length (not too short, not essays) ---
REASONING_MIN_CHARS: int = 15
REASONING_MAX_CHARS: int = 200

# --- Confidence (0-1 inclusive) ---
CONFIDENCE_MIN: float = 0.0
CONFIDENCE_MAX: float = 1.0

# --- Input text limits ---
RESPONSE_MIN_CHARS: int = 1
RESPONSE_MAX_CHARS: int = 5000
TURN_CONTENT_MAX_CHARS: int = 2000
CONVERSATION_TURNS_MIN: int = 3
CONVERSATION_TURNS_MAX: int = 20
CONTEXT_WINDOW_MAX_CHARS: int = 15000  # Total conversation context size
SUGGESTIONS_MAX: int = 3

# --- Semantic versioning for prompt_version (e.g. v2.1, v3.0) ---
PROMPT_VERSION_PATTERN: re.Pattern[str] = re.compile(
    r"^v?(0|[1-9]\d*)\.(0|[1-9]\d*)(?:\.(0|[1-9]\d*))?$",
    re.IGNORECASE,
)

# --- Low-content / ambiguous patterns (allowed but may be flagged) ---
LOW_CONTENT_PATTERNS: tuple[str, ...] = (
    "...",
    "..",
    ".",
    "—",
    "-",
    "n/a",
    "na",
    "none",
    "idk",
    "unknown",
)


def normalize_string(value: str | None) -> str:
    """
    Strip and collapse whitespace. Never return None; empty becomes ''.
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())


def is_low_content(text: str) -> bool:
    """
    True if the text is effectively empty or a known low-content token.
    Used for edge cases like response = "...".
    """
    normalized = normalize_string(text).lower()
    if not normalized:
        return True
    if normalized in LOW_CONTENT_PATTERNS:
        return True
    if len(normalized) <= 2 and all(c in ".-,_ " for c in normalized):
        return True
    return False


def truncate_to_max(value: str, max_chars: int) -> str:
    """Safely truncate string to max_chars (no mid-character cut)."""
    if not value or len(value) <= max_chars:
        return value
    return value[:max_chars].rsplit(maxsplit=1)[0] or value[:max_chars]
