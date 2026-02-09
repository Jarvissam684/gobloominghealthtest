"""
OpenAI-backed LLM judge and improvement for usecase_3.

- Judge: uses prompts/judge_prompt.md (system + user template), returns structured JSON.
- Improve: uses improve.build_improve_prompt, returns improved response text only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
JUDGE_PROMPT_PATH = PROMPTS_DIR / "judge_prompt.md"

# Default model for judge (good balance of cost and quality)
JUDGE_MODEL = "gpt-4o-mini"
IMPROVE_MODEL = "gpt-4o-mini"


def _load_judge_prompt() -> tuple[str, str]:
    """Load judge_prompt.md and split into system (up to USER MESSAGE TEMPLATE) and user template."""
    if not JUDGE_PROMPT_PATH.exists():
        raise FileNotFoundError(f"Judge prompt not found: {JUDGE_PROMPT_PATH}")
    text = JUDGE_PROMPT_PATH.read_text(encoding="utf-8")

    # Split at USER MESSAGE TEMPLATE
    marker = "## USER MESSAGE TEMPLATE"
    if marker not in text:
        # Fallback: use full doc as system, minimal user
        return text.strip(), "Evaluate this healthcare voice AI response. Output only valid JSON."

    system_part, rest = text.split(marker, 1)
    system_str = system_part.strip()

    # Extract user template from ``` block
    user_template = "Evaluate this healthcare voice AI response. Output only valid JSON."
    if "```" in rest:
        blocks = rest.split("```")
        for i, block in enumerate(blocks):
            if "context_type" in block and "{{context_type}}" in block:
                user_template = block.strip()
                if user_template.lower().startswith("evaluate"):
                    break
    return system_str, user_template


def _build_judge_user_message(
    context_type: str,
    directive: str,
    conversation_history: Any,
    user_input: str | None,
    response: str,
) -> str:
    """Fill the user message template."""
    conv = conversation_history
    if conv is not None and hasattr(conv, "model_dump"):
        conv = conv.model_dump()
    conv_str = json.dumps(conv, default=str) if conv is not None else "[]"
    return (
        f"Evaluate this healthcare voice AI response.\n\n"
        f"**context_type:** {context_type}\n\n"
        f"**directive:** {directive}\n\n"
        f"**conversation_history:**\n{conv_str}\n\n"
        f"**user_input (current turn):**\n{user_input or '(none)'}\n\n"
        f"**AI response to evaluate:**\n{response}\n\n"
        "---\n\n"
        "Apply the dimension definitions, reasoning requirements, context-weighted overall score, "
        "and flag/suggestion rules. Output only valid JSON, no markdown fences or preamble."
    )


def _normalize_llm_dimensions(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Validate and normalize dimension output from judge. All six dimensions must be present
    with numeric score; no invented values.
    """
    dim_names = (
        "task_completion",
        "empathy",
        "conciseness",
        "naturalness",
        "safety",
        "clarity",
    )
    out = {}
    for d in dim_names:
        data = raw.get(d)
        if not isinstance(data, dict):
            raise ValueError(f"Incomplete judge output: missing dimension '{d}'")
        score = data.get("score")
        if score is None or not isinstance(score, (int, float)):
            raise ValueError(f"Incomplete judge output: dimension '{d}' missing or invalid score")
        score = max(1, min(10, int(score)))
        reasoning = data.get("reasoning")
        reasoning_str = str(reasoning)[:500] if reasoning is not None else ""
        if not reasoning_str.strip():
            raise ValueError(f"Incomplete judge output: dimension '{d}' missing reasoning")
        try:
            conf = float(data.get("confidence", 0.7))
        except (TypeError, ValueError):
            conf = 0.5
        conf = max(0.1, min(1.0, conf))
        out[d] = {"score": score, "reasoning": reasoning_str, "confidence": conf}
    return out


def _normalize_flags(flags: list[Any]) -> list[dict]:
    """Map flag_type -> code, ensure code/message/severity."""
    out = []
    for f in flags if isinstance(flags, list) else []:
        if not isinstance(f, dict):
            continue
        code = f.get("flag_type") or f.get("code") or "unknown"
        out.append({
            "code": code,
            "message": str(f.get("message", f.get("reasoning", "")) or code)[:300],
            "severity": str(f.get("severity", "medium")).lower()[:20],
        })
    return out


def call_judge(
    context_type: str,
    directive: str,
    conversation_history: Any,
    user_input: str | None,
    response: str,
    weights: dict[str, float],
    model: str = JUDGE_MODEL,
) -> dict[str, Any]:
    """
    Call OpenAI to evaluate the response. Returns dict with dimensions, overall_score, flags, suggestions.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Install openai: pip install openai") from None

    client = OpenAI()
    system_str, _ = _load_judge_prompt()
    user_msg = _build_judge_user_message(
        context_type=context_type,
        directive=directive,
        conversation_history=conversation_history,
        user_input=user_input,
        response=response,
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_str},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=4000,
    )
    raw_text = (resp.choices[0].message.content or "").strip()
    if not raw_text:
        raise ValueError("Empty response from judge model")

    # Strip markdown code fence if present
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw_text = "\n".join(lines)
    data = json.loads(raw_text)

    dimensions = _normalize_llm_dimensions(data.get("dimensions", {}))
    overall = float(data.get("overall_score", 0))
    if overall <= 0:
        # Recompute from weights if LLM didn't provide
        total = 0.0
        for dim, w in weights.items():
            if dim in dimensions:
                total += dimensions[dim].get("score", 0) * w
        overall = round(total * 2) / 2.0
    flags = _normalize_flags(data.get("flags", []))
    suggestions = list(data.get("suggestions", []))[:3]
    if not isinstance(suggestions, list):
        suggestions = []
    return {
        "dimensions": dimensions,
        "overall_score": overall,
        "flags": flags,
        "suggestions": suggestions,
        "review_recommended": bool(data.get("review_recommended", False)),
    }


def call_improve(
    original_response: str,
    context_type: str,
    directive: str,
    target_dimensions: list[str],
    current_scores: dict[str, float],
    variant_style: str | None = None,
    model: str = IMPROVE_MODEL,
) -> str:
    """Call OpenAI to generate an improved response. Returns improved text only."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Install openai: pip install openai") from None

    from .improve import build_improve_prompt

    client = OpenAI()
    user_msg = build_improve_prompt(
        original_response=original_response,
        context_type=context_type,
        directive=directive,
        target_dimensions=target_dimensions,
        current_scores=current_scores,
        variant_style=variant_style,
    )
    system = (
        "You are an expert at improving healthcare voice AI responses. "
        "Keep the same intent and directive. Output only the improved response text, no explanation or JSON."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=1000,
    )
    improved = (resp.choices[0].message.content or "").strip()
    return improved[:5000] if improved else original_response
