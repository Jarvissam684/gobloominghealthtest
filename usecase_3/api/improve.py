"""
Improvement suggestion: generate improved response, score prediction, change tracking.

- Categorize changes: lexical, structural, additive, subtractive.
- Confidence: (predicted - original) / (target - original); recommend human review if < 50%.
- Supports chaining: improved response can be fed to another /api/improve call.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from .schemas import ChangeRecord

# Minimum improvement to consider "meaningful" (points on a dimension)
MEANINGFUL_IMPROVEMENT_THRESHOLD = 0.5
CONFIDENCE_HUMAN_REVIEW_THRESHOLD = 0.5
CONFIDENCE_ITERATION_STOP = 0.4


def _words(s: str) -> list[str]:
    return s.split()


def _normalize(s: str) -> str:
    return " ".join(s.strip().split())


def categorize_changes(
    original: str,
    improved: str,
    target_dimensions: list[str],
) -> list[ChangeRecord]:
    """
    Diff original vs improved and categorize each change as lexical, structural, additive, or subtractive.
    Returns up to top 3 changes that drive improvement (by impact / dimension).
    """
    original = _normalize(original)
    improved = _normalize(improved)
    if original == improved:
        return []

    changes: list[ChangeRecord] = []
    # Use SequenceMatcher to find opcodes (replace, insert, delete)
    matcher = SequenceMatcher(None, original, improved)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        orig_slice = original[i1:i2].strip()
        impr_slice = improved[j1:j2].strip()
        if tag == "replace":
            if orig_slice and impr_slice:
                # Lexical: same length/word count, different words
                w_orig = len(_words(orig_slice))
                w_impr = len(_words(impr_slice))
                if abs(w_orig - w_impr) <= 1 and w_orig <= 5:
                    category = "lexical"
                else:
                    category = "lexical" if w_orig == w_impr else "structural"
                dim = target_dimensions[0] if target_dimensions else "naturalness"
                changes.append(
                    ChangeRecord(
                        category=category,
                        dimension=dim,
                        original_text=orig_slice[:200],
                        improved_text=impr_slice[:200],
                        description=f"Replaced '{orig_slice[:50]}...' with '{impr_slice[:50]}...'",
                    )
                )
        elif tag == "insert":
            category = "additive"
            dim = "empathy" if target_dimensions and "empathy" in target_dimensions else (target_dimensions[0] if target_dimensions else "naturalness")
            changes.append(
                ChangeRecord(
                    category=category,
                    dimension=dim,
                    original_text=None,
                    improved_text=impr_slice[:200],
                    description=f"Added: '{impr_slice[:80]}...'",
                )
            )
        elif tag == "delete":
            category = "subtractive"
            dim = "conciseness" if target_dimensions and "conciseness" in target_dimensions else (target_dimensions[0] if target_dimensions else "conciseness")
            changes.append(
                ChangeRecord(
                    category=category,
                    dimension=dim,
                    original_text=orig_slice[:200],
                    improved_text=None,
                    description=f"Removed: '{orig_slice[:80]}...'",
                )
            )

    # Return top 3 by relevance (prefer target dimensions)
    def score_change(c: ChangeRecord) -> int:
        s = 0
        if c.dimension in target_dimensions:
            s += 2
        if c.category in ("additive", "structural"):
            s += 1
        return s

    changes.sort(key=score_change, reverse=True)
    return changes[:3]


def confidence_in_improvement(
    original_score: float,
    predicted_score: float,
    target_score: float = 8.0,
) -> float:
    """
    confidence = (predicted - original) / (target - original) when target > original,
    capped to 1.0. If target <= original, return 1.0 if predicted > original else 0.0.
    """
    if predicted_score <= original_score:
        return 0.0
    gap = target_score - original_score
    if gap <= 0:
        return 1.0 if predicted_score > original_score else 0.0
    delta = predicted_score - original_score
    return min(1.0, max(0.0, delta / gap))


def is_improvement_meaningful(
    original_scores: dict[str, float],
    predicted_scores: dict[str, float],
    target_dimensions: list[str],
    threshold: float = MEANINGFUL_IMPROVEMENT_THRESHOLD,
) -> bool:
    """True if any target dimension improved by more than threshold."""
    for dim in target_dimensions:
        o = original_scores.get(dim, 0)
        p = predicted_scores.get(dim, 0)
        if p - o >= threshold:
            return True
    return False


def build_improve_prompt(
    original_response: str,
    context_type: str,
    directive: str,
    target_dimensions: list[str],
    current_scores: dict[str, float],
    variant_style: str | None = None,
) -> str:
    """Build user message for the improvement LLM from template."""
    scores_str = ", ".join(f"{d}: {s}" for d, s in current_scores.items())
    target_str = ", ".join(target_dimensions) if target_dimensions else "overall quality"
    prompt = f"""Rewrite this response to improve **{target_str}** while maintaining task completion and safety.

**Context type:** {context_type}
**Directive:** {directive}

**Current scores (for reference):**
{scores_str}

**Original response:**
{original_response}

**Constraints:**
- Keep length within ±20% of original.
- Preserve the question or ask; do not change the directive.
- Improve only the target dimension(s).

"""
    if variant_style == "empathy_focused":
        prompt += "Prioritize adding brief acknowledgment or validation before the main ask. Keep the rest unchanged where possible.\n\n"
    elif variant_style == "conciseness_focused":
        prompt += "Remove filler words and redundancy. Make the ask direct. Do not add new sentences.\n\n"
    elif variant_style == "naturalness_focused":
        prompt += "Use more conversational phrasing and contractions. Avoid formal or stiff wording. Keep the same structure.\n\n"
    prompt += "Output only the improved response, nothing else."
    return prompt
