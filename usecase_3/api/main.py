"""
POST /api/evaluate: real-time evaluation of AI responses.

- Input: EvaluationRequest (Phase 1.1) + optional directive
- Output: EvaluationResult (scores, reasoning, flags, suggestions)
- Latency target: <5s; on LLM timeout return cached result if available
- Auth: Bearer token (service-to-service)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from concurrent.futures import TimeoutError as FuturesTimeoutError
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# FastAPI
try:
    from fastapi import FastAPI, HTTPException, Request, Depends, status
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
except ImportError:
    raise ImportError("Install fastapi: pip install fastapi uvicorn")

from pydantic import ValidationError

# Project schemas
from schemas import ContextTypeEnum, DimensionNameEnum, is_low_content
from schemas.output_schema import DIMENSION_WEIGHTS

from .compare_logic import (
    BIAS_ALERT_THRESHOLD,
    CONFIDENCE_HUMAN_REVIEW_THRESHOLD,
    DIMENSION_NAMES as COMPARE_DIMENSION_NAMES,
    aggregate_batch_with_stats,
    build_recommendation as compare_build_recommendation,
    check_bias,
    confidence_in_winner as compare_confidence_in_winner,
    dimension_winner as compare_dimension_winner,
    overall_winner as compare_overall_winner,
    weighted_diff as compare_weighted_diff,
)
from . import batch_eval
from . import improve as improve_module
from .schemas import (
    BatchEvaluateRequest,
    BatchEvaluationResult,
    CompareBatchRequest,
    CompareBatchResult,
    CompareRequest,
    ComparisonResult,
    DimensionComparison,
    DimensionScoreResult,
    EvaluateRequestBody,
    EvaluationResult,
    FlagResult,
    ImproveRequest,
    ImprovementResult,
    ImprovementVariant,
    ChangeRecord,
    VALID_IMPROVE_DIMENSIONS,
)

# -----------------------------------------------------------------------------
# Config: weights, timeouts, cache
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = BASE_DIR / "docs" / "context_weights.json"
LLM_TIMEOUT_SEC = 30.0  # OpenAI can take 10–25s; 5s was too short
POST_PROCESS_TIMEOUT_SEC = 1.0
CACHE_TTL_SEC = 24 * 3600  # 24 hours
VALID_CONTEXT_TYPES = [e.value for e in ContextTypeEnum]

# In-memory cache (key -> { result, cached_at }). Production: use Redis.
_cache: dict[str, dict[str, Any]] = {}
_cache_hits = 0
_cache_misses = 0

# Metrics for observability
_metrics = {"eval_count": 0, "error_count": 0, "timeout_count": 0, "latencies": []}
# Compare (A/B) bias detection: alert if a_wins or b_wins > 65%
_compare_metrics = {"a_wins": 0, "b_wins": 0, "ties": 0}

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Load context weights (versioned: weights_v1.0)
# -----------------------------------------------------------------------------


def load_weights(version: str = "v1.0") -> dict[str, dict[str, float]]:
    """Load dimension weights by context_type. Keys: default_weights, by_context."""
    path = WEIGHTS_PATH
    if not path.exists():
        return {"default_weights": _default_weights_dict(), "by_context": {}}
    with open(path) as f:
        data = json.load(f)
    return {
        "default_weights": data.get("default_weights", _default_weights_dict()),
        "by_context": data.get("by_context", {}),
    }


def _default_weights_dict() -> dict[str, float]:
    return {d.value: w for d, w in DIMENSION_WEIGHTS.items()}


def get_weights_for_context(context_type: str | None, weights_config: dict) -> dict[str, float]:
    """Return weight map for context_type; fallback to default_weights."""
    if context_type and context_type in weights_config.get("by_context", {}):
        return weights_config["by_context"][context_type].copy()
    return weights_config["default_weights"].copy()


# -----------------------------------------------------------------------------
# Cache key and lookup
# -----------------------------------------------------------------------------


def cache_key(conversation_history: Any, directive: str | None, response: str) -> str:
    """Stable hash for identical requests. Key: hash(conversation_history + directive + response)."""
    ch = conversation_history
    if ch is not None and hasattr(ch, "model_dump"):
        ch = ch.model_dump()
    payload = json.dumps(
        {"conversation_history": ch, "directive": directive or "", "response": response},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def cache_get(key: str) -> dict | None:
    global _cache_hits, _cache_misses
    entry = _cache.get(key)
    if entry is None:
        _cache_misses += 1
        return None
    if time.time() - entry.get("cached_at_ts", 0) > CACHE_TTL_SEC:
        _cache.pop(key, None)
        _cache_misses += 1
        return None
    _cache_hits += 1
    return entry.get("result")


def cache_set(key: str, result: dict) -> None:
    now = time.time()
    _cache[key] = {"result": result, "cached_at_ts": now, "cached_at": datetime.now(timezone.utc).isoformat()}


# -----------------------------------------------------------------------------
# LLM judge invocation (stub: replace with Claude/GPT-4 call)
# -----------------------------------------------------------------------------


def call_llm_judge(
    context_type: str,
    directive: str,
    conversation_history: Any,
    user_input: str | None,
    response: str,
    weights: dict[str, float],
) -> dict[str, Any]:
    """
    Call LLM judge (OpenAI) with judge prompt from prompts/judge_prompt.md.
    Requires OPENAI_API_KEY; no stub fallback.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail={"detail": "Evaluation service requires OPENAI_API_KEY.", "code": "configuration_required"},
        )
    from . import llm_openai
    return llm_openai.call_judge(
        context_type=context_type,
        directive=directive,
        conversation_history=conversation_history,
        user_input=user_input,
        response=response,
        weights=weights,
    )


def parse_llm_response(raw: str) -> dict[str, Any]:
    """Parse JSON from LLM; strip markdown fences if present. Raises ValueError on failure."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return json.loads(text)


# -----------------------------------------------------------------------------
# Output enrichment: overall_score from weights, auto-flags, suggestions
# -----------------------------------------------------------------------------


def compute_overall_score(dimensions: dict[str, Any], weights: dict[str, float]) -> float:
    """Overall = sum(dimension_score * weight); round to 0.5."""
    total = 0.0
    for dim, w in weights.items():
        if dim in dimensions and isinstance(dimensions[dim], dict):
            total += dimensions[dim].get("score", 0) * w
    return round(total * 2) / 2.0


def add_auto_flags(
    dimensions: dict[str, Any],
    overall_score: float,
    flags: list[dict],
    review_recommended: bool,
) -> list[dict]:
    """Append auto-flags: dimension < 4, safety < 8, review_recommended, low confidence."""
    out = list(flags)
    for dim, data in dimensions.items():
        if not isinstance(data, dict):
            continue
        score = data.get("score")
        conf = data.get("confidence", 1.0)
        if score is not None and score < 4:
            out.append({
                "code": "dimension_unacceptable",
                "message": f"{dim} score {score} below acceptable threshold.",
                "severity": "critical" if score <= 2 else "high",
            })
        if dim == "safety" and score is not None and score < 8:
            out.append({
                "code": "low_safety",
                "message": "Safety score below 8.",
                "severity": "high",
            })
        if conf < 0.6:
            out.append({
                "code": "low_confidence",
                "message": f"{dim} confidence {conf} below 0.6.",
                "severity": "medium",
            })
    if review_recommended and not any(f.get("code") == "review_recommended" for f in out):
        out.append({
            "code": "review_recommended",
            "message": "Human review recommended.",
            "severity": "low",
        })
    return out


def ensure_suggestions_if_low(overall_score: float, suggestions: list[str]) -> list[str]:
    """Return up to 3 suggestions from judge; no injected placeholder."""
    return (suggestions or [])[:3]


# -----------------------------------------------------------------------------
# Compare (A/B): uses api/compare_logic.py (COMPARE_LOGIC.md)
# -----------------------------------------------------------------------------


def _run_one_eval(body: EvaluateRequestBody) -> tuple[dict | None, str, bool]:
    """
    Run a single evaluation (same logic as /api/evaluate). Returns (result_dict, eval_id, from_cache).
    On timeout/error without cache, returns (None, eval_id, False); caller should raise 408/500.
    """
    eval_id = str(uuid.uuid4())[:12]
    weights_config = load_weights()
    req = body
    context_type = "verification"
    if req.metadata and req.metadata.context_type is not None:
        context_type = req.metadata.context_type.value
    if context_type not in VALID_CONTEXT_TYPES:
        context_type = "verification"
    weights = get_weights_for_context(context_type, weights_config)
    directive = body.directive or f"Evaluate the response for context_type={context_type}."
    ckey = cache_key(req.conversation_history, body.directive, req.response)
    cached = cache_get(ckey)
    try:
        future = ThreadPoolExecutor(max_workers=1).submit(
            call_llm_judge,
            context_type=context_type,
            directive=directive,
            conversation_history=req.conversation_history,
            user_input=req.user_input,
            response=req.response,
            weights=weights,
        )
        result = future.result(timeout=LLM_TIMEOUT_SEC)
    except FuturesTimeoutError:
        _metrics["timeout_count"] += 1
        if cached:
            cached["source"] = "cache"
            cached["cached_at"] = _cache.get(ckey, {}).get("cached_at")
            return (cached, eval_id, True)
        return (None, eval_id, False)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("LLM judge failed: %s", e)
        _metrics["error_count"] += 1
        if cached:
            cached["source"] = "cache"
            cached["cached_at"] = _cache.get(ckey, {}).get("cached_at")
            return (cached, eval_id, True)
        return (None, eval_id, False)
    overall = compute_overall_score(result.get("dimensions", {}), weights)
    result["overall_score"] = overall
    result["flags"] = add_auto_flags(
        result.get("dimensions", {}),
        overall,
        result.get("flags", []),
        result.get("review_recommended", False),
    )
    result["suggestions"] = ensure_suggestions_if_low(overall, result.get("suggestions", []))
    if is_low_content(req.response):
        result.setdefault("flags", []).append({
            "code": "low_content_response",
            "message": "Response is empty or very low content.",
            "severity": "medium",
        })
    cache_set(ckey, result)
    return (result, eval_id, False)


# -----------------------------------------------------------------------------
# Auth
# -----------------------------------------------------------------------------

security = HTTPBearer(auto_error=False)


def require_bearer(credentials: HTTPAuthorizationCredentials | None = Depends(security)) -> None:
    """Require Bearer token if EVAL_REQUIRE_AUTH=1."""
    if os.environ.get("EVAL_REQUIRE_AUTH") != "1":
        return
    if credentials is None or credentials.credentials is None:
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    # Optional: validate JWT or API key here
    return


# -----------------------------------------------------------------------------
# App and endpoint
# -----------------------------------------------------------------------------

app = FastAPI(title="LLM Response Quality Evaluator API", version="1.0.0")

# Mount under /api so route is /api/evaluate; docs at /api/docs
api_app = FastAPI(title="Evaluator API", version="1.0.0", docs_url="/docs", openapi_url="/openapi.json")


@api_app.post(
    "/evaluate",
    response_model=EvaluationResult,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Bad request (validation, unknown context_type)"},
        408: {"description": "Request timeout; may return cached result"},
        500: {"description": "Server error (e.g. parse failure)"},
    },
)
def api_evaluate(
    body: EvaluateRequestBody,
    request: Request,
    _: None = Depends(require_bearer),
):
    """
    Evaluate a single AI response. Latency <5s. On LLM timeout returns cached result if exists.
    """
    t0 = time.perf_counter()
    try:
        req = body
    except ValidationError as e:
        _metrics["error_count"] += 1
        raise HTTPException(status_code=400, detail={"detail": e.errors(), "code": "validation_error"})
    if req.metadata and req.metadata.context_type is not None and req.metadata.context_type.value not in VALID_CONTEXT_TYPES:
        _metrics["error_count"] += 1
        raise HTTPException(
            status_code=400,
            detail={"detail": f"Invalid context_type. Valid: {VALID_CONTEXT_TYPES}", "code": "unknown_context"},
        )
    result, eval_id, from_cache = _run_one_eval(body)
    if result is None:
        raise HTTPException(
            status_code=408,
            detail={"detail": "LLM evaluation timed out.", "code": "timeout"},
        )
    latency_ms = (time.perf_counter() - t0) * 1000
    _metrics["eval_count"] += 1
    _metrics["latencies"] = (_metrics["latencies"] + [latency_ms])[-100:]
    logger.info("eval_id=%s overall=%.1f latency_ms=%.0f", eval_id, result.get("overall_score"), latency_ms)
    return _to_evaluation_result(result, eval_id, from_cache)


def _to_evaluation_result(payload: dict, eval_id: str, from_cache: bool) -> EvaluationResult:
    """Build EvaluationResult from judge payload."""
    dims = payload.get("dimensions", {})
    dim_result = {}
    for k, v in dims.items():
        if isinstance(v, dict):
            dim_result[k] = DimensionScoreResult(score=v.get("score", 0), reasoning=v.get("reasoning", ""), confidence=v.get("confidence", 0.5))
    flags_result = [
        FlagResult(code=f.get("code", ""), message=f.get("message", ""), severity=f.get("severity", "medium"))
        for f in payload.get("flags", [])
    ]
    return EvaluationResult(
        eval_id=eval_id,
        source="cache" if from_cache else "live",
        cached_at=payload.get("cached_at") if from_cache else None,
        dimensions=dim_result,
        overall_score=payload.get("overall_score", 0.0),
        flags=flags_result,
        suggestions=payload.get("suggestions", [])[:3],
    )


BATCH_CONCURRENCY = 10  # up to 10 concurrent evals (rate limit friendly)


def _run_batch(
    evaluations: list[EvaluateRequestBody],
    batch_id: str | None,
) -> BatchEvaluationResult:
    """
    Run batch evaluation: prioritize by context_type then agent_id (cache efficiency),
    run up to BATCH_CONCURRENCY in parallel, then compute aggregate stats and anomalies.
    """
    t0 = time.perf_counter()
    cache_hits_before = _cache_hits
    cache_misses_before = _cache_misses

    # Group by (context_type, agent_id) for cache efficiency; process in original order for output alignment
    individual_results: list[EvaluationResult] = []
    result_dicts: list[dict] = []
    latencies_ms: list[float] = []
    errors: list[str] = []
    warnings: list[str] = []

    def run_one(i: int, body: EvaluateRequestBody) -> tuple[int, EvaluationResult | None, dict | None, float]:
        start = time.perf_counter()
        result, eval_id, from_cache = _run_one_eval(body)
        latency = (time.perf_counter() - start) * 1000
        if result is None:
            return (i, None, None, latency)
        er = _to_evaluation_result(result, eval_id, from_cache)
        meta = {}
        if body.metadata:
            if hasattr(body.metadata, "agent_id") and body.metadata.agent_id is not None:
                meta["agent_id"] = body.metadata.agent_id.value
            if hasattr(body.metadata, "prompt_version"):
                meta["prompt_version"] = body.metadata.prompt_version or "unknown"
            if hasattr(body.metadata, "context_type") and body.metadata.context_type is not None:
                meta["context_type"] = body.metadata.context_type.value
        rd = {
            "eval_id": eval_id,
            "dimensions": result.get("dimensions", {}),
            "overall_score": result.get("overall_score", 0),
            "flags": result.get("flags", []),
            "metadata": meta,
        }
        return (i, er, rd, latency)

    # Prioritize: sort by context_type then agent_id so similar requests are grouped (cache-friendly)
    def sort_key(req: EvaluateRequestBody) -> tuple[str, str]:
        ctx = "verification"
        aid = "unknown"
        if req.metadata:
            if hasattr(req.metadata, "context_type") and req.metadata.context_type is not None:
                ctx = req.metadata.context_type.value
            if hasattr(req.metadata, "agent_id") and req.metadata.agent_id is not None:
                aid = req.metadata.agent_id.value
        return (ctx, aid)

    ordered = sorted(enumerate(evaluations), key=lambda x: (sort_key(x[1]), x[0]))
    with ThreadPoolExecutor(max_workers=BATCH_CONCURRENCY) as executor:
        futures = [executor.submit(run_one, orig_i, req) for orig_i, req in ordered]
        by_idx: dict[int, tuple[EvaluationResult | None, dict | None, float]] = {}
        for fut in futures:
            try:
                i, er, rd, lat = fut.result(timeout=LLM_TIMEOUT_SEC + 2)
                by_idx[i] = (er, rd, lat)
            except Exception as e:
                errors.append(str(e))
        for idx in range(len(evaluations)):
            if idx in by_idx:
                er, rd, lat = by_idx[idx]
                if er is not None and rd is not None:
                    individual_results.append(er)
                    result_dicts.append(rd)
                    latencies_ms.append(lat)
            else:
                errors.append(f"Evaluation {idx} failed or timed out.")

    batch_cache_hits = _cache_hits - cache_hits_before
    batch_cache_misses = _cache_misses - cache_misses_before
    if len(individual_results) < len(evaluations):
        warnings.append(f"Only {len(individual_results)}/{len(evaluations)} evaluations succeeded.")

    aggregate_stats = batch_eval.compute_aggregate_stats(result_dicts)
    flags_summary = batch_eval.compute_flags_summary(result_dicts)
    quality_distribution = batch_eval.compute_quality_distribution(result_dicts)
    anomalies = batch_eval.compute_anomalies(result_dicts)
    metadata = batch_eval.build_batch_metadata(
        batch_id=batch_id,
        total_evaluated=len(individual_results),
        cache_hits=batch_cache_hits,
        cache_misses=batch_cache_misses,
        latencies_ms=latencies_ms,
        warnings=warnings if warnings else None,
        errors=errors if errors else None,
    )

    logger.info(
        "batch batch_id=%s n=%d cache_hit_rate=%.2f latency_ms=%.0f",
        batch_id or "none", len(individual_results), metadata.cache_hit_rate, metadata.avg_latency_ms,
    )
    return BatchEvaluationResult(
        individual_scores=individual_results,
        aggregate_stats=aggregate_stats,
        flags_summary=flags_summary,
        quality_distribution=quality_distribution,
        anomalies=anomalies,
        metadata=metadata,
    )


@api_app.post(
    "/evaluate/batch",
    response_model=BatchEvaluationResult,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Bad request (e.g. evaluations length > 500)"},
        408: {"description": "Timeout"},
    },
)
def api_evaluate_batch(
    body: BatchEvaluateRequest,
    _: None = Depends(require_bearer),
):
    """
    Batch evaluate up to 500 responses. Returns individual scores + aggregate statistics
    (mean, std, percentiles per dimension), by agent_id/prompt_version/context_type,
    flags summary, quality distribution, anomalies. Cost-optimized; target ~$0.015/response.
    """
    return _run_batch(body.evaluations, body.batch_id)


def _call_improvement_llm(
    original_response: str,
    context_type: str,
    directive: str,
    target_dimensions: list[str],
    current_scores: dict[str, float],
    variant_style: str | None = None,
) -> str:
    """
    Call OpenAI to rewrite response for target dimensions. Requires OPENAI_API_KEY.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail={"detail": "Improve endpoint requires OPENAI_API_KEY.", "code": "configuration_required"},
        )
    from . import llm_openai
    return llm_openai.call_improve(
        original_response=original_response,
        context_type=context_type,
        directive=directive,
        target_dimensions=target_dimensions,
        current_scores=current_scores,
        variant_style=variant_style,
    )


def _run_improve(body: ImproveRequest) -> ImprovementResult:
    """Run improvement: evaluate original, generate improved, evaluate improved, diff and confidence."""
    weights_config = load_weights()
    context_type = "verification"
    directive = "Evaluate the response."
    if body.context:
        if hasattr(body.context, "directive") and body.context.directive:
            directive = body.context.directive
        if getattr(body.context, "metadata", None):
            meta = body.context.metadata
            if isinstance(meta, dict):
                context_type = meta.get("context_type") or context_type
            elif hasattr(meta, "context_type") and meta.context_type is not None:
                context_type = meta.context_type.value

    eval_body = _build_evaluate_body_from_compare_context(body.context, body.response)
    result_orig, eval_id_orig, _ = _run_one_eval(eval_body)
    if result_orig is None:
        raise HTTPException(status_code=408, detail={"detail": "Evaluation of original response timed out.", "code": "timeout"})

    dims = result_orig.get("dimensions", {})
    original_scores = {d: (dims.get(d) or {}).get("score", 0) for d in VALID_IMPROVE_DIMENSIONS if dims.get(d)}
    original_scores["overall"] = result_orig.get("overall_score", 0)

    target_dimensions = body.target_dimensions or []
    if not target_dimensions:
        # Focus on 2 lowest-scoring dimensions
        scored = [(d, original_scores.get(d, 0)) for d in VALID_IMPROVE_DIMENSIONS if d in original_scores]
        scored.sort(key=lambda x: x[1])
        target_dimensions = [d for d, _ in scored[:2]]

    improved_response = _call_improvement_llm(
        body.response,
        context_type,
        directive,
        target_dimensions,
        original_scores,
        variant_style=None,
    )

    eval_body_improved = _build_evaluate_body_from_compare_context(body.context, improved_response)
    result_improved, eval_id_improved, _ = _run_one_eval(eval_body_improved)
    if result_improved is None:
        raise HTTPException(status_code=408, detail={"detail": "Evaluation of improved response timed out.", "code": "timeout"})

    dims_imp = result_improved.get("dimensions", {})
    predicted_scores = {d: (dims_imp.get(d) or {}).get("score", 0) for d in VALID_IMPROVE_DIMENSIONS if dims_imp.get(d)}
    predicted_scores["overall"] = result_improved.get("overall_score", 0)

    changes_made = improve_module.categorize_changes(
        body.response,
        improved_response,
        target_dimensions,
    )
    conf = improve_module.confidence_in_improvement(
        original_scores.get("overall", 0),
        predicted_scores.get("overall", 0),
        target_score=8.0,
    )
    recommend_human_review = conf < improve_module.CONFIDENCE_HUMAN_REVIEW_THRESHOLD

    variants: list[ImprovementVariant] = []
    if body.num_variants > 1:
        for i, style in enumerate(["empathy_focused", "conciseness_focused", "naturalness_focused"][: body.num_variants]):
            alt = _call_improvement_llm(body.response, context_type, directive, target_dimensions, original_scores, variant_style=style)
            alt_body = _build_evaluate_body_from_compare_context(body.context, alt)
            alt_result, _, _ = _run_one_eval(alt_body)
            alt_scores = {}
            if alt_result:
                dims_alt = alt_result.get("dimensions", {})
                alt_scores = {d: (dims_alt.get(d) or {}).get("score", 0) for d in VALID_IMPROVE_DIMENSIONS if dims_alt.get(d)}
                alt_scores["overall"] = alt_result.get("overall_score", 0)
            alt_changes = improve_module.categorize_changes(body.response, alt, target_dimensions)
            alt_conf = improve_module.confidence_in_improvement(original_scores.get("overall", 0), alt_scores.get("overall", 0), 8.0)
            variants.append(
                ImprovementVariant(
                    variant_id=f"v{i+1}",
                    style=style,
                    improved_response=alt,
                    predicted_scores=alt_scores,
                    changes_made=alt_changes,
                    confidence_in_improvement=round(alt_conf, 3),
                )
            )

    return ImprovementResult(
        original_response=body.response,
        original_scores=original_scores,
        improved_response=improved_response,
        predicted_scores=predicted_scores,
        changes_made=changes_made,
        confidence_in_improvement=round(conf, 3),
        recommend_human_review=recommend_human_review,
        variants=variants,
        eval_id_original=eval_id_orig,
        eval_id_improved=eval_id_improved,
    )


@api_app.post(
    "/improve",
    response_model=ImprovementResult,
    status_code=status.HTTP_200_OK,
    responses={400: {"description": "Bad request"}, 408: {"description": "Timeout"}, 500: {"description": "Server error"}},
)
def api_improve(body: ImproveRequest, _: None = Depends(require_bearer)):
    """
    Generate improved version of response; optional target_dimensions. Returns improved text,
    predicted scores, categorized changes, confidence. Supports 2–3 variants (empathy/conciseness/naturalness).
    """
    if body.target_dimensions:
        for d in body.target_dimensions:
            if d not in VALID_IMPROVE_DIMENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail={"detail": f"Invalid target_dimension: {d}. Valid: {list(VALID_IMPROVE_DIMENSIONS)}", "code": "validation_error"},
                )
    return _run_improve(body)


def _build_evaluate_body_from_compare_context(context: Any, response: str) -> EvaluateRequestBody:
    """Build EvaluateRequestBody from CompareRequest.context + response_a or response_b."""
    if context is None:
        return EvaluateRequestBody(response=response)
    return EvaluateRequestBody(
        response=response,
        directive=getattr(context, "directive", None),
        conversation_history=getattr(context, "conversation_history", None),
        metadata=getattr(context, "metadata", None),
        user_input=getattr(context, "user_input", None),
    )


def _do_compare(body: CompareRequest) -> ComparisonResult:
    """Run A/B comparison (used by both /compare and /compare/batch)."""
    compare_id = str(uuid.uuid4())[:12]
    t0 = time.perf_counter()
    weights_config = load_weights()
    context_type = "verification"
    if body.context and getattr(body.context, "metadata", None):
        meta = body.context.metadata
        if isinstance(meta, dict):
            context_type = meta.get("context_type") or context_type
        elif hasattr(meta, "context_type") and meta.context_type is not None:
            context_type = meta.context_type.value if hasattr(meta.context_type, "value") else str(meta.context_type)
    if context_type not in VALID_CONTEXT_TYPES:
        context_type = "verification"
    weights = get_weights_for_context(context_type, weights_config)

    try:
        body_a = _build_evaluate_body_from_compare_context(body.context, body.response_a)
        body_b = _build_evaluate_body_from_compare_context(body.context, body.response_b)
    except ValidationError as e:
        _metrics["error_count"] += 1
        raise HTTPException(status_code=400, detail={"detail": e.errors(), "code": "validation_error"})

    result_a, eval_id_a, _ = _run_one_eval(body_a)
    result_b, eval_id_b, _ = _run_one_eval(body_b)

    if result_a is None or result_b is None:
        raise HTTPException(
            status_code=408,
            detail={"detail": "LLM evaluation timed out for one or both responses.", "code": "timeout"},
        )

    dims_a = result_a.get("dimensions", {})
    dims_b = result_b.get("dimensions", {})

    # Per-dimension comparison (unbiased: |diff| > 1 => winner) — compare_logic.dimension_winner
    dimension_comparisons = []
    a_better_dims: list[str] = []
    b_better_dims: list[str] = []
    for dim in COMPARE_DIMENSION_NAMES:
        da = dims_a.get(dim) or {}
        db = dims_b.get(dim) or {}
        sa = da.get("score", 0) if isinstance(da, dict) else 0
        sb = db.get("score", 0) if isinstance(db, dict) else 0
        winner = compare_dimension_winner(sa, sb)
        if winner == "a":
            a_better_dims.append(dim)
        elif winner == "b":
            b_better_dims.append(dim)
        reasoning_a = (da.get("reasoning") or "") if isinstance(da, dict) else ""
        reasoning_b = (db.get("reasoning") or "") if isinstance(db, dict) else ""
        reasoning = f"A: {reasoning_a[:80]}... B: {reasoning_b[:80]}..." if winner == "tie" else (reasoning_a if winner == "a" else reasoning_b)
        example = None
        if winner == "a":
            example = body.response_a[:120] + "..." if len(body.response_a) > 120 else body.response_a
        elif winner == "b":
            example = body.response_b[:120] + "..." if len(body.response_b) > 120 else body.response_b
        dimension_comparisons.append(
            DimensionComparison(
                dimension=dim,
                winner=winner,
                score_a=sa,
                score_b=sb,
                reasoning=reasoning[:500],
                example=example,
            )
        )

    # Context-weighted overall winner — compare_logic.weighted_diff, overall_winner
    wdiff = compare_weighted_diff(dims_a, dims_b, weights)
    overall_winner = compare_overall_winner(wdiff)
    confidence = compare_confidence_in_winner(dims_a, dims_b, dimensions=COMPARE_DIMENSION_NAMES)
    recommend_human_review = confidence < CONFIDENCE_HUMAN_REVIEW_THRESHOLD

    # Recommendation text — compare_logic.build_recommendation
    recommendation = compare_build_recommendation(overall_winner, context_type, a_better_dims, b_better_dims)

    # Bias detection — compare_logic.check_bias; log if alert
    _compare_metrics["a_wins" if overall_winner == "a" else "b_wins" if overall_winner == "b" else "ties"] += 1
    bias_alert, a_pct, b_pct = check_bias(
        _compare_metrics["a_wins"],
        _compare_metrics["b_wins"],
        _compare_metrics["ties"],
        threshold=BIAS_ALERT_THRESHOLD,
    )
    if bias_alert:
        logger.warning(
            "compare_bias_detected compare_id=%s a_wins_pct=%.2f b_wins_pct=%.2f (alert if >65%%)",
            compare_id, a_pct, b_pct,
        )

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "compare_id=%s winner=%s confidence=%.2f latency_ms=%.0f",
        compare_id, overall_winner, confidence, latency_ms,
    )

    return ComparisonResult(
        compare_id=compare_id,
        winner=overall_winner,
        overall_winner=overall_winner,
        dimension_comparisons=dimension_comparisons,
        recommendation=recommendation,
        confidence_in_winner=confidence,
        recommend_human_review=recommend_human_review,
        eval_id_a=eval_id_a,
        eval_id_b=eval_id_b,
    )


@api_app.post(
    "/compare",
    response_model=ComparisonResult,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Bad request (validation)"},
        408: {"description": "Request timeout (evaluation of one or both responses)"},
        500: {"description": "Server error"},
    },
)
def api_compare(
    body: CompareRequest,
    _: None = Depends(require_bearer),
):
    """
    A/B compare two responses to the same context. Unbiased: both evaluated independently.
    Returns winner (a/b/tie), per-dimension comparison, recommendation. Alert if a or b wins >65% (bias).
    """
    return _do_compare(body)


@api_app.post(
    "/compare/batch",
    response_model=CompareBatchResult,
    status_code=status.HTTP_200_OK,
)
def api_compare_batch(
    body: CompareBatchRequest,
    _: None = Depends(require_bearer),
):
    """
    Run multiple A/B comparisons (same variant pair across contexts). Returns per-context
    results and aggregate (a_wins, b_wins, ties). Alert if a or b wins >65% (bias).
    """
    results = []
    for cmp_req in body.comparisons:
        res = _do_compare(cmp_req)
        results.append(res)
    # Aggregate + optional stats (p-value for "statistically better") — compare_logic.aggregate_batch_with_stats
    result_dicts = [{"winner": r.winner} for r in results]
    aggregate = aggregate_batch_with_stats(result_dicts, winner_key="winner")
    return CompareBatchResult(results=results, aggregate=aggregate)


# Mount API under /api
app.mount("/api", api_app)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/metrics")
def metrics():
    """Cache hit rate and latency (for alerts: avg latency > 4s, error rate > 5%)."""
    total = _cache_hits + _cache_misses
    hit_rate = _cache_hits / total if total else 0
    latencies = _metrics.get("latencies") or []
    avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0
    evals = _metrics.get("eval_count") or 0
    errs = _metrics.get("error_count") or 0
    error_rate = errs / evals if evals else 0
    out = {
        "cache_hit_rate": round(hit_rate, 3),
        "cache_hits": _cache_hits,
        "cache_misses": _cache_misses,
        "eval_count": evals,
        "error_count": errs,
        "error_rate": round(error_rate, 3),
        "avg_latency_ms": round(avg_latency_ms, 1),
    }
    # Compare (A/B) bias: alert if a_wins or b_wins > 65%
    ca, cb, ct = _compare_metrics["a_wins"], _compare_metrics["b_wins"], _compare_metrics["ties"]
    compare_total = ca + cb + ct
    if compare_total > 0:
        out["compare_a_wins"] = ca
        out["compare_b_wins"] = cb
        out["compare_ties"] = ct
        out["compare_a_win_pct"] = round(ca / compare_total, 3)
        out["compare_b_win_pct"] = round(cb / compare_total, 3)
        out["compare_bias_alert"] = out["compare_a_win_pct"] > 0.65 or out["compare_b_win_pct"] > 0.65
    return out
