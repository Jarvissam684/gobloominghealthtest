"""
Prompt Similarity Service — REST API Layer.

FastAPI app with 4 endpoints: embeddings/generate, prompts/{id}/similar,
search/semantic, analysis/duplicates. CORS enabled for localhost:3000.
Embeddings and similarity/duplicate results cached for 5 minutes.

Example curl commands:
  # Regenerate all embeddings
  curl -X POST http://localhost:8000/api/embeddings/generate -H "Content-Type: application/json" -d '{"prompt_ids": null}'

  # Regenerate specific prompts
  curl -X POST http://localhost:8000/api/embeddings/generate -H "Content-Type: application/json" -d '{"prompt_ids": ["survey.question.base"]}'

  # Get similar prompts
  curl "http://localhost:8000/api/prompts/survey.question.base/similar?limit=5&threshold=0.8"

  # Semantic search
  curl -X POST http://localhost:8000/api/search/semantic -H "Content-Type: application/json" -d '{"query": "how to handle user interruptions", "limit": 10}'

  # Get duplicate analysis
  curl "http://localhost:8000/api/analysis/duplicates?threshold=0.9&same_layer=true&tier=Tier1"
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from data_layer import PromptRecord, PromptStore
from embedding_layer import EmbeddingGenerator, PromptEmbeddingStore, PromptNormalizer
from similarity_layer import (
    DuplicateClusterer,
    MergeRecommendationBuilder,
    MetadataAwareMatcher,
    SimilarityComputer,
)

# --- Config (override via env or app state) ---
DEFAULT_PROMPTS_DB = "prompts.db"
DEFAULT_EMBEDDINGS_DB = "embeddings.db"
CONTENT_PREVIEW_MAX = 100
CACHE_TTL_SECONDS = 300  # 5 minutes
# Rate limiting: placeholder for future implementation
# RATE_LIMIT_REQUESTS = 100
# RATE_LIMIT_WINDOW = 60

# --- In-memory TTL cache ---
_cache: Dict[str, tuple[Any, float]] = {}


def _cache_get(key: str) -> Optional[Any]:
    if key not in _cache:
        return None
    val, expiry = _cache[key]
    if time.monotonic() > expiry:
        del _cache[key]
        return None
    return val


def _cache_set(key: str, value: Any, ttl: int = CACHE_TTL_SECONDS) -> None:
    _cache[key] = (value, time.monotonic() + ttl)


def _content_preview(content: str, max_len: int = CONTENT_PREVIEW_MAX) -> str:
    s = (content or "").replace("\n", " ").replace("\r", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3].rstrip() + "..."


# --- App ---
app = FastAPI(
    title="Prompt Similarity Service",
    description="REST API for prompt embeddings, similarity, semantic search, and duplicate analysis.",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response models ---
class EmbeddingsGenerateRequest(BaseModel):
    prompt_ids: Optional[List[str]] = None


class EmbeddingsGenerateResponse(BaseModel):
    status: Literal["success"] = "success"
    generated_count: int = 0
    updated_count: int = 0
    errors: List[str] = []


class SimilarResultItem(BaseModel):
    prompt_id: str
    similarity_score: float
    content_preview: str
    layer: str
    category: str


class SimilarPromptsResponse(BaseModel):
    query_prompt_id: str
    query_content_preview: str
    results: List[SimilarResultItem]
    count: int


class SemanticSearchRequest(BaseModel):
    query: str = Field(...)
    limit: int = Field(default=10, ge=1, le=100)


class SemanticSearchResultItem(BaseModel):
    prompt_id: str
    similarity_score: float
    content_preview: str
    layer: str
    category: str


class SemanticSearchResponse(BaseModel):
    query: str
    results: List[SemanticSearchResultItem]
    count: int


class DuplicateClusterItem(BaseModel):
    cluster_id: str
    tier: str
    confidence: float
    reason: str
    recommendation: str
    target_prompt_id: str
    prompts: List[Dict[str, Any]]
    merge_candidates: List[str]
    variable_summary: str


class DuplicatesResponse(BaseModel):
    duplicates: List[DuplicateClusterItem]
    total_clusters: int
    tier_breakdown: Dict[str, int]


class PipelineRunRequest(BaseModel):
    data_file: str = Field(..., description="Path to JSON/JSONL prompt file")
    db_path: Optional[str] = Field(default=None, description="SQLite prompts DB path")
    index_path: Optional[str] = Field(default=None, description="SQLite embeddings DB path")


class PipelineRunResponse(BaseModel):
    status: Literal["success"] = "success"
    prompts_loaded: int = 0
    embeddings_generated: int = 0
    embeddings_updated: int = 0
    errors: List[str] = []


class StatusResponse(BaseModel):
    prompts_db: str
    embeddings_db: str
    prompt_count: int
    embedding_count: int


# --- App state (set in lifespan or dependency; main.py may override for initialize_service) ---
_override_prompts_db_path: Optional[str] = None
_override_embeddings_db_path: Optional[str] = None


def _get_prompts_db_path() -> str:
    if _override_prompts_db_path is not None:
        return _override_prompts_db_path
    return str(Path(__file__).parent / DEFAULT_PROMPTS_DB)


def _get_embeddings_db_path() -> str:
    if _override_embeddings_db_path is not None:
        return _override_embeddings_db_path
    return str(Path(__file__).parent / DEFAULT_EMBEDDINGS_DB)


def _get_prompt_store() -> PromptStore:
    store = PromptStore()
    store.open_db(_get_prompts_db_path())
    return store


def _get_embedding_store() -> PromptEmbeddingStore:
    return PromptEmbeddingStore(_get_embeddings_db_path())


def _get_embedding_generator() -> EmbeddingGenerator:
    return EmbeddingGenerator(model_name="all-MiniLM-L6-v2")


# --- Endpoints ---


@app.post(
    "/api/embeddings/generate",
    response_model=EmbeddingsGenerateResponse,
    summary="Generate or regenerate embeddings",
)
async def post_embeddings_generate(body: EmbeddingsGenerateRequest) -> EmbeddingsGenerateResponse:
    """
    Regenerate embeddings for all prompts (prompt_ids=null) or for the given prompt_ids.
    Returns generated_count (new), updated_count (overwritten), and any errors.
    """
    def _run() -> EmbeddingsGenerateResponse:
        store = _get_prompt_store()
        emb_store = _get_embedding_store()
        generator = _get_embedding_generator()
        normalizer = PromptNormalizer()
        all_prompts = store.get_all_prompts()
        if body.prompt_ids is None:
            to_process = all_prompts
        else:
            if not body.prompt_ids:
                raise HTTPException(status_code=400, detail={"error": "prompt_ids must be null or a non-empty list"})
            by_id = {p.prompt_id: p for p in all_prompts}
            missing = [pid for pid in body.prompt_ids if pid not in by_id]
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail={"error": f"Prompt(s) not found: {', '.join(missing)}"},
                )
            to_process = [by_id[pid] for pid in body.prompt_ids]
        if not to_process:
            return EmbeddingsGenerateResponse(generated_count=0, updated_count=0, errors=[])
        existing = set(emb_store.load_all_embeddings().keys())
        generated = 0
        updated = 0
        errors: List[str] = []
        try:
            contents = [normalizer.normalize(p.content) for p in to_process]
            embeddings = generator.embed_batch(contents)
            for i, p in enumerate(to_process):
                try:
                    emb_store.save_embedding(p.prompt_id, contents[i], embeddings[i])
                    if p.prompt_id in existing:
                        updated += 1
                    else:
                        generated += 1
                except Exception as e:
                    errors.append(f"{p.prompt_id}: {e}")
        except Exception as e:
            raise HTTPException(status_code=500, detail={"error": f"Embedding model error: {e}"})
        _cache.clear()  # Invalidate caches after regeneration
        return EmbeddingsGenerateResponse(
            generated_count=generated,
            updated_count=updated,
            errors=errors,
        )

    try:
        return await asyncio.to_thread(_run)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})


@app.get(
    "/api/prompts/{prompt_id}/similar",
    response_model=SimilarPromptsResponse,
    summary="Get similar prompts by embedding",
)
async def get_prompts_similar(
    prompt_id: str,
    limit: int = Query(default=5, ge=1, le=100, description="Max number of results"),
    threshold: float = Query(default=0.8, description="Minimum similarity score (0.0-1.0)"),
) -> SimilarPromptsResponse:
    """
    Return prompts most similar to the given prompt_id, with score >= threshold.
    """
    if not (0.0 <= threshold <= 1.0):
        raise HTTPException(status_code=400, detail={"error": "threshold must be between 0.0 and 1.0"})

    def _run() -> SimilarPromptsResponse:
        store = _get_prompt_store()
        emb_store = _get_embedding_store()
        try:
            prompt = store.get_prompt(prompt_id)
        except KeyError:
            raise HTTPException(status_code=404, detail={"error": f"Prompt not found: {prompt_id}"})
        try:
            query_emb, _ = emb_store.load_embedding(prompt_id)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Embedding not found for prompt: {prompt_id}. Run POST /api/embeddings/generate first."},
            )
        all_emb = _cache_get("embeddings")
        if all_emb is None:
            all_emb = emb_store.load_all_embeddings()
            _cache_set("embeddings", all_emb)
        by_id = {p.prompt_id: p for p in store.get_all_prompts()}
        computer = SimilarityComputer()
        results: List[SimilarResultItem] = []
        for pid, emb in all_emb.items():
            if pid == prompt_id:
                continue
            sim = computer.compute_similarity(query_emb, emb)
            if sim < threshold:
                continue
            rec = by_id.get(pid)
            if not rec:
                continue
            results.append(
                SimilarResultItem(
                    prompt_id=pid,
                    similarity_score=round(sim, 4),
                    content_preview=_content_preview(rec.content),
                    layer=rec.layer,
                    category=rec.category,
                )
            )
        results.sort(key=lambda r: r.similarity_score, reverse=True)
        results = results[:limit]
        return SimilarPromptsResponse(
            query_prompt_id=prompt_id,
            query_content_preview=_content_preview(prompt.content),
            results=results,
            count=len(results),
        )

    try:
        return await asyncio.to_thread(_run)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})


@app.post(
    "/api/search/semantic",
    response_model=SemanticSearchResponse,
    summary="Semantic search over prompts",
)
async def post_search_semantic(body: SemanticSearchRequest) -> SemanticSearchResponse:
    """
    Embed the query (normalized, no variables), then return nearest prompts by cosine similarity.
    """
    query = (body.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail={"error": "query must be non-empty"})

    def _run() -> SemanticSearchResponse:
        store = _get_prompt_store()
        emb_store = _get_embedding_store()
        generator = _get_embedding_generator()
        normalizer = PromptNormalizer()
        norm_query = normalizer.normalize(query)
        query_emb = generator.embed_single(norm_query)
        all_emb = _cache_get("embeddings")
        if all_emb is None:
            all_emb = emb_store.load_all_embeddings()
            _cache_set("embeddings", all_emb)
        by_id = {p.prompt_id: p for p in store.get_all_prompts()}
        computer = SimilarityComputer()
        results: List[SemanticSearchResultItem] = []
        for pid, emb in all_emb.items():
            sim = computer.compute_similarity(query_emb, emb)
            rec = by_id.get(pid)
            if not rec:
                continue
            results.append(
                SemanticSearchResultItem(
                    prompt_id=pid,
                    similarity_score=round(sim, 4),
                    content_preview=_content_preview(rec.content),
                    layer=rec.layer,
                    category=rec.category,
                )
            )
        results.sort(key=lambda r: r.similarity_score, reverse=True)
        results = results[: body.limit]
        return SemanticSearchResponse(query=body.query, results=results, count=len(results))

    try:
        return await asyncio.to_thread(_run)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})


VALID_TIERS = ("Tier1", "Tier2", "Tier3")


@app.get(
    "/api/analysis/duplicates",
    response_model=DuplicatesResponse,
    summary="Get duplicate/cluster analysis",
)
async def get_analysis_duplicates(
    threshold: float = Query(default=0.9, description="Similarity threshold for clustering (0.0-1.0)"),
    same_layer: bool = Query(default=True, description="If true, exclude Tier3 (different layer)"),
    tier: Optional[str] = Query(default=None, description="Filter to Tier1, Tier2, or Tier3 only"),
) -> DuplicatesResponse:
    """
    Run metadata-aware clustering and return merge recommendations.
    same_layer=true returns only Tier1 and Tier2. tier=Tier1 returns only Tier1 clusters.
    """
    if not (0.0 <= threshold <= 1.0):
        raise HTTPException(status_code=400, detail={"error": "threshold must be between 0.0 and 1.0"})
    if tier is not None and tier not in VALID_TIERS:
        raise HTTPException(status_code=400, detail={"error": f"tier must be one of: {', '.join(VALID_TIERS)}"})

    def _run() -> DuplicatesResponse:
        store = _get_prompt_store()
        emb_store = _get_embedding_store()
        prompts = store.get_all_prompts()
        embeddings = _cache_get("embeddings")
        if embeddings is None:
            embeddings = emb_store.load_all_embeddings()
            _cache_set("embeddings", embeddings)
        if not prompts or not embeddings:
            return DuplicatesResponse(duplicates=[], total_clusters=0, tier_breakdown={"Tier1": 0, "Tier2": 0, "Tier3": 0})
        matcher = MetadataAwareMatcher()
        sims_raw = matcher.compute_pairwise_similarities(prompts, embeddings)
        by_id = {p.prompt_id: p for p in prompts}
        filtered: Dict[tuple, tuple] = {}
        for (id1, id2), sim in sims_raw.items():
            if sim < threshold:
                continue
            t, conf = matcher.apply_metadata_filter((by_id[id1], by_id[id2]), sim)
            if t == "NoMatch":
                continue
            if same_layer and t == "Tier3":
                continue
            if tier is not None and t != tier:
                continue
            filtered[(id1, id2)] = (t, conf)
        clusterer = DuplicateClusterer(
            tier1_threshold=max(threshold, 0.92),
            tier2_threshold=max(threshold, 0.90),
            tier3_threshold=max(threshold, 0.88),
        )
        clusters = clusterer.cluster_by_tier(filtered, min_cluster_size=2)
        builder = MergeRecommendationBuilder()
        tier_breakdown: Dict[str, int] = {"Tier1": 0, "Tier2": 0, "Tier3": 0}
        duplicates: List[DuplicateClusterItem] = []
        for c in clusters:
            tier_breakdown[c["tier"]] = tier_breakdown.get(c["tier"], 0) + 1
            rec = builder.suggest_merge(c, by_id)
            duplicates.append(
                DuplicateClusterItem(
                    cluster_id=rec["cluster_id"],
                    tier=c["tier"],
                    confidence=rec["confidence"],
                    reason=rec["reason"],
                    recommendation=rec["recommendation"],
                    target_prompt_id=rec["target_prompt_id"],
                    prompts=c["prompts"],
                    merge_candidates=rec["merge_candidates"],
                    variable_summary=rec["variable_summary"],
                )
            )
        return DuplicatesResponse(
            duplicates=duplicates,
            total_clusters=len(duplicates),
            tier_breakdown=tier_breakdown,
        )

    try:
        return await asyncio.to_thread(_run)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})


@app.get("/api/status", response_model=StatusResponse, summary="Service status")
async def get_status() -> StatusResponse:
    """Return DB paths and counts (prompts, embeddings)."""
    def _run() -> StatusResponse:
        store = _get_prompt_store()
        emb_store = _get_embedding_store()
        prompts = store.get_all_prompts()
        try:
            embeddings = emb_store.load_all_embeddings()
        except Exception:
            embeddings = {}
        return StatusResponse(
            prompts_db=_get_prompts_db_path(),
            embeddings_db=_get_embeddings_db_path(),
            prompt_count=len(prompts),
            embedding_count=len(embeddings),
        )
    try:
        return await asyncio.to_thread(_run)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})


@app.post("/api/pipeline/run", response_model=PipelineRunResponse, summary="Run full pipeline")
async def post_pipeline_run(body: PipelineRunRequest) -> PipelineRunResponse:
    """Load prompts from file → SQLite, generate embeddings (idempotent). Single pipeline entry."""
    data_file = (body.data_file or "").strip()
    if not data_file:
        raise HTTPException(status_code=400, detail={"error": "data_file is required"})
    path = Path(data_file)
    if not path.is_absolute():
        path = Path(__file__).parent / path
    if not path.exists():
        raise HTTPException(status_code=400, detail={"error": f"Data file not found: {data_file}"})
    db_path = body.db_path or str(Path(__file__).parent / DEFAULT_PROMPTS_DB)
    index_path = body.index_path or str(Path(__file__).parent / DEFAULT_EMBEDDINGS_DB)
    # Set module-level overrides so _get_* use these paths
    globals()["_override_prompts_db_path"] = db_path
    globals()["_override_embeddings_db_path"] = index_path

    def _run() -> PipelineRunResponse:
        store = PromptStore()
        store.load_prompts(str(path))
        prompts = store.get_all_prompts()
        store.save_prompts(prompts, db_path)
        n_prompts = len(prompts)
        emb_store = PromptEmbeddingStore(index_path)
        generator = _get_embedding_generator()
        normalizer = PromptNormalizer()
        existing = set(emb_store.load_all_embeddings().keys())
        prompt_ids = {p.prompt_id for p in prompts}
        missing = prompt_ids - existing
        generated, updated, errors = 0, 0, []
        if missing:
            to_embed = [p for p in prompts if p.prompt_id in missing]
            contents = [normalizer.normalize(p.content) for p in to_embed]
            embs = generator.embed_batch(contents)
            for i, p in enumerate(to_embed):
                try:
                    emb_store.save_embedding(p.prompt_id, contents[i], embs[i])
                    if p.prompt_id in existing:
                        updated += 1
                    else:
                        generated += 1
                except Exception as e:
                    errors.append(f"{p.prompt_id}: {e}")
        all_emb = emb_store.load_all_embeddings()
        _cache_set("embeddings", all_emb)
        return PipelineRunResponse(
            prompts_loaded=n_prompts,
            embeddings_generated=generated,
            embeddings_updated=updated,
            errors=errors,
        )

    try:
        return await asyncio.to_thread(_run)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})


# --- Error handler for consistent JSON ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    from fastapi.responses import JSONResponse
    body = exc.detail if isinstance(exc.detail, dict) else {"error": str(exc.detail)}
    return JSONResponse(status_code=exc.status_code, content=body)
