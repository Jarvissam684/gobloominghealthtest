"""
Prompt Similarity Service — Main Orchestrator.

Initializes the service: load prompts from JSON into SQLite, generate embeddings
into the index DB, optionally pre-warm caches. Returns a FastAPI app ready for uvicorn.

Idempotent: calling initialize_service twice with the same paths does not double-embed;
embeddings are skipped when all prompts already have entries in the index.

Usage:
  from main import initialize_service
  app = initialize_service("sample_prompts.json", "prompts.db", "embeddings.db")
  # Then: uvicorn main:app (using app variable below) or pass app to uvicorn programmatically.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from data_layer import PromptStore
from embedding_layer import EmbeddingGenerator, PromptEmbeddingStore, PromptNormalizer

# Import app; initialize_service sets api_layer path overrides before returning app
import api_layer as _api  # noqa: E402


def initialize_service(
    data_file: str,
    db_path: str,
    index_path: str,
) -> FastAPI:
    """
    Load prompts from JSON into PromptStore (SQLite at db_path), normalize + embed
    all prompts into PromptEmbeddingStore (SQLite at index_path), pre-warm embeddings
    cache. Return FastAPI app configured to use db_path and index_path.

    Idempotent: if every prompt already has an embedding in index_path, embedding
    generation is skipped on subsequent calls.
    """
    # 1. Configure API to use these paths
    _api._override_prompts_db_path = db_path
    _api._override_embeddings_db_path = index_path

    # 2. Load prompts from JSON → PromptStore (SQLite)
    path = Path(data_file)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    store = PromptStore()
    store.load_prompts(data_file)
    store.save_prompts(store.get_all_prompts(), db_path)

    # 3. Normalize + embed all prompts → PromptEmbeddingStore (idempotent)
    emb_store = PromptEmbeddingStore(index_path)
    generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    prompts = store.get_all_prompts()
    existing_ids = set(emb_store.load_all_embeddings().keys())
    prompt_ids = {p.prompt_id for p in prompts}
    missing = prompt_ids - existing_ids
    if missing:
        to_embed = [p for p in prompts if p.prompt_id in missing]
        normalizer = PromptNormalizer()
        contents = [normalizer.normalize(p.content) for p in to_embed]
        embeddings = generator.embed_batch(contents)
        for i, p in enumerate(to_embed):
            emb_store.save_embedding(p.prompt_id, contents[i], embeddings[i])
    # If no missing, skip (idempotent: second call does not re-embed)

    # 4. Pre-warm cache: compute pairwise similarities and cluster by tiers (cache embeddings)
    # API layer caches embeddings on first request; we pre-warm so first request is fast
    all_emb = emb_store.load_all_embeddings()
    _api._cache_set("embeddings", all_emb)

    # 5. Optionally precompute pairwise + clusters and cache (API computes on demand; we just warmed embeddings)
    # Pairwise and clusters are computed per-request in get_analysis_duplicates; no separate cache key for them.
    # So step 4 is sufficient.

    return _api.app


# Default app for uvicorn when run as module with default paths
_DEFAULT_DATA = str(Path(__file__).parent / "sample_prompts.json")
_DEFAULT_DB = str(Path(__file__).parent / "prompts.db")
_DEFAULT_INDEX = str(Path(__file__).parent / "embeddings.db")

app: FastAPI
try:
    app = initialize_service(_DEFAULT_DATA, _DEFAULT_DB, _DEFAULT_INDEX)
except FileNotFoundError:
    # If sample_prompts.json missing, expose API anyway (endpoints will 404 until data/embeddings exist)
    _api._override_prompts_db_path = _DEFAULT_DB
    _api._override_embeddings_db_path = _DEFAULT_INDEX
    app = _api.app
