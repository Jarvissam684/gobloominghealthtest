"""
Prompt Similarity Service — Single pipeline entry.

Runs the full flow: load prompts from JSON → SQLite → normalize → embed → prewarm cache.
Use from API (POST /api/pipeline/run), CLI, or programmatically.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from data_layer import PromptStore
from embedding_layer import EmbeddingGenerator, PromptEmbeddingStore, PromptNormalizer


def run_pipeline(
    data_file: str,
    db_path: str = "prompts.db",
    index_path: str = "embeddings.db",
) -> dict:
    """
    Load prompts from data_file into db_path, generate embeddings into index_path (idempotent).
    Returns dict: prompts_loaded, embeddings_generated, embeddings_updated, errors.
    """
    path = Path(data_file)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    store = PromptStore()
    store.load_prompts(str(path))
    prompts = store.get_all_prompts()
    store.save_prompts(prompts, db_path)
    prompts_loaded = len(prompts)
    emb_store = PromptEmbeddingStore(index_path)
    generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    normalizer = PromptNormalizer()
    existing = set(emb_store.load_all_embeddings().keys())
    missing = {p.prompt_id for p in prompts} - existing
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
    return {
        "prompts_loaded": prompts_loaded,
        "embeddings_generated": generated,
        "embeddings_updated": updated,
        "errors": errors,
    }


if __name__ == "__main__":
    import sys
    data = sys.argv[1] if len(sys.argv) > 1 else "sample_prompts.json"
    db = sys.argv[2] if len(sys.argv) > 2 else "prompts.db"
    idx = sys.argv[3] if len(sys.argv) > 3 else "embeddings.db"
    out = run_pipeline(data, db, idx)
    print("Pipeline result:", out)
