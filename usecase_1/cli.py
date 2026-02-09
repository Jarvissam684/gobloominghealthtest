"""
Prompt Similarity Service — CLI.

Commands: analyze-duplicates, search-similar, semantic-search, generate-embeddings.
Uses Click for argument parsing.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from data_layer import PromptStore
from embedding_layer import EmbeddingGenerator, PromptEmbeddingStore, PromptNormalizer
from similarity_layer import (
    DuplicateClusterer,
    MergeRecommendationBuilder,
    MetadataAwareMatcher,
    SimilarityComputer,
)


def _index_path_default(db_path: str) -> str:
    p = Path(db_path)
    return str(p.parent / "embeddings.db")


def _table(rows: List[tuple], headers: tuple, col_widths: Optional[List[int]] = None) -> str:
    """Format rows as a box-drawing table. rows are (col1, col2, ...); headers same length."""
    if not headers:
        return ""
    if col_widths is None:
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    col_widths = [max(c, len(str(h))) for c, h in zip(col_widths, headers)]
    w = col_widths
    top = "┌" + "┬".join("─" * (wi + 2) for wi in w) + "┐"
    sep = "├" + "┼".join("─" * (wi + 2) for wi in w) + "┤"
    bot = "└" + "┴".join("─" * (wi + 2) for wi in w) + "┘"
    lines = [top]
    lines.append("│" + "│".join(f" {str(headers[i]):<{w[i]}} " for i in range(len(headers))) + "│")
    lines.append(sep)
    for row in rows:
        lines.append("│" + "│".join(f" {str(row[i]):<{w[i]}} " for i in range(len(row))) + "│")
    lines.append(bot)
    return "\n".join(lines)


@click.group()
def cli() -> None:
    """Prompt Similarity Service CLI."""
    pass


@cli.command("analyze-duplicates")
@click.option("--db-path", required=True, type=click.Path(exists=True), help="Path to prompts SQLite DB")
@click.option("--index-path", type=click.Path(), default=None, help="Path to embeddings DB (default: same dir as db-path/embeddings.db)")
@click.option("--tier", type=click.Choice(["Tier1", "Tier2", "Tier3"]), default=None, help="Filter to specific tier")
@click.option("--threshold", type=float, default=0.9, help="Similarity threshold (0.0-1.0)")
@click.option("--output", type=click.Path(), default=None, help="Output JSON file path")
def analyze_duplicates(
    db_path: str,
    index_path: Optional[str],
    tier: Optional[str],
    threshold: float,
    output: Optional[str],
) -> None:
    """Write duplicate clusters to JSON file."""
    if not (0.0 <= threshold <= 1.0):
        click.echo("Error: threshold must be between 0.0 and 1.0", err=True)
        sys.exit(1)
    idx = index_path or _index_path_default(db_path)
    store = PromptStore()
    store.open_db(db_path)
    emb_store = PromptEmbeddingStore(idx)
    prompts = store.get_all_prompts()
    try:
        embeddings = emb_store.load_all_embeddings()
    except Exception as e:
        click.echo(f"Error loading embeddings: {e}", err=True)
        sys.exit(1)
    if not prompts or not embeddings:
        clusters_out: List[Dict[str, Any]] = []
    else:
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
            if tier is not None and t != tier:
                continue
            filtered[(id1, id2)] = (t, conf)
        clusterer = DuplicateClusterer(
            tier1_threshold=max(threshold, 0.92),
            tier2_threshold=max(threshold, 0.90),
            tier3_threshold=max(threshold, 0.88),
        )
        clusters = clusterer.cluster_by_tier(filtered, min_cluster_size=2)
        clusters_out = [
            {
                "cluster_id": c["cluster_id"],
                "tier": c["tier"],
                "confidence": c["confidence"],
                "prompts": c["prompts"],
                "recommendation": (
                    "MERGE" if c["tier"] == "Tier1" else "REVIEW" if c["tier"] == "Tier2" else "KEEP_SEPARATE"
                ),
                "reason": c["reason"],
            }
            for c in clusters
        ]
    payload = {"clusters": clusters_out}
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(payload, f, indent=2)
        click.echo(f"Wrote {len(clusters_out)} clusters to {output}")
    else:
        click.echo(json.dumps(payload, indent=2))


@cli.command("search-similar")
@click.option("--db-path", required=True, type=click.Path(exists=True), help="Path to prompts SQLite DB")
@click.option("--index-path", type=click.Path(), default=None, help="Path to embeddings DB")
@click.option("--prompt-id", required=True, help="Prompt ID to find similar prompts for")
@click.option("--limit", type=int, default=5, help="Max number of results")
@click.option("--threshold", type=float, default=0.8, help="Minimum similarity (0.0-1.0)")
def search_similar(
    db_path: str,
    index_path: Optional[str],
    prompt_id: str,
    limit: int,
    threshold: float,
) -> None:
    """Print similar prompts in table format."""
    if not (0.0 <= threshold <= 1.0):
        click.echo("Error: threshold must be between 0.0 and 1.0", err=True)
        sys.exit(1)
    idx = index_path or _index_path_default(db_path)
    store = PromptStore()
    store.open_db(db_path)
    emb_store = PromptEmbeddingStore(idx)
    try:
        prompt = store.get_prompt(prompt_id)
    except KeyError:
        click.echo(f"Error: Prompt not found: {prompt_id}", err=True)
        sys.exit(1)
    try:
        query_emb, _ = emb_store.load_embedding(prompt_id)
    except KeyError:
        click.echo(f"Error: Embedding not found for {prompt_id}. Run generate-embeddings first.", err=True)
        sys.exit(1)
    all_emb = emb_store.load_all_embeddings()
    by_id = {p.prompt_id: p for p in store.get_all_prompts()}
    computer = SimilarityComputer()
    results: List[tuple] = []
    for pid, emb in all_emb.items():
        if pid == prompt_id:
            continue
        sim = computer.compute_similarity(query_emb, emb)
        if sim < threshold:
            continue
        rec = by_id.get(pid)
        if not rec:
            continue
        results.append((pid, f"{sim:.2f}", rec.layer))
    results.sort(key=lambda r: float(r[1]), reverse=True)
    results = results[:limit]
    headers = ("Prompt ID", "Similarity", "Layer")
    click.echo(_table(results, headers))


@cli.command("semantic-search")
@click.option("--db-path", required=True, type=click.Path(exists=True), help="Path to prompts SQLite DB")
@click.option("--index-path", type=click.Path(), default=None, help="Path to embeddings DB")
@click.option("--query", required=True, help="Search query text")
@click.option("--limit", type=int, default=5, help="Max number of results")
def semantic_search(
    db_path: str,
    index_path: Optional[str],
    query: str,
    limit: int,
) -> None:
    """Print semantic search results in table format."""
    if not query.strip():
        click.echo("Error: query must be non-empty", err=True)
        sys.exit(1)
    idx = index_path or _index_path_default(db_path)
    store = PromptStore()
    store.open_db(db_path)
    emb_store = PromptEmbeddingStore(idx)
    generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    normalizer = PromptNormalizer()
    norm_query = normalizer.normalize(query.strip())
    query_emb = generator.embed_single(norm_query)
    all_emb = emb_store.load_all_embeddings()
    by_id = {p.prompt_id: p for p in store.get_all_prompts()}
    computer = SimilarityComputer()
    results: List[tuple] = []
    for pid, emb in all_emb.items():
        sim = computer.compute_similarity(query_emb, emb)
        rec = by_id.get(pid)
        if not rec:
            continue
        results.append((pid, f"{sim:.2f}", rec.layer))
    results.sort(key=lambda r: float(r[1]), reverse=True)
    results = results[:limit]
    headers = ("Prompt ID", "Similarity", "Layer")
    click.echo(_table(results, headers))


@cli.command("generate-embeddings")
@click.option("--db-path", required=True, type=click.Path(), help="Path to prompts SQLite DB (created/overwritten)")
@click.option("--index-path", type=click.Path(), default=None, help="Path to embeddings DB")
@click.option("--data-file", required=True, type=click.Path(exists=True), help="JSON file with prompts")
def generate_embeddings(
    db_path: str,
    index_path: Optional[str],
    data_file: str,
) -> None:
    """Load prompts from JSON, save to DB, generate and store embeddings. Print progress."""
    idx = index_path or _index_path_default(db_path)
    store = PromptStore()
    click.echo("Loading prompts... ", nl=False)
    store.load_prompts(data_file)
    n = len(store.get_all_prompts())
    click.echo(f"[{n}/{n}]")
    store.save_prompts(store.get_all_prompts(), db_path)
    prompts = store.get_all_prompts()
    if not prompts:
        click.echo("No prompts to embed.")
        return
    normalizer = PromptNormalizer()
    generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    click.echo("Generating embeddings... ", nl=False)
    contents = [normalizer.normalize(p.content) for p in prompts]
    t0 = time.perf_counter()
    embeddings = generator.embed_batch(contents)
    click.echo(f"[{len(prompts)}/{len(prompts)}]")
    emb_store = PromptEmbeddingStore(idx)
    click.echo("Storing embeddings... ", nl=False)
    for i, p in enumerate(prompts):
        emb_store.save_embedding(p.prompt_id, contents[i], embeddings[i])
    click.echo(f"[{len(prompts)}/{len(prompts)}]")
    elapsed = time.perf_counter() - t0
    click.echo(f"Done in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    cli()
