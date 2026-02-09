"""
Unit and integration tests for the Prompt Similarity Service REST API.

Unit tests: valid and invalid inputs for each endpoint.
Integration tests: full flow (load data → embed → query → get results).
"""
import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api_layer import app
from data_layer import PromptRecord, PromptStore
from embedding_layer import EmbeddingGenerator, PromptEmbeddingStore, PromptNormalizer


# --- Fixtures: in-memory DBs and app override ---
@pytest.fixture
def prompts_db_path(tmp_path):
    path = str(tmp_path / "prompts.db")
    store = PromptStore()
    store.load_prompts(str(Path(__file__).parent / "sample_prompts.json"))
    store.save_prompts(store.get_all_prompts(), path)
    return path


@pytest.fixture
def embeddings_db_path(tmp_path):
    return str(tmp_path / "embeddings.db")


@pytest.fixture
def client_with_dbs(prompts_db_path, embeddings_db_path):
    """Override app defaults to use temp DBs."""
    from api_layer import DEFAULT_EMBEDDINGS_DB, DEFAULT_PROMPTS_DB

    # Patch paths so app uses our temp DBs
    import api_layer as api_module
    original_prompts = getattr(api_module, "_get_prompts_db_path", None)
    original_embeddings = getattr(api_module, "_get_embeddings_db_path", None)

    def _prompts_path():
        return prompts_db_path

    def _embeddings_path():
        return embeddings_db_path

    api_module._get_prompts_db_path = _prompts_path
    api_module._get_embeddings_db_path = _embeddings_path
    try:
        with TestClient(app) as c:
            yield c
    finally:
        if original_prompts is not None:
            api_module._get_prompts_db_path = original_prompts
        if original_embeddings is not None:
            api_module._get_embeddings_db_path = original_embeddings


# --- POST /api/embeddings/generate ---


class TestEmbeddingsGenerate:
    def test_generate_all_null_success(self, client_with_dbs):
        r = client_with_dbs.post("/api/embeddings/generate", json={"prompt_ids": None})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "success"
        assert "generated_count" in data
        assert "updated_count" in data
        assert "errors" in data
        assert isinstance(data["errors"], list)

    def test_generate_specific_ids_success(self, client_with_dbs):
        r = client_with_dbs.post(
            "/api/embeddings/generate",
            json={"prompt_ids": ["survey.question.base", "survey.question.with_options"]},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "success"

    def test_generate_invalid_prompt_ids_400(self, client_with_dbs):
        r = client_with_dbs.post(
            "/api/embeddings/generate",
            json={"prompt_ids": ["survey.question.base", "nonexistent.id"]},
        )
        assert r.status_code == 400
        data = r.json()
        assert "error" in data
        assert "nonexistent" in data["error"].lower() or "not found" in data["error"].lower()

    def test_generate_empty_list_400(self, client_with_dbs):
        r = client_with_dbs.post("/api/embeddings/generate", json={"prompt_ids": []})
        assert r.status_code == 400
        data = r.json()
        assert "error" in data


# --- GET /api/prompts/{prompt_id}/similar ---


class TestPromptsSimilar:
    def test_similar_success_after_embed(self, client_with_dbs):
        client_with_dbs.post("/api/embeddings/generate", json={"prompt_ids": None})
        r = client_with_dbs.get(
            "/api/prompts/survey.question.base/similar",
            params={"limit": 5, "threshold": 0.8},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["query_prompt_id"] == "survey.question.base"
        assert "query_content_preview" in data
        assert "results" in data
        assert "count" in data
        assert len(data["query_content_preview"]) <= 100
        for item in data["results"]:
            assert "prompt_id" in item
            assert "similarity_score" in item
            assert "content_preview" in item
            assert "layer" in item
            assert "category" in item
            assert len(item["content_preview"]) <= 100

    def test_similar_404_prompt_not_found(self, client_with_dbs):
        r = client_with_dbs.get("/api/prompts/nonexistent.prompt/similar?limit=5&threshold=0.8")
        assert r.status_code == 404
        data = r.json()
        assert "error" in data

    def test_similar_400_invalid_threshold(self, client_with_dbs):
        # Endpoint returns 400 with clear message for invalid threshold
        r = client_with_dbs.get(
            "/api/prompts/survey.question.base/similar",
            params={"limit": 5, "threshold": 1.5},
        )
        assert r.status_code == 400
        data = r.json()
        assert "error" in data
        assert "threshold" in data["error"].lower() and "0.0" in data["error"] and "1.0" in data["error"]
        r2 = client_with_dbs.get(
            "/api/prompts/survey.question.base/similar",
            params={"limit": 5, "threshold": -0.1},
        )
        assert r2.status_code == 400

    def test_similar_limit_validation(self, client_with_dbs):
        r = client_with_dbs.get(
            "/api/prompts/survey.question.base/similar",
            params={"limit": 0, "threshold": 0.8},
        )
        assert r.status_code == 422
        # limit=5 without embeddings yields 404 (embedding not found)
        r2 = client_with_dbs.get(
            "/api/prompts/survey.question.base/similar",
            params={"limit": 5, "threshold": 0.8},
        )
        assert r2.status_code in (200, 404)


# --- POST /api/search/semantic ---


class TestSearchSemantic:
    def test_semantic_success(self, client_with_dbs):
        client_with_dbs.post("/api/embeddings/generate", json={"prompt_ids": None})
        r = client_with_dbs.post(
            "/api/search/semantic",
            json={"query": "how to handle user interruptions", "limit": 10},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["query"] == "how to handle user interruptions"
        assert "results" in data
        assert "count" in data
        for item in data["results"]:
            assert "prompt_id" in item
            assert "similarity_score" in item
            assert "content_preview" in item
            assert "layer" in item
            assert "category" in item

    def test_semantic_400_empty_query(self, client_with_dbs):
        r = client_with_dbs.post(
            "/api/search/semantic",
            json={"query": "", "limit": 10},
        )
        assert r.status_code == 400
        data = r.json()
        assert "error" in data and "query" in data["error"].lower()
        r2 = client_with_dbs.post(
            "/api/search/semantic",
            json={"query": "   ", "limit": 10},
        )
        assert r2.status_code == 400
        assert "error" in r2.json()


# --- GET /api/analysis/duplicates ---


class TestAnalysisDuplicates:
    def test_duplicates_success(self, client_with_dbs):
        client_with_dbs.post("/api/embeddings/generate", json={"prompt_ids": None})
        r = client_with_dbs.get(
            "/api/analysis/duplicates",
            params={"threshold": 0.9, "same_layer": True},
        )
        assert r.status_code == 200
        data = r.json()
        assert "duplicates" in data
        assert "total_clusters" in data
        assert "tier_breakdown" in data
        assert data["tier_breakdown"].keys() >= {"Tier1", "Tier2", "Tier3"}
        for dup in data["duplicates"]:
            assert "cluster_id" in dup
            assert "tier" in dup
            assert "confidence" in dup
            assert "reason" in dup
            assert "recommendation" in dup
            assert "target_prompt_id" in dup
            assert "prompts" in dup
            assert "merge_candidates" in dup
            assert "variable_summary" in dup

    def test_duplicates_same_layer_excludes_tier3(self, client_with_dbs):
        client_with_dbs.post("/api/embeddings/generate", json={"prompt_ids": None})
        r = client_with_dbs.get(
            "/api/analysis/duplicates",
            params={"threshold": 0.8, "same_layer": True},
        )
        assert r.status_code == 200
        data = r.json()
        for dup in data["duplicates"]:
            assert dup["tier"] != "Tier3", "same_layer=true should exclude Tier3"

    def test_duplicates_tier_filter(self, client_with_dbs):
        client_with_dbs.post("/api/embeddings/generate", json={"prompt_ids": None})
        r = client_with_dbs.get(
            "/api/analysis/duplicates",
            params={"threshold": 0.8, "same_layer": False, "tier": "Tier1"},
        )
        assert r.status_code == 200
        data = r.json()
        for dup in data["duplicates"]:
            assert dup["tier"] == "Tier1"

    def test_duplicates_400_invalid_tier(self, client_with_dbs):
        r = client_with_dbs.get(
            "/api/analysis/duplicates",
            params={"threshold": 0.9, "tier": "InvalidTier"},
        )
        assert r.status_code == 400
        data = r.json()
        assert "error" in data
        assert "tier" in data["error"].lower()

    def test_duplicates_threshold_validation(self, client_with_dbs):
        r = client_with_dbs.get(
            "/api/analysis/duplicates",
            params={"threshold": 1.5, "same_layer": True},
        )
        assert r.status_code == 400
        data = r.json()
        assert "error" in data and "threshold" in data["error"].lower()


# --- Integration: full flow ---


class TestIntegrationFullFlow:
    def test_load_embed_query_similar_and_duplicates(self, client_with_dbs):
        # 1. Generate embeddings for all
        r1 = client_with_dbs.post("/api/embeddings/generate", json={"prompt_ids": None})
        assert r1.status_code == 200
        assert r1.json()["status"] == "success"
        assert r1.json()["generated_count"] >= 1 or r1.json()["updated_count"] >= 1

        # 2. Get similar prompts
        r2 = client_with_dbs.get(
            "/api/prompts/survey.question.base/similar",
            params={"limit": 3, "threshold": 0.5},
        )
        assert r2.status_code == 200
        assert r2.json()["query_prompt_id"] == "survey.question.base"
        assert r2.json()["count"] >= 0

        # 3. Semantic search
        r3 = client_with_dbs.post(
            "/api/search/semantic",
            json={"query": "ask the user a question", "limit": 5},
        )
        assert r3.status_code == 200
        assert r3.json()["query"] == "ask the user a question"

        # 4. Duplicates
        r4 = client_with_dbs.get(
            "/api/analysis/duplicates",
            params={"threshold": 0.85, "same_layer": True},
        )
        assert r4.status_code == 200
        assert "duplicates" in r4.json()
        assert "tier_breakdown" in r4.json()
