"""
Unit tests for similarity_layer: similarity, metadata filter, clustering, merge recommendations.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from data_layer import PromptRecord, PromptStore
from embedding_layer import EmbeddingGenerator, PromptEmbeddingStore, PromptNormalizer
from similarity_layer import (
    MetadataAwareMatcher,
    DuplicateClusterer,
    MergeRecommendationBuilder,
    SimilarityComputer,
    TIER1_THRESHOLD,
    TIER2_THRESHOLD,
    TIER3_THRESHOLD,
)


# --- SimilarityComputer ---


class TestSimilarityComputer:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert SimilarityComputer.compute_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert SimilarityComputer.compute_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        sim = SimilarityComputer.compute_similarity(a, b)
        assert 0.0 <= sim <= 1.0
        assert sim == pytest.approx(0.0, abs=1e-5)

    def test_zero_vector(self):
        z = np.zeros(5, dtype=np.float32)
        v = np.ones(5, dtype=np.float32)
        assert SimilarityComputer.compute_similarity(z, v) == 0.0
        assert SimilarityComputer.compute_similarity(z, z) == 0.0

    def test_nan_returns_zero(self):
        a = np.array([1.0, np.nan, 3.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert SimilarityComputer.compute_similarity(a, b) == 0.0

    def test_range_zero_to_one(self):
        a = np.random.randn(32).astype(np.float32)
        b = np.random.randn(32).astype(np.float32)
        sim = SimilarityComputer.compute_similarity(a, b)
        assert 0.0 <= sim <= 1.0


# --- MetadataAwareMatcher ---


def _make_record(prompt_id: str, category: str, layer: str, name: str, content: str) -> PromptRecord:
    return PromptRecord(
        prompt_id=prompt_id,
        category=category,
        layer=layer,
        name=name,
        content=content,
    )


class TestMetadataFilter:
    def test_tier1_same_layer_same_category_high_sim(self):
        matcher = MetadataAwareMatcher()
        a = _make_record("a", "survey", "engine", "A", "x" * 15)
        b = _make_record("b", "survey", "engine", "B", "y" * 15)
        tier, conf = matcher.apply_metadata_filter((a, b), 0.93)
        assert tier == "Tier1"
        assert conf == 0.93

    def test_tier1_below_threshold_returns_no_match(self):
        matcher = MetadataAwareMatcher()
        a = _make_record("a", "survey", "engine", "A", "x" * 15)
        b = _make_record("b", "survey", "engine", "B", "y" * 15)
        tier, conf = matcher.apply_metadata_filter((a, b), 0.91)
        assert tier == "NoMatch"
        assert conf == 0.0

    def test_tier2_same_layer_different_category_high_sim(self):
        matcher = MetadataAwareMatcher()
        a = _make_record("warm", "os", "os", "Warm", "x" * 15)
        b = _make_record("emp", "empathy", "os", "Emp", "y" * 15)
        tier, conf = matcher.apply_metadata_filter((a, b), 0.91)
        assert tier == "Tier2"
        assert conf == 0.91

    def test_tier3_different_layer_high_sim(self):
        matcher = MetadataAwareMatcher()
        a = _make_record("a", "survey", "engine", "A", "x" * 15)
        b = _make_record("b", "org", "org", "B", "y" * 15)
        tier, conf = matcher.apply_metadata_filter((a, b), 0.89)
        assert tier == "Tier3"
        assert conf == 0.89

    def test_no_match_below_all_thresholds(self):
        matcher = MetadataAwareMatcher()
        a = _make_record("a", "survey", "engine", "A", "x" * 15)
        b = _make_record("b", "survey", "engine", "B", "y" * 15)
        tier, conf = matcher.apply_metadata_filter((a, b), 0.80)
        assert tier == "NoMatch"
        assert conf == 0.0


class TestPairwiseSimilarities:
    def test_upper_triangle_only(self):
        matcher = MetadataAwareMatcher()
        prompts = [
            _make_record("a", "survey", "engine", "A", "x" * 15),
            _make_record("b", "survey", "engine", "B", "y" * 15),
        ]
        emb = np.random.randn(384).astype(np.float32) * 0.01
        embeddings = {"a": emb, "b": emb.copy()}
        sims = matcher.compute_pairwise_similarities(prompts, embeddings)
        assert list(sims.keys()) == [("a", "b")]
        assert 0 <= sims[("a", "b")] <= 1

    def test_no_duplicate_reverse_pair(self):
        matcher = MetadataAwareMatcher()
        prompts = [
            _make_record("x", "survey", "engine", "X", "x" * 15),
            _make_record("y", "survey", "engine", "Y", "y" * 15),
        ]
        embeddings = {"x": np.ones(384, dtype=np.float32), "y": np.ones(384, dtype=np.float32)}
        sims = matcher.compute_pairwise_similarities(prompts, embeddings)
        assert ("x", "y") in sims
        assert ("y", "x") not in sims


# --- DuplicateClusterer + MergeRecommendationBuilder ---


class TestMergeRecommendationBuilder:
    def test_tier1_merge(self):
        builder = MergeRecommendationBuilder()
        cluster = {
            "cluster_id": "dup_001",
            "tier": "Tier1",
            "prompts": [{"prompt_id": "a", "similarity": 0.95}, {"prompt_id": "b", "similarity": 0.95}],
            "confidence": 0.95,
            "reason": "Same layer + category + high semantic similarity",
        }
        prompts = {
            "a": _make_record("a", "survey", "engine", "A", "short content here"),
            "b": _make_record("b", "survey", "engine", "B", "longer content here for canonical"),
        }
        rec = builder.suggest_merge(cluster, prompts)
        assert rec["recommendation"] == "MERGE"
        assert rec["target_prompt_id"] == "b"
        assert rec["merge_candidates"] == ["a"]
        assert rec["confidence"] == 0.95

    def test_tier2_review(self):
        builder = MergeRecommendationBuilder()
        cluster = {"cluster_id": "dup_002", "tier": "Tier2", "prompts": [{"prompt_id": "x", "similarity": 0.9}], "confidence": 0.9, "reason": "Same layer, different category"}
        prompts = {"x": _make_record("x", "survey", "engine", "X", "x" * 15)}
        rec = builder.suggest_merge(cluster, prompts)
        assert rec["recommendation"] == "REVIEW"

    def test_tier3_keep_separate(self):
        builder = MergeRecommendationBuilder()
        cluster = {"cluster_id": "dup_003", "tier": "Tier3", "prompts": [{"prompt_id": "p", "similarity": 0.88}], "confidence": 0.88, "reason": "Different layer"}
        prompts = {"p": _make_record("p", "survey", "engine", "P", "p" * 15)}
        rec = builder.suggest_merge(cluster, prompts)
        assert rec["recommendation"] == "KEEP_SEPARATE"

    def test_variable_summary(self):
        builder = MergeRecommendationBuilder()
        cluster = {
            "cluster_id": "dup_004",
            "tier": "Tier1",
            "prompts": [{"prompt_id": "q", "similarity": 0.95}, {"prompt_id": "r", "similarity": 0.95}],
            "confidence": 0.95,
            "reason": "Same layer + category",
        }
        prompts = {
            "q": _make_record("q", "survey", "engine", "Q", "Ask {{question_text}} here."),
            "r": _make_record("r", "survey", "engine", "R", "Use {{question_text}} and {{options}}."),
        }
        rec = builder.suggest_merge(cluster, prompts)
        assert "question_text" in rec["variable_summary"] or "{{" in rec["variable_summary"]


# --- DuplicateClusterer (no embeddings) ---


class TestDuplicateClusterer:
    def test_cluster_by_tier_forms_clusters(self):
        """Two Tier1 pairs (a,b) and (b,c) should yield one Tier1 cluster {a,b,c}."""
        similarities = {
            ("a", "b"): ("Tier1", 0.95),
            ("b", "c"): ("Tier1", 0.93),
        }
        clusterer = DuplicateClusterer()
        clusters = clusterer.cluster_by_tier(similarities, min_cluster_size=2)
        tier1 = [c for c in clusters if c["tier"] == "Tier1"]
        assert len(tier1) >= 1
        ids_in_tier1 = [p["prompt_id"] for c in tier1 for p in c["prompts"]]
        assert "a" in ids_in_tier1 and "b" in ids_in_tier1 and "c" in ids_in_tier1

    def test_clusters_sorted_by_confidence_desc(self):
        similarities = {
            ("x", "y"): ("Tier1", 0.92),
            ("p", "q"): ("Tier1", 0.98),
        }
        clusterer = DuplicateClusterer()
        clusters = clusterer.cluster_by_tier(similarities, min_cluster_size=2)
        assert len(clusters) >= 1
        confs = [c["confidence"] for c in clusters]
        assert confs == sorted(confs, reverse=True)


# --- Full pipeline on 12 sample prompts ---


@pytest.fixture(scope="module")
def sample_prompts_and_embeddings():
    """Load 12 sample prompts, generate embeddings, return (prompts, embeddings dict)."""
    sample_path = Path(__file__).parent / "sample_prompts.json"
    if not sample_path.exists():
        pytest.skip("sample_prompts.json not found")
    store = PromptStore()
    records = store.load_prompts(str(sample_path))
    prompts = records[:12]
    normalizer = PromptNormalizer()
    generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    contents = [normalizer.normalize(p.content) for p in prompts]
    embs = generator.embed_batch(contents)
    embeddings = {p.prompt_id: embs[i] for i, p in enumerate(prompts)}
    return prompts, embeddings


class TestFullPipelineSamplePrompts:
    def test_warm_and_empathetic_not_in_same_tier1_cluster(self, sample_prompts_and_embeddings):
        """Run on 12 sample prompts; verify os.style.warm and os.style.empathetic are NOT in the same Tier1 cluster."""
        prompts, embeddings = sample_prompts_and_embeddings
        matcher = MetadataAwareMatcher()
        sims_raw = matcher.compute_pairwise_similarities(prompts, embeddings)
        by_id = {p.prompt_id: p for p in prompts}
        filtered = {}
        for (id1, id2), sim in sims_raw.items():
            tier, conf = matcher.apply_metadata_filter((by_id[id1], by_id[id2]), sim)
            if tier != "NoMatch":
                filtered[(id1, id2)] = (tier, conf)
        clusterer = DuplicateClusterer()
        clusters = clusterer.cluster_by_tier(filtered, min_cluster_size=2)
        tier1_clusters = [c for c in clusters if c["tier"] == "Tier1"]
        for c in tier1_clusters:
            ids = [p["prompt_id"] for p in c["prompts"]]
            # Must not have both warm and empathetic in same Tier1 cluster
            has_warm = "os.style.warm" in ids
            has_emp = "os.style.empathetic" in ids
            assert not (has_warm and has_emp), "os.style.warm and os.style.empathetic must not be in same Tier1 cluster"

    def test_common_error_recovery_and_handling_tier1_if_high_sim(self, sample_prompts_and_embeddings):
        """common.error_recovery vs common.error_handling: same layer+category; if sim >= 0.92 they cluster as Tier1."""
        prompts, embeddings = sample_prompts_and_embeddings
        by_id = {p.prompt_id: p for p in prompts}
        if "common.error_recovery" not in by_id or "common.error_handling" not in by_id:
            pytest.skip("sample does not contain common.error_recovery and common.error_handling")
        matcher = MetadataAwareMatcher()
        sim = matcher.compute_pairwise_similarities(
            [by_id["common.error_recovery"], by_id["common.error_handling"]],
            {k: embeddings[k] for k in ["common.error_recovery", "common.error_handling"]},
        )
        key = ("common.error_recovery", "common.error_handling")
        if key not in sim:
            key = ("common.error_handling", "common.error_recovery")
        assert key in sim
        tier, conf = matcher.apply_metadata_filter(
            (by_id["common.error_recovery"], by_id["common.error_handling"]),
            sim[key],
        )
        if sim[key] >= TIER1_THRESHOLD:
            assert tier == "Tier1"
        else:
            assert tier in ("NoMatch", "Tier2", "Tier3")

    def test_clustering_produces_expected_tiers(self, sample_prompts_and_embeddings):
        """Clustering returns clusters with tier in Tier1, Tier2, Tier3; confidence and reason set."""
        prompts, embeddings = sample_prompts_and_embeddings
        matcher = MetadataAwareMatcher()
        sims_raw = matcher.compute_pairwise_similarities(prompts, embeddings)
        by_id = {p.prompt_id: p for p in prompts}
        filtered = {}
        for (id1, id2), sim in sims_raw.items():
            tier, conf = matcher.apply_metadata_filter((by_id[id1], by_id[id2]), sim)
            if tier != "NoMatch":
                filtered[(id1, id2)] = (tier, conf)
        clusterer = DuplicateClusterer()
        clusters = clusterer.cluster_by_tier(filtered, min_cluster_size=2)
        for c in clusters:
            assert c["tier"] in ("Tier1", "Tier2", "Tier3")
            assert "cluster_id" in c and c["cluster_id"].startswith("dup_")
            assert "prompts" in c and len(c["prompts"]) >= 2
            assert "confidence" in c and 0 <= c["confidence"] <= 1
            assert "reason" in c
        # Merge recommendations
        builder = MergeRecommendationBuilder()
        for c in clusters:
            rec = builder.suggest_merge(c, by_id)
            assert rec["recommendation"] in ("MERGE", "REVIEW", "KEEP_SEPARATE")
            assert rec["target_prompt_id"] in by_id
            assert rec["cluster_id"] == c["cluster_id"]
