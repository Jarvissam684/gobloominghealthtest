"""
Unit tests for embedding_layer: normalization, embedding shape/dtype, batch vs single, benchmark.
"""

import json
import time
from pathlib import Path

import numpy as np
import pytest

from data_layer import PromptRecord, PromptStore
from embedding_layer import (
    MINILM_EMBEDDING_DIM,
    EmbeddingGenerator,
    PromptEmbeddingStore,
    PromptNormalizer,
)


# --- Sample dataset content for normalization tests ---
SAMPLE_CONTENTS = [
    "Present the following question to the user naturally: {{question_text}}. Listen carefully to their response.",
    "Ask the user this question: {{question_text}}. The valid responses are: {{options}}. Accept their answer if it reasonably matches.",
    "You need to collect: {{field_name}}. Ask the user for this information in a conversational way.",
    "Maintain a warm, friendly tone throughout the conversation. Use encouraging language.",
    "Maintain a professional, courteous demeanor. Be respectful and efficient. Use clear language.",
    "Show empathy and understanding in your responses. Acknowledge the user's feelings.",
]


class TestPromptNormalizer:
    def test_normalize_single_variable(self):
        out = PromptNormalizer.normalize("Ask {{question}}")
        assert out == "Ask [VARIABLE_QUESTION]"

    def test_normalize_snake_case_variable(self):
        out = PromptNormalizer.normalize("Use {{question_text}} here.")
        assert out == "Use [VARIABLE_QUESTION_TEXT] here."

    def test_normalize_multiple_variables(self):
        out = PromptNormalizer.normalize("Ask {{question_text}}. Options: {{options}}.")
        assert "[VARIABLE_QUESTION_TEXT]" in out
        assert "[VARIABLE_OPTIONS]" in out
        assert "{{" not in out and "}}" not in out

    def test_normalize_zero_variables(self):
        text = "No variables in this prompt at all."
        assert PromptNormalizer.normalize(text) == text

    def test_normalize_five_plus_variables(self):
        text = "{{a}} {{b}} {{c}} {{d}} {{e}} {{f}}"
        out = PromptNormalizer.normalize(text)
        assert out == "[VARIABLE_A] [VARIABLE_B] [VARIABLE_C] [VARIABLE_D] [VARIABLE_E] [VARIABLE_F]"

    def test_extract_variables_empty(self):
        assert PromptNormalizer.extract_variables("No vars") == set()

    def test_extract_variables_one(self):
        assert PromptNormalizer.extract_variables("Ask {{question}}") == {"question"}

    def test_extract_variables_multiple_deduplicated(self):
        s = PromptNormalizer.extract_variables("{{x}} and {{y}} and {{x}}")
        assert s == {"x", "y"}

    def test_normalize_sample_dataset_contents(self):
        for content in SAMPLE_CONTENTS:
            out = PromptNormalizer.normalize(content)
            assert "{{" not in out
            assert "}}" not in out
            # All {{...}} should become [VARIABLE_...]
            vars_found = PromptNormalizer.extract_variables(content)
            for v in vars_found:
                assert f"[VARIABLE_{v.upper()}]" in out


class TestEmbeddingGenerator:
    @pytest.fixture
    def generator(self):
        return EmbeddingGenerator(model_name="all-MiniLM-L6-v2")

    def test_embed_single_shape_and_dtype(self, generator):
        emb = generator.embed_single("Hello world.")
        assert emb.shape == (MINILM_EMBEDDING_DIM,)
        assert emb.dtype == np.float32

    def test_embed_batch_shape_and_dtype(self, generator):
        texts = ["First sentence.", "Second sentence."]
        emb = generator.embed_batch(texts)
        assert emb.shape == (2, MINILM_EMBEDDING_DIM)
        assert emb.dtype == np.float32

    def test_embed_batch_empty(self, generator):
        emb = generator.embed_batch([])
        assert emb.shape == (0, MINILM_EMBEDDING_DIM)
        assert emb.dtype == np.float32

    def test_batch_vs_single_equivalence(self, generator):
        text = "The same content for single and batch."
        single = generator.embed_single(text)
        batch = generator.embed_batch([text])[0]
        np.testing.assert_array_almost_equal(single, batch, decimal=5)

    def test_embed_normalized_content_not_raw(self, generator):
        raw = "Ask the user: {{question_text}}."
        normalized = PromptNormalizer.normalize(raw)
        assert "{{" in raw and "}}" in raw
        assert "[VARIABLE_" in normalized
        emb_norm = generator.embed_single(normalized)
        emb_raw = generator.embed_single(raw)
        assert emb_norm.shape == emb_raw.shape
        # Normalized and raw should differ (different text)
        assert not np.allclose(emb_norm, emb_raw)

    def test_os_style_warm_vs_empathetic_differ(self, generator):
        warm = "Maintain a warm, friendly tone throughout the conversation. Use encouraging language."
        empathetic = "Show empathy and understanding in your responses. Acknowledge the user's feelings."
        emb_warm = generator.embed_single(warm)
        emb_emp = generator.embed_single(empathetic)
        diff = np.linalg.norm(emb_warm - emb_emp)
        assert diff > 0.1, "os.style.warm and os.style.empathetic should have measurably different embeddings"


class TestPromptEmbeddingStore:
    @pytest.fixture
    def db_path(self, tmp_path):
        return str(tmp_path / "embeddings.db")

    @pytest.fixture
    def generator(self):
        return EmbeddingGenerator(model_name="all-MiniLM-L6-v2")

    def test_save_and_load_embedding(self, db_path, generator):
        store = PromptEmbeddingStore(db_path)
        norm = "Ask [VARIABLE_QUESTION] here."
        emb = generator.embed_single(norm)
        store.save_embedding("test.id", norm, emb)
        loaded_emb, loaded_norm = store.load_embedding("test.id")
        np.testing.assert_array_almost_equal(loaded_emb, emb)
        assert loaded_norm == norm

    def test_load_embedding_raises_key_error(self, db_path):
        store = PromptEmbeddingStore(db_path)
        with pytest.raises(KeyError):
            store.load_embedding("nonexistent")

    def test_load_all_embeddings(self, db_path, generator):
        store = PromptEmbeddingStore(db_path)
        store.save_embedding("a", "text a", generator.embed_single("text a"))
        store.save_embedding("b", "text b", generator.embed_single("text b"))
        all_ = store.load_all_embeddings()
        assert set(all_.keys()) == {"a", "b"}
        assert all_["a"].shape == (MINILM_EMBEDDING_DIM,)
        assert all_["b"].dtype == np.float32

    def test_batch_generate(self, db_path, generator):
        store = PromptEmbeddingStore(db_path)
        prompts = [
            PromptRecord(
                prompt_id="p1",
                category="survey",
                layer="engine",
                name="P1",
                content="Ask {{question_text}} here.",
            ),
            PromptRecord(
                prompt_id="p2",
                category="form",
                layer="engine",
                name="P2",
                content="Collect {{field_name}} from the user.",
            ),
        ]
        store.batch_generate(prompts, generator)
        emb1, norm1 = store.load_embedding("p1")
        emb2, norm2 = store.load_embedding("p2")
        assert "[VARIABLE_QUESTION_TEXT]" in norm1
        assert "[VARIABLE_FIELD_NAME]" in norm2
        assert emb1.shape == (MINILM_EMBEDDING_DIM,)
        assert emb2.dtype == np.float32


class TestBenchmark:
    """Benchmark: time to embed all 12 sample prompts."""

    def test_benchmark_embed_12_sample_prompts(self):
        sample_path = Path(__file__).parent / "sample_prompts.json"
        if not sample_path.exists():
            pytest.skip("sample_prompts.json not found")
        store = PromptStore()
        records = store.load_prompts(str(sample_path))
        assert len(records) >= 12
        prompts = records[:12]

        generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        normalizer = PromptNormalizer()
        contents = [normalizer.normalize(p.content) for p in prompts]

        start = time.perf_counter()
        embeddings = generator.embed_batch(contents)
        elapsed = time.perf_counter() - start

        assert embeddings.shape == (12, MINILM_EMBEDDING_DIM)
        assert embeddings.dtype == np.float32
        # 12 prompts should complete in reasonable time (e.g. < 60s on CPU)
        assert elapsed < 60, f"Embedding 12 prompts took {elapsed:.2f}s (expected < 60s)"
        print(f"\nBenchmark: embedded 12 prompts in {elapsed:.3f}s ({elapsed/12*1000:.1f}ms per prompt)")

    def test_retrieval_latency_under_5ms(self, tmp_path):
        """Embedding retrieval from DB should be < 5ms per prompt (per spec)."""
        db_path = str(tmp_path / "latency.db")
        store = PromptEmbeddingStore(db_path)
        generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        # Store one embedding
        emb = generator.embed_single("Test content for retrieval latency.")
        store.save_embedding("latency_test", "Test content for retrieval latency.", emb)
        # Warm-up one load
        store.load_embedding("latency_test")
        # Measure repeated loads
        n = 20
        start = time.perf_counter()
        for _ in range(n):
            store.load_embedding("latency_test")
        elapsed_ms = (time.perf_counter() - start) / n * 1000
        assert elapsed_ms < 5, f"Retrieval took {elapsed_ms:.2f}ms per prompt (expected < 5ms)"
        print(f"\nBenchmark: embedding retrieval {elapsed_ms:.2f}ms per prompt (n={n})")
