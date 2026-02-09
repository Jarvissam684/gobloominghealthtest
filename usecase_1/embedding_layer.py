"""
Prompt Similarity Service — Embedding Pipeline.

Variable normalization: {{variable_name}} → [VARIABLE_VARIABLE_NAME] (uppercase).
Embeddings are computed on normalized content. Stored as float32 BLOB.

SQLite schema for embeddings (raw SQL, no ORM):
  CREATE TABLE embeddings (
    prompt_id TEXT PRIMARY KEY,
    normalized_content TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );
  Embedding BLOB: int32 (4 bytes) dimension, then N * float32 (dimension 384 for MiniLM).
"""
from __future__ import annotations

import re
import sqlite3
import struct
from datetime import datetime, timezone
from typing import Dict, List, Set, Tuple

import numpy as np

# Lazy import to avoid loading model at import time
_sentence_transformers = None

def _get_sentence_transformers():
    global _sentence_transformers
    if _sentence_transformers is None:
        from sentence_transformers import SentenceTransformer
        _sentence_transformers = SentenceTransformer
    return _sentence_transformers


# --- Variable pattern: {{ name }} where name is [a-zA-Z_][a-zA-Z0-9_]*
VARIABLE_PATTERN = re.compile(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}")


class PromptNormalizer:
    """
    Replaces {{variable_name}} with [VARIABLE_VARIABLE_NAME] (uppercase)
    so embeddings are stable across template instances.
    """

    @staticmethod
    def extract_variables(content: str) -> Set[str]:
        """Return the set of variable names found in content (without braces)."""
        return set(VARIABLE_PATTERN.findall(content))

    @staticmethod
    def normalize(content: str) -> str:
        """
        Replace each {{var}} with [VARIABLE_<UPPERCASE_NAME>].
        Escaped braces (e.g. \\{\\{) are not supported; only the regex pattern is replaced.
        """
        def repl(match: re.Match) -> str:
            name = match.group(1)
            anchor = name.upper().replace("-", "_")
            return f"[VARIABLE_{anchor}]"
        return VARIABLE_PATTERN.sub(repl, content)


# --- Embedding dimension for default model (all-MiniLM-L6-v2)
MINILM_EMBEDDING_DIM = 384


class EmbeddingGenerator:
    """
    Generates embeddings via sentence-transformers. Model is cached on first use.
    Output dtype: float32. Shape: (384,) for single, (N, 384) for batch.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            cls = _get_sentence_transformers()
            self._model = cls(self._model_name)
        return self._model

    def embed_single(self, content: str) -> np.ndarray:
        """Embed one string. Returns shape (384,) float32."""
        model = self._get_model()
        out = model.encode([content], convert_to_numpy=True, precision="float32")
        arr = np.asarray(out[0] if out.ndim > 1 else out, dtype=np.float32)
        return arr.ravel()

    def embed_batch(self, contents: List[str]) -> np.ndarray:
        """Embed a list of strings. Returns shape (N, 384) float32."""
        if not contents:
            return np.zeros((0, MINILM_EMBEDDING_DIM), dtype=np.float32)
        model = self._get_model()
        out = model.encode(contents, convert_to_numpy=True, precision="float32")
        return np.asarray(out, dtype=np.float32)


class PromptEmbeddingStore:
    """
    Stores prompt embeddings in SQLite. BLOB format: 4-byte int32 dimension, then dim * float32.
    """

    # Schema:
    # CREATE TABLE embeddings (
    #   prompt_id TEXT PRIMARY KEY,
    #   normalized_content TEXT NOT NULL,
    #   embedding BLOB NOT NULL,
    #   created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    # );
    _CREATE_SQL = """
    CREATE TABLE IF NOT EXISTS embeddings (
        prompt_id TEXT PRIMARY KEY,
        normalized_content TEXT NOT NULL,
        embedding BLOB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """

    def __init__(self, db_path: str, embedding_dim: int = MINILM_EMBEDDING_DIM) -> None:
        self._db_path = db_path
        self._embedding_dim = embedding_dim
        self._conn: sqlite3.Connection | None = None

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.execute(self._CREATE_SQL)
            self._conn.commit()
        return self._conn

    @staticmethod
    def _serialize_embedding(embedding: np.ndarray) -> bytes:
        """Pack (dim,) float32 array as: 4-byte dim (int32 LE) + dim * 4 bytes float32 LE."""
        arr = np.asarray(embedding, dtype=np.float32).ravel()
        dim = arr.shape[0]
        return struct.pack(f"<i{dim}f", dim, *arr.tolist())

    @staticmethod
    def _deserialize_embedding(blob: bytes) -> np.ndarray:
        """Unpack BLOB to float32 array of shape (dim,)."""
        dim = struct.unpack_from("<i", blob)[0]
        floats = struct.unpack_from(f"<{dim}f", blob, 4)
        return np.array(floats, dtype=np.float32)

    def save_embedding(
        self,
        prompt_id: str,
        normalized_content: str,
        embedding: np.ndarray,
    ) -> None:
        """Persist one embedding. Overwrites if prompt_id exists."""
        conn = self._ensure_conn()
        blob = self._serialize_embedding(embedding)
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT OR REPLACE INTO embeddings (prompt_id, normalized_content, embedding, created_at) VALUES (?, ?, ?, ?)",
            (prompt_id, normalized_content, blob, now),
        )
        conn.commit()

    def load_embedding(self, prompt_id: str) -> Tuple[np.ndarray, str]:
        """Load embedding and normalized_content for prompt_id. Raises KeyError if not found."""
        conn = self._ensure_conn()
        row = conn.execute(
            "SELECT normalized_content, embedding FROM embeddings WHERE prompt_id = ?",
            (prompt_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Embedding not found: {prompt_id}")
        norm_content, blob = row[0], row[1]
        emb = self._deserialize_embedding(blob)
        return emb, norm_content

    def load_all_embeddings(self) -> Dict[str, np.ndarray]:
        """Load all embeddings as dict prompt_id -> array of shape (dim,)."""
        conn = self._ensure_conn()
        cur = conn.execute("SELECT prompt_id, embedding FROM embeddings")
        out = {}
        for prompt_id, blob in cur.fetchall():
            out[prompt_id] = self._deserialize_embedding(blob)
        return out

    def batch_generate(self, prompts: List["PromptRecord"], generator: EmbeddingGenerator) -> None:
        """
        Normalize each prompt's content, embed, and store. Uses batch embedding.
        Imports PromptRecord locally to avoid circular dependency.
        """
        from data_layer import PromptRecord  # noqa: F401

        if not prompts:
            return
        normalizer = PromptNormalizer()
        contents = [normalizer.normalize(p.content) for p in prompts]
        embeddings = generator.embed_batch(contents)
        for i, p in enumerate(prompts):
            self.save_embedding(p.prompt_id, contents[i], embeddings[i])
