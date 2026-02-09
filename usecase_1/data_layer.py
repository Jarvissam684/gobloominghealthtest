"""
Prompt Similarity Service — Data Model & Storage Layer.

SQLite schema (raw SQL, no ORM):
  CREATE TABLE prompts (
    prompt_id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    layer TEXT NOT NULL CHECK (layer IN ('org', 'os', 'team', 'engine', 'directive')),
    name TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );
"""

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# --- Validation constants ---
PROMPT_ID_PATTERN = re.compile(r"^[a-z0-9._]+$")
VALID_LAYERS = frozenset({"org", "os", "team", "engine", "directive"})
MIN_CONTENT_LENGTH = 10


class PromptRecord(BaseModel):
    """Single prompt record. prompt_id is the unique identifier."""

    prompt_id: str
    category: str
    layer: str
    name: str
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("prompt_id")
    @classmethod
    def validate_prompt_id(cls, v: str) -> str:
        if not PROMPT_ID_PATTERN.match(v):
            raise ValueError(
                f"prompt_id must match [a-z0-9._]+, got: {v!r}"
            )
        return v

    @field_validator("layer")
    @classmethod
    def validate_layer(cls, v: str) -> str:
        if v not in VALID_LAYERS:
            raise ValueError(
                f"layer must be one of {sorted(VALID_LAYERS)}, got: {v!r}"
            )
        return v

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        if len(v) < MIN_CONTENT_LENGTH:
            raise ValueError(
                f"content must be at least {MIN_CONTENT_LENGTH} characters, got {len(v)}"
            )
        return v


class PromptStore:
    """
    In-memory + SQLite store for prompt records.
    Use load_prompts() then save_prompts() to persist, or connect to an existing DB.
    """

    # Schema for reference and init:
    # CREATE TABLE prompts (
    #   prompt_id TEXT PRIMARY KEY,
    #   category TEXT NOT NULL,
    #   layer TEXT NOT NULL,
    #   name TEXT NOT NULL,
    #   content TEXT NOT NULL,
    #   created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    # );
    _CREATE_SQL = """
    CREATE TABLE IF NOT EXISTS prompts (
        prompt_id TEXT PRIMARY KEY,
        category TEXT NOT NULL,
        layer TEXT NOT NULL,
        name TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """

    def __init__(self) -> None:
        self._records: dict[str, PromptRecord] = {}
        self._db_path: Optional[str] = None
        self._conn: Optional[sqlite3.Connection] = None

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            if not self._db_path:
                raise RuntimeError("No database path set. Call save_prompts first or open_db.")
            self._conn = sqlite3.connect(self._db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def open_db(self, db_path: str) -> None:
        """Open an existing SQLite database and load prompts into memory."""
        self._db_path = db_path
        conn = self._ensure_conn()
        cur = conn.execute(
            "SELECT prompt_id, category, layer, name, content, created_at FROM prompts"
        )
        self._records = {}
        for row in cur.fetchall():
            r = PromptRecord(
                prompt_id=row["prompt_id"],
                category=row["category"],
                layer=row["layer"],
                name=row["name"],
                content=row["content"],
                created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now(timezone.utc),
            )
            self._records[r.prompt_id] = r

    def load_prompts(self, json_file_path: str) -> List[PromptRecord]:
        """
        Load prompts from a JSON file (array of objects) or JSONL (one JSON object per line).
        Returns list of validated PromptRecords. Replaces in-memory store with loaded data.
        """
        path = Path(json_file_path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_file_path}")

        text = path.read_text(encoding="utf-8").strip()
        records: List[PromptRecord] = []

        # Try single JSON array first
        if text.startswith("["):
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                # Fallback: treat as JSONL (one object per line)
                data = []
                for line in text.splitlines():
                    line = line.strip()
                    if not line or line == "[" or line == "]":
                        continue
                    if line.endswith(","):
                        line = line[:-1]
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            if not isinstance(data, list):
                raise ValueError("JSON root must be an array of prompt objects")
        else:
            # JSONL: one JSON object per line
            data = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        for i, obj in enumerate(data):
            if not isinstance(obj, dict):
                raise ValueError(f"Item at index {i} is not a JSON object")
            # Allow missing "name" for fixed dataset compatibility
            name = obj.get("name")
            if name is None:
                name = obj.get("prompt_id", "")
            record = PromptRecord(
                prompt_id=obj["prompt_id"],
                category=obj["category"],
                layer=obj["layer"],
                name=name,
                content=obj["content"],
            )
            records.append(record)

        self._records = {r.prompt_id: r for r in records}
        return records

    def save_prompts(self, records: List[PromptRecord], db_path: str) -> None:
        """
        Persist records to SQLite. Creates table if needed. Replaces all rows.
        """
        self._db_path = db_path
        conn = sqlite3.connect(db_path)
        conn.execute(self._CREATE_SQL)
        conn.execute("DELETE FROM prompts")
        for r in records:
            conn.execute(
                "INSERT INTO prompts (prompt_id, category, layer, name, content, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    r.prompt_id,
                    r.category,
                    r.layer,
                    r.name,
                    r.content,
                    r.created_at.isoformat(),
                ),
            )
        conn.commit()
        conn.close()
        self._records = {r.prompt_id: r for r in records}
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def get_prompt(self, prompt_id: str) -> PromptRecord:
        """Return the prompt with the given prompt_id. Raises KeyError if not found."""
        if prompt_id not in self._records:
            raise KeyError(f"Prompt not found: {prompt_id}")
        return self._records[prompt_id]

    def get_all_prompts(self) -> List[PromptRecord]:
        """Return all stored prompts."""
        return list(self._records.values())

    def update_prompt(self, prompt_id: str, updated_record: PromptRecord) -> None:
        """Update the prompt with the given prompt_id. Id in updated_record must match."""
        if updated_record.prompt_id != prompt_id:
            raise ValueError(
                f"updated_record.prompt_id ({updated_record.prompt_id!r}) must match prompt_id ({prompt_id!r})"
            )
        if prompt_id not in self._records:
            raise KeyError(f"Prompt not found: {prompt_id}")
        self._records[prompt_id] = updated_record
        if self._db_path and self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
        if self._conn is not None:
            r = updated_record
            self._conn.execute(
                "UPDATE prompts SET category=?, layer=?, name=?, content=?, created_at=? WHERE prompt_id=?",
                (
                    r.category,
                    r.layer,
                    r.name,
                    r.content,
                    r.created_at.isoformat(),
                    r.prompt_id,
                ),
            )
            self._conn.commit()
