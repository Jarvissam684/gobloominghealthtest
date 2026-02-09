"""
Unit tests for data_layer: load, save, validation, get_prompt, get_all_prompts, update_prompt.
"""

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from data_layer import (
    MIN_CONTENT_LENGTH,
    VALID_LAYERS,
    PromptRecord,
    PromptStore,
)


# --- Fixtures ---

def valid_record(**overrides) -> dict:
    base = {
        "prompt_id": "test.prompt.one",
        "category": "survey",
        "layer": "engine",
        "name": "Test Prompt",
        "content": "This is at least ten characters long.",
    }
    base.update(overrides)
    return base


@pytest.fixture
def sample_json_path(tmp_path):
    """Write a valid 3-prompt JSON array to a temp file."""
    data = [
        valid_record(prompt_id="a.one", content="Content for prompt one here."),
        valid_record(prompt_id="b.two", content="Content for prompt two here."),
        valid_record(prompt_id="c.three", content="Content for prompt three here."),
    ]
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


@pytest.fixture
def sample_jsonl_path(tmp_path):
    """Write valid JSONL (one JSON object per line)."""
    data = [
        valid_record(prompt_id="x.one", content="Content for x one here."),
        valid_record(prompt_id="x.two", content="Content for x two here."),
    ]
    path = tmp_path / "prompts.jsonl"
    path.write_text("\n".join(json.dumps(obj) for obj in data), encoding="utf-8")
    return str(path)


@pytest.fixture
def twelve_prompt_sample_path():
    """Path to the 12-prompt sample dataset (valid JSON)."""
    return str(Path(__file__).parent / "sample_prompts.json")


# --- PromptRecord validation ---


class TestPromptRecordValidation:
    def test_valid_record_creates(self):
        r = PromptRecord(**valid_record())
        assert r.prompt_id == "test.prompt.one"
        assert r.layer == "engine"
        assert r.content == "This is at least ten characters long."
        assert r.created_at is not None

    def test_prompt_id_rejects_uppercase(self):
        with pytest.raises(ValidationError) as exc:
            PromptRecord(**valid_record(prompt_id="Test.Prompt"))
        assert "prompt_id" in str(exc.value).lower() or "match" in str(exc.value).lower()

    def test_prompt_id_rejects_invalid_chars(self):
        with pytest.raises(ValidationError):
            PromptRecord(**valid_record(prompt_id="test prompt"))
        with pytest.raises(ValidationError):
            PromptRecord(**valid_record(prompt_id="test-prompt"))

    def test_prompt_id_accepts_valid_pattern(self):
        for pid in ["a", "a.b", "a_b", "a1.b2.c3", "org.layer.name"]:
            r = PromptRecord(**valid_record(prompt_id=pid))
            assert r.prompt_id == pid

    def test_layer_rejects_invalid(self):
        with pytest.raises(ValidationError) as exc:
            PromptRecord(**valid_record(layer="invalid"))
        assert "layer" in str(exc.value).lower()

    def test_layer_accepts_all_valid(self):
        for layer in VALID_LAYERS:
            r = PromptRecord(**valid_record(layer=layer))
            assert r.layer == layer

    def test_content_rejects_short(self):
        with pytest.raises(ValidationError) as exc:
            PromptRecord(**valid_record(content="short"))
        assert "content" in str(exc.value).lower() or str(MIN_CONTENT_LENGTH) in str(exc.value)

    def test_content_accepts_exactly_ten(self):
        r = PromptRecord(**valid_record(content="0123456789"))
        assert len(r.content) == 10


# --- PromptStore load ---


class TestPromptStoreLoad:
    def test_load_prompts_from_json_array(self, sample_json_path):
        store = PromptStore()
        records = store.load_prompts(sample_json_path)
        assert len(records) == 3
        ids = {r.prompt_id for r in records}
        assert ids == {"a.one", "b.two", "c.three"}

    def test_load_prompts_from_jsonl(self, sample_jsonl_path):
        store = PromptStore()
        records = store.load_prompts(sample_jsonl_path)
        assert len(records) == 2
        assert records[0].prompt_id == "x.one"
        assert records[1].prompt_id == "x.two"

    def test_load_prompts_rejects_nonexistent_file(self):
        store = PromptStore()
        with pytest.raises(FileNotFoundError):
            store.load_prompts("/nonexistent/prompts.json")

    def test_load_prompts_populates_store(self, sample_json_path):
        store = PromptStore()
        store.load_prompts(sample_json_path)
        assert store.get_prompt("a.one").content == "Content for prompt one here."
        assert len(store.get_all_prompts()) == 3

    def test_load_prompts_accepts_missing_name(self, tmp_path):
        path = tmp_path / "no_name.json"
        path.write_text(
            json.dumps([
                {
                    "prompt_id": "no.name.prompt",
                    "category": "common",
                    "layer": "engine",
                    "content": "Content without name field here.",
                },
            ]),
            encoding="utf-8",
        )
        store = PromptStore()
        records = store.load_prompts(str(path))
        assert len(records) == 1
        assert records[0].name == "no.name.prompt"  # defaulted from prompt_id

    def test_load_twelve_prompt_sample_without_errors(self, twelve_prompt_sample_path):
        """Load the provided 12-prompt sample dataset without errors."""
        store = PromptStore()
        records = store.load_prompts(twelve_prompt_sample_path)
        assert len(records) == 12
        ids = {r.prompt_id for r in records}
        assert "survey.question.base" in ids
        assert "directive.confidentiality" in ids
        for r in records:
            assert r.prompt_id
            assert r.layer in VALID_LAYERS
            assert len(r.content) >= MIN_CONTENT_LENGTH


# --- PromptStore save ---


class TestPromptStoreSave:
    def test_save_prompts_creates_db(self, sample_json_path, tmp_path):
        store = PromptStore()
        records = store.load_prompts(sample_json_path)
        db_path = str(tmp_path / "prompts.db")
        store.save_prompts(records, db_path)
        assert Path(db_path).exists()

    def test_save_prompts_reloads_via_new_store(self, sample_json_path, tmp_path):
        store = PromptStore()
        records = store.load_prompts(sample_json_path)
        db_path = str(tmp_path / "prompts.db")
        store.save_prompts(records, db_path)
        other = PromptStore()
        other.open_db(db_path)
        reloaded = other.get_all_prompts()
        assert len(reloaded) == len(records)
        by_id = {r.prompt_id: r for r in reloaded}
        for r in records:
            assert r.prompt_id in by_id
            assert by_id[r.prompt_id].content == r.content
            assert by_id[r.prompt_id].layer == r.layer


# --- PromptStore get / update ---


class TestPromptStoreGetUpdate:
    def test_get_prompt_returns_record(self, sample_json_path):
        store = PromptStore()
        store.load_prompts(sample_json_path)
        r = store.get_prompt("b.two")
        assert r.prompt_id == "b.two"
        assert "two" in r.content

    def test_get_prompt_raises_key_error_when_missing(self, sample_json_path):
        store = PromptStore()
        store.load_prompts(sample_json_path)
        with pytest.raises(KeyError):
            store.get_prompt("nonexistent.id")

    def test_get_all_prompts_returns_list(self, sample_json_path):
        store = PromptStore()
        store.load_prompts(sample_json_path)
        all_ = store.get_all_prompts()
        assert isinstance(all_, list)
        assert len(all_) == 3

    def test_update_prompt_in_memory(self, sample_json_path):
        store = PromptStore()
        store.load_prompts(sample_json_path)
        r = store.get_prompt("a.one")
        updated = PromptRecord(
            prompt_id="a.one",
            category="survey",
            layer="engine",
            name="Updated Name",
            content="Updated content that is long enough to pass validation.",
        )
        store.update_prompt("a.one", updated)
        assert store.get_prompt("a.one").name == "Updated Name"
        assert store.get_prompt("a.one").content == updated.content

    def test_update_prompt_raises_when_id_mismatch(self, sample_json_path):
        store = PromptStore()
        store.load_prompts(sample_json_path)
        updated = PromptRecord(
            prompt_id="other.id",
            category="survey",
            layer="engine",
            name="Other",
            content="Content for other id that is long enough.",
        )
        with pytest.raises(ValueError):
            store.update_prompt("a.one", updated)

    def test_update_prompt_raises_when_not_found(self, sample_json_path):
        store = PromptStore()
        store.load_prompts(sample_json_path)
        updated = PromptRecord(
            prompt_id="nonexistent.id",
            category="survey",
            layer="engine",
            name="Nonexistent",
            content="Content for nonexistent that is long enough.",
        )
        with pytest.raises(KeyError):
            store.update_prompt("nonexistent.id", updated)
