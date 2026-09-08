"""
Tests for sage_memory/memory_models.py and sage_memory/config.py — Phase 1.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sage_memory.config import (
    MEMORY_COLD_COLLECTION,
    MEMORY_COLD_DIR,
    MEMORY_DEFAULT_TOP_K,
    MEMORY_ENABLED,
    MEMORY_HOT_COLLECTION,
    MEMORY_HOT_DIR,
    MEMORY_HOT_SUMMARY_THRESHOLD,
    MEMORY_STORE_PATH,
)
from sage_memory.memory_models import (
    KNOWN_CATEGORIES,
    MemoryRecord,
    MemorySearchResult,
    MemoryValidationError,
    make_memory_id,
    validate_memory_record,
)
from sage_memory.memory_store import MemoryStore
from sage_memory.memory_index import MemoryChromaStore
from sage_memory.memory_manager import MemoryManager
from sage_memory.memory_summarizer import MemorySummarizer


def _sample_hot(**overrides) -> MemoryRecord:
    base = {
        "memory_id": make_memory_id(),
        "content": "Use implementation X for this task.",
        "memory_type": "hot",
        "category": "task",
        "source": "user_message",
        "session_id": "sess_abc123",
        "created_at": "2026-09-08T12:00:00+00:00",
        "updated_at": "2026-09-08T12:00:00+00:00",
    }
    base.update(overrides)
    return MemoryRecord(**base)


def _sample_cold(**overrides) -> MemoryRecord:
    base = {
        "memory_id": make_memory_id(),
        "content": "User prefers FastAPI for backend projects.",
        "memory_type": "cold",
        "category": "preference",
        "source": "user_message",
        "session_id": None,
        "created_at": "2026-09-08T12:00:00+00:00",
        "updated_at": "2026-09-08T12:00:00+00:00",
    }
    base.update(overrides)
    return MemoryRecord(**base)


class TestMemoryConfig:
    def test_default_constants(self):
        assert MEMORY_ENABLED is True
        assert MEMORY_HOT_COLLECTION == "sage_memory_hot"
        assert MEMORY_COLD_COLLECTION == "sage_memory_cold"
        assert MEMORY_DEFAULT_TOP_K == 5
        assert MEMORY_HOT_SUMMARY_THRESHOLD == 20
        assert MEMORY_HOT_DIR == MEMORY_STORE_PATH / "hot"
        assert MEMORY_COLD_DIR == MEMORY_STORE_PATH / "cold"

    def test_env_overrides(self, monkeypatch, tmp_path):
        store = tmp_path / "custom_memory"
        monkeypatch.setenv("MEMORY_ENABLED", "0")
        monkeypatch.setenv("MEMORY_STORE_PATH", str(store))
        monkeypatch.setenv("MEMORY_HOT_COLLECTION", "custom_hot")
        monkeypatch.setenv("MEMORY_COLD_COLLECTION", "custom_cold")
        monkeypatch.setenv("MEMORY_DEFAULT_TOP_K", "8")
        monkeypatch.setenv("MEMORY_HOT_SUMMARY_THRESHOLD", "15")

        import importlib
        import sage_memory.config as cfg
        importlib.reload(cfg)

        assert cfg.MEMORY_ENABLED is False
        assert cfg.MEMORY_STORE_PATH == store
        assert cfg.MEMORY_HOT_COLLECTION == "custom_hot"
        assert cfg.MEMORY_COLD_COLLECTION == "custom_cold"
        assert cfg.MEMORY_DEFAULT_TOP_K == 8
        assert cfg.MEMORY_HOT_SUMMARY_THRESHOLD == 15

        # Restore defaults for other tests
        for key in (
            "MEMORY_ENABLED",
            "MEMORY_STORE_PATH",
            "MEMORY_HOT_COLLECTION",
            "MEMORY_COLD_COLLECTION",
            "MEMORY_DEFAULT_TOP_K",
            "MEMORY_HOT_SUMMARY_THRESHOLD",
        ):
            monkeypatch.delenv(key, raising=False)
        importlib.reload(cfg)


class TestMemoryRecordValidation:
    def test_valid_hot_memory(self):
        record = _sample_hot()
        assert record.memory_type == "hot"
        assert record.session_id == "sess_abc123"

    def test_valid_cold_memory(self):
        record = _sample_cold()
        assert record.memory_type == "cold"
        assert record.session_id is None

    def test_custom_category_accepted(self):
        record = _sample_cold(category="custom_domain_fact")
        assert record.category == "custom_domain_fact"

    def test_known_categories_documented(self):
        assert "preference" in KNOWN_CATEGORIES
        assert "summary" in KNOWN_CATEGORIES

    def test_hot_requires_session_id(self):
        with pytest.raises(MemoryValidationError, match="session_id is required"):
            _sample_hot(session_id=None)

    def test_empty_content_rejected(self):
        with pytest.raises(MemoryValidationError, match="content must be"):
            _sample_hot(content="   ")

    def test_invalid_memory_type_rejected(self):
        with pytest.raises(MemoryValidationError, match="memory_type must be"):
            _sample_hot(memory_type="warm")  # type: ignore[arg-type]

    def test_invalid_memory_id_prefix(self):
        with pytest.raises(MemoryValidationError, match="memory_id must start"):
            _sample_hot(memory_id="bad_id_123")

    def test_importance_out_of_range(self):
        with pytest.raises(MemoryValidationError, match="importance must be"):
            _sample_hot(importance=1.5)

    def test_confidence_out_of_range(self):
        with pytest.raises(MemoryValidationError, match="confidence must be"):
            _sample_cold(confidence=-0.1)

    def test_invalid_timestamp(self):
        with pytest.raises(MemoryValidationError, match="created_at must be ISO"):
            _sample_hot(created_at="not-a-date")

    def test_summary_with_parents(self):
        parent_id = make_memory_id()
        record = _sample_hot(
            category="summary",
            is_summary=True,
            parent_memory_ids=[parent_id],
        )
        assert record.is_summary is True
        assert record.parent_memory_ids == [parent_id]

    def test_invalid_parent_id_rejected(self):
        with pytest.raises(MemoryValidationError, match="parent_memory_ids"):
            _sample_hot(parent_memory_ids=[""])


class TestMemoryRecordSerialization:
    def test_to_dict_round_trip(self):
        original = _sample_hot()
        restored = MemoryRecord.from_dict(original.to_dict())
        assert restored.memory_id == original.memory_id
        assert restored.content == original.content
        assert restored.memory_type == original.memory_type
        assert restored.parent_memory_ids == original.parent_memory_ids

    def test_to_json_is_valid_json(self):
        record = _sample_cold()
        parsed = json.loads(record.to_json())
        assert parsed["memory_id"] == record.memory_id
        assert parsed["memory_type"] == "cold"

    def test_validate_memory_record_function(self):
        record = _sample_cold()
        validate_memory_record(record)  # should not raise


class TestMemorySearchResult:
    def test_derived_similarity(self):
        result = MemorySearchResult(
            memory_id=make_memory_id(),
            content="test",
            memory_type="cold",
            category="preference",
            distance=0.2,
        )
        assert result.derived_similarity == 0.8

    def test_to_dict_json_safe(self):
        result = MemorySearchResult(
            memory_id=make_memory_id(),
            content="test",
            memory_type="hot",
            category="task",
            distance=0.1,
            session_id="sess_1",
        )
        serialized = json.dumps(result.to_dict())
        assert "derived_similarity" in serialized


class TestMakeMemoryId:
    def test_format(self):
        mid = make_memory_id()
        assert mid.startswith("mem_")
        assert len(mid) == 16  # mem_ + 12 hex chars


class TestPhase1Skeletons:
    def test_memory_store_initializes(self, tmp_path):
        store = MemoryStore(tmp_path / "memory")
        assert store.store_root == tmp_path / "memory"
        assert store.hot_dir.is_dir()
        assert store.cold_dir.is_dir()

    def test_memory_chroma_store_initializes(self, tmp_chroma_root):
        from sage_document_db.embeddings import EmbeddingService
        index = MemoryChromaStore(
            chroma_root=tmp_chroma_root,
            embedding_service=EmbeddingService(),
            hot_collection="sage_memory_hot",
            cold_collection="sage_memory_cold",
        )
        assert index.count("hot") == 0
        assert index.count("cold") == 0

    def test_memory_manager_skeleton(self):
        manager = MemoryManager(store=None, index=None)
        assert hasattr(manager, "add_hot_memory")
        assert hasattr(manager, "rebuild_memory_indexes")
        assert callable(manager.rebuild_memory_indexes)
        with pytest.raises(Exception):
            manager.rebuild_memory_indexes()

    def test_memory_summarizer_threshold(self):
        summarizer = MemorySummarizer(manager=None, threshold=5)
        assert summarizer.should_summarize(4) is False
        assert summarizer.should_summarize(5) is True
        assert summarizer.should_summarize(10) is True

    def test_memory_summarizer_create_summary_record(self):
        summarizer = MemorySummarizer(manager=None)
        parent = make_memory_id()
        record = summarizer.create_summary_record(
            memory_id=make_memory_id(),
            content="Summary of recent discussion.",
            session_id="sess_xyz",
            parent_memory_ids=[parent],
            created_at="2026-09-08T12:00:00+00:00",
            updated_at="2026-09-08T12:00:00+00:00",
        )
        assert record.is_summary is True
        assert record.category == "summary"
        assert record.parent_memory_ids == [parent]

    def test_package_imports(self):
        import sage_memory
        assert hasattr(sage_memory, "MemoryRecord")
        assert hasattr(sage_memory, "MemoryManager")
        assert hasattr(sage_memory, "MEMORY_ENABLED")
