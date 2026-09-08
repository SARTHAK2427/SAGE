"""
Tests for sage_memory/memory_index.py — Phase 3 Chroma memory index.
"""

from __future__ import annotations

import pytest

from sage_document_db.embeddings import EmbeddingService
from sage_memory.memory_index import MemoryChromaStore, MemoryIndexError
from sage_memory.memory_models import MemoryRecord, make_memory_id
from sage_memory.memory_store import MemoryStore


def _hot(**overrides) -> MemoryRecord:
    base = {
        "memory_id": make_memory_id(),
        "content": "For this task, use implementation X.",
        "memory_type": "hot",
        "category": "task",
        "source": "user_message",
        "session_id": "sess_a",
        "created_at": "2026-09-08T10:00:00+00:00",
        "updated_at": "2026-09-08T10:00:00+00:00",
    }
    base.update(overrides)
    return MemoryRecord(**base)


def _cold(**overrides) -> MemoryRecord:
    base = {
        "memory_id": make_memory_id(),
        "content": "User prefers FastAPI for backend projects.",
        "memory_type": "cold",
        "category": "preference",
        "source": "user_message",
        "session_id": None,
        "created_at": "2026-09-08T10:00:00+00:00",
        "updated_at": "2026-09-08T10:00:00+00:00",
    }
    base.update(overrides)
    return MemoryRecord(**base)


@pytest.fixture()
def embedding_service():
    return EmbeddingService()


@pytest.fixture()
def memory_index(tmp_chroma_root, embedding_service):
    return MemoryChromaStore(
        chroma_root=tmp_chroma_root,
        embedding_service=embedding_service,
    )


class TestMemoryChromaIndexing:
    def test_add_hot_memory(self, memory_index):
        record = _hot()
        mid = memory_index.add_memory(record)
        assert mid == record.memory_id
        assert memory_index.count("hot") == 1
        assert record.memory_id in memory_index.get_indexed_memory_ids("hot")

    def test_add_cold_memory(self, memory_index):
        record = _cold()
        memory_index.add_memory(record)
        assert memory_index.count("cold") == 1
        assert memory_index.count("hot") == 0

    def test_separate_collections(self, memory_index):
        memory_index.add_memory(_hot())
        memory_index.add_memory(_cold())
        assert memory_index.count("hot") == 1
        assert memory_index.count("cold") == 1

    def test_update_memory_reindexes(self, memory_index):
        record = _hot(content="Original content")
        memory_index.add_memory(record)

        updated = MemoryRecord.from_dict({
            **record.to_dict(),
            "content": "Updated content for the task.",
            "updated_at": "2026-09-08T11:00:00+00:00",
        })
        memory_index.update_memory(updated)

        assert memory_index.count("hot") == 1
        results = memory_index.search_memory(
            "Updated content",
            memory_type="hot",
            top_k=1,
            session_id="sess_a",
        )
        assert len(results) == 1
        assert results[0].content == "Updated content for the task."

    def test_delete_memory(self, memory_index):
        record = _hot()
        memory_index.add_memory(record)
        assert memory_index.delete_memory(record.memory_id, "hot") is True
        assert memory_index.count("hot") == 0
        assert memory_index.delete_memory(record.memory_id, "hot") is False

    def test_delete_cold_memory(self, memory_index):
        record = _cold()
        memory_index.add_memory(record)
        assert memory_index.delete_memory(record.memory_id, "cold") is True
        assert memory_index.count("cold") == 0


class TestMemoryChromaSearch:
    def test_search_hot_returns_results(self, memory_index):
        record = _hot(content="Use FastAPI with pydantic models.")
        memory_index.add_memory(record)
        results = memory_index.search_memory(
            "FastAPI pydantic",
            memory_type="hot",
            top_k=5,
            session_id="sess_a",
        )
        assert len(results) == 1
        assert results[0].memory_id == record.memory_id
        assert results[0].memory_type == "hot"

    def test_search_cold_returns_results(self, memory_index):
        record = _cold(content="My name is Ojasvi.")
        memory_index.add_memory(record)
        results = memory_index.search_memory(
            "What is my name",
            memory_type="cold",
            top_k=5,
        )
        assert len(results) == 1
        assert results[0].memory_id == record.memory_id

    def test_search_empty_collection(self, memory_index):
        results = memory_index.search_memory(
            "anything",
            memory_type="hot",
            top_k=5,
        )
        assert results == []

    def test_top_k_limits_results(self, memory_index):
        for i in range(5):
            memory_index.add_memory(_hot(
                content=f"Task instruction number {i}",
                session_id="sess_a",
            ))
        results = memory_index.search_memory(
            "Task instruction",
            memory_type="hot",
            top_k=2,
            session_id="sess_a",
        )
        assert len(results) <= 2

    def test_session_id_filter(self, memory_index):
        r_a = _hot(content="Session A task alpha", session_id="sess_a")
        r_b = _hot(content="Session B task beta", session_id="sess_b")
        memory_index.add_memory(r_a)
        memory_index.add_memory(r_b)

        results_a = memory_index.search_memory(
            "task",
            memory_type="hot",
            top_k=10,
            session_id="sess_a",
        )
        assert len(results_a) == 1
        assert results_a[0].session_id == "sess_a"
        assert results_a[0].memory_id == r_a.memory_id

        results_b = memory_index.search_memory(
            "task",
            memory_type="hot",
            top_k=10,
            session_id="sess_b",
        )
        assert len(results_b) == 1
        assert results_b[0].memory_id == r_b.memory_id

    def test_session_isolation_in_search(self, memory_index):
        memory_index.add_memory(_hot(content="Secret for session A", session_id="sess_a"))
        memory_index.add_memory(_hot(content="Secret for session B", session_id="sess_b"))

        results = memory_index.search_memory(
            "Secret",
            memory_type="hot",
            top_k=10,
            session_id="sess_b",
        )
        assert all(r.session_id == "sess_b" for r in results)
        assert not any("session A" in r.content for r in results)

    def test_category_filter(self, memory_index):
        memory_index.add_memory(_cold(content="Prefers FastAPI", category="preference"))
        memory_index.add_memory(_cold(content="Project uses PostgreSQL", category="project"))

        pref_results = memory_index.search_memory(
            "FastAPI PostgreSQL",
            memory_type="cold",
            top_k=10,
            category="preference",
        )
        assert len(pref_results) == 1
        assert pref_results[0].category == "preference"

    def test_is_summary_filter(self, memory_index):
        memory_index.add_memory(_hot(
            content="Raw discussion detail.",
            session_id="sess_a",
            is_summary=False,
        ))
        summary = _hot(
            content="Summary of recent discussion.",
            session_id="sess_a",
            category="summary",
            is_summary=True,
            parent_memory_ids=[make_memory_id()],
        )
        memory_index.add_memory(summary)

        summary_results = memory_index.search_memory(
            "discussion",
            memory_type="hot",
            top_k=10,
            session_id="sess_a",
            is_summary=True,
        )
        assert len(summary_results) == 1
        assert summary_results[0].is_summary is True

    def test_importance_filter(self, memory_index):
        memory_index.add_memory(_cold(content="Low importance fact", importance=0.2))
        memory_index.add_memory(_cold(content="High importance fact", importance=0.9))

        results = memory_index.search_memory(
            "importance fact",
            memory_type="cold",
            top_k=10,
            min_importance=0.8,
        )
        assert len(results) == 1
        assert results[0].importance >= 0.8

    def test_created_at_filter(self, memory_index):
        memory_index.add_memory(_cold(
            content="Older preference",
            created_at="2026-09-01T10:00:00+00:00",
            updated_at="2026-09-01T10:00:00+00:00",
        ))
        memory_index.add_memory(_cold(
            content="Newer preference",
            created_at="2026-09-08T10:00:00+00:00",
            updated_at="2026-09-08T10:00:00+00:00",
        ))

        results = memory_index.search_memory(
            "preference",
            memory_type="cold",
            top_k=10,
            created_after="2026-09-05T00:00:00+00:00",
        )
        assert len(results) == 1
        assert "Newer" in results[0].content

    def test_created_before_filter(self, memory_index):
        memory_index.add_memory(_cold(
            content="Older preference",
            created_at="2026-09-01T10:00:00+00:00",
            updated_at="2026-09-01T10:00:00+00:00",
        ))
        memory_index.add_memory(_cold(
            content="Newer preference",
            created_at="2026-09-08T10:00:00+00:00",
            updated_at="2026-09-08T10:00:00+00:00",
        ))

        results = memory_index.search_memory(
            "preference",
            memory_type="cold",
            top_k=10,
            created_before="2026-09-05T00:00:00+00:00",
        )
        assert len(results) == 1
        assert "Older" in results[0].content

    def test_created_range_filter(self, memory_index):
        memory_index.add_memory(_cold(content="First", created_at="2026-09-01T10:00:00+00:00"))
        memory_index.add_memory(_cold(content="Second", created_at="2026-09-05T10:00:00+00:00"))
        memory_index.add_memory(_cold(content="Third", created_at="2026-09-10T10:00:00+00:00"))

        results = memory_index.search_memory(
            "item",
            memory_type="cold",
            top_k=10,
            created_after="2026-09-03T00:00:00+00:00",
            created_before="2026-09-07T00:00:00+00:00",
        )
        assert len(results) == 1
        assert results[0].content == "Second"

    def test_max_importance_filter(self, memory_index):
        memory_index.add_memory(_cold(content="Low priority task", importance=0.2))
        memory_index.add_memory(_cold(content="High priority task", importance=0.9))

        results = memory_index.search_memory(
            "task",
            memory_type="cold",
            top_k=10,
            max_importance=0.5,
        )
        assert len(results) == 1
        assert results[0].importance <= 0.5

    def test_importance_range_filter(self, memory_index):
        memory_index.add_memory(_cold(content="Low", importance=0.2))
        memory_index.add_memory(_cold(content="Medium", importance=0.5))
        memory_index.add_memory(_cold(content="High", importance=0.9))

        results = memory_index.search_memory(
            "item",
            memory_type="cold",
            top_k=10,
            min_importance=0.3,
            max_importance=0.7,
        )
        assert len(results) == 1
        assert results[0].content == "Medium"

    def test_combined_multiple_filters(self, memory_index):
        memory_index.add_memory(_hot(
            content="Session A task alpha",
            session_id="sess_1",
            category="task",
            importance=0.8,
            is_summary=False,
            created_at="2026-09-08T10:00:00+00:00",
        ))
        memory_index.add_memory(_hot(
            content="Session A task beta summary",
            session_id="sess_1",
            category="summary",
            importance=0.8,
            is_summary=True,
            created_at="2026-09-08T10:00:00+00:00",
        ))
        memory_index.add_memory(_hot(
            content="Session B task gamma",
            session_id="sess_2",
            category="task",
            importance=0.8,
            is_summary=False,
            created_at="2026-09-08T10:00:00+00:00",
        ))

        results = memory_index.search_memory(
            "task",
            memory_type="hot",
            top_k=10,
            session_id="sess_1",
            category="task",
            is_summary=False,
            min_importance=0.5,
            created_after="2026-09-07T00:00:00+00:00",
        )
        assert len(results) == 1
        assert results[0].content == "Session A task alpha"

    def test_search_result_to_dict(self, memory_index):
        memory_index.add_memory(_cold())
        results = memory_index.search_memory("FastAPI", memory_type="cold", top_k=1)
        d = results[0].to_dict()
        assert "derived_similarity" in d
        assert "distance" in d


class TestMemoryChromaWithCanonicalStore:
    def test_index_matches_canonical_store(self, tmp_memory_root, tmp_chroma_root, embedding_service):
        store = MemoryStore(tmp_memory_root)
        index = MemoryChromaStore(tmp_chroma_root, embedding_service)

        hot = _hot()
        cold = _cold()
        store.create(hot)
        store.create(cold)
        index.add_memory(hot)
        index.add_memory(cold)

        assert index.count("hot") == 1
        assert index.count("cold") == 1
        assert len(store.list_hot()) == 1
        assert len(store.list_cold()) == 1


class TestMemoryChromaErrors:
    def test_invalid_memory_type_raises(self, memory_index):
        with pytest.raises(MemoryIndexError, match="memory_type must be"):
            memory_index.search_memory("q", memory_type="warm", top_k=5)

    def test_to_dict_includes_counts(self, memory_index):
        memory_index.add_memory(_hot())
        d = memory_index.to_dict()
        assert d["hot_count"] == 1
        assert d["cold_count"] == 0

    def test_uses_separate_collection_names(self, tmp_chroma_root, embedding_service):
        index = MemoryChromaStore(
            tmp_chroma_root,
            embedding_service,
            hot_collection="test_memory_hot",
            cold_collection="test_memory_cold",
        )
        assert index.to_dict()["hot_collection"] == "test_memory_hot"
        assert index.to_dict()["cold_collection"] == "test_memory_cold"

    def test_add_memory_invalid_type_raises(self, memory_index):
        record = _hot()
        object.__setattr__(record, "memory_type", "invalid_type")
        with pytest.raises(MemoryIndexError, match="Cannot index memory_type"):
            memory_index.add_memory(record)

    def test_delete_nonexistent_returns_false(self, memory_index):
        assert memory_index.delete_memory("mem_nonexistent", "hot") is False
        assert memory_index.delete_memory("mem_nonexistent", "cold") is False
