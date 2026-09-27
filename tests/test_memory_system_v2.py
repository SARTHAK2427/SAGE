"""Contract tests for the Hot/Cold/Global memory and Observer implementation."""

from __future__ import annotations

import os

import pytest

from core.observer import ObserverStore
from flash.memory_worker import FlashMemoryWorker
from memory_system.global_identity import global_memory_identity, same_global_memory
from memory_system.job_store import memory_job_store
from memory_system.coordinator import memory_coordinator
from memory_system.quality import is_valid_memory, memory_rejection_reason
from sage_memory import sage_memory, set_sqlite_path


@pytest.fixture()
def memory_db(tmp_path, monkeypatch):
    previous = os.environ.get("SAGE_MEMORY_DB")
    monkeypatch.setenv("SAGE_MEMORY_DB", "sqlite")
    set_sqlite_path(str(tmp_path / "memory-v2.db"))
    sage_memory._initialized_backends.clear()
    monkeypatch.setattr("memory_system.index_outbox.index_outbox.enqueue", lambda *_args, **_kwargs: "test-index-event")
    yield
    set_sqlite_path(None)
    if previous is None:
        os.environ.pop("SAGE_MEMORY_DB", None)
    else:
        os.environ["SAGE_MEMORY_DB"] = previous


def test_hot_context_is_last_five_completed_turns(memory_db):
    for number in range(1, 7):
        sage_memory.write_message("chat-a", "user-a", "user", f"user-{number}")
        sage_memory.write_message("chat-a", "user-a", "assistant", f"assistant-{number}")

    recent = memory_coordinator.recent_turns("chat-a", max_turns=5, token_budget=5000)

    assert recent["turn_count"] == 5
    assert len(recent["messages"]) == 10
    assert "user-1" not in recent["text"]
    assert "user-2" in recent["text"]
    assert "assistant-6" in recent["text"]


def test_global_extraction_is_immediate_and_cold_compression_waits_for_eviction(memory_db, monkeypatch):
    calls = []
    seen = set()

    def fake_schedule_turn(**kwargs):
        key = (kwargs["job_type"], kwargs["user_message"]["msg_id"], kwargs["assistant_message"]["msg_id"])
        if key in seen:
            return None
        seen.add(key)
        calls.append(kwargs)
        return f"job-{len(calls)}"

    monkeypatch.setattr("flash.memory_worker.memory_worker.schedule_turn", fake_schedule_turn)

    for number in range(1, 7):
        memory_coordinator.persist_turn(
            chat_id="chat-a", user_id="user-a",
            user_text=f"user-{number}", answer=f"assistant-{number}", run_id=f"run-{number}",
        )

    global_jobs = [call for call in calls if call["job_type"] == FlashMemoryWorker.GLOBAL_EXTRACT]
    cold_jobs = [call for call in calls if call["job_type"] == FlashMemoryWorker.COLD_COMPRESS]

    assert len(global_jobs) == 6
    assert len(cold_jobs) == 1
    assert cold_jobs[0]["user_message"]["content"] == "user-1"
    assert global_jobs[-1]["user_message"]["content"] == "user-6"


def test_scoped_recall_hydrates_only_current_chat_and_global(memory_db, monkeypatch):
    current = sage_memory.store_memory(
        user_id="user-a", content="current chat project", category="project",
        source_chat_id="chat-a", memory_tier="cold",
    )
    other = sage_memory.store_memory(
        user_id="user-a", content="another chat secret", category="project",
        source_chat_id="chat-b", memory_tier="cold",
    )
    global_memory = sage_memory.store_memory(
        user_id="user-a", content="User identity: Rakshit Jain", category="personal",
        source_chat_id=None, memory_tier="cold",
    )
    hits = [
        {"memory_id": current["memory_id"], "distance": 0.1},
        {"memory_id": other["memory_id"], "distance": 0.01},
        {"memory_id": global_memory["memory_id"], "distance": 0.2},
    ]
    monkeypatch.setattr("sage_memory_index.memory_vector_index.search_memory_vectors", lambda **_kwargs: hits)

    result = memory_coordinator.search(
        query="project preferences", user_id="user-a", chat_id="chat-a", scope="both", limit=10,
    )

    ids = {memory["memory_id"] for memory in result["memories"]}
    assert ids == {current["memory_id"], global_memory["memory_id"]}
    assert other["memory_id"] not in ids


def test_curator_parser_rejects_invalid_objects():
    raw = """{"memories":[
      {"content":"Useful episode","scope":"cold","category":"summary","importance":0.7,"confidence":0.9},
      {"content":"","scope":"global","category":"fact"},
      {"content":"Bad scope","scope":"other","category":"fact"}
    ]}"""
    parsed = FlashMemoryWorker._parse(raw)
    assert parsed == [{
        "content": "Useful episode", "scope": "cold", "category": "summary",
        "importance": 0.7, "confidence": 0.9,
    }]


def test_curator_allows_only_personal_category_for_global_memory():
    raw = """{"memories":[
      {"content":"User prefers Python.","scope":"global","category":"preference","importance":0.8,"confidence":0.9},
      {"content":"User name is Rakshit Jain.","scope":"global","category":"personal","importance":0.9,"confidence":0.99},
      {"content":"User is planning a baking job application.","scope":"cold","category":"project","importance":0.7,"confidence":0.9}
    ]}"""
    parsed = FlashMemoryWorker._parse(raw)
    assert parsed == [
        {"content": "User name is Rakshit Jain.", "scope": "global", "category": "personal", "importance": 0.9, "confidence": 0.99},
        {"content": "User is planning a baking job application.", "scope": "cold", "category": "project", "importance": 0.7, "confidence": 0.9},
    ]


def test_api_allows_only_personal_global_memory(memory_db):
    from fastapi.testclient import TestClient
    from app import app

    client = TestClient(app)
    rejected = client.post("/api/memory", json={
        "content": "User prefers Python.", "category": "preference", "scope": "global",
    })
    accepted = client.post("/api/memory", json={
        "content": "User name is Rakshit Jain.", "category": "personal", "scope": "global",
    })

    assert rejected.status_code == 400
    assert "reserved for personal information" in rejected.json()["detail"]
    assert accepted.status_code == 200
    assert accepted.json()["memory"]["category"] == "personal"


def test_global_memory_quality_rejects_uncertainty_without_rejecting_real_negative_preferences():
    assert memory_rejection_reason("User's name has not yet been disclosed.", "global")
    assert memory_rejection_reason("The assistant does not know the user's age.", "global")
    assert memory_rejection_reason("User initiated a greeting.", "global")
    assert memory_rejection_reason("It is recommended that the user upgrade RAM.", "global")
    assert memory_rejection_reason("User is questioning the assistant's knowledge of previous conversations.", "global")
    assert memory_rejection_reason("User prefers concise replies and is currently inquiring about PCs.", "global")
    assert memory_rejection_reason("Context suggests a preference for GPU-equipped laptops.", "global")
    assert is_valid_memory("User identity: Rakshit Jain, 21 years old.", "global")
    assert is_valid_memory("User owns an RTX 5060 Ti 16GB desktop GPU.", "global")
    assert is_valid_memory("User does not like coffee.", "global")


def test_global_identity_normalizes_rephrased_name_facts():
    assert global_memory_identity("The user's name is Rakshit Jain.") == ("identity.name", "rakshit jain")
    assert global_memory_identity("The user identified themselves as Rakshit Jain.") == ("identity.name", "rakshit jain")
    assert global_memory_identity("User's age is 21.") == ("identity.age", "21")
    assert same_global_memory("User prefers concise technical replies.", "User prefers concise technical replies.")


def test_global_curator_collapses_existing_name_duplicates(memory_db, monkeypatch):
    first = sage_memory.store_memory(
        user_id="user-a", content="The user's name is Rakshit Jain.", category="personal",
        source_chat_id=None, memory_tier="cold",
    )
    duplicate = sage_memory.store_memory(
        user_id="user-a", content="The user identified themselves as Rakshit Jain.", category="personal",
        source_chat_id=None, memory_tier="cold",
    )
    payload = {
        "session_id": "chat-a", "user_id": "user-a", "run_id": "run-dedup",
        "job_type": FlashMemoryWorker.GLOBAL_EXTRACT,
        "user_message": {"msg_id": "user-msg", "content": "My name is Rakshit Jain."},
        "assistant_message": {"msg_id": "assistant-msg", "content": "Nice to meet you."},
    }
    job_id = memory_job_store.create(key="dedup-test", user_id="user-a", chat_id="chat-a", payload=payload)
    assert job_id
    monkeypatch.setattr(
        "flash.memory_worker.flash_transport.invoke",
        lambda *_args, **_kwargs: {
            "content": '{"memories":[{"scope":"global","category":"personal","content":"User name is Rakshit Jain","importance":0.9,"confidence":0.99}]}',
            "duration": 0.001,
        },
    )
    worker = FlashMemoryWorker()
    assert worker._slots.acquire(blocking=False)
    worker._run(job_id, payload)

    active = sage_memory.list_memories(user_id="user-a", status="active", memory_tier="cold", limit=20)
    active_global = [memory for memory in active if not memory.get("source_chat_id")]
    assert len(active_global) == 1
    assert active_global[0]["memory_id"] in {first["memory_id"], duplicate["memory_id"]}


def test_scoped_recall_ignores_invalid_global_records(memory_db, monkeypatch):
    good = sage_memory.store_memory(
        user_id="user-a", content="User identity: Rakshit Jain, 21 years old.", category="personal",
        source_chat_id=None, memory_tier="cold",
    )
    bad = sage_memory.store_memory(
        user_id="user-a", content="User's name has not yet been disclosed.", category="personal",
        source_chat_id=None, memory_tier="cold",
    )
    monkeypatch.setattr("sage_memory_index.memory_vector_index.search_memory_vectors", lambda **_kwargs: [
        {"memory_id": bad["memory_id"], "distance": 0.01},
        {"memory_id": good["memory_id"], "distance": 0.1},
    ])

    result = memory_coordinator.search(
        query="What is my name?", user_id="user-a", chat_id="a-different-chat", scope="global", limit=5,
    )

    assert [item["memory_id"] for item in result["memories"]] == [good["memory_id"]]
    assert "not yet been disclosed" not in result["context"]


def test_observer_redacts_credentials_but_preserves_usage_tokens():
    store = ObserverStore()
    store.emit("run", "gemma", "model_input", "started", "test", {
        "api_key": "secret", "authorization": "Bearer secret",
        "token_estimate": 123, "usage": {"completion_tokens": 45},
        "image": "data:image/png;base64,AAAA",
    })
    payload = store.events("run")[0]["payload"]
    assert payload["api_key"] == "[redacted]"
    assert payload["authorization"] == "[redacted]"
    assert payload["token_estimate"] == 123
    assert payload["usage"]["completion_tokens"] == 45
    assert payload["image"] == "[base64 image redacted]"
