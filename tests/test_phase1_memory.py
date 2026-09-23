"""
tests/test_phase1_memory.py
Phase 1: Chat Continuity + Conversation Ledger Tests
"""

import os
import uuid
import pytest
from fastapi.testclient import TestClient

import config
from core.run_state import RunState
from sage_memory import sage_memory, set_sqlite_path, get_backend_type
from app import app, _prepare_chat_request


@pytest.fixture(autouse=True)
def use_sqlite_memory(tmp_path):
    """Force SQLite memory database backend for all Phase 1 tests."""
    orig_env = os.environ.get("SAGE_MEMORY_DB")
    os.environ["SAGE_MEMORY_DB"] = "sqlite"

    db_path = str(tmp_path / "test_memory.db")
    set_sqlite_path(db_path)
    sage_memory._initialized_backends.clear()

    yield

    set_sqlite_path(None)
    if orig_env is not None:
        os.environ["SAGE_MEMORY_DB"] = orig_env
    else:
        os.environ.pop("SAGE_MEMORY_DB", None)


def test_run_state_fields():
    """Verify chat_id and user_id fields exist on RunState with valid defaults."""
    state = RunState(user_text="hello")
    assert hasattr(state, "chat_id")
    assert hasattr(state, "user_id")
    assert state.chat_id == ""
    assert state.user_id == ""

    state_with_ids = RunState(chat_id="chat_123", user_id="user_abc", user_text="hello")
    assert state_with_ids.chat_id == "chat_123"
    assert state_with_ids.user_id == "user_abc"
    assert state_with_ids.to_dict()["chat_id"] == "chat_123"
    assert state_with_ids.to_dict()["user_id"] == "user_abc"


@pytest.mark.asyncio
async def test_prepare_chat_request_chat_id_handling():
    """Verify _prepare_chat_request generates missing chat_id and preserves supplied chat_id."""
    # Case 1: Missing chat_id -> generated
    state1, _, _ = await _prepare_chat_request(objective="Test prompt", files=None, chat_id=None)
    assert state1.chat_id.startswith("chat_")
    assert state1.user_id == config.DEFAULT_USER_ID

    # Case 2: Supplied chat_id -> preserved
    existing_id = "chat_custom_9999"
    state2, _, _ = await _prepare_chat_request(objective="Test prompt 2", files=None, chat_id=existing_id)
    assert state2.chat_id == existing_id
    assert state2.user_id == config.DEFAULT_USER_ID


def test_write_and_get_messages():
    """Verify write_message and get_messages persist user and assistant messages."""
    chat_id = "chat_test_ledger"
    user_id = "local_user"

    sage_memory.write_message(chat_id, user_id, "user", "What is Python?")
    sage_memory.write_message(chat_id, user_id, "assistant", "Python is a programming language.")

    msgs = sage_memory.get_messages(chat_id)
    assert len(msgs) == 2

    assert msgs[0]["chat_id"] == chat_id
    assert msgs[0]["user_id"] == user_id
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "What is Python?"

    assert msgs[1]["chat_id"] == chat_id
    assert msgs[1]["user_id"] == user_id
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "Python is a programming language."


def test_isolation_between_chats():
    """Verify messages from chat_A never appear in chat_B."""
    chat_a = "chat_AAA_111"
    chat_b = "chat_BBB_222"
    user_id = "local_user"

    sage_memory.write_message(chat_a, user_id, "user", "Message in Chat A")
    sage_memory.write_message(chat_b, user_id, "user", "Message in Chat B")

    window_a = sage_memory.get_recent_window(chat_a)
    window_b = sage_memory.get_recent_window(chat_b)

    assert "Message in Chat A" in window_a
    assert "Message in Chat B" not in window_a

    assert "Message in Chat B" in window_b
    assert "Message in Chat A" not in window_b


def test_recent_window_slicing_and_ordering():
    """Verify get_recent_window returns only the newest 5 messages in chronological order."""
    chat_id = "chat_window_test"
    user_id = "local_user"

    # Insert 7 sequential messages
    for i in range(1, 8):
        role = "user" if i % 2 == 1 else "assistant"
        sage_memory.write_message(chat_id, user_id, role, f"Message number {i}")

    # Fetch recent window with n=5
    window_str = sage_memory.get_recent_window(chat_id, n=5)
    lines = [line.strip() for line in window_str.splitlines() if line.strip()]

    assert lines[0] == "RECENT CONVERSATION:"
    # The header is line 0, so lines 1..5 contain the 5 messages
    msg_lines = lines[1:]
    assert len(msg_lines) == 5

    # Messages 1 and 2 should be excluded; messages 3..7 should be included in chronological order
    assert "Message number 1" not in window_str
    assert "Message number 2" not in window_str

    assert msg_lines[0] == "user: Message number 3"
    assert msg_lines[1] == "assistant: Message number 4"
    assert msg_lines[2] == "user: Message number 5"
    assert msg_lines[3] == "assistant: Message number 6"
    assert msg_lines[4] == "user: Message number 7"


def test_empty_chat_window():
    """Verify get_recent_window returns empty string for unknown/empty chat_id."""
    assert sage_memory.get_recent_window("") == ""
    assert sage_memory.get_recent_window("nonexistent_chat_id") == ""


def test_api_chat_endpoint_chat_id(monkeypatch):
    """Verify /api/chat handles chat_id in request and exposes chat_id in response."""
    monkeypatch.setenv("SAGE_MOCK_MODE", "1")
    client = TestClient(app)

    # 1. Request without chat_id -> API generates chat_id
    res1 = client.post("/api/chat", data={"objective": "My name is Sarthak."})
    assert res1.status_code == 200
    data1 = res1.json()
    assert "chat_id" in data1
    generated_chat_id = data1["chat_id"]
    assert generated_chat_id.startswith("chat_")

    # Verify message ledger contains the persisted turn
    msgs1 = sage_memory.get_messages(generated_chat_id)
    assert len(msgs1) == 2
    assert msgs1[0]["content"] == "My name is Sarthak."

    # 2. Request with supplied chat_id -> API preserves chat_id
    res2 = client.post(
        "/api/chat",
        data={"objective": "What is my name?", "chat_id": generated_chat_id}
    )
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["chat_id"] == generated_chat_id

    # Verify message ledger now contains 4 messages
    msgs2 = sage_memory.get_messages(generated_chat_id)
    assert len(msgs2) == 4
    assert msgs2[2]["content"] == "What is my name?"


def test_api_chat_stream_endpoint_chat_id(monkeypatch):
    """Verify /api/chat/stream includes chat_id in the done SSE event result."""
    monkeypatch.setenv("SAGE_MOCK_MODE", "1")
    client = TestClient(app)

    custom_chat_id = "chat_streaming_999"
    res = client.post(
        "/api/chat/stream",
        data={"objective": "Stream test prompt", "chat_id": custom_chat_id}
    )
    assert res.status_code == 200
    content = res.text
    assert "event: done" in content
    assert custom_chat_id in content


def test_postgres_configuration_and_sanitized_errors(monkeypatch):
    """Verify postgres backend fails clearly when unconfigured and error messages do not leak URL/credentials."""
    monkeypatch.setenv("SAGE_MEMORY_DB", "postgres")
    monkeypatch.setattr(config, "SAGE_DATABASE_URL", "")
    monkeypatch.delenv("SAGE_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    # 1. Unconfigured Postgres URL -> RuntimeError asking to set SAGE_DATABASE_URL
    with pytest.raises(RuntimeError) as exc_info:
        sage_memory._get_connection()
    assert "no database URL is configured" in str(exc_info.value)

    # 2. Configured invalid Postgres URL -> RuntimeError without exposing raw secret URL
    secret_url = "postgresql://user_secret:password_secret@127.0.0.1:54321/db_secret"
    monkeypatch.setenv("SAGE_DATABASE_URL", secret_url)

    with pytest.raises(RuntimeError) as exc_info2:
        sage_memory._get_connection()

    err_msg = str(exc_info2.value)
    assert (
        "Failed to connect to PostgreSQL" in err_msg
        or "psycopg2 is not installed" in err_msg
    )
    assert "password_secret" not in err_msg
    assert "user_secret" not in err_msg
    assert "db_secret" not in err_msg
