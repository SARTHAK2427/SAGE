"""Smoke tests for chat list/clear helpers and Flash role probe messaging."""

from __future__ import annotations

import os
import uuid

import pytest

from sage_memory import sage_memory, set_sqlite_path


@pytest.fixture()
def isolated_sqlite(tmp_path):
    db_path = str(tmp_path / "chat_api.db")
    os.environ["SAGE_MEMORY_DB"] = "sqlite"
    set_sqlite_path(db_path)
    yield db_path
    set_sqlite_path(None)


def test_list_and_clear_chat(isolated_sqlite):
    chat_id = f"chat_{uuid.uuid4().hex[:8]}"
    user_id = "local_user"
    sage_memory.write_message(chat_id, user_id, "user", "Hello from history test")
    sage_memory.write_message(chat_id, user_id, "assistant", "Reply stored")
    chats = sage_memory.list_chats(user_id)
    assert any(item["chat_id"] == chat_id for item in chats)
    messages = sage_memory.get_messages(chat_id)
    assert len(messages) == 2
    cleared = sage_memory.clear_chat(chat_id, user_id=user_id)
    assert cleared["messages_removed"] == 2
    assert sage_memory.get_messages(chat_id) == []
    assert all(item["chat_id"] != chat_id for item in sage_memory.list_chats(user_id))


def test_mock_role_probe_reports_healthy(monkeypatch):
    monkeypatch.setenv("SAGE_MOCK_MODE", "1")
    from flash.transport import flash_transport
    result = flash_transport.test_role("gemma")
    assert result["healthy"] is True
    assert "mock" in (result.get("message") or "").lower() or result.get("probe") == "mock"
