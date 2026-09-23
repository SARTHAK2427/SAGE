import json

import pytest

from flash.service import FlashService, _document_context, _parse_route


def test_flash_route_parser_accepts_flash_schema_without_legacy_type():
    assert _parse_route('{"flash_case":"A","gemma_answer":"hello"}')["gemma_answer"] == "hello"


def test_simple_text_attachment_enters_shared_context(tmp_path):
    document = tmp_path / "notes.txt"
    document.write_text("remember this exact requirement", encoding="utf-8")
    context = _document_context(
        "summarize",
        [{"ref": "file_1", "name": "notes.txt", "type": "txt"}],
        {"file_1": {"path": str(document), "type": "txt"}},
    )
    assert "remember this exact requirement" in context


@pytest.mark.parametrize(
    ("route", "worker", "expected", "calls"),
    [
        ({"flash_case": "A", "gemma_answer": "direct"}, None, "direct", ["gemma"]),
        ({"flash_case": "B", "qwen": {"request": "read", "final": True}}, "visual", "visual", ["gemma", "qwen"]),
        ({"flash_case": "C", "qwen": {"request": "extract", "final": False}}, "evidence", "synthesized", ["gemma", "qwen", "gemma"]),
        ({"flash_case": "D", "gemma_answer": "text", "qwen": {"request": "inspect", "final": True}}, "visual", "text\n\nvisual", ["gemma", "qwen"]),
    ],
)
def test_flash_cases(monkeypatch, tmp_path, route, worker, expected, calls):
    image = tmp_path / "sample.png"
    image.write_bytes(b"not-a-real-image-but-sufficient-for-transport")
    seen = []
    gemma_count = 0

    def fake_invoke(role, messages, **kwargs):
        nonlocal gemma_count
        seen.append(role)
        if role == "gemma":
            gemma_count += 1
            content = json.dumps(route) if gemma_count == 1 else "synthesized"
        else:
            content = worker
        return {"content": content, "duration": 0.01, "usage": {}, "timings": {}}

    monkeypatch.setattr("flash.service.flash_transport.invoke", fake_invoke)

    needs_image = route["flash_case"] != "A"
    result = FlashService().run(
        objective="test",
        attachments=[{"ref": "file_1", "name": "sample.png", "type": "png"}] if needs_image else [],
        file_map={"file_1": {"path": str(image), "type": "png"}} if needs_image else {},
        session_id="session_test",
    )
    assert result["answer"] == expected
    assert result["flash_case"] == route["flash_case"]
    assert seen == calls


def test_gemma_remains_authoritative_for_ambiguous_image_turn(monkeypatch, tmp_path):
    image = tmp_path / "screen.png"
    image.write_bytes(b"image")
    seen = []

    def fake_invoke(role, messages, **kwargs):
        seen.append(role)
        content = '{"flash_case":"A","gemma_answer":"The attachment is unrelated to your question."}'
        return {"content": content, "duration": 0.01, "usage": {}, "timings": {}}

    monkeypatch.setattr("flash.service.flash_transport.invoke", fake_invoke)

    result = FlashService().run(
        objective="What did I say before?",
        attachments=[{"ref": "file_1", "name": "screen.png", "type": "png"}],
        file_map={"file_1": {"path": str(image), "type": "png"}},
        session_id="session_test",
    )

    assert result["answer"] == "The attachment is unrelated to your question."
    assert result["flash_case"] == "A"
    assert "routing_override" not in result
    assert seen == ["gemma"]


def test_flash_injects_canonical_memory_and_persists_completed_turn(monkeypatch):
    captured = {}

    def fake_invoke(role, messages, **_kwargs):
        captured["messages"] = messages
        return {"content": '{"flash_case":"A","gemma_answer":"done"}', "duration": 0.01, "usage": {}, "timings": {}}

    persisted = []
    monkeypatch.setattr("flash.service.flash_transport.invoke", fake_invoke)
    monkeypatch.setattr("flash.service._durable_memory_context", lambda _session, _user: "PERSISTENT MEMORY:\n- [preference] User prefers Python.")
    monkeypatch.setattr("flash.service._persist_flash_turn", lambda *args: persisted.append(args))

    result = FlashService().run(
        objective="What language do I prefer?",
        attachments=[],
        file_map={},
        session_id="session_test",
        user_id="local_user",
    )

    assert "PERSISTENT MEMORY:" in captured["messages"][1]["content"]
    assert persisted == [("session_test", "local_user", "What language do I prefer?", "done")]
    assert result["chat_id"] == "session_test"
    assert result["memory_job"] is None


def test_flash_persists_and_reloads_from_local_sqlite(monkeypatch, tmp_path):
    from sage_memory import sage_memory, set_sqlite_path

    db_path = str(tmp_path / "flash_local_memory.db")
    monkeypatch.setenv("SAGE_MEMORY_DB", "sqlite")
    set_sqlite_path(db_path)
    chat_id = "verify_local_memory"
    seen_roles = []
    captured = {}

    def fake_invoke(role, messages, **_kwargs):
        seen_roles.append(role)
        captured["prompt"] = messages[1]["content"]
        return {"content": '{"flash_case":"A","gemma_answer":"You like espresso."}', "duration": 0.01, "usage": {}, "timings": {}}

    monkeypatch.setattr("flash.service.flash_transport.invoke", fake_invoke)
    service = FlashService()
    first = service.run(
        objective="Remember that I like espresso.",
        attachments=[],
        file_map={},
        session_id=chat_id,
        user_id="local_user",
    )
    messages = sage_memory.get_messages(chat_id)
    second = service.run(
        objective="What do I like?",
        attachments=[],
        file_map={},
        session_id=chat_id,
        user_id="local_user",
    )
    set_sqlite_path(None)

    assert first["memory_job"] is None
    assert "memory" not in seen_roles
    assert len(messages) == 2
    assert messages[0]["content"] == "Remember that I like espresso."
    assert "RECENT CONVERSATION:" in captured["prompt"]
    assert "espresso" in captured["prompt"]
    assert second["history_saved"] is True
