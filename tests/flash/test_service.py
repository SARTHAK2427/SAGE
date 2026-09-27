import json
from types import SimpleNamespace

import pytest

from flash.service import FlashService, _document_context, _parse_route, _recent_attachment_evidence


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


def test_ingested_document_uses_rich_rag_socket_and_observer(monkeypatch):
    import db_service

    events = []
    socket = {
        "status": "success",
        "operation": "rag_search",
        "result": {
            "count": 1,
            "records": [{
                "record_id": "chunk_7",
                "doc_id": "doc_123",
                "page": 4,
                "text": "The warranty period is three years.",
                "derived_cosine_similarity": 0.91,
            }],
        },
        "execution": {"embedding_model": "BAAI/bge-m3", "reranker_enabled": False},
        "timing": {"duration_ms": 12.5, "stages": {"embedding_ms": 8.0}},
        "warnings": [],
        "error": None,
    }
    fake_db = SimpleNamespace(rag_search_socket=lambda **kwargs: socket)
    monkeypatch.setattr(db_service, "document_db", fake_db)
    monkeypatch.setattr("flash.service.observer.emit", lambda *args, **kwargs: events.append((args, kwargs)))

    context = _document_context(
        "How long is the warranty?",
        [{"ref": "file_1", "name": "manual.pdf", "type": "pdf", "doc_id": "doc_123"}],
        {"file_1": {"type": "pdf", "doc_id": "doc_123"}},
        run_id="run-rag",
    )

    assert "The warranty period is three years." in context
    assert "record=chunk_7" in context
    assert "similarity=0.91" in context
    assert [event[0][3] for event in events] == ["started", "completed"]
    assert events[1][0][5]["execution"]["embedding_model"] == "BAAI/bge-m3"


def test_ingested_document_rag_failure_is_visible_but_non_blocking(monkeypatch):
    import db_service

    events = []
    fake_db = SimpleNamespace(rag_search_socket=lambda **kwargs: {
        "status": "error",
        "result": {"records": []},
        "timing": {"duration_ms": 2.0},
        "error": {"message": "index unavailable"},
    })
    monkeypatch.setattr(db_service, "document_db", fake_db)
    monkeypatch.setattr("flash.service.observer.emit", lambda *args, **kwargs: events.append((args, kwargs)))

    context = _document_context(
        "Summarize it",
        [{"ref": "file_1", "name": "manual.pdf", "type": "pdf", "doc_id": "doc_123"}],
        {"file_1": {"type": "pdf", "doc_id": "doc_123"}},
        run_id="run-rag-error",
    )

    assert context == ""
    assert [event[0][3] for event in events] == ["started", "failed"]
    assert "index unavailable" in events[1][0][5]["error"]


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
        save_history=False,
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
        save_history=False,
    )

    assert result["answer"] == "The attachment is unrelated to your question."
    assert result["flash_case"] == "A"
    assert "routing_override" not in result
    assert seen == ["gemma"]


def test_document_visual_route_is_corrected_to_grounded_gemma_answer(monkeypatch):
    seen = []
    responses = iter([
        '{"flash_case":"C","qwen":{"request":"Read the document","final":false}}',
        '{"flash_case":"A","gemma_answer":"Quick overview: the warranty lasts three years.\\n\\nDetailed: coverage is valid for three years."}',
    ])

    def fake_invoke(role, _messages, **_kwargs):
        seen.append(role)
        return {"content": next(responses), "duration": 0.01, "usage": {}, "timings": {}}

    monkeypatch.setattr("flash.service.flash_transport.invoke", fake_invoke)
    monkeypatch.setattr("flash.service._document_context", lambda *_args, **_kwargs: "[DOCUMENT file_1] Warranty period: three years.")

    result = FlashService().run(
        objective="Give a quick overview, then explain this document in detail.",
        attachments=[{"ref": "file_1", "name": "manual.pdf", "type": "pdf", "doc_id": "doc_123"}],
        file_map={"file_1": {"type": "pdf", "doc_id": "doc_123"}},
        session_id="document-session",
        save_history=False,
    )

    assert result["flash_case"] == "A"
    assert result["answer"].startswith("Quick overview")
    assert seen == ["gemma", "gemma"]


def test_flash_performs_model_requested_memory_recall(monkeypatch):
    captured = []
    gemma_calls = 0

    def fake_invoke(role, messages, **_kwargs):
        nonlocal gemma_calls
        gemma_calls += 1
        captured.append(messages)
        if gemma_calls == 1:
            return {"content": '{"flash_case":"A","memory":{"query":"language preference","scope":"global"}}', "duration": 0.01, "usage": {}, "timings": {}}
        return {"content": '{"flash_case":"A","gemma_answer":"done"}', "duration": 0.01, "usage": {}, "timings": {}}

    monkeypatch.setattr("flash.service.flash_transport.invoke", fake_invoke)
    monkeypatch.setattr("flash.service.memory_coordinator.search", lambda **_kwargs: {
        "memories": [{"memory_id": "mem-1", "content": "User prefers Python."}],
        "context": "[GLOBAL MEMORY mem-1] User prefers Python.", "count": 1,
    })

    result = FlashService().run(
        objective="What language do I prefer?",
        attachments=[],
        file_map={},
        session_id="session_test",
        user_id="local_user", save_history=False,
    )

    assert len(captured) == 2
    assert "GLOBAL MEMORY" in captured[1][2]["content"]
    assert result["chat_id"] == "session_test"
    assert result["recalled_memory_ids"] == ["mem-1"]


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
    # This test verifies the synchronous conversation ledger. Background
    # curator scheduling has its own contract tests and must not outlive this
    # test's temporary SQLite override.
    monkeypatch.setattr("flash.memory_worker.memory_worker.schedule_turn", lambda **_kwargs: None)
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
    assert "RECENT CONVERSATION (verbatim, bounded):" in captured["prompt"]
    assert "espresso" in captured["prompt"]
    assert second["history_saved"] is True


def test_flash_frames_a_short_reply_as_a_current_user_message(monkeypatch):
    captured = {}

    monkeypatch.setattr("flash.service.memory_coordinator.recent_turns", lambda *_a, **_k: {
        "text": "USER: What is my name?\nASSISTANT: I do not know your name yet.",
        "turn_count": 1, "token_estimate": 20,
    })
    monkeypatch.setattr("flash.service._recent_attachment_evidence", lambda *_a, **_k: ("", []))
    monkeypatch.setattr("flash.service._global_memory_context", lambda *_a, **_k: ("", []))
    monkeypatch.setattr("flash.service.memory_coordinator.persist_turn", lambda **_k: {"memory_jobs": []})

    def fake_invoke(_role, messages, **_kwargs):
        captured["messages"] = messages
        return {"content": '{"flash_case":"A","gemma_answer":"Thanks, Rakshit Jain."}',
                "duration": 0.01, "usage": {}, "timings": {}}

    monkeypatch.setattr("flash.service.flash_transport.invoke", fake_invoke)
    result = FlashService().run(
        objective="Rakshit Jain", attachments=[], file_map={}, session_id="reply-chat", user_id="local_user",
    )

    assert "CURRENT USER MESSAGE (authoritative):\nRakshit Jain" in captured["messages"][1]["content"]
    assert "short fragment can directly answer" in captured["messages"][0]["content"]
    assert result["answer"] == "Thanks, Rakshit Jain."


def test_flash_prompt_requires_context_check_before_missing_user_details():
    import config

    prompt = (config.PROMPTS_DIR / "flash_system.txt").read_text(encoding="utf-8")
    assert "Before saying that a user detail is unknown" in prompt
    assert "request memory before saying it is unavailable" in prompt
    assert "Replace a placeholder when the matching fact is known" in prompt
    assert "Never claim that you edited, filled, or updated a prior already-sent response" in prompt


def test_flash_does_not_eagerly_search_memory_for_a_simple_turn(monkeypatch):
    captured = {}
    searches = []

    def fake_invoke(role, messages, **_kwargs):
        captured["messages"] = messages
        return {"content": '{"flash_case":"A","gemma_answer":"Hello!"}',
                "duration": 0.01, "usage": {}, "timings": {}}

    monkeypatch.setattr("flash.service.flash_transport.invoke", fake_invoke)
    monkeypatch.setattr("flash.service.memory_coordinator.recent_turns", lambda *_args, **_kwargs: {
        "text": "", "turn_count": 0, "token_estimate": 0,
    })
    monkeypatch.setattr("flash.service.memory_coordinator.search", lambda **kwargs: searches.append(kwargs))
    monkeypatch.setattr("flash.service.memory_coordinator.persist_turn", lambda **_kwargs: {"memory_jobs": []})

    result = FlashService().run(
        objective="hi", attachments=[], file_map={},
        session_id="brand-new-chat", user_id="local_user", save_history=True,
    )

    assert "RELEVANT GLOBAL MEMORY" not in captured["messages"][1]["content"]
    assert searches == []
    assert result["answer"] == "Hello!"


def test_flash_injects_valid_global_memory_without_semantic_search(monkeypatch):
    from sage_memory import sage_memory

    records = [
        {"memory_id": "bad-1", "category": "personal", "source_chat_id": None, "content": "User's name is not disclosed; user explicitly stated they do not know their name."},
        {"memory_id": "name-1", "category": "personal", "source_chat_id": None, "content": "User name is Rakshit Jain"},
        {"memory_id": "cold-1", "source_chat_id": "other-chat", "content": "A chat-specific project detail."},
        {"memory_id": "bad-2", "category": "personal", "source_chat_id": None, "content": "User explicitly states they do not know the assistant's name."},
    ]
    captured = {}

    monkeypatch.setattr(sage_memory, "list_memories", lambda **_kwargs: records)
    monkeypatch.setattr("flash.service.memory_coordinator.recent_turns", lambda *_a, **_k: {
        "text": "", "turn_count": 0, "token_estimate": 0,
    })
    monkeypatch.setattr("flash.service._recent_attachment_evidence", lambda *_a, **_k: ("", []))
    monkeypatch.setattr("flash.service.memory_coordinator.search", lambda **_kwargs: pytest.fail("simple global injection must not use semantic search"))
    monkeypatch.setattr("flash.service.memory_coordinator.persist_turn", lambda **_kwargs: {"memory_jobs": []})
    monkeypatch.setattr(
        "flash.service.flash_transport.invoke",
        lambda _role, messages, **_kwargs: captured.setdefault("messages", messages) and {
            "content": '{"flash_case":"A","gemma_answer":"Your name is Rakshit Jain."}',
            "duration": 0.01, "usage": {}, "timings": {},
        },
    )

    result = FlashService().run(
        objective="What is my name?", attachments=[], file_map={},
        session_id="new-chat", user_id="local_user", save_history=True,
    )

    system_prompt = captured["messages"][0]["content"]
    user_prompt = captured["messages"][1]["content"]
    assert "AUTHORITATIVE KNOWN USER PERSONAL INFORMATION" in system_prompt
    assert "User name is Rakshit Jain" in system_prompt
    assert "not disclosed" not in system_prompt
    assert "assistant's name" not in system_prompt
    assert "chat-specific project" not in system_prompt
    assert "User name is Rakshit Jain" not in user_prompt
    assert result["recalled_memory_ids"] == ["name-1"]


def test_global_memory_quality_rejects_unknown_name_variants():
    from memory_system.quality import is_valid_memory

    assert not is_valid_memory("User's name is not disclosed; user explicitly stated they do not know their name.", "global")
    assert not is_valid_memory("User explicitly states they do not know the assistant's name.", "global")
    assert is_valid_memory("User name is Rakshit Jain", "global")


def test_flash_injects_prior_chat_attachment_evidence(monkeypatch):
    captured = {}

    def fake_invoke(role, messages, **_kwargs):
        captured["prompt"] = messages[1]["content"]
        return {"content": '{"flash_case":"A","gemma_answer":"The button is disabled."}', "duration": 0.01, "usage": {}, "timings": {}}

    monkeypatch.setattr("flash.service.flash_transport.invoke", fake_invoke)
    monkeypatch.setattr("flash.service.memory_coordinator.recent_turns", lambda *_a, **_k: {"text": "", "turn_count": 0, "token_estimate": 0})
    monkeypatch.setattr("flash.service.memory_coordinator.search", lambda **_k: {"memories": [], "context": "", "count": 0})
    monkeypatch.setattr("flash.service.memory_coordinator.persist_turn", lambda **_k: {"memory_jobs": []})
    monkeypatch.setattr("flash.service._recent_attachment_evidence", lambda *_a, **_k: (
        "CHAT ATTACHMENT EVIDENCE\\nAttachments: screen.png (png)\\n\\nVISION EVIDENCE: The save button is disabled.", ["evidence-1"]
    ))

    result = FlashService().run(
        objective="Why was that button disabled?", attachments=[], file_map={},
        session_id="attachment-chat", user_id="local_user", save_history=True,
    )

    assert "CHAT-SCOPED ATTACHMENT EVIDENCE" in captured["prompt"]
    assert "save button is disabled" in captured["prompt"]
    assert result["attachment_memory_ids"] == ["evidence-1"]


def test_attachment_evidence_is_persisted_after_visual_review(monkeypatch, tmp_path):
    image = tmp_path / "screen.png"
    image.write_bytes(b"image")
    saved = {}

    def fake_invoke(role, _messages, **_kwargs):
        content = '{"flash_case":"B","qwen":{"request":"inspect","final":true}}' if role == "gemma" else "The save button is disabled."
        return {"content": content, "duration": 0.01, "usage": {}, "timings": {}}

    monkeypatch.setattr("flash.service.flash_transport.invoke", fake_invoke)
    monkeypatch.setattr("flash.service.memory_coordinator.recent_turns", lambda *_a, **_k: {"text": "", "turn_count": 0, "token_estimate": 0})
    monkeypatch.setattr("flash.service.memory_coordinator.search", lambda **_k: {"memories": [], "context": "", "count": 0})
    monkeypatch.setattr("flash.service.memory_coordinator.persist_turn", lambda **_k: {"memory_jobs": []})
    def capture_evidence(**kwargs):
        saved["kwargs"] = kwargs
        return ["evidence-new"]

    monkeypatch.setattr("flash.service._store_attachment_evidence", capture_evidence)

    FlashService().run(
        objective="Describe this image", attachments=[{"ref": "file_1", "name": "screen.png", "type": "png"}],
        file_map={"file_1": {"path": str(image), "type": "png"}}, session_id="attachment-chat", user_id="local_user",
    )

    assert saved["kwargs"]["visual_evidence"] == "The save button is disabled."
