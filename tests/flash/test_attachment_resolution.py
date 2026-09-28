from pathlib import Path

from flash.actions import flash_action_executor
from flash.service import _attachment_reference


def test_unnamed_reference_prefers_unique_current_attachment():
    items = [
        {"attachment_id": "old-image", "name": "old.png", "media_type": "png", "turn_number": 2, "turn_distance": 3, "uploaded_at": "old"},
        {"attachment_id": "new-pdf", "name": "network.pdf", "media_type": "pdf", "turn_number": 5, "turn_distance": 0, "uploaded_at": "new"},
    ]

    resolved = _attachment_reference(items, {"new-pdf"})

    assert resolved["status"] == "resolved"
    assert resolved["attachment_id"] == "new-pdf"
    assert resolved["reason"] == "attached_to_current_message"


def test_unnamed_reference_keeps_same_message_tie_ambiguous():
    items = [
        {"attachment_id": "a", "name": "a.pdf", "media_type": "pdf", "turn_number": 4, "turn_distance": 1, "uploaded_at": "same"},
        {"attachment_id": "b", "name": "b.png", "media_type": "png", "turn_number": 4, "turn_distance": 1, "uploaded_at": "same"},
    ]

    resolved = _attachment_reference(items, set())

    assert resolved["status"] == "ambiguous"
    assert {item["attachment_id"] for item in resolved["candidates"]} == {"a", "b"}


def test_document_image_uses_document_child_not_chat_screenshot(tmp_path, monkeypatch):
    import db_service
    import flash.actions as actions_module

    artifacts_root = tmp_path / "artifacts"
    image_path = artifacts_root / "doc_1" / "images" / "img_000003.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"third-document-image")

    class FakeDocumentDB:
        def list_artifacts(self, doc_id, artifact_type=None):
            assert doc_id == "doc_1"
            assert artifact_type == "image"
            return [
                {"element_id": "img_000001", "order": 1, "page": 1},
                {"element_id": "img_000002", "order": 2, "page": 2},
                {"element_id": "img_000003", "order": 3, "page": 4, "caption": "Third figure"},
            ]

        def artifact_fetch(self, doc_id, element_id):
            assert doc_id == "doc_1"
            assert element_id == "img_000003"
            return {"local_path": str(image_path), "exists": True, "type": "image"}

    captured = {}

    def fake_invoke(role, messages, temperature=None, **kwargs):
        captured["role"] = role
        captured["messages"] = messages
        return {"content": "The third PDF image shows a router diagram.", "duration": 0.1, "usage": {}}

    monkeypatch.setattr(db_service, "document_db", FakeDocumentDB())
    monkeypatch.setattr(actions_module.config, "ARTIFACTS_ROOT", artifacts_root)
    monkeypatch.setattr(actions_module.flash_transport, "invoke", fake_invoke)
    monkeypatch.setattr(actions_module.observer, "emit", lambda *args, **kwargs: None)

    result = flash_action_executor._document_image(
        {"attachment_ids": ["pdf-att"], "image_number": 3, "instruction": "Explain the third image."},
        "What is the third image in the document about?",
        [
            {"attachment_id": "screen-att", "name": "dashboard.png", "media_type": "png", "storage_path": "ignored"},
            {"attachment_id": "pdf-att", "name": "network.pdf", "media_type": "pdf", "doc_id": "doc_1"},
        ],
        "test-run", 0.1,
    )

    assert captured["role"] == "qwen"
    assert result["attachment_id"] == "pdf-att"
    assert result["image_number"] == 3
    assert result["element_id"] == "img_000003"
    assert result["page"] == 4
    assert "router diagram" in result["evidence"]
