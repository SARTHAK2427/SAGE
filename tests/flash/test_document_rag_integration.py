"""Connected regression test for Daksh document RAG inside Flash mode."""

from flash.service import _document_context
from sage_document_db import SageDocumentDB


def test_ingest_retrieve_and_supply_document_context_to_flash(tmp_path, monkeypatch):
    import db_service

    source = tmp_path / "warranty.txt"
    source.write_text(
        "SAGE DEVICE MANUAL\n\nThe cobalt-series warranty period is exactly three years.\n",
        encoding="utf-8",
    )
    document_db = SageDocumentDB(
        artifacts_root=tmp_path / "artifacts",
        chroma_root=tmp_path / "chroma",
    )
    ingestion = document_db.ingest_document(source)
    assert ingestion["status"] == "success"
    doc_id = ingestion["doc_id"]

    socket = document_db.rag_search_socket(
        query="How long is the cobalt-series warranty?",
        doc_ids=[doc_id],
        top_k=3,
    )
    assert socket["status"] == "success"
    assert socket["execution"]["embedding_model"] == "BAAI/bge-m3"
    assert socket["result"]["records"]

    monkeypatch.setattr(db_service, "document_db", document_db)
    context = _document_context(
        "How long is the cobalt-series warranty?",
        [{"ref": "file_1", "name": "warranty.txt", "type": "pdf", "doc_id": doc_id}],
        {"file_1": {"type": "pdf", "doc_id": doc_id}},
        run_id="connected-rag-test",
    )

    assert "warranty period is exactly three years" in context
    assert f"doc={doc_id}" in context
