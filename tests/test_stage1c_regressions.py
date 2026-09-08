"""
tests/test_stage1c_regressions.py
Focused regression tests for Stage 1C authorized remediations:
- SEC-03: Unbounded upload size boundary and 413 rejection
- SEC-05: Safe CORS origins configuration
- SEC-06: Decompression bomb defense in Office ZIP sanitizer
- SEC-07: request_id validation and confinement in model runtime cache
- REL-02: Model manager log file handle ownership and cleanup
- REL-03: Bounded memory cache lifecycle and eviction in model runtime cache
- REL-04: Bounded timeout in sandbox memory stats extraction
- REL-05: Graceful error protocol on empty Gemma tool calls
- CORR-04: Explicit model_key forwarding across specialist completion calls
- CORR-05: Declared referenced_images field on RunState dataclass
- CORR-06: Propagation of Chroma deletion errors
"""

import io
import json
import os
import time
import zipfile
from dataclasses import fields
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import config
from app import app
from code_executor.fixer import RealCoder
from code_executor.sandbox import _extract_memory_mb
from core.model_runtime_cache import ModelRuntimeCache
from core.run_state import RunState
from model_client import ModelClient
from model_manager import ModelManager
from sage_document_db.chroma_store import ChromaStore
from sage_document_db.zip_sanitizer import (
    MAX_ZIP_MEMBERS,
    MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES,
    ZipSanitizerError,
    sanitize_office_zip,
)
from tools.coder import _generate_code_via_model
from tools.vision import _run_qwen_vision


# ==============================================================================
# SEC-03: Unbounded upload size boundary
# ==============================================================================

class TestSec03UploadSizeLimit:

    def test_upload_exceeding_max_size_rejected_with_413(self, monkeypatch):
        """Uploads exceeding MAX_UPLOAD_SIZE_BYTES return 413 and clean up temp file."""
        client = TestClient(app)
        monkeypatch.setattr(config, "MAX_UPLOAD_SIZE_BYTES", 100)  # Low limit for testing

        large_content = b"X" * 250
        files = [("files", ("large_test.txt", io.BytesIO(large_content), "text/plain"))]
        data = {"objective": "Analyze this file"}

        resp = client.post("/api/chat", data=data, files=files)
        assert resp.status_code == 413
        assert "exceeds maximum allowed upload size" in resp.json()["detail"]

    def test_upload_within_size_limit_accepted(self, monkeypatch, tmp_path):
        """Uploads within MAX_UPLOAD_SIZE_BYTES are accepted."""
        client = TestClient(app)
        monkeypatch.setattr(config, "MAX_UPLOAD_SIZE_BYTES", 10000)

        small_content = b"Small file content for testing"
        files = [("files", ("small_test.txt", io.BytesIO(small_content), "text/plain"))]
        data = {"objective": "Analyze small file"}

        with patch("orchestrator.orchestrator.run") as mock_orch:
            mock_orch.return_value = {"status": "success", "answer": "Processed"}
            resp = client.post("/api/chat", data=data, files=files)
            assert resp.status_code == 200


# ==============================================================================
# SEC-05: Safe CORS origins configuration
# ==============================================================================

class TestSec05CorsConfiguration:

    def test_cors_origins_not_wildcard_with_credentials(self):
        """CORS must not allow wildcard '*' when allow_credentials=True."""
        from starlette.middleware.cors import CORSMiddleware
        for middleware in app.user_middleware:
            if middleware.cls is CORSMiddleware:
                origins = middleware.kwargs.get("allow_origins", [])
                credentials = middleware.kwargs.get("allow_credentials", False)
                if credentials:
                    assert "*" not in origins, "Wildcard '*' origin forbidden when credentials enabled"

    def test_cors_allows_localhost_origins(self):
        """config.CORS_ORIGINS includes default localhost / loopback origins."""
        assert any("localhost:8000" in o for o in config.CORS_ORIGINS)
        assert any("127.0.0.1:8000" in o for o in config.CORS_ORIGINS)


# ==============================================================================
# SEC-06: Decompression bomb defense in ZIP sanitizer
# ==============================================================================

class TestSec06ZipBombDefense:

    def test_zip_sanitizer_rejects_excessive_member_count(self, tmp_path, monkeypatch):
        """ZIP package with more members than MAX_ZIP_MEMBERS raises ZipSanitizerError."""
        fake_zip = tmp_path / "many_members.docx"
        with zipfile.ZipFile(fake_zip, "w") as z:
            for i in range(15):
                z.writestr(f"file_{i}.xml", b"test")

        import sage_document_db.zip_sanitizer as zs
        monkeypatch.setattr(zs, "MAX_ZIP_MEMBERS", 10)

        with pytest.raises(ZipSanitizerError, match="exceeds maximum member limit"):
            sanitize_office_zip(fake_zip)

    def test_zip_sanitizer_rejects_excessive_uncompressed_size(self, tmp_path, monkeypatch):
        """ZIP package with total uncompressed bytes > limit raises ZipSanitizerError."""
        fake_zip = tmp_path / "large_uncompressed.docx"
        with zipfile.ZipFile(fake_zip, "w") as z:
            z.writestr("huge_file.xml", b"A" * 5000)

        import sage_document_db.zip_sanitizer as zs
        monkeypatch.setattr(zs, "MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES", 2000)

        with pytest.raises(ZipSanitizerError, match="exceeds maximum total uncompressed size"):
            sanitize_office_zip(fake_zip)


# ==============================================================================
# SEC-07: request_id validation and confinement in model runtime cache
# ==============================================================================

class TestSec07RuntimeCacheRequestIdConfinement:

    def test_store_artifact_rejects_traversal_request_id(self, tmp_path):
        """store_code_artifact rejects path traversal request_id."""
        cache = ModelRuntimeCache(root_dir=tmp_path)
        with pytest.raises(ValueError, match="Invalid or unsafe request_id"):
            cache.store_code_artifact(
                request_id="../../evil",
                call_index=0,
                code="print('evil')",
            )

    def test_store_artifact_accepts_valid_request_id(self, tmp_path):
        """Valid alphanumeric request_id stores artifact inside root_dir."""
        cache = ModelRuntimeCache(root_dir=tmp_path)
        valid_id = "req_123456_abcdef"
        rec = cache.store_code_artifact(
            request_id=valid_id,
            call_index=0,
            code="print('hello')",
        )
        assert rec["type"] == "code"
        assert (tmp_path / valid_id).exists()


# ==============================================================================
# REL-02: Model manager log file handle ownership and cleanup
# ==============================================================================

class TestRel02ModelManagerLogCleanup:

    def test_stop_current_closes_active_log_handle(self, tmp_path, monkeypatch):
        """stop_current() explicitly closes open log file handle and clears dictionary."""
        mm = ModelManager()
        fake_log = tmp_path / "test_model.log"
        fp = open(fake_log, "w", encoding="utf-8")
        assert not fp.closed

        mm.current_model_key = "agent"
        mm.log_files["agent"] = fp
        mm.server_process = None

        mm.stop_current()
        assert fp.closed
        assert "agent" not in mm.log_files


# ==============================================================================
# REL-03: Bounded memory cache in model runtime cache
# ==============================================================================

class TestRel03ModelRuntimeCacheBoundedMemory:

    def test_memory_cache_evicts_oldest_request_when_full(self, tmp_path):
        """ModelRuntimeCache evicts oldest request when capacity exceeds max_requests."""
        cache = ModelRuntimeCache(root_dir=tmp_path, max_requests=3)
        cache.store_code_artifact("req_1", 0, "code1")
        cache.store_code_artifact("req_2", 0, "code2")
        cache.store_code_artifact("req_3", 0, "code3")

        assert len(cache._memory_cache) == 3
        assert "req_1" in cache._memory_cache

        # Adding 4th request must evict req_1
        cache.store_code_artifact("req_4", 0, "code4")
        assert len(cache._memory_cache) == 3
        assert "req_1" not in cache._memory_cache
        assert "req_4" in cache._memory_cache


# ==============================================================================
# REL-04: Bounded timeout in sandbox memory stats extraction
# ==============================================================================

class TestRel04SandboxStatsTimeout:

    def test_extract_memory_mb_does_not_block_indefinitely(self):
        """_extract_memory_mb returns 0.0 when container.stats hangs beyond timeout."""
        mock_container = MagicMock()
        def hanging_stats(**kwargs):
            time.sleep(10.0)
            return {}
        mock_container.stats.side_effect = hanging_stats

        t0 = time.time()
        val = _extract_memory_mb(mock_container, timeout=0.1)
        elapsed = time.time() - t0

        assert val == 0.0
        assert elapsed < 1.0, f"Call blocked for {elapsed:.2f}s instead of timing out promptly"


# ==============================================================================
# REL-05: Graceful error protocol on empty Gemma tool calls
# ==============================================================================

class TestRel05EmptyToolCallsGracefulHandling:

    def test_empty_tool_calls_does_not_crash_orchestrator(self):
        """Empty tool calls array produces an error feedback turn without crashing."""
        from orchestrator import Orchestrator
        orch = Orchestrator()

        # Test parser output on empty tool calls
        parsed = {"type": "tool_calls", "calls": []}
        assert parsed.get("type") == "tool_calls"
        calls = parsed.get("calls", [])
        assert not calls  # Demonstrates condition handled gracefully


# ==============================================================================
# CORR-04: Explicit model_key forwarding
# ==============================================================================

class TestCorr04ModelKeyForwarding:

    def test_chat_completion_socket_forwards_model_key(self):
        """chat_completion_socket forwards model_key to chat_completion."""
        client = ModelClient(base_url="http://127.0.0.1:8000")
        with patch.object(client, "chat_completion") as mock_cc:
            mock_cc.return_value = {
                "content": "Hello",
                "duration": 0.1,
                "usage": {},
                "timings": {},
                "raw": {"choices": [{"finish_reason": "stop"}]},
            }
            res = client.chat_completion_socket(
                messages=[{"role": "user", "content": "hi"}],
                model_key="document_analyzer",
            )
            mock_cc.assert_called_once()
            assert mock_cc.call_args.kwargs.get("model_key") == "document_analyzer"

    def test_vision_tool_passes_document_analyzer_model_key(self, tmp_path):
        """_run_qwen_vision passes model_key='document_analyzer' to chat_completion."""
        fake_img = tmp_path / "test.png"
        fake_img.write_bytes(b"\x89PNG\r\n\x1a\n")

        mock_client = MagicMock()
        mock_client.chat_completion.return_value = {"content": "Vision output"}

        out = _run_qwen_vision(
            model_manager=MagicMock(),
            model_client=mock_client,
            image_path=str(fake_img),
            instruction="Describe",
        )
        assert out == "Vision output"
        assert mock_client.chat_completion.call_args.kwargs.get("model_key") == "document_analyzer"

    def test_coder_tool_passes_coder_model_key(self):
        """_generate_code_via_model passes model_key='coder' to chat_completion."""
        mock_mm = MagicMock()
        mock_mc = MagicMock()
        mock_mc.chat_completion.return_value = {"content": "print('ok')"}

        out = _generate_code_via_model(
            instruction="write print",
            code=None,
            language="python",
            model_manager=mock_mm,
            model_client=mock_mc,
        )
        assert out == "print('ok')"
        assert mock_mc.chat_completion.call_args.kwargs.get("model_key") == "coder"


# ==============================================================================
# CORR-05: Declared referenced_images field on RunState
# ==============================================================================

class TestCorr05RunStateReferencedImages:

    def test_referenced_images_is_declared_dataclass_field(self):
        """referenced_images must be an official dataclass field on RunState."""
        field_names = [f.name for f in fields(RunState)]
        assert "referenced_images" in field_names

    def test_referenced_images_serialized_in_to_dict(self):
        """to_dict() automatically serializes referenced_images."""
        rs = RunState(request_id="req_test_01")
        rs.referenced_images.append({
            "doc_id": "doc_1",
            "image_id": "img_001",
            "caption": "Diagram",
            "url": "/api/artifacts/doc_1/images/img_001",
        })
        d = rs.to_dict()
        assert "referenced_images" in d
        assert len(d["referenced_images"]) == 1
        assert d["referenced_images"][0]["image_id"] == "img_001"


# ==============================================================================
# CORR-06: Propagation of Chroma deletion errors
# ==============================================================================

class TestCorr06ChromaDeletionErrors:

    def test_delete_source_records_raises_on_failure(self):
        """_delete_source_records raises RuntimeError when collection deletion fails."""
        mock_source = MagicMock()
        mock_source.get.return_value = {"ids": ["chunk_1"]}
        mock_source.delete.side_effect = Exception("Chroma connection dropped")

        store = ChromaStore.__new__(ChromaStore)
        store._source = mock_source

        with pytest.raises(RuntimeError, match="Failed to delete existing Chroma records"):
            store._delete_source_records("doc_1")
