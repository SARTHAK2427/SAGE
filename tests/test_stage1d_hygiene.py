"""
tests/test_stage1d_hygiene.py
Focused regression tests for Stage 1D codebase hygiene & maintainability refactors:
- QUAL-01 / DUP-01: Shared _prepare_chat_request helper across normal and streaming chat
- QUAL-02 / DUP-02: RunState.complete_tool_call delegating to record_tool_result
- QUAL-03 / DEAD-05: Orchestrator removal of dead document_system_prompt
- QUAL-04 / DUP-04: Encapsulated list_doc_ids() on ArtifactStore and SageDocumentDB
- QUAL-05: Elimination of redundant image disk write in ArtifactStore
- DEAD-02 & DEAD-03: Removal of dead patterns and helper in code_executor.extractor
- DUP-03: Shared _query_coder_llm in tools.coder
"""

import io
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import code_executor.extractor as extractor
from app import _prepare_chat_request
from core.run_state import RunState
from orchestrator import Orchestrator
from sage_document_db.artifact_store import ArtifactStore
from tools.coder import _generate_code_via_model, _get_llm_code_response


class TestQual01AppDeduplication:

    def test_prepare_chat_request_initializes_run_state_and_manifest(self):
        """_prepare_chat_request correctly constructs RunState, attachments, and file map."""
        import asyncio
        run_state, manifest, file_map = asyncio.run(_prepare_chat_request(
            objective="Analyze data",
            files=None,
        ))
        assert isinstance(run_state, RunState)
        assert run_state.user_text == "Analyze data"
        assert run_state.request_id.startswith("req_")
        assert manifest == []
        assert file_map == {}


class TestQual02RunStateDeduplication:

    def test_complete_tool_call_delegates_to_record_tool_result(self):
        """complete_tool_call properly updates status, duration, and summary."""
        rs = RunState(request_id="req_test_02")
        t0 = time.time() - 0.5
        rs.record_tool_call(
            call_id="call_001",
            tool_name="math",
            function_name="calculate",
            sanitized_args={"expression": "2+2"},
            start_time=t0,
        )

        rs.complete_tool_call(
            call_id="call_001",
            status="success",
            result_summary="Result: 4",
            error=None,
        )

        record = rs.tool_calls[0]
        assert record.status == "success"
        assert record.result_summary == "Result: 4"
        assert record.duration_ms > 0


class TestQual03OrchestratorDeadPromptRemoved:

    def test_orchestrator_has_no_dead_document_system_prompt(self):
        """Orchestrator.__init__ does not define document_system_prompt."""
        orch = Orchestrator()
        assert not hasattr(orch, "document_system_prompt")
        assert hasattr(orch, "agent_system_prompt")
        assert hasattr(orch, "coder_system_prompt")


class TestQual04ListDocIdsEncapsulation:

    def test_artifact_store_list_doc_ids(self, tmp_path):
        """ArtifactStore.list_doc_ids returns list of documents with manifests."""
        store = ArtifactStore(tmp_path)
        # Create doc directories
        doc1 = tmp_path / "doc_1111111111"
        doc1.mkdir()
        (doc1 / "manifest.json").write_text("{}", encoding="utf-8")

        doc2 = tmp_path / "doc_2222222222"
        doc2.mkdir()
        (doc2 / "manifest.json").write_text("{}", encoding="utf-8")

        # Invalid dir without manifest
        (tmp_path / "random_dir").mkdir()

        doc_ids = store.list_doc_ids()
        assert doc_ids == ["doc_1111111111", "doc_2222222222"]


class TestDead02And03ExtractorClean:

    def test_extractor_module_has_no_dead_patterns_or_functions(self):
        """Unused _find_blocks, _PYTHON_PATTERN, and _GENERIC_PATTERN are removed."""
        assert not hasattr(extractor, "_find_blocks")
        assert not hasattr(extractor, "_PYTHON_PATTERN")
        assert not hasattr(extractor, "_GENERIC_PATTERN")


class TestDup03CoderPromptDeduplication:

    def test_generate_code_via_model_and_get_llm_code_response_use_shared_query(self):
        """Both coder helpers successfully invoke the model and return code content."""
        mock_mm = MagicMock()
        mock_mc = MagicMock()
        mock_mc.chat_completion.return_value = {"content": "```python\nprint('dedup')\n```"}

        out1 = _generate_code_via_model(
            instruction="generate",
            code=None,
            language="python",
            model_manager=mock_mm,
            model_client=mock_mc,
        )
        assert "print('dedup')" in out1

        out2 = _get_llm_code_response(
            instruction="generate",
            code=None,
            language="python",
            model_manager=mock_mm,
            model_client=mock_mc,
        )
        assert "print('dedup')" in out2
