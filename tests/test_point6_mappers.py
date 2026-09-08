"""
tests/test_point6_mappers.py
Comprehensive Point-6 tests verifying:
    - All 7 Gemma result mappers
    - Multiple RAG matches preserved in original order
    - Exact-search multiple matches
    - Artifact fetch text/table/code/image variants
    - List artifacts inventory
    - Vision mapping
    - Coder success/error/not_executed states + language preservation
    - Math evaluation mapping
    - Common error mapping
    - Partial-result mapping
    - Coder input formatting and exact indentation preservation
    - Vision TASK input and deterministic image resolution
    - Tool-results packet order and call_index sequencing
    - JSON serializability of all outputs
    - Comprehensive leakage guard (no internal paths, PIDs, containers, Chroma, vectors)
    - Prompt and tool contract consistency
    - 5-call heterogeneous batch integration test
    - Multi-loop end-to-end orchestration test (Gemma → tools → results → Gemma → tools → results → Gemma → final)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.mappers.gemma_results import (
    _STRIP_ALWAYS,
    build_tool_results_packet,
    map_artifact_fetch_result,
    map_coder_result,
    map_error_for_gemma,
    map_exact_result,
    map_list_artifacts_result,
    map_math_result,
    map_rag_result,
    map_tool_result_for_gemma,
    map_vision_result,
    strip_internal_fields,
    CANONICAL_GEMMA_MAPPERS,
)
from core.mappers.coder_input import (
    SANDBOX_SUPPORTED_LANGUAGES,
    build_coder_input,
    is_sandbox_supported,
)
from core.mappers.vision_input import (
    build_vision_text_prompt,
    resolve_vision_image,
)
from core.run_state import RunState
from orchestrator import Orchestrator
from core.dispatcher import ToolRegistry, ToolResult


# ─── 1. All 7 Result Mappers: Core & Variants ─────────────────────────────────

class TestGemmaResultMappers:

    def test_rag_search_multiple_matches_preserved_in_order(self):
        """rag_search returns multiple ranked matches; all forwarded in order."""
        rich_socket = {
            "status": "success",
            "result": {
                "records": [
                    {
                        "record_id": "rec_001",
                        "text": "First ranked match: revenue was $4.2B.",
                        "doc_id": "doc_fin_2024",
                        "record_type": "text",
                        "raw_distance": 0.15,
                        "derived_cosine_similarity": 0.85,
                        "origin": "source",
                        "source_element_ids": ["txt_001"],
                        "page": 1,
                        "image_refs": ["img_chart_1"],
                        "embedding": [0.1, 0.2, 0.3],
                        "local_path": "/internal/docs/doc_fin_2024.pdf",
                    },
                    {
                        "record_id": "rec_002",
                        "text": "Second ranked match: expenses grew 8%.",
                        "doc_id": "doc_fin_2024",
                        "record_type": "text",
                        "raw_distance": 0.28,
                        "derived_cosine_similarity": 0.72,
                        "origin": "source",
                        "source_element_ids": ["txt_002"],
                        "page": 2,
                        "embedding": [0.4, 0.5, 0.6],
                    },
                    {
                        "record_id": "rec_003",
                        "text": "Third match from OCR table.",
                        "doc_id": "doc_fin_2024",
                        "record_type": "table",
                        "raw_distance": 0.41,
                        "derived_cosine_similarity": 0.59,
                        "origin": "vision",
                        "source_element_ids": ["tbl_001"],
                        "page": 5,
                    },
                ]
            },
            "context": {"requested_top_k": 5},
        }

        mapped = map_rag_result(rich_socket)

        assert mapped["returned"] == 3
        assert mapped["top_k"] == 5
        matches = mapped["matches"]
        assert len(matches) == 3

        # Order must be strictly preserved
        assert matches[0]["text"] == "First ranked match: revenue was $4.2B."
        assert matches[0]["cosine_similarity"] == 0.85
        assert matches[0]["element_id"] == "txt_001"
        assert matches[0]["source"] == "canonical"
        assert matches[0]["image_refs"] == ["img_chart_1"]
        assert matches[0]["page"] == 1

        assert matches[1]["text"] == "Second ranked match: expenses grew 8%."
        assert matches[1]["cosine_similarity"] == 0.72
        assert matches[1]["source"] == "canonical"

        assert matches[2]["text"] == "Third match from OCR table."
        assert matches[2]["source"] == "derived"
        assert matches[2]["element_type"] == "table"

        # Internal keys must not be present in mapped output
        for m in matches:
            assert "embedding" not in m
            assert "local_path" not in m
            assert "record_id" not in m

    def test_map_rag_result_with_dispatcher_results_format(self):
        """rag_search mapper must also accept the direct tool_rag_search output schema."""
        socket_data = {
            "status": "success",
            "results": [
                {
                    "record_id": "doc_1:chunk:1",
                    "origin": "source",
                    "record_type": "text_chunk",
                    "text": "Linear and Binary Search analysis.",
                    "distance": 0.15,
                    "doc_id": "doc_1",
                    "source_element_ids": ["txt_001"],
                    "image_refs": ["img_01"],
                }
            ],
            "count": 1,
        }
        mapped = map_rag_result(socket_data)
        assert mapped["returned"] == 1
        assert len(mapped["matches"]) == 1
        assert mapped["matches"][0]["text"] == "Linear and Binary Search analysis."
        assert mapped["matches"][0]["element_id"] == "txt_001"
        assert mapped["matches"][0]["cosine_similarity"] == 0.85
        assert mapped["matches"][0]["source"] == "canonical"

    def test_exact_search_multiple_matches(self):
        """exact_search returns multiple literal matches with context & offsets."""
        rich_socket = {
            "status": "success",
            "result": {
                "matches": [
                    {
                        "doc_id": "doc_sec_10k",
                        "element_id": "sec_04",
                        "element_type": "text",
                        "matched_text": "Item 1A. Risk Factors",
                        "snippet": "...pursuant to Section 13... Item 1A. Risk Factors ...",
                        "match_start": 120,
                        "match_end": 141,
                        "page": 12,
                        "local_path": "/var/data/sec.pdf",
                    },
                    {
                        "doc_id": "doc_sec_10k",
                        "element_id": "tbl_07",
                        "element_type": "table",
                        "matched_text": "$142,500,000",
                        "snippet": "Operating Income: $142,500,000",
                        "match_start": 18,
                        "match_end": 30,
                        "page": 45,
                        "table_row": 3,
                        "table_col": 2,
                    },
                ]
            },
        }

        mapped = map_exact_result(rich_socket)
        matches = mapped["matches"]
        assert len(matches) == 2

        assert matches[0]["element_id"] == "sec_04"
        assert matches[0]["matched_text"] == "Item 1A. Risk Factors"
        assert matches[0]["context"] == "...pursuant to Section 13... Item 1A. Risk Factors ..."
        assert matches[0]["start_offset"] == 120
        assert matches[0]["end_offset"] == 141
        assert matches[0]["page"] == 12
        assert "local_path" not in matches[0]

        assert matches[1]["element_id"] == "tbl_07"
        assert matches[1]["table_row"] == 3
        assert matches[1]["table_col"] == 2

    def test_artifact_fetch_text_variant(self):
        """artifact_fetch text element."""
        rich_socket = {
            "status": "success",
            "element": {
                "id": "txt_99",
                "doc_id": "doc_notes",
                "type": "text",
                "text": "Meeting notes: project approved.",
                "page": 3,
                "local_path": "C:\\tmp\\notes.txt",
                "char_count": 32,
            },
        }
        res = map_artifact_fetch_result(rich_socket, doc_id="doc_notes", element_id="txt_99")
        art = res["artifact"]
        assert art["type"] == "text"
        assert art["element_id"] == "txt_99"
        assert art["text"] == "Meeting notes: project approved."
        assert art["page"] == 3
        assert "local_path" not in art
        assert "char_count" not in art

    def test_artifact_fetch_table_variant(self):
        """artifact_fetch table element with rows and headers."""
        rich_socket = {
            "status": "success",
            "element": {
                "id": "tbl_42",
                "doc_id": "doc_budget",
                "type": "table",
                "rows": [["Engineering", "$1M"], ["Marketing", "$500K"]],
                "headers": ["Department", "Budget"],
                "markdown": "| Department | Budget |\n|---|---|\n| Engineering | $1M |",
                "page": 7,
                "bbox": [10, 20, 100, 200],
            },
        }
        res = map_artifact_fetch_result(rich_socket, doc_id="doc_budget", element_id="tbl_42")
        art = res["artifact"]
        assert art["type"] == "table"
        assert art["rows"] == [["Engineering", "$1M"], ["Marketing", "$500K"]]
        assert art["headers"] == ["Department", "Budget"]
        assert "markdown" in art
        assert art["page"] == 7
        assert "bbox" not in art

    def test_artifact_fetch_code_variant(self):
        """artifact_fetch code element."""
        rich_socket = {
            "status": "success",
            "element": {
                "id": "code_01",
                "doc_id": "doc_manual",
                "type": "code",
                "code": "def process_data(x):\n    return x * 2\n",
                "language": "python",
                "page": 10,
            },
        }
        res = map_artifact_fetch_result(rich_socket, doc_id="doc_manual", element_id="code_01")
        art = res["artifact"]
        assert art["type"] == "code"
        assert art["code"] == "def process_data(x):\n    return x * 2\n"
        assert art["language"] == "python"

    def test_artifact_fetch_image_variant(self):
        """artifact_fetch image element retains image_id for reusable vision call, strips path."""
        rich_socket = {
            "status": "success",
            "element": {
                "id": "img_figure_3",
                "doc_id": "doc_paper",
                "type": "image",
                "ref": "doc_paper/img_figure_3",
                "caption": "Figure 3: System Architecture",
                "derived_available": True,
                "page": 4,
                "local_path": "/var/storage/images/doc_paper_fig3.png",
                "width": 800,
                "height": 600,
            },
        }
        res = map_artifact_fetch_result(rich_socket, doc_id="doc_paper", element_id="img_figure_3")
        art = res["artifact"]
        assert art["type"] == "image"
        assert art["image_id"] == "img_figure_3"
        assert art["image_ref"] == "doc_paper/img_figure_3"
        assert art["caption"] == "Figure 3: System Architecture"
        assert art["derived_available"] is True
        assert "local_path" not in art
        assert "width" not in art
        assert "height" not in art

    def test_list_artifacts_inventory(self):
        """list_artifacts returns lightweight inventory."""
        rich_socket = {
            "status": "success",
            "result": {
                "artifacts": [
                    {"element_id": "img_01", "type": "image", "order": 1, "page": 1, "caption": "Logo"},
                    {"element_id": "tbl_01", "type": "table", "order": 2, "page": 2, "heading": "Summary"},
                    {"element_id": "img_02", "type": "image", "order": 3, "page": 3, "derived_available": True},
                ]
            },
        }
        mapped = map_list_artifacts_result(rich_socket)
        arts = mapped["artifacts"]
        assert len(arts) == 3
        assert arts[0] == {"element_id": "img_01", "type": "image", "order": 1, "page": 1, "caption": "Logo"}
        assert arts[1] == {"element_id": "tbl_01", "type": "table", "order": 2, "page": 2, "heading": "Summary"}
        assert arts[2] == {"element_id": "img_02", "type": "image", "order": 3, "page": 3, "derived_available": True}

    def test_vision_mapping(self):
        """map_vision_result extracts textual analysis, strips transport/local paths."""
        rich_socket = {
            "status": "success",
            "image_id": "img_scan_01",
            "doc_id": "doc_invoices",
            "analysis": "INVOICE #9981\nDate: 2024-03-01\nTotal: $1,250.00",
            "derived_analysis_id": "analysis_9981",
            "model": "qwen-vl-7b",
            "duration_ms": 1420,
            "local_path": "/tmp/img_scan_01.png",
        }
        mapped = map_vision_result(rich_socket, doc_id="doc_invoices", image_id="img_scan_01")
        assert mapped["doc_id"] == "doc_invoices"
        assert mapped["image_id"] == "img_scan_01"
        assert "INVOICE #9981" in mapped["analysis"]
        assert mapped["derived_analysis_id"] == "analysis_9981"
        assert "local_path" not in mapped
        assert "model" not in mapped
        assert "duration_ms" not in mapped

    def test_coder_mapping_success(self):
        """map_coder_result on success preserves language, code, stdout, exit_code."""
        rich_socket = {
            "status": "success",
            "succeeded": True,
            "final_code": "print(42)\n",
            "language": "python",
            "stdout": "42\n",
            "stderr": "",
            "exit_code": 0,
            "attempts": 1,
            "container_id": "dock_abc123",
            "wall_time_ms": 320,
        }
        mapped = map_coder_result(rich_socket)
        assert mapped["execution_status"] == "success"
        assert mapped["code"] == "print(42)\n"
        assert mapped["language"] == "python"
        assert mapped["stdout"] == "42\n"
        assert mapped["exit_code"] == 0
        assert mapped["attempts_used"] == 1
        assert "container_id" not in mapped
        assert "wall_time_ms" not in mapped

    def test_coder_mapping_python_error(self):
        """map_coder_result on Python error preserves failure details, stderr, and exit_code."""
        rich_socket = {
            "status": "error",
            "succeeded": False,
            "final_code": "1 / 0\n",
            "language": "python",
            "stdout": "",
            "stderr": "ZeroDivisionError: division by zero",
            "exit_code": 1,
            "attempts": 3,
            "repair_prompt": "Fix the error above",
        }
        mapped = map_coder_result(rich_socket)
        assert mapped["execution_status"] == "error"
        assert mapped["language"] == "python"
        assert mapped["exit_code"] == 1
        assert "ZeroDivisionError" in mapped["stderr"]
        assert mapped["attempts_used"] == 3
        assert "repair_prompt" not in mapped

    def test_coder_mapping_unsupported_language_not_executed(self):
        """Unsupported languages (e.g. C, C++) return not_executed with no exit_code and no fabricated errors."""
        # Case 1: Raw socket explicitly marked not_executed
        rich_socket_1 = {
            "execution_status": "not_executed",
            "code": "#include <iostream>\nint main() { std::cout << 1; }",
            "language": "cpp",
        }
        mapped_1 = map_coder_result(rich_socket_1)
        assert mapped_1["execution_status"] == "not_executed"
        assert mapped_1["language"] == "cpp"
        assert mapped_1["code"] == "#include <iostream>\nint main() { std::cout << 1; }"
        assert "exit_code" not in mapped_1
        assert "stdout" not in mapped_1
        assert "stderr" not in mapped_1

        # Case 2: Language gating automatically catches unsupported language (C)
        rich_socket_2 = {
            "status": "success",
            "succeeded": False,
            "final_code": "int main() { return 0; }",
            "language": "c",
        }
        mapped_2 = map_coder_result(rich_socket_2)
        assert mapped_2["execution_status"] == "not_executed"
        assert mapped_2["language"] == "c"
        assert mapped_2["code"] == "int main() { return 0; }"
        assert "exit_code" not in mapped_2
        assert "stdout" not in mapped_2
        assert "stderr" not in mapped_2

    def test_math_mapping(self):
        """map_math_result produces compact {expression, value}."""
        rich_socket = {"expression": "(17 * 23) + 4", "result": 395}
        mapped = map_math_result(rich_socket)
        assert mapped == {"expression": "(17 * 23) + 4", "value": 395}


# ─── 2. Error and Partial-Result Mapping ───────────────────────────────────────

class TestEnvelopeAndStatusMapping:

    def test_common_error_mapping(self):
        """map_error_for_gemma strips tracebacks, exposes clean {code, message, retryable}."""
        err_dict = {
            "code": "DOC_NOT_FOUND",
            "message": "Document doc_999 does not exist",
            "retryable": False,
            "traceback": "Traceback (most recent call last):\n  File '...'\nKeyError: doc_999",
            "source_file": "db.py",
        }
        mapped_err = map_error_for_gemma(err_dict)
        assert mapped_err == {
            "code": "DOC_NOT_FOUND",
            "message": "Document doc_999 does not exist",
            "retryable": False,
        }
        assert "traceback" not in mapped_err

        # String error fallback
        str_err = map_error_for_gemma("Simple failure string")
        assert str_err == {"code": "TOOL_ERROR", "message": "Simple failure string", "retryable": False}

    def test_partial_result_mapping(self):
        """map_tool_result_for_gemma produces status: partial with usable result and warning."""
        rich_socket = {
            "status": "partial",
            "warning": "2 of 5 documents could not be queried.",
            "result": {
                "records": [
                    {
                        "text": "Partial match from available doc.",
                        "doc_id": "doc_available",
                        "record_type": "text",
                        "derived_cosine_similarity": 0.81,
                    }
                ]
            },
        }

        envelope = map_tool_result_for_gemma(
            tool="document_database",
            function="rag_search",
            rich_socket=rich_socket,
            call_index=0,
            mapper_fn=map_rag_result,
        )

        assert envelope["status"] == "partial"
        assert envelope["call_index"] == 0
        assert envelope["tool"] == "document_database"
        assert envelope["function"] == "rag_search"
        assert "warning" in envelope
        assert "2 of 5 documents" in envelope["warning"]
        assert len(envelope["result"]["matches"]) == 1
        assert envelope["result"]["matches"][0]["doc_id"] == "doc_available"

    def test_tool_failure_produces_error_envelope(self):
        """map_tool_result_for_gemma wraps errors in standard shell."""
        rich_socket = {
            "status": "error",
            "error": {"code": "FILE_CORRUPT", "message": "PDF header damaged", "retryable": False},
        }

        envelope = map_tool_result_for_gemma(
            tool="document_database",
            function="artifact_fetch",
            rich_socket=rich_socket,
            call_index=2,
            mapper_fn=map_artifact_fetch_result,
        )

        assert envelope["status"] == "error"
        assert envelope["call_index"] == 2
        assert envelope["error"]["code"] == "FILE_CORRUPT"
        assert "result" not in envelope


# ─── 3. Input Mappers: Coder and Vision ────────────────────────────────────────

class TestSpecialistInputMappers:

    def test_coder_input_formatting_exact_indentation(self):
        """build_coder_input preserves tabs, 4-spaces, docstrings exactly."""
        code_with_indentation = (
            "def calculate_total(items):\n"
            "    total = 0\n"
            "    for item in items:\n"
            "        if item.get('active'):\n"
            "            total += item['price'] * item.get('qty', 1)\n"
            "    return total"
        )
        instruction = "Fix tax calculation"
        language = "python"

        prompt = build_coder_input(instruction=instruction, code=code_with_indentation, language=language)

        expected = (
            "TASK:\n"
            "Fix tax calculation\n\n"
            "TARGET LANGUAGE:\n"
            "python\n\n"
            "CODE:\n"
            f"{code_with_indentation}"
        )
        assert prompt == expected
        # Indentation within the code must match character-for-character
        assert "            total += item['price'] * item.get('qty', 1)\n" in prompt

    def test_coder_input_optional_sections_omitted(self):
        """build_coder_input omits TARGET LANGUAGE and CODE when not provided."""
        prompt_task_only = build_coder_input(instruction="Write hello world")
        assert prompt_task_only == "TASK:\nWrite hello world"
        assert "TARGET LANGUAGE" not in prompt_task_only
        assert "CODE" not in prompt_task_only

    def test_coder_sandbox_language_support_check(self):
        """is_sandbox_supported approves python/python3 and rejects others."""
        assert is_sandbox_supported("python") is True
        assert is_sandbox_supported("Python") is True
        assert is_sandbox_supported("python3") is True
        assert is_sandbox_supported(None) is True  # default assumption
        assert is_sandbox_supported("") is True

        assert is_sandbox_supported("cpp") is False
        assert is_sandbox_supported("c++") is False
        assert is_sandbox_supported("rust") is False
        assert is_sandbox_supported("javascript") is False

    def test_vision_task_text_prompt(self):
        """build_vision_text_prompt wraps instruction in TASK: plain text."""
        p = build_vision_text_prompt("Transcribe the handwritten notes in this image")
        assert p == "TASK:\nTranscribe the handwritten notes in this image"

    def test_vision_image_resolution_from_db(self, tmp_path):
        """resolve_vision_image retrieves local_path from db artifact element."""
        fake_img = tmp_path / "diagram.png"
        fake_img.write_text("fake_bytes")

        mock_db = MagicMock()
        mock_db.artifact_fetch.return_value = {
            "id": "img_001",
            "type": "image",
            "local_path": str(fake_img),
        }

        resolved = resolve_vision_image(mock_db, doc_id="doc_1", image_id="img_001")
        assert resolved == str(fake_img)
        mock_db.artifact_fetch.assert_called_once_with("doc_1", "img_001")

    def test_vision_image_resolution_non_image_fails(self):
        """resolve_vision_image raises ValueError if element is not image type."""
        mock_db = MagicMock()
        mock_db.artifact_fetch.return_value = {
            "id": "txt_001",
            "type": "text",
            "local_path": "/some/path",
        }
        with pytest.raises(ValueError, match="not an image"):
            resolve_vision_image(mock_db, doc_id="doc_1", image_id="txt_001")


# ─── 4. Packet Order and JSON Serializability ──────────────────────────────────

class TestPacketAndSerialization:

    def test_build_tool_results_packet_preserves_order(self):
        """build_tool_results_packet preserves listed call results in exact order."""
        call_results = [
            {"call_index": 0, "tool": "math", "function": "calculate", "status": "success", "result": {"value": 1}},
            {"call_index": 1, "tool": "document_database", "function": "rag_search", "status": "success", "result": {"matches": []}},
            {"call_index": 2, "tool": "code_specialist", "function": "solve_code_task", "status": "success", "result": {"code": "x=1"}},
        ]
        packet = build_tool_results_packet(call_results)
        assert packet["type"] == "tool_results"
        assert len(packet["results"]) == 3
        for i, res in enumerate(packet["results"]):
            assert res["call_index"] == i

    def test_json_serializability_all_outputs(self):
        """All mapped outputs serialize cleanly via json.dumps without circular or non-serializable objects."""
        # 1. RAG result
        rag = map_tool_result_for_gemma("document_database", "rag_search", {
            "result": {"records": [{"text": "t", "derived_cosine_similarity": 0.9}]}
        }, mapper_fn=map_rag_result)

        # 2. Exact result
        exact = map_tool_result_for_gemma("document_database", "exact_search", {
            "result": {"matches": [{"matched_text": "m", "doc_id": "d"}]}
        }, mapper_fn=map_exact_result)

        # 3. Artifact fetch
        art = map_tool_result_for_gemma("document_database", "artifact_fetch", {
            "element": {"id": "1", "type": "text", "text": "body"}
        }, mapper_fn=map_artifact_fetch_result)

        # 4. List artifacts
        la = map_tool_result_for_gemma("document_database", "list_artifacts", {
            "result": {"artifacts": [{"element_id": "e", "type": "image"}]}
        }, mapper_fn=map_list_artifacts_result)

        # 5. Vision
        vis = map_tool_result_for_gemma("vision_ocr", "analyze_image", {
            "analysis": "ans", "doc_id": "d", "image_id": "i"
        }, mapper_fn=map_vision_result)

        # 6. Coder
        coder = map_tool_result_for_gemma("code_specialist", "solve_code_task", {
            "succeeded": True, "final_code": "a=1", "language": "python"
        }, mapper_fn=map_coder_result)

        # 7. Math
        math = map_tool_result_for_gemma("math", "calculate", {
            "expression": "1+1", "result": 2
        }, mapper_fn=map_math_result)

        packet = build_tool_results_packet([rag, exact, art, la, vis, coder, math])

        serialized = json.dumps(packet, ensure_ascii=False)
        assert isinstance(serialized, str)
        deserialized = json.loads(serialized)
        assert len(deserialized["results"]) == 7


# ─── 5. Leakage Guard ──────────────────────────────────────────────────────────

class TestLeakageGuard:

    @staticmethod
    def _find_leaked_keys(obj: Any, forbidden_tokens: set[str], path: str = "") -> list[str]:
        leaks = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                curr_path = f"{path}.{k}" if path else k
                # Check key name
                k_lower = k.lower()
                for token in forbidden_tokens:
                    if token in k_lower:
                        leaks.append(f"Forbidden key token '{token}' in '{curr_path}'")
                # Recurse
                leaks.extend(TestLeakageGuard._find_leaked_keys(v, forbidden_tokens, curr_path))
        elif isinstance(obj, (list, tuple)):
            for idx, item in enumerate(obj):
                leaks.extend(TestLeakageGuard._find_leaked_keys(item, forbidden_tokens, f"{path}[{idx}]"))
        elif isinstance(obj, str):
            # Check string values for sensitive path indicators
            if "/var/run/docker" in obj or "chromadb" in obj.lower() and "sqlite" in obj.lower():
                leaks.append(f"Suspicious path in value at '{path}': {obj}")
        return leaks

    def test_comprehensive_leakage_guard(self):
        """Verify that internal fields never leak into any Gemma-facing mapped output."""
        forbidden_tokens = {
            "local_path", "temp_path", "cache_path", "derived_file", "manifest_path",
            "traceback", "stack_trace", "exc_info", "source_file", "source_function",
            "pid", "process_id", "container_id", "docker_id",
            "raw_chroma", "chroma_ids", "raw_chroma_responses",
            "embedding", "embeddings", "embedding_model", "embedding_device",
            "stage_timings", "throughput", "token_rate",
            "repair_prompt", "fix_history", "fix_attempts", "infra_error",
        }

        # Stuffed rich socket full of internal secrets
        polluted_rag_socket = {
            "status": "success",
            "local_path": "/secret/storage/doc.pdf",
            "temp_path": "/tmp/extract_123",
            "cache_path": "/var/cache/sage",
            "pid": 12345,
            "port": 8080,
            "container_id": "docker_cont_999",
            "docker_id": "daemon_999",
            "traceback": "Traceback...\n",
            "raw_chroma_responses": {"ids": ["ch_1"], "distances": [0.1]},
            "embeddings": [[0.1, 0.2]],
            "embedding_model": "all-MiniLM-L6-v2",
            "stage_timings": {"embed_ms": 45, "query_ms": 12},
            "repair_prompt": "Fix this code...",
            "result": {
                "records": [
                    {
                        "record_id": "rec_01",
                        "text": "Valid document text.",
                        "doc_id": "doc_1",
                        "record_type": "text",
                        "raw_distance": 0.22,
                        "derived_cosine_similarity": 0.78,
                        "origin": "source",
                        "source_element_ids": ["txt_1"],
                        "embedding": [0.1, 0.2],
                        "local_path": "/secret/storage/doc.pdf",
                    }
                ]
            },
        }

        mapped_rag = map_tool_result_for_gemma(
            "document_database", "rag_search",
            polluted_rag_socket, call_index=0, mapper_fn=map_rag_result
        )

        leaks = self._find_leaked_keys(mapped_rag, forbidden_tokens)
        assert not leaks, f"Leakage detected in RAG output: {leaks}"

        # Test polluted coder socket
        polluted_coder_socket = {
            "status": "success",
            "succeeded": True,
            "code": "print(1)",
            "language": "python",
            "stdout": "1\n",
            "container_id": "d_12345",
            "docker_id": "dock_main",
            "repair_prompt": "Retry syntax",
            "fix_history": [{"attempt": 1, "error": "none"}],
            "wall_time_ms": 400,
            "memory_peak_mb": 120,
        }
        mapped_coder = map_tool_result_for_gemma(
            "code_specialist", "solve_code_task",
            polluted_coder_socket, call_index=1, mapper_fn=map_coder_result
        )
        coder_leaks = self._find_leaked_keys(mapped_coder, forbidden_tokens)
        assert not coder_leaks, f"Leakage detected in coder output: {coder_leaks}"

        # Test polluted error socket
        polluted_error_socket = {
            "status": "error",
            "error": {
                "code": "EXEC_FAIL",
                "message": "Process failed",
                "traceback": "Traceback (most recent call last): line 45",
                "source_file": "executor.py",
                "pid": 54321,
            },
        }
        mapped_err = map_tool_result_for_gemma(
            "code_specialist", "solve_code_task",
            polluted_error_socket, call_index=2, mapper_fn=map_coder_result
        )
        err_leaks = self._find_leaked_keys(mapped_err, forbidden_tokens)
        assert not err_leaks, f"Leakage detected in error output: {err_leaks}"


# ─── 6. Prompt and Tool Contract Tests ────────────────────────────────────────

class TestContractConsistency:

    def test_tools_json_has_exact_7_functions_with_returns(self):
        """prompts/tools.json contains canonical functions with returns defined."""
        tools_path = Path("prompts/tools.json")
        assert tools_path.exists(), "prompts/tools.json must exist"

        with open(tools_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "tools" in data
        tools_map = {t["name"]: t for t in data["tools"]}

        # Canonical tools
        expected_tools = {"document_database", "vision_ocr", "code_specialist", "math", "memory"}
        assert set(tools_map.keys()) == expected_tools

        # Check all 15 canonical functions
        expected_functions = {
            "document_database": {"rag_search", "exact_search", "artifact_fetch", "list_artifacts"},
            "vision_ocr": {"analyze_image"},
            "code_specialist": {"solve_code_task"},
            "math": {"calculate"},
            "memory": {
                "memory_search_hot",
                "memory_search_cold",
                "memory_store_hot",
                "memory_store_cold",
                "memory_update",
                "memory_delete",
                "memory_summarize",
                "memory_promote",
            },
        }

        total_fn_count = 0
        for tool_name, expected_fns in expected_functions.items():
            tool_entry = tools_map[tool_name]
            actual_fns = {fn["name"]: fn for fn in tool_entry["functions"]}
            assert set(actual_fns.keys()) == expected_fns, f"Mismatch in {tool_name} functions"

            for fn_name, fn_def in actual_fns.items():
                total_fn_count += 1
                assert "returns" in fn_def, f"Function {tool_name}/{fn_name} missing 'returns' definition"
                assert isinstance(fn_def["returns"], dict), f"{tool_name}/{fn_name} 'returns' must be an object"
                assert "properties" in fn_def["returns"], f"{tool_name}/{fn_name} returns missing 'properties'"

        assert total_fn_count == 15

    def test_abilities_json_matches_canonical_tools(self):
        """prompts/abilities.json matches the 15 canonical functions in tools.json."""
        abilities_path = Path("prompts/abilities.json")
        assert abilities_path.exists(), "prompts/abilities.json must exist"

        with open(abilities_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        delegated = data.get("delegated", [])
        delegated_pairs = {(d["tool"], d["function"]) for d in delegated}

        expected_pairs = {
            ("document_database", "rag_search"),
            ("document_database", "exact_search"),
            ("document_database", "artifact_fetch"),
            ("document_database", "list_artifacts"),
            ("vision_ocr", "analyze_image"),
            ("code_specialist", "solve_code_task"),
            ("math", "calculate"),
            ("memory", "memory_search_hot"),
            ("memory", "memory_search_cold"),
            ("memory", "memory_store_hot"),
            ("memory", "memory_store_cold"),
            ("memory", "memory_update"),
            ("memory", "memory_delete"),
            ("memory", "memory_summarize"),
            ("memory", "memory_promote"),
        }
        assert delegated_pairs == expected_pairs

    def test_canonical_mapper_coverage_all_7_functions(self):
        """Programmatically assert that all 15 canonical functions have an explicit Gemma mapper in CANONICAL_GEMMA_MAPPERS."""
        expected_15 = {
            ("document_database", "rag_search"),
            ("document_database", "exact_search"),
            ("document_database", "artifact_fetch"),
            ("document_database", "list_artifacts"),
            ("vision_ocr", "analyze_image"),
            ("code_specialist", "solve_code_task"),
            ("math", "calculate"),
            ("memory", "memory_search_hot"),
            ("memory", "memory_search_cold"),
            ("memory", "memory_store_hot"),
            ("memory", "memory_store_cold"),
            ("memory", "memory_update"),
            ("memory", "memory_delete"),
            ("memory", "memory_summarize"),
            ("memory", "memory_promote"),
        }
        assert set(CANONICAL_GEMMA_MAPPERS.keys()) == expected_15
        for key, mapper_fn in CANONICAL_GEMMA_MAPPERS.items():
            assert callable(mapper_fn), f"Mapper for {key} must be a callable function"

    def test_agent_system_prompt_defines_input_and_output_plugs(self):
        """prompts/agent_system.txt explicitly defines the input plug and output sockets."""
        prompt_path = Path("prompts/agent_system.txt")
        assert prompt_path.exists(), "prompts/agent_system.txt must exist"

        text = prompt_path.read_text(encoding="utf-8")

        # Input plug
        assert "tool_results" in text
        assert "call_index" in text

        # Output sockets
        assert "tool_calls" in text
        assert "final" in text

        # No rich socket terminology leaked
        forbidden_in_prompt = ["raw_chroma", "docker_id", "container_id", "local_path", "stage_timings"]
        for forbidden in forbidden_in_prompt:
            assert forbidden not in text, f"Forbidden term '{forbidden}' exposed in agent_system.txt"


# ─── 7. Fail Closed on Unmapped Tool Results ──────────────────────────────────

class TestFailClosedOnUnmappedToolResults:

    def test_unmapped_registered_tool_returns_unmapped_error(self):
        """A registered tool executed without an explicit Gemma mapper must FAIL CLOSED with UNMAPPED_TOOL_RESULT."""
        orch = Orchestrator()
        # Register a custom tool that returns a rich socket full of internal data
        rich_internal_payload = {
            "status": "success",
            "internal_secret": "super_secret_token",
            "local_path": "/var/run/internal/file.dat",
            "pid": 99999,
        }
        orch.registry.register("custom_tool", "do_custom", lambda: rich_internal_payload)

        run_state = RunState(user_text="Run custom tool")
        res = orch._dispatch_tool_call(
            call={"tool": "custom_tool", "function": "do_custom"},
            run_state=run_state,
            telemetry={},
            trace=[],
            file_map={},
            call_index=0,
        )

        assert res["status"] == "error"
        assert res["call_index"] == 0
        assert res["tool"] == "custom_tool"
        assert res["function"] == "do_custom"
        assert res["error"]["code"] == "UNMAPPED_TOOL_RESULT"
        assert res["error"]["message"] == "No Gemma-facing mapper exists for this tool result."
        assert res["error"]["retryable"] is False
        # Absolute guarantee: no rich socket content is exposed
        assert "result" not in res
        assert "internal_secret" not in json.dumps(res)
        assert "super_secret_token" not in json.dumps(res)
        assert "/var/run" not in json.dumps(res)

    def test_map_tool_result_for_gemma_fails_closed_when_mapper_is_none(self):
        """map_tool_result_for_gemma fails closed with UNMAPPED_TOOL_RESULT when mapper_fn is None."""
        rich_socket = {"status": "success", "secret": "123"}
        res = map_tool_result_for_gemma(
            tool="some_tool",
            function="some_fn",
            rich_socket=rich_socket,
            call_index=3,
            mapper_fn=None,
        )
        assert res["status"] == "error"
        assert res["error"]["code"] == "UNMAPPED_TOOL_RESULT"
        assert res["error"]["message"] == "No Gemma-facing mapper exists for this tool result."
        assert "result" not in res
        assert "secret" not in json.dumps(res)

    def test_document_db_unmapped_function_fails_closed(self):
        """document_database with an unknown/unmapped function fails closed."""
        orch = Orchestrator()
        run_state = RunState(user_text="Run unsupported db fn")
        res = orch._dispatch_tool_call(
            call={"tool": "document_database", "function": "unsupported_fn"},
            run_state=run_state,
            telemetry={},
            trace=[],
            file_map={},
            call_index=1,
        )
        assert res["status"] == "error"
        assert res["error"]["code"] == "UNMAPPED_TOOL_RESULT"
        assert res["error"]["message"] == "No Gemma-facing mapper exists for this tool result."
        assert "result" not in res


# ─── 7. Integration: 5 Heterogeneous Calls in One Gemma Batch ─────────────────

class TestHeterogeneousBatchIntegration:

    def test_5_heterogeneous_calls_in_one_batch(self):
        """Orchestrator dispatches a Gemma batch of 5 heterogeneous calls through the mapper layer."""
        orch = Orchestrator()
        run_state = RunState(user_text="Perform analysis")
        run_state.register_document(doc_id="doc_q4", display_name="q4_report.pdf")

        telemetry: dict[str, Any] = {}
        trace: list[dict[str, Any]] = []
        file_map: dict[str, Any] = {}

        # 5 heterogeneous calls in a single batch
        batch_calls = [
            {"tool": "math", "function": "calculate", "arguments": {"expression": "25 * 40"}},
            {"tool": "document_database", "function": "rag_search", "arguments": {"query": "revenue", "top_k": 3}},
            {"tool": "document_database", "function": "exact_search", "arguments": {"query": "EBITDA"}},
            {"tool": "document_database", "function": "artifact_fetch", "arguments": {"doc_id": "doc_q4", "element_id": "txt_01"}},
            {"tool": "code_specialist", "function": "solve_code_task", "arguments": {"instruction": "Compute sum", "language": "python"}},
        ]

        mapped_results = []
        for idx, call in enumerate(batch_calls):
            res = orch._dispatch_tool_call(
                call=call,
                run_state=run_state,
                telemetry=telemetry,
                trace=trace,
                file_map=file_map,
                call_index=idx,
            )
            mapped_results.append(res)

        # Verify envelope structure for all 5
        assert len(mapped_results) == 5

        # Call 0: Math
        assert mapped_results[0]["call_index"] == 0
        assert mapped_results[0]["tool"] == "math"
        assert mapped_results[0]["function"] == "calculate"
        assert mapped_results[0]["status"] == "success"
        assert mapped_results[0]["result"]["value"] == 1000

        # Call 1: RAG Search
        assert mapped_results[1]["call_index"] == 1
        assert mapped_results[1]["tool"] == "document_database"
        assert mapped_results[1]["function"] == "rag_search"
        assert mapped_results[1]["status"] in ("success", "partial", "error")
        if mapped_results[1]["status"] == "success":
            assert "matches" in mapped_results[1]["result"]

        # Call 2: Exact Search
        assert mapped_results[2]["call_index"] == 2
        assert mapped_results[2]["tool"] == "document_database"
        assert mapped_results[2]["function"] == "exact_search"

        # Call 3: Artifact Fetch
        assert mapped_results[3]["call_index"] == 3
        assert mapped_results[3]["tool"] == "document_database"
        assert mapped_results[3]["function"] == "artifact_fetch"

        # Call 4: Code Specialist
        assert mapped_results[4]["call_index"] == 4
        assert mapped_results[4]["tool"] == "code_specialist"
        assert mapped_results[4]["function"] == "solve_code_task"

        # Build packet for Gemma
        packet = build_tool_results_packet(mapped_results)
        assert packet["type"] == "tool_results"
        assert len(packet["results"]) == 5

        # Verify JSON serializability of the entire 5-call batch
        json_packet = json.dumps(packet)
        assert len(json_packet) > 100


# ─── 8. Multi-Loop Test (Gemma → Tools → Results → Gemma → Tools → Results → Gemma → Final)

class TestMultiLoopOrchestration:

    def test_multi_loop_conversation_flow(self):
        """Simulate Gemma → tools → results → Gemma → tools → results → Gemma → final answer."""
        orch = Orchestrator()
        run_state = RunState(user_text="Analyze Q3 figures and compute average")
        run_state.register_document(doc_id="doc_q3", display_name="q3.pdf")

        telemetry: dict[str, Any] = {}
        trace: list[dict[str, Any]] = []
        file_map: dict[str, Any] = {}

        history: list[dict[str, Any]] = [
            {"role": "system", "content": orch.agent_system_prompt},
            {"role": "user", "content": "Objective: Analyze Q3 figures and compute average"},
        ]

        # ── Loop 1: Gemma requests document search and math ───────────────────
        gemma_turn_1 = {
            "type": "tool_calls",
            "calls": [
                {"tool": "document_database", "function": "rag_search", "arguments": {"query": "Q3 figures", "top_k": 2}},
                {"tool": "math", "function": "calculate", "arguments": {"expression": "100 + 200"}},
            ],
        }
        history.append({"role": "assistant", "content": json.dumps(gemma_turn_1)})

        results_loop_1 = []
        for idx, call in enumerate(gemma_turn_1["calls"]):
            res = orch._dispatch_tool_call(
                call=call, run_state=run_state, telemetry=telemetry,
                trace=trace, file_map=file_map, call_index=idx
            )
            results_loop_1.append(res)

        packet_1 = build_tool_results_packet(results_loop_1)
        history.append({"role": "user", "content": json.dumps(packet_1)})

        assert len(results_loop_1) == 2
        assert results_loop_1[0]["tool"] == "document_database"
        assert results_loop_1[1]["tool"] == "math"
        assert results_loop_1[1]["result"]["value"] == 300

        # ── Loop 2: Gemma requests code specialist ────────────────────────────
        gemma_turn_2 = {
            "type": "tool_calls",
            "calls": [
                {"tool": "code_specialist", "function": "solve_code_task", "arguments": {
                    "instruction": "Compute average of [100, 200]",
                    "language": "python",
                }},
            ],
        }
        history.append({"role": "assistant", "content": json.dumps(gemma_turn_2)})

        results_loop_2 = []
        for idx, call in enumerate(gemma_turn_2["calls"]):
            res = orch._dispatch_tool_call(
                call=call, run_state=run_state, telemetry=telemetry,
                trace=trace, file_map=file_map, call_index=idx
            )
            results_loop_2.append(res)

        packet_2 = build_tool_results_packet(results_loop_2)
        history.append({"role": "user", "content": json.dumps(packet_2)})

        assert len(results_loop_2) == 1
        assert results_loop_2[0]["tool"] == "code_specialist"
        assert results_loop_2[0]["status"] == "success"

        # ── Loop 3: Gemma synthesizes evidence and returns final answer ────────
        gemma_turn_3 = {
            "type": "final",
            "answer": "The average of the Q3 figures (100 and 200) is 150.",
        }
        history.append({"role": "assistant", "content": json.dumps(gemma_turn_3)})

        # Verify conversation trajectory integrity
        assert len(history) == 7
        assert history[0]["role"] == "system"
        assert history[1]["role"] == "user"
        assert history[2]["role"] == "assistant"
        assert json.loads(history[2]["content"])["type"] == "tool_calls"
        assert history[3]["role"] == "user"
        assert json.loads(history[3]["content"])["type"] == "tool_results"
        assert history[4]["role"] == "assistant"
        assert json.loads(history[4]["content"])["type"] == "tool_calls"
        assert history[5]["role"] == "user"
        assert json.loads(history[5]["content"])["type"] == "tool_results"
        assert history[6]["role"] == "assistant"
        assert json.loads(history[6]["content"])["type"] == "final"
        assert "150" in json.loads(history[6]["content"])["answer"]
