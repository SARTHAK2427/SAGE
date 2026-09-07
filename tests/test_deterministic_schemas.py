"""
tests/test_deterministic_schemas.py
Comprehensive validation suite for SAGE deterministic output sockets (A through W).

Verifies:
1. All 23 schema files in schemas/deterministic/ are valid Draft 2020-12 JSON Schemas.
2. Real or representative outputs from all deterministic components validate successfully.
3. Representative failure/error objects validate against the same schemas.
4. All emitted socket objects are 100% JSON-serializable (json.dumps succeeds).
5. Required provenance and timing fields exist when available.
6. Internal-only metadata (e.g. local host paths, PIDs) are preserved with visibility tags.
"""

from __future__ import annotations
import json
from pathlib import Path
import pytest
import jsonschema
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from core.sockets import (
    Visibility,
    build_ingestion_socket,
    build_rag_search_socket,
    build_exact_search_socket,
    build_artifact_fetch_socket,
    build_list_artifacts_socket,
    build_embedding_socket,
    build_chroma_query_socket,
    build_chroma_upsert_socket,
    build_artifact_write_socket,
    build_derived_analysis_socket,
    build_reindex_document_socket,
    build_rebuild_indexes_socket,
    build_math_socket,
    build_dispatcher_socket,
    build_run_state_socket,
    build_sandbox_execution_socket,
    build_code_extraction_socket,
    build_repair_loop_socket,
    build_model_manager_socket,
    build_model_transport_socket,
    build_upload_socket,
    build_json_repair_socket,
    build_cache_socket,
    build_derived_analysis_cache_socket,
    build_timing_payload,
    build_error_payload,
)

SCHEMAS_DIR = Path(__file__).resolve().parent.parent / "schemas" / "deterministic"


def _build_registry() -> Registry:
    registry = Registry()
    for p in SCHEMAS_DIR.glob("*.schema.json"):
        data = json.loads(p.read_text(encoding="utf-8"))
        res = Resource.from_contents(data, default_specification=DRAFT202012)
        registry = registry.with_resource(p.name, res)
        if "$id" in data:
            registry = registry.with_resource(data["$id"], res)
    return registry


REGISTRY = _build_registry()


def _load_schema(schema_name: str) -> dict:
    schema_path = SCHEMAS_DIR / schema_name
    assert schema_path.exists(), f"Schema file {schema_name} not found at {schema_path}"
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    Draft202012Validator.check_schema(schema)
    return schema


def _validate_socket(instance: dict, schema_name: str) -> None:
    # 1. Must serialize to JSON cleanly
    dumped = json.dumps(instance)
    assert isinstance(dumped, str)

    # 2. Must validate against schema using referencing registry
    schema = _load_schema(schema_name)
    validator = Draft202012Validator(schema, registry=REGISTRY)
    validator.validate(instance=instance)


# ─── Schema Syntax Check ──────────────────────────────────────────────────────

def test_all_schemas_valid_syntax():
    """Ensure all 23 schema files are valid Draft 2020-12 schemas."""
    schema_files = list(SCHEMAS_DIR.glob("*.schema.json"))
    assert len(schema_files) >= 22, f"Expected at least 22 schema files, found {len(schema_files)}"
    for sf in schema_files:
        with open(sf, "r", encoding="utf-8") as f:
            schema = json.load(f)
        Draft202012Validator.check_schema(schema)


# ─── A. Document Ingestion ────────────────────────────────────────────────────

def test_socket_ingestion_success():
    socket = build_ingestion_socket(
        status="success",
        doc_id="doc_a81f42c91e",
        original_filename="sample.pdf",
        sha256="a81f42c91e920d3215be1234567890abcdef",
        file_size_bytes=10240,
        extension="pdf",
        parser_name="docling",
        parser_tag="docling",
        counts={"text": 10, "images": 2, "tables": 1, "code": 0, "links": 3},
        artifact_dir="artifacts/doc_a81f42c91e",
        manifest_path="artifacts/doc_a81f42c91e/manifest.json",
        indexed_source_records=8,
        timing=build_timing_payload(duration_ms=250.0, stages={"parse_ms": 150.0, "chroma_index_ms": 100.0}),
    )
    _validate_socket(socket, "ingestion.schema.json")


def test_socket_ingestion_error():
    err = build_error_payload("CORRUPT_FILE", "ValueError", "Cannot parse document", recoverable=False)
    socket = build_ingestion_socket(
        status="error",
        doc_id="doc_failed",
        original_filename="corrupt.docx",
        sha256="00000000000000000000",
        file_size_bytes=100,
        extension="docx",
        parser_name="unknown",
        parser_tag="unknown",
        counts={"text": 0, "images": 0, "tables": 0, "code": 0},
        artifact_dir="",
        manifest_path="",
        indexed_source_records=0,
        timing=build_timing_payload(duration_ms=10.0),
        error=err,
    )
    _validate_socket(socket, "ingestion.schema.json")


# ─── B. RAG Search ────────────────────────────────────────────────────────────

def test_socket_rag_search_success():
    records = [
        {
            "record_id": "doc_1:chunk:001",
            "origin": "source",
            "record_type": "text_chunk",
            "text": "Revenue grew by 25 percent in Q3.",
            "distance": 0.15,
            "raw_distance": 0.15,
            "derived_similarity": 0.85,
            "doc_id": "doc_1",
            "page": 2,
            "source_element_ids": ["txt_001"],
            "image_refs": [],
            "table_refs": [],
            "code_refs": [],
            "derived_cache_ref": None,
        }
    ]
    socket = build_rag_search_socket(
        query="revenue growth Q3",
        records=records,
        requested_top_k=5,
        doc_ids_filter=["doc_1"],
        timing=build_timing_payload(duration_ms=15.2, stages={"embedding_ms": 5.0, "chroma_query_ms": 10.2}),
    )
    _validate_socket(socket, "rag_search.schema.json")


def test_socket_rag_search_error():
    err = build_error_payload("EMPTY_QUERY", "ValueError", "Query cannot be empty")
    socket = build_rag_search_socket(
        query="",
        records=[],
        requested_top_k=5,
        timing=build_timing_payload(duration_ms=0.5),
        error=err,
    )
    _validate_socket(socket, "rag_search.schema.json")


# ─── C. Exact Search ──────────────────────────────────────────────────────────

def test_socket_exact_search_success():
    from sage_document_db.models import ExactSearchResult
    res = ExactSearchResult(
        doc_id="doc_1",
        element_id="txt_002",
        element_type="text",
        page=1,
        order=2,
        ref="text/document.json",
        match_start=10,
        match_end=15,
        snippet="invoice: 12345 due",
        matched_text="12345",
    )
    socket = build_exact_search_socket(
        query="12345",
        matches=[res.to_dict()],
        max_results=20,
        case_sensitive=True,
        regex=False,
        timing=build_timing_payload(duration_ms=4.1),
    )
    _validate_socket(socket, "exact_search.schema.json")


def test_socket_exact_search_invalid_regex():
    err = build_error_payload("INVALID_REGEX", "ValueError", "Unbalanced parenthesis", recoverable=True)
    socket = build_exact_search_socket(
        query="(invalid[regex",
        matches=[],
        max_results=20,
        regex=True,
        timing=build_timing_payload(duration_ms=1.0),
        error=err,
    )
    _validate_socket(socket, "exact_search.schema.json")


# ─── D. Artifact Fetch ────────────────────────────────────────────────────────

def test_socket_artifact_fetch_success():
    el = {
        "id": "img_000001",
        "type": "image",
        "subtype": "diagram",
        "order": 3,
        "page": 2,
        "ref": "images/img_000001.png",
        "local_path": "c:/path/to/artifacts/doc_1/images/img_000001.png",
        "exists": True,
    }
    socket = build_artifact_fetch_socket(
        doc_id="doc_1",
        element_id="img_000001",
        element=el,
        found=True,
        timing=build_timing_payload(duration_ms=1.5),
    )
    _validate_socket(socket, "artifact_fetch.schema.json")
    # Verify local_path is retained internally
    assert socket["storage"]["local_path"] is not None


def test_socket_artifact_fetch_not_found():
    err = build_error_payload("ELEMENT_NOT_FOUND", "KeyError", "Element 'txt_999' not found")
    socket = build_artifact_fetch_socket(
        doc_id="doc_1",
        element_id="txt_999",
        found=False,
        timing=build_timing_payload(duration_ms=0.8),
        error=err,
    )
    _validate_socket(socket, "artifact_fetch.schema.json")


# ─── E. List Artifacts ────────────────────────────────────────────────────────

def test_socket_list_artifacts():
    artifacts = [
        {"element_id": "txt_001", "type": "text", "order": 0, "page": 1, "ref": "text/doc.json"},
        {"element_id": "img_001", "type": "image", "order": 1, "page": 1, "ref": "images/img_1.png", "derived_available": True},
    ]
    socket = build_list_artifacts_socket(
        doc_id="doc_1",
        artifacts=artifacts,
        artifact_type_filter=None,
        total_elements_in_doc=2,
        timing=build_timing_payload(duration_ms=2.3),
    )
    _validate_socket(socket, "list_artifacts.schema.json")


# ─── F. Embedding ─────────────────────────────────────────────────────────────

def test_socket_embedding():
    socket = build_embedding_socket(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        dimension=384,
        input_count=2,
        vectors=[[0.1] * 384, [-0.1] * 384],
        is_mock=True,
        approx_tokens=15,
        timing=build_timing_payload(duration_ms=5.0),
    )
    _validate_socket(socket, "embedding.schema.json")


# ─── G. Chroma Query ──────────────────────────────────────────────────────────

def test_socket_chroma_query():
    raw_responses = {
        "sage_source": {
            "ids": [["doc_1:chunk:1"]],
            "distances": [[0.12]],
            "documents": [["text"]],
            "metadatas": [[{"doc_id": "doc_1"}]],
        }
    }
    socket = build_chroma_query_socket(
        collections_queried=["sage_source"],
        query_count=1,
        requested_top_k=5,
        returned_count=1,
        collection_responses=raw_responses,
        timing=build_timing_payload(duration_ms=12.0),
    )
    _validate_socket(socket, "chroma_query.schema.json")


# ─── H. Chroma Upsert ─────────────────────────────────────────────────────────

def test_socket_chroma_upsert():
    socket = build_chroma_upsert_socket(
        collection_name="sage_source",
        record_count=10,
        embedded_count=10,
        doc_ids_affected=["doc_1"],
        ids_written=["doc_1:chunk:000001"],
        timing=build_timing_payload(duration_ms=45.0),
    )
    _validate_socket(socket, "chroma_upsert.schema.json")


# ─── I. Artifact Write ────────────────────────────────────────────────────────

def test_socket_artifact_write():
    files = [
        {"path": "text/document.json", "size_bytes": 1024, "suffix": ".json"},
        {"path": "manifest.json", "size_bytes": 512, "suffix": ".json"},
    ]
    socket = build_artifact_write_socket(
        doc_id="doc_1",
        artifact_dir="artifacts/doc_1",
        manifest_path="artifacts/doc_1/manifest.json",
        files_written=files,
        element_counts={"text": 5, "images": 0, "tables": 0, "code": 0},
        timing=build_timing_payload(duration_ms=18.0),
    )
    _validate_socket(socket, "artifact_write.schema.json")


# ─── J. Derived Analysis Persistence ──────────────────────────────────────────

def test_socket_derived_analysis():
    socket = build_derived_analysis_socket(
        doc_id="doc_1",
        image_id="img_000001",
        analysis_id="analysis_000001",
        attempt_number=1,
        model="qwen-vl",
        task_type="ocr",
        raw_output="Invoice #9988 total $4,500",
        derived_file_ref="derived/vision/img_000001.json",
        indexed_in_chroma=True,
        chroma_record_id="doc_1:derived:image:img_000001",
        ocr_text="Invoice #9988 total $4,500",
        description="A black and white receipt",
        observations={"total": 4500},
        timing=build_timing_payload(duration_ms=35.0, stages={"cache_write_ms": 10.0, "chroma_upsert_ms": 25.0}),
    )
    _validate_socket(socket, "derived_analysis.schema.json")


# ─── K. Reindex Document ──────────────────────────────────────────────────────

def test_socket_reindex_document():
    socket = build_reindex_document_socket(
        doc_id="doc_1",
        source_records_rebuilt=8,
        embedded_records=8,
        timing=build_timing_payload(duration_ms=80.0),
    )
    _validate_socket(socket, "reindex_document.schema.json")


# ─── L. Rebuild All Indexes ───────────────────────────────────────────────────

def test_socket_rebuild_indexes():
    socket = build_rebuild_indexes_socket(
        source_summary={"documents": 5, "records": 45, "errors": []},
        derived_summary={"images": 3, "records": 3, "errors": []},
        total_documents=5,
        total_records=48,
        chroma_root="chroma_db",
        artifacts_root="artifacts",
        timing=build_timing_payload(duration_ms=350.0),
    )
    _validate_socket(socket, "rebuild_indexes.schema.json")


# ─── M. Safe Math Engine ──────────────────────────────────────────────────────

def test_socket_math_success():
    from tools.math_tool import tool_calculate_socket
    socket = tool_calculate_socket("sqrt(144) + 10 * 2")
    _validate_socket(socket, "math.schema.json")
    assert socket["result"]["value"] == 32.0


def test_socket_math_error():
    from tools.math_tool import tool_calculate_socket
    socket = tool_calculate_socket("10 / 0")
    _validate_socket(socket, "math.schema.json")
    assert socket["status"] == "error"
    assert socket["error"]["code"] == "DIVISION_BY_ZERO"


# ─── N. Dispatcher Execution ──────────────────────────────────────────────────

def test_socket_dispatcher():
    from core.dispatcher import ToolResult
    tr = ToolResult(
        call_id="call_abc123",
        tool_name="math",
        function_name="calculate",
        status="success",
        result={"value": 42},
        duration_ms=2.5,
    )
    socket = tr.to_socket()
    _validate_socket(socket, "dispatcher.schema.json")
    assert socket["references"]["tool_output"] == {"value": 42}


# ─── O. Run State ─────────────────────────────────────────────────────────────

def test_socket_run_state():
    from core.run_state import RunState
    rs = RunState(request_id="req_001", user_text="analyze budget")
    rs.register_document("doc_1", "budget.xlsx", "xlsx", "budget.xlsx")
    rs.record_tool_call("call_01", "document_db", "rag_search", {"query": "budget"})
    socket = rs.to_socket()
    _validate_socket(socket, "run_state.schema.json")


# ─── P. Sandbox Execution ─────────────────────────────────────────────────────

def test_socket_sandbox():
    from code_executor.sandbox import SandboxResult
    res = SandboxResult(
        stdout="hello world\n",
        stderr="",
        exit_code=0,
        wall_time_ms=120.0,
        memory_peak_mb=14.5,
        timed_out=False,
    )
    socket = res.to_socket(code_text="print('hello world')")
    _validate_socket(socket, "sandbox_execution.schema.json")


# ─── Q. Code Extraction ───────────────────────────────────────────────────────

def test_socket_code_extraction():
    from code_executor.extractor import extract_code
    extracted = extract_code("Here is the code:\n```python\nprint(123)\n```")
    socket = extracted.to_socket("Here is the code:\n```python\nprint(123)\n```")
    _validate_socket(socket, "code_extraction.schema.json")
    assert socket["result"]["language"] == "python"
    assert socket["result"]["strategy"] == "named_python_fence"


# ─── R. Repair Loop ───────────────────────────────────────────────────────────

def test_socket_repair_loop():
    from code_executor.pipeline import PipelineResult
    from code_executor.fixer import FixAttempt
    from code_executor.sandbox import SandboxResult
    sbx_res = SandboxResult(
        stdout="1\n",
        stderr="",
        exit_code=0,
        wall_time_ms=10.0,
        memory_peak_mb=12.0,
        timed_out=False,
    )
    attempt = FixAttempt(
        attempt_number=1,
        original_code="print('bad')",
        fixed_code="print(1)",
        sandbox_result=sbx_res,
        repair_prompt="Fix this code",
    )
    pr = PipelineResult(
        status="success",
        final_code="print(1)",
        stdout="1\n",
        stderr="",
        exit_code=0,
        wall_time_ms=150.0,
        memory_peak_mb=12.0,
        timed_out=False,
        attempts=1,
        fix_history=[attempt],
    )
    socket = pr.to_socket(task="print number 1")
    _validate_socket(socket, "repair_loop.schema.json")


# ─── S. Model Manager ─────────────────────────────────────────────────────────

def test_socket_model_manager():
    from model_manager import model_manager
    socket = model_manager.get_lifecycle_socket(
        action="already_loaded",
        model_key="gemma",
        duration_ms=1.2,
    )
    _validate_socket(socket, "model_manager.schema.json")


# ─── T. Model Transport Telemetry ─────────────────────────────────────────────

def test_socket_model_transport():
    socket = build_model_transport_socket(
        endpoint="/v1/chat/completions",
        url="http://127.0.0.1:8080/v1/chat/completions",
        duration_seconds=0.45,
        content='{"type": "final", "answer": "done"}',
        usage={"prompt_tokens": 120, "completion_tokens": 25, "total_tokens": 145},
        timings={"prompt_ms": 150.0, "predicted_ms": 300.0, "prompt_per_second": 800.0, "predicted_per_second": 83.3},
        model_key="gemma",
        http_status=200,
        request_id="req_123",
    )
    _validate_socket(socket, "model_transport.schema.json")


# ─── U. Backend File Upload ───────────────────────────────────────────────────

def test_socket_upload():
    socket = build_upload_socket(
        request_id="req_001",
        original_filename="receipt.png",
        safe_filename="receipt.png",
        file_ref="file_1",
        file_type="png",
        byte_size=204800,
        temp_path="temp/req_001/receipt.png",
        doc_id="doc_abc",
        timing=build_timing_payload(duration_ms=15.0),
    )
    _validate_socket(socket, "upload.schema.json")


# ─── V. JSON Parse / Repair ───────────────────────────────────────────────────

def test_socket_json_repair_success():
    from core.json_repair import parse_agent_json_socket
    raw = "<thought>Thinking...</thought>\n```json\n{\"type\": \"final\", \"answer\": \"All set\"}\n```"
    socket = parse_agent_json_socket(raw)
    _validate_socket(socket, "json_repair.schema.json")
    assert socket["status"] == "repaired"
    assert socket["result"]["is_valid"] is True
    assert socket["context"]["steps"]["thought_tags_stripped"] is True
    assert socket["context"]["steps"]["markdown_fence_extracted"] is True


def test_socket_json_repair_error():
    from core.json_repair import parse_agent_json_socket
    raw = "Just some unformatted prose without any JSON"
    socket = parse_agent_json_socket(raw)
    _validate_socket(socket, "json_repair.schema.json")
    assert socket["status"] == "error"
    assert socket["result"]["is_valid"] is False


# ─── W. Cache Lookup / Write ──────────────────────────────────────────────────

def test_socket_cache():
    socket = build_cache_socket(
        operation="cache_lookup",
        doc_id="doc_1",
        image_id="img_001",
        cache_file_ref="derived/vision/img_001.json",
        hit=True,
        cache_path="artifacts/doc_1/derived/vision/img_001.json",
        analyses_count=2,
        current_keys=["ocr_text", "description"],
        byte_size=1240,
        duration_ms=0.8,
    )
    _validate_socket(socket, "cache.schema.json")
    _validate_socket(socket, "derived_analysis_cache.schema.json")
    assert socket["status"] == "hit"


# ==============================================================================
# SECTION 22: REPRESENTATIVE SEMANTIC TESTS
# ==============================================================================

def test_semantic_error_without_fake_fields():
    """Verify pre-result failures validate without inventing fake doc_id, paths, or record counts."""
    err = build_error_payload(
        code="UNSUPPORTED_FORMAT",
        type_name="ValueError",
        message="Cannot parse file with extension .xyz",
        recoverable=False,
        retryable=False,
    )
    pre_result_failure = {
        "status": "error",
        "operation": "document_ingestion",
        "warnings": [],
        "error": err,
    }
    _validate_socket(pre_result_failure, "ingestion.schema.json")


def test_semantic_derived_indexing_lifecycle_unindexed():
    """Verify indexed_in_chroma=False allows chroma_record_id=None across pending/accepted states."""
    socket = build_derived_analysis_socket(
        doc_id="doc_test_derived",
        image_id="img_000002",
        analysis_id="analysis_pending_01",
        attempt_number=1,
        model="qwen-vl",
        task_type="detailed_description",
        raw_output="A complex architectural floor plan.",
        derived_file_ref="derived/vision/img_000002.json",
        indexed_in_chroma=False,
        chroma_record_id=None,
        lifecycle_status="pending",
        became_current=False,
        description="A complex architectural floor plan.",
    )
    assert socket["result"]["chroma_record_id"] is None
    assert socket["execution"]["chroma_record_id"] is None
    assert socket["execution"]["indexed_in_chroma"] is False
    assert socket["context"]["status"] == "pending"
    _validate_socket(socket, "derived_analysis.schema.json")


def test_semantic_rag_unclipped_negative_similarity():
    """Verify negative cosine similarity (e.g. opposite vectors) is preserved without clamping to 0."""
    raw_dist = 1.85  # cosine distance > 1 -> cosine similarity = 1 - 1.85 = -0.85
    records = [
        {
            "record_id": "doc_neg:chunk:001",
            "origin": "source",
            "record_type": "text_chunk",
            "text": "Diametrically opposed semantic content.",
            "distance": raw_dist,
            "raw_distance": raw_dist,
            "derived_cosine_similarity": round(1.0 - raw_dist, 6),
            "derived_similarity": 0.0,  # Legacy clamped field
            "doc_id": "doc_neg",
            "page": 1,
            "source_element_ids": ["txt_neg"],
            "image_refs": [],
            "table_refs": [],
            "code_refs": [],
            "derived_cache_ref": None,
        }
    ]
    socket = build_rag_search_socket(
        query="unrelated semantic query",
        records=records,
        requested_top_k=5,
    )
    rec = socket["result"]["records"][0]
    assert rec["derived_cosine_similarity"] == -0.85
    assert rec["raw_distance"] == 1.85
    _validate_socket(socket, "rag_search.schema.json")


def test_semantic_artifact_variants_text_image_table_code():
    """Verify all four canonical artifact types validate against artifact_fetch.schema.json."""
    variants = [
        ("text", {
            "id": "txt_canonical_01",
            "type": "text",
            "text": "Quarterly financial summary paragraph.",
            "order": 0,
            "page": 1,
            "ref": "text/doc.json",
        }),
        ("image", {
            "id": "img_canonical_01",
            "type": "image",
            "ref": "images/chart.png",
            "local_path": "c:/data/artifacts/doc_1/images/chart.png",
            "dimensions": [1024, 768],
            "order": 1,
            "page": 1,
            "exists": True,
            "derived_available": False,
        }),
        ("table", {
            "id": "tbl_canonical_01",
            "type": "table",
            "rows": [["Year", "Revenue"], ["2025", "$10M"]],
            "dimensions": [2, 2],
            "headers": ["Year", "Revenue"],
            "order": 2,
            "page": 2,
            "ref": "tables/tbl_1.json",
        }),
        ("code", {
            "id": "code_canonical_01",
            "type": "code",
            "code": "import numpy as np\nprint(np.pi)",
            "text": "import numpy as np\nprint(np.pi)",
            "language": "python",
            "order": 3,
            "page": 3,
            "ref": "code/code_1.py",
        }),
    ]
    for el_type, el_dict in variants:
        socket = build_artifact_fetch_socket(
            doc_id="doc_variants",
            element_id=el_dict["id"],
            element=el_dict,
            found=True,
            timing=build_timing_payload(duration_ms=1.0),
        )
        assert socket["result"]["element"]["type"] == el_type
        _validate_socket(socket, "artifact_fetch.schema.json")


def test_semantic_code_extraction_preserves_raw_and_blocks():
    """Verify code extractor preserves raw input text and multi-block detected fences."""
    from code_executor.extractor import extract_code
    multi_fence_response = (
        "Here is the setup:\n"
        "```bash\npip install sympy\n```\n"
        "And the main calculation:\n"
        "```python\nimport sympy\nx = sympy.Symbol('x')\nprint(sympy.diff(x**3, x))\n```\n"
        "Done!"
    )
    result = extract_code(multi_fence_response)
    socket = result.to_socket(multi_fence_response)
    assert socket["raw"]["raw_response"] == multi_fence_response
    assert socket["result"]["block_count"] == 2
    assert len(socket["result"]["detected_blocks"]) == 2
    assert socket["result"]["detected_blocks"][0]["language"] == "bash"
    assert socket["result"]["detected_blocks"][1]["language"] == "python"
    assert socket["result"]["selected_block_index"] == 1
    assert "sympy.diff" in socket["result"]["code"]
    _validate_socket(socket, "code_extraction.schema.json")


def test_semantic_repair_loop_structured_history():
    """Verify fix_history structures all attempts and nullable exit_code when no execution occurs."""
    from code_executor.pipeline import PipelineResult
    from code_executor.fixer import FixAttempt
    from code_executor.sandbox import SandboxResult

    sbx = SandboxResult(stdout="result: 42\n", stderr="", exit_code=0, wall_time_ms=15.0, memory_peak_mb=10.0, timed_out=False)
    attempt = FixAttempt(
        attempt_number=1,
        original_code="bad syntax :(",
        fixed_code="print('result: 42')",
        sandbox_result=sbx,
        repair_prompt="Fix Python syntax error",
        raw_response="```python\nprint('result: 42')\n```",
        failure_reason="SyntaxError on line 1",
        retryable=True,
    )
    pipe_res = PipelineResult(
        status="success",
        final_code="print('result: 42')",
        stdout="result: 42\n",
        stderr="",
        exit_code=0,
        wall_time_ms=45.0,
        memory_peak_mb=10.0,
        timed_out=False,
        attempts=1,
        fix_history=[attempt],
    )
    socket = pipe_res.to_socket(task="Compute 42")
    assert len(socket["references"]["fix_history"]) == 1
    hist = socket["references"]["fix_history"][0]
    assert hist["attempt_number"] == 1
    assert hist["model"]["raw_response"] == "```python\nprint('result: 42')\n```"
    assert hist["failure"]["retryable"] is True
    _validate_socket(socket, "repair_loop.schema.json")

    # Test that pre-execution extract_failed allows null exit_code
    extract_failed_socket = build_repair_loop_socket(
        status="extract_failed",
        final_code="",
        stdout="",
        stderr="No Python code block found in model response",
        exit_code=None,
        wall_time_ms=0.0,
        memory_peak_mb=0.0,
        timed_out=False,
        attempts=1,
        task="Generate script",
    )
    assert extract_failed_socket["result"]["exit_code"] is None
    _validate_socket(extract_failed_socket, "repair_loop.schema.json")


def test_semantic_strictness_rejects_misspelled_property():
    """Verify that unexpected SAGE-controlled properties are rejected by additionalProperties: false."""
    socket = build_ingestion_socket(
        status="success",
        doc_id="doc_strict_test",
        original_filename="sample.pdf",
        sha256="abc1234567890abcdef",
        file_size_bytes=2048,
        extension="pdf",
        parser_name="docling",
        parser_tag="docling",
        counts={"text": 1, "images": 0, "tables": 0, "code": 0},
        artifact_dir="artifacts/doc_strict_test",
        manifest_path="artifacts/doc_strict_test/manifest.json",
        indexed_source_records=1,
    )
    # Inject a misspelled property at top level
    invalid_socket = dict(socket)
    invalid_socket["unknown_field_typo"] = "should_be_rejected"
    with pytest.raises(jsonschema.ValidationError):
        _validate_socket(invalid_socket, "ingestion.schema.json")

    # Inject a misspelled property inside SAGE-controlled result
    invalid_result = dict(socket)
    invalid_result["result"] = dict(socket["result"])
    invalid_result["result"]["extra_metric_typo"] = 123
    with pytest.raises(jsonschema.ValidationError):
        _validate_socket(invalid_result, "ingestion.schema.json")


def test_semantic_json_serializability_across_all_sockets():
    """Verify 100% JSON-serializability for all socket types without custom json encoders."""
    from model_manager import model_manager

    test_sockets = [
        build_ingestion_socket("success", "doc_s", "f.pdf", "hash", 100, "pdf", "p", "t", {"text": 1, "images": 0, "tables": 0, "code": 0}, "a", "m", 1),
        build_rag_search_socket("q", [], 5),
        build_exact_search_socket("q", [], 10),
        build_artifact_fetch_socket("doc_s", "txt_01", {"id": "txt_01", "type": "text"}),
        build_list_artifacts_socket("doc_s", []),
        build_embedding_socket("model", "cpu", 384, 1, [[0.1] * 384]),
        build_chroma_query_socket(["sage_source"], 1, 5, 0, {}),
        build_chroma_upsert_socket("sage_source", 0, 0, []),
        build_artifact_write_socket("doc_s", "a", "m", [], {"text": 0, "images": 0, "tables": 0, "code": 0}),
        build_derived_analysis_socket("doc_s", "img_1", "an_1", 1, "m", "t", "out", "ref", False),
        build_reindex_document_socket("doc_s", 0, 0),
        build_rebuild_indexes_socket({}, {}, 0, 0, "c", "a"),
        build_math_socket("1+1", 2.0, "float"),
        build_dispatcher_socket("c_1", "math", "calc", "success", {"val": 2}),
        build_run_state_socket("run_1", "req_1", 0, "hi", [], []),
        build_sandbox_execution_socket("ok", "", 0, 10.0, 5.0, False),
        build_code_extraction_socket("code", "python", True, 1, "```python\ncode\n```"),
        build_repair_loop_socket("success", "code", "ok", "", 0, 10.0, 5.0, False, 1, "task"),
        model_manager.get_lifecycle_socket("loaded", "gemma", 1.0),
        build_model_transport_socket("/url", "http://", 0.1, "out"),
        build_upload_socket("req_1", "f.png", "f.png", "ref", "png", 100, "tmp", "doc_1"),
        build_derived_analysis_cache_socket("cache_lookup", "doc_s", "img_1", "ref", False),
    ]

    for sock in test_sockets:
        dumped = json.dumps(sock)
        assert isinstance(dumped, str)
        reloaded = json.loads(dumped)
        assert isinstance(reloaded, dict)
        assert reloaded["status"] in ("success", "error", "partial", "hit", "miss", "write_success", "fallback", "empty", "active", "completed")

