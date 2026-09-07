# SAGE SIH 2026 — Deterministic & Internal Socket Schemas Reference

## 1. Architectural Foundation

In SAGE, deterministic, rule-based, infrastructure, database, parser, retrieval, execution, and runtime components communicate via **rich output sockets**.

```
    COMPONENT
       ↓
    FULL RICH OUTPUT SOCKET  (100% JSON-serializable, maximum useful data)
       ↓
    PYTHON MAPPER / CONNECTOR
       ↓
    SELECTED SUBSET / TRANSFORMED DATA
       ↓
    NEXT COMPONENT INPUT PLUG
```

### Core Invariants
1. **No Premature Discarding**: A producer does not discard information simply because one immediate downstream consumer (such as Gemma 4B) does not require it.
2. **Deterministic Internals**: Registered tools do not autonomously invoke other registered tools. Deterministic plumbing (e.g. `rag_search` → `EmbeddingService` → `ChromaStore`, `dispatcher` → logger/timer, `vision` → filesystem path resolution) is allowed internally.
3. **JSON-Serializable Guarantee**: No Python dataclasses, `Path` objects, exceptions, generators, or custom class instances leak out of a socket. All data types are strictly JSON primitives (`string`, `number`, `boolean`, `null`, `array`, `object`).
4. **Decoupled Data Visibility**: The rich socket represents the complete internal truth. A downstream Python mapping layer filters fields according to visibility classification before exposing data to Gemma, the UI, or logs.

---

## 2. Common Envelope & Standards

### 2.1 Standard Top-Level Envelope
Where applicable, all deterministic sockets follow this structural pattern:
```json
{
  "status": "success | partial | error | hit | miss | timeout | infra_error",
  "operation": "machine_readable_operation_name",
  "result": { ... },
  "identity": { ... },
  "context": { ... },
  "provenance": { ... },
  "references": { ... },
  "execution": { ... },
  "timing": { ... },
  "storage": { ... },
  "warnings": [ ... ],
  "error": null | { ... },
  "raw": { ... }
}
```

### 2.2 Standard Error Schema
```json
{
  "code": "STABLE_MACHINE_READABLE_CODE",
  "type": "ExceptionClassOrLogicalErrorType",
  "message": "Human-readable explanation",
  "recoverable": true,
  "retryable": false,
  "details": { ... },
  "debug": {
    "traceback": "...",
    "source_file": "...",
    "source_function": "..."
  }
}
```

### 2.3 Standard Timing Schema
```json
{
  "started_at": "2026-09-07T12:00:00.000Z",
  "finished_at": "2026-09-07T12:00:00.125Z",
  "duration_ms": 125.45,
  "stages": {
    "parse_ms": 45.2,
    "embedding_ms": 30.1,
    "chroma_ms": 50.15
  }
}
```

### 2.4 Data Visibility Classification
| Classification | Meaning | Allowed Consumers |
| :--- | :--- | :--- |
| `SAFE_FOR_AGENT` | Clean, semantic information safe for Gemma context | Gemma, UI, Logs, Persistence |
| `SAFE_FOR_UI` | Human-facing presentation data, summaries, progress | Frontend UI, Logs, Persistence |
| `INTERNAL_ONLY` | Backend plumbing (host paths, PIDs, container IDs, internal tokens) | Python runtime only (stripped by mappers) |
| `LOG_ONLY` | Detailed stack traces, raw request bodies, debug traces | Loggers, audit files only |
| `PERSISTENCE_ONLY` | Cold storage, historical caches, metadata manifests | SQLite, Chroma, JSON files |

---

## 3. Directory of Deterministic Output Sockets

---

### Socket A: Document Ingestion
- **Purpose**: Captures complete auditable output of the document ingestion pipeline.
- **Producer**: [`sage_document_db/pipeline.py::ingest_document`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/sage_document_db/pipeline.py) & [`SageDocumentDB.ingest_document`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/sage_document_db/__init__.py).
- **Schema File**: [`schemas/deterministic/ingestion.schema.json`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/schemas/deterministic/ingestion.schema.json)
- **Field Breakdown & Visibility**:
  - `status` (`string`, Always): `"success"` (fully parsed and indexed), `"partial"` (parsed, artifacts written, vector index failed), or `"error"`. [SAFE_FOR_UI, SAFE_FOR_AGENT]
  - `operation` (`string`, Always): `"document_ingestion"`. [SAFE_FOR_UI]
  - `result.doc_id` (`string`, Always): Deterministic document ID (`doc_<sha256[:10]>`). [SAFE_FOR_AGENT, SAFE_FOR_UI]
  - `result.counts` (`object`, Always): Extracted element counts (`text`, `images`, `tables`, `code`, `links`). [SAFE_FOR_AGENT, SAFE_FOR_UI]
  - `result.indexed_source_records` (`integer`, Always): Count of chunks indexed in Chroma. [SAFE_FOR_UI]
  - `result.is_duplicate` (`boolean`, Always): Whether ingestion skipped re-parsing due to matching SHA-256. [SAFE_FOR_UI]
  - `identity.original_filename` (`string`, Always): Name of source file. [SAFE_FOR_AGENT, SAFE_FOR_UI]
  - `identity.extension` (`string`, Always): Normalized file extension. [SAFE_FOR_AGENT]
  - `identity.mime_type` (`string|null`, Conditional): MIME type if detected. [SAFE_FOR_AGENT]
  - `identity.sha256` (`string`, Always): SHA-256 hash of source file. [PERSISTENCE_ONLY]
  - `identity.file_size_bytes` (`integer`, Always): Source file byte length. [SAFE_FOR_UI]
  - `context.parser` (`object`, Always): Parser selected and version. [INTERNAL_ONLY]
  - `storage.artifact_dir` (`string`, Always): Local directory containing canonical artifacts. [INTERNAL_ONLY]
  - `storage.manifest_path` (`string`, Always): Path to canonical `manifest.json`. [INTERNAL_ONLY]
  - `timing` (`object`, Always): Total duration and sub-stages (`parse_ms`, `artifact_write_ms`, `chroma_index_ms`). [SAFE_FOR_UI, LOG_ONLY]
  - `warnings` (`array[string]`, Always): Parser and chunker warnings. [SAFE_FOR_UI, LOG_ONLY]
  - `error` (`object|null`, Always): Structured error on failure. [SAFE_FOR_UI, LOG_ONLY]
- **Example Success Object**:
```json
{
  "status": "success",
  "operation": "document_ingestion",
  "result": {
    "doc_id": "doc_a81f42c91e",
    "counts": { "text": 42, "images": 3, "tables": 2, "code": 1, "links": 5 },
    "indexed_source_records": 18,
    "is_duplicate": false
  },
  "identity": {
    "doc_id": "doc_a81f42c91e",
    "original_filename": "annual_report.pdf",
    "extension": "pdf",
    "mime_type": "application/pdf",
    "sha256": "a81f42c91e920d3215be...",
    "file_size_bytes": 1048576
  },
  "storage": {
    "artifact_dir": "artifacts/doc_a81f42c91e",
    "manifest_path": "artifacts/doc_a81f42c91e/manifest.json",
    "raw_source_retained": true
  },
  "timing": {
    "started_at": "2026-09-07T12:00:00.000Z",
    "finished_at": "2026-09-07T12:00:00.540Z",
    "duration_ms": 540.2,
    "stages": { "parse_ms": 320.1, "artifact_write_ms": 45.3, "chroma_index_ms": 174.8 }
  },
  "warnings": [],
  "error": null
}
```

---

### Socket B: RAG Search
- **Purpose**: Exposes complete semantic search information across `sage_source` and `sage_derived` vector collections.
- **Producer**: [`sage_document_db/retrieval.py::RetrievalService.rag_search_socket`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/sage_document_db/retrieval.py).
- **Schema File**: [`schemas/deterministic/rag_search.schema.json`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/schemas/deterministic/rag_search.schema.json)
- **Field Breakdown & Visibility**:
  - `result.records` (`array`, Always): List of retrieved chunk records. [SAFE_FOR_AGENT]
    - `record_id`: Globally unique Chroma ID.
    - `origin`: `"source"` (canonical document chunk) or `"derived"` (VLM analysis).
    - `record_type`: `"text_chunk"`, `"table"`, `"code"`, `"derived_image"`.
    - `text`: Chunk content text.
    - `raw_distance`: Cosine distance from Chroma (in [0, 2]).
    - `derived_cosine_similarity`: Unclipped cosine similarity (`1.0 - raw_distance` in [-1.0, 1.0]), preserving true diametric opposition.
    - `derived_similarity`: Backward-compatible clamped similarity (`max(0.0, 1.0 - raw_distance)` in [0.0, 1.0]).
    - `doc_id`: Originating document ID.
    - `page`: Source page number (if paginated).
    - `source_element_ids`: Canonical artifact element IDs contributing to chunk.
    - `image_refs`, `table_refs`, `code_refs`: Connected artifact IDs.
    - `derived_cache_ref`: Path to JSON cache if originating from derived analysis.
  - `context.query` (`string`, Always): Raw search query text. [SAFE_FOR_AGENT]
  - `context.requested_top_k` (`integer`, Always): Maximum requested results. [INTERNAL_ONLY]
  - `context.collections_queried` (`array[string]`, Always): Vector collections searched. [INTERNAL_ONLY]
  - `execution.embedding_model` (`string`, Always): Name of embedding model used. [LOG_ONLY]
  - `execution.distance_metric` (`string`, Always): `"cosine"`. [LOG_ONLY]
  - `execution.similarity_formula` (`string`, Always): `"derived_cosine_similarity = 1.0 - raw_distance"`. [LOG_ONLY]
  - `timing.stages` (`object`, Always): Breakdown of `embedding_ms` and `chroma_query_ms`. [LOG_ONLY]

---

### Socket C: Exact Search
- **Purpose**: Exposes literal and regex match results across canonical source artifacts (`text/`, `tables/`, `code/`).
- **Producer**: [`sage_document_db/retrieval.py::RetrievalService.exact_search_socket`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/sage_document_db/retrieval.py).
- **Schema File**: [`schemas/deterministic/exact_search.schema.json`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/schemas/deterministic/exact_search.schema.json)
- **Field Breakdown & Visibility**:
  - `result.matches` (`array`, Always): List of matches. [SAFE_FOR_AGENT]
    - `doc_id`, `element_id`, `element_type`: Identification of matched element.
    - `matched_text`: Exact matched substring.
    - `snippet`: Surrounding context window.
    - `match_start`, `match_end`: Exact character offsets in element text.
    - `table_row`, `table_col`: Coordinate in table cells if applicable.
    - `ref`: Relative path to artifact JSON file.
  - `context.searched_canonical_subdirs` (`array`, Always): `["text", "tables", "code"]`.
  - `context.derived_excluded` (`boolean`, Always): `true` (enforces rule that exact search only touches source truth).

---

### Socket D: Artifact Fetch
- **Purpose**: Direct retrieval of atomic canonical elements by known ID.
- **Producer**: [`sage_document_db/artifact_store.py::ArtifactStore.fetch_element_socket`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/sage_document_db/artifact_store.py).
- **Schema File**: [`schemas/deterministic/artifact_fetch.schema.json`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/schemas/deterministic/artifact_fetch.schema.json)
- **Field Breakdown & Visibility**:
  - `result.element` (`object`, Always): Canonical element contents. [SAFE_FOR_AGENT]
  - `provenance` (`object`, Always): Bounding box, page, slide, sheet, and reading order position. [SAFE_FOR_AGENT]
  - `storage.local_path` (`string|null`, Conditional): Host filesystem path for image binary. Marked `INTERNAL_ONLY` (must not be sent to Gemma). [INTERNAL_ONLY]
  - `storage.ref` (`string|null`, Conditional): Relative artifact path. [SAFE_FOR_AGENT]

---

### Socket E: List Artifacts
- **Purpose**: Enumerates document contents from canonical manifest without loading binary pixels or dumping entire texts.
- **Producer**: [`sage_document_db/artifact_store.py::ArtifactStore.list_elements_socket`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/sage_document_db/artifact_store.py).
- **Schema File**: [`schemas/deterministic/list_artifacts.schema.json`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/schemas/deterministic/list_artifacts.schema.json)
- **Field Breakdown & Visibility**:
  - `result.artifacts` (`array`, Always): Metadata list sorted deterministically by reading order.
  - `result.total_in_document` (`integer`, Always): Total elements in manifest before filtering.
  - `context.artifact_type_filter` (`string|null`, Conditional): Filter applied (`text`, `image`, `table`, `code`).

---

### Socket F: Embedding Operation
- **Purpose**: Programmatic vector outputs and model inference telemetry from SentenceTransformer.
- **Producer**: [`sage_document_db/embeddings.py::EmbeddingService.embed_documents_socket`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/sage_document_db/embeddings.py).
- **Schema File**: [`schemas/deterministic/embedding.schema.json`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/schemas/deterministic/embedding.schema.json)
- **Field Breakdown & Visibility**:
  - `result.vectors` (`array[array[number]]`, Always): Dense float embedding vectors. [INTERNAL_ONLY]
  - `identity.device` (`string`, Always): `"cuda"` or `"cpu"`. [LOG_ONLY]
  - `identity.dimension` (`integer`, Always): Vector dimensionality (384). [INTERNAL_ONLY]
  - `identity.is_mock` (`boolean`, Always): Flag indicating mock execution. [LOG_ONLY]
  - `context.approx_tokens` (`integer`, Always): Token count evaluated via tokenizer. [LOG_ONLY]

---

### Socket G: Chroma Query
- **Purpose**: Captures low-level database query response prior to application normalization.
- **Producer**: [`sage_document_db/chroma_store.py::ChromaStore.rag_query_socket`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/sage_document_db/chroma_store.py).
- **Schema File**: [`schemas/deterministic/chroma_query.schema.json`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/schemas/deterministic/chroma_query.schema.json)
- **Field Breakdown & Visibility**:
  - `result.collection_responses` (`object`, Always): Raw unflattened Chroma result dictionary mapping collections to IDs, documents, metadatas, and distances. [INTERNAL_ONLY]
  - `context.where_filter` (`object|null`, Conditional): Chroma query where-clause applied. [INTERNAL_ONLY]

---

### Socket H: Chroma Upsert
- **Purpose**: Vector database insertion and indexing telemetry.
- **Producer**: [`sage_document_db/chroma_store.py::ChromaStore.index_document_socket`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/sage_document_db/chroma_store.py).
- **Schema File**: [`schemas/deterministic/chroma_upsert.schema.json`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/schemas/deterministic/chroma_upsert.schema.json)
- **Field Breakdown & Visibility**:
  - `result.record_count` (`integer`, Always): Number of records inserted. [INTERNAL_ONLY]
  - `result.embedded_count` (`integer`, Always): Number of chunks embedded. [INTERNAL_ONLY]
  - `identity.collection_name` (`string`, Always): Collection written (`sage_source` or `sage_derived`). [INTERNAL_ONLY]

---

### Socket I: Artifact Write
- **Purpose**: Low-level disk writing telemetry for canonical artifact files.
- **Producer**: [`sage_document_db/artifact_store.py::ArtifactStore.write_document_socket`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/sage_document_db/artifact_store.py).
- **Schema File**: [`schemas/deterministic/artifact_write.schema.json`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/schemas/deterministic/artifact_write.schema.json)
- **Field Breakdown & Visibility**:
  - `storage.files_written` (`array`, Always): List of written files, relative paths, and sizes in bytes. [INTERNAL_ONLY]
  - `result.element_counts` (`object`, Always): Breakdown of written element counts. [INTERNAL_ONLY]

---

### Socket J: Derived Image Analysis Persistence
- **Purpose**: Records persistent storage of Qwen3-VL analysis attempts and Chroma indexing state.
- **Producer**: [`sage_document_db/derived.py::add_image_analysis_socket`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/sage_document_db/derived.py).
- **Schema File**: [`schemas/deterministic/derived_analysis.schema.json`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/schemas/deterministic/derived_analysis.schema.json)
- **Field Breakdown & Visibility**:
  - `result.analysis_id` (`string`, Always): Generated ID (`analysis_000001`). [SAFE_FOR_AGENT]
  - `result.attempt_number` (`integer`, Always): Historical attempt count for this image. [SAFE_FOR_AGENT]
  - `storage.derived_file_ref` (`string`, Always): Relative path to JSON cache (`derived/vision/<img_id>.json`). [INTERNAL_ONLY]
  - `raw.raw_output` (`string`, Always): Complete unedited model output preserved verbatim. [PERSISTENCE_ONLY]
  - `raw.ocr_text` (`string|null`, Conditional): Extracted text. [SAFE_FOR_AGENT]
  - `execution.indexed_in_chroma` (`boolean`, Always): Whether record was upserted into `sage_derived`. [INTERNAL_ONLY]

---

### Socket K: Reindex Document
- **Purpose**: Single-document vector index rebuilding telemetry.
- **Producer**: [`SageDocumentDB.reindex_document_socket`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/sage_document_db/__init__.py).
- **Schema File**: [`schemas/deterministic/reindex_document.schema.json`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/schemas/deterministic/reindex_document.schema.json)

---

### Socket L: Rebuild All Indexes
- **Purpose**: Global vector database rebuilding telemetry across all documents and caches.
- **Producer**: [`SageDocumentDB.rebuild_all_indexes_socket`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/sage_document_db/__init__.py).
- **Schema File**: [`schemas/deterministic/rebuild_indexes.schema.json`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/schemas/deterministic/rebuild_indexes.schema.json)

---

### Socket M: Safe Math Engine
- **Purpose**: Deterministic, AST-verified arithmetic evaluation without unrestricted `eval`.
- **Producer**: [`tools/math_tool.py::tool_calculate_socket`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/tools/math_tool.py).
- **Schema File**: [`schemas/deterministic/math.schema.json`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/schemas/deterministic/math.schema.json)
- **Field Breakdown & Visibility**:
  - `result.value` (`number|null`, Always): Computed numeric result. [SAFE_FOR_AGENT, SAFE_FOR_UI]
  - `result.value_type` (`string|null`, Always): `"int"` or `"float"`. [SAFE_FOR_AGENT]
  - `execution.operators_used` (`array[string]`, Always): AST operator nodes evaluated (e.g. `["Add", "Mult"]`). [LOG_ONLY]
  - `execution.functions_used` (`array[string]`, Always): Whitelisted math functions called (e.g. `["sqrt"]`). [LOG_ONLY]

---

### Socket N: Dispatcher Execution
- **Purpose**: Structured outcome of generic tool invocation, preserving nested underlying tool output.
- **Producer**: [`core/dispatcher.py::ToolResult.to_socket`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/core/dispatcher.py).
- **Schema File**: [`schemas/deterministic/dispatcher.schema.json`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/schemas/deterministic/dispatcher.schema.json)
- **Field Breakdown & Visibility**:
  - `identity.call_id` (`string`, Always): Unique execution identifier (`call_<uuid>`). [SAFE_FOR_UI]
  - `references.tool_output` (`any`, Always): Complete nested output socket of invoked tool. [SAFE_FOR_AGENT]
  - `execution.is_mock` (`boolean`, Always): Whether tool was mocked. [LOG_ONLY]

---

### Socket O: Run State & Tool Call Record
- **Purpose**: Backend-owned runtime lifecycle execution state and audit records.
- **Producer**: [`core/run_state.py::RunState.to_socket`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/core/run_state.py).
- **Schema File**: [`schemas/deterministic/run_state.schema.json`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/schemas/deterministic/run_state.schema.json)
- **Field Breakdown & Visibility**:
  - `references.registered_documents` (`array`, Always): Runtime metadata of active user documents. [SAFE_FOR_AGENT, SAFE_FOR_UI]
  - `references.tool_calls` (`array`, Always): Chronological log of tool invocations and durations. [SAFE_FOR_UI]

---

### Socket P: Code Sandbox Execution
- **Purpose**: Deterministic Docker container isolation, execution results, and resource telemetry.
- **Producer**: [`code_executor/sandbox.py::SandboxResult.to_socket`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/code_executor/sandbox.py).
- **Schema File**: [`schemas/deterministic/sandbox_execution.schema.json`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/schemas/deterministic/sandbox_execution.schema.json)
- **Field Breakdown & Visibility**:
  - `result.stdout` (`string`, Always): Container standard output. [SAFE_FOR_AGENT]
  - `result.stderr` (`string`, Always): Container standard error. [SAFE_FOR_AGENT, LOG_ONLY]
  - `result.exit_code` (`integer`, Always): Process exit code. [SAFE_FOR_AGENT]
  - `execution.network` (`string`, Always): `"none"`. [LOG_ONLY]
  - `execution.read_only_mount` (`boolean`, Always): `true`. [LOG_ONLY]
  - `execution.memory_limit_mb` (`integer`, Always): Configured memory ceiling. [LOG_ONLY]

---

### Socket Q: Code Extraction
- **Purpose**: Deterministic code block extraction, fence detection, and language tagging from specialist outputs.
- **Producer**: [`code_executor/extractor.py::CodeExtractResult.to_socket`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/code_executor/extractor.py).
- **Schema File**: [`schemas/deterministic/code_extraction.schema.json`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/schemas/deterministic/code_extraction.schema.json)
- **Field Breakdown & Visibility**:
  - `result.code` (`string`, Always): Clean extracted Python code. [SAFE_FOR_AGENT]
  - `result.strategy` (`string`, Always): `"named_python_fence"`, `"generic_fence"`, `"unfenced_fallback"`, or `"empty"`. [LOG_ONLY]

---

### Socket R: Code Repair Loop Result
- **Purpose**: Complete audit trail of code execution, failure detection, and iterative repair cycles.
- **Producer**: [`code_executor/pipeline.py::PipelineResult.to_socket`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/code_executor/pipeline.py).
- **Schema File**: [`schemas/deterministic/repair_loop.schema.json`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/schemas/deterministic/repair_loop.schema.json)
- **Field Breakdown & Visibility**:
  - `result.succeeded` (`boolean`, Always): Whether execution exited 0 within retry limits. [SAFE_FOR_AGENT, SAFE_FOR_UI]
  - `result.attempts` (`integer`, Always): Total execution cycles used. [SAFE_FOR_UI]
  - `references.fix_history` (`array`, Always): Chronological audit of repair attempts and error prompts. [LOG_ONLY]

---

### Socket S: Model Manager Lifecycle
- **Purpose**: Process lifecycle, port binding, and health monitoring for `llama-server.exe`.
- **Producer**: [`model_manager.py::ModelManager.get_lifecycle_socket`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/model_manager.py).
- **Schema File**: [`schemas/deterministic/model_manager.schema.json`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/schemas/deterministic/model_manager.schema.json)
- **Field Breakdown & Visibility**:
  - `identity.requested_model_key` (`string`, Always): E.g. `"gemma"`, `"coder"`, `"vision"`. [INTERNAL_ONLY]
  - `execution.pid` (`integer|null`, Conditional): Host process ID. Marked `INTERNAL_ONLY`. [INTERNAL_ONLY]
  - `execution.model_path` (`string|null`, Conditional): Host filesystem path to GGUF model weights. Marked `INTERNAL_ONLY`. [INTERNAL_ONLY]

---

### Socket T: Model HTTP Transport & Telemetry
- **Purpose**: Deterministic HTTP transport metrics, token usage, and inference speed from llama.cpp.
- **Producer**: [`model_client.py::ModelClient.chat_completion_socket`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/model_client.py).
- **Schema File**: [`schemas/deterministic/model_transport.schema.json`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/schemas/deterministic/model_transport.schema.json)
- **Field Breakdown & Visibility**:
  - `result.content` (`string`, Always): Raw string from LLM (semantic structure intentionally unfinalized). [INTERNAL_ONLY]
  - `result.tokens` (`object`, Always): `prompt_tokens`, `completion_tokens`, `total_tokens`. [SAFE_FOR_UI]
  - `execution.telemetry` (`object`, Always): Prompt tokens/sec and generation tokens/sec. [SAFE_FOR_UI, LOG_ONLY]

---

### Socket U: Backend File Upload
- **Purpose**: Validates and tracks user document uploads and triggers DocumentDB ingestion.
- **Producer**: [`core/sockets.py::build_upload_socket`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/core/sockets.py) (used by FastAPI upload handlers).
- **Schema File**: [`schemas/deterministic/upload.schema.json`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/schemas/deterministic/upload.schema.json)
- **Field Breakdown & Visibility**:
  - `identity.doc_id` (`string`, Always): Ingested DocumentDB ID. [SAFE_FOR_AGENT, SAFE_FOR_UI]
  - `storage.temp_path` (`string`, Always): Host staging path. Marked `INTERNAL_ONLY`. [INTERNAL_ONLY]

---

### Socket V: JSON Parse & Repair
- **Purpose**: Robust JSON boundary extraction, fence stripping, and syntax repair for model outputs.
- **Producer**: [`core/json_repair.py::parse_agent_json_socket`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/core/json_repair.py).
- **Schema File**: [`schemas/deterministic/json_repair.schema.json`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/schemas/deterministic/json_repair.schema.json)
- **Field Breakdown & Visibility**:
  - `result.is_valid` (`boolean`, Always): Whether valid JSON was produced. [INTERNAL_ONLY]
  - `result.parsed_json` (`any`, Always): Parsed data structure. [INTERNAL_ONLY]
  - `context.steps` (`object`, Always): Which recovery steps fired (`thought_tags_stripped`, `markdown_fence_extracted`, `outermost_braces_extracted`). [LOG_ONLY]

---

### Socket W: Derived Analysis Cache Operation
- **Purpose**: Deterministic cache hit/miss and persistence tracking specifically for derived vision/image analyses (`artifacts/<doc_id>/derived/vision/<image_id>.json`).
- **Producer**: [`sage_document_db/derived.py::derived_cache_socket`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/sage_document_db/derived.py) via `build_derived_analysis_cache_socket`.
- **Schema File**: [`schemas/deterministic/derived_analysis_cache.schema.json`](file:///c:/Users/Rakshit%20Jain/OneDrive/desktop/SAGE/schemas/deterministic/derived_analysis_cache.schema.json) (aliased by `cache.schema.json`).
- **Field Breakdown & Visibility**:
  - `status` (`string`, Always): `"hit"`, `"miss"`, `"write_success"`, or `"error"`. [INTERNAL_ONLY]
  - `result.analyses_count` (`integer`, Always): Existing historical attempts cached. [INTERNAL_ONLY]
  - `result.current_keys` (`array[string]`, Always): Keys present in current analysis. [INTERNAL_ONLY]
  - `storage.cache_file_ref` (`string`, Always): Relative cache path. [INTERNAL_ONLY]
  - `storage.cache_path` (`string|null`, Conditional): Host path, marked `INTERNAL_ONLY`. [INTERNAL_ONLY]
