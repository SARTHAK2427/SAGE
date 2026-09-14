# SAGE — Complete Project Reference

> **Last updated:** September 2026 (from USB snapshot)
>
> This file is the canonical, single-source reference for anyone (human or AI) trying to understand
> or continue work on the full merged SAGE system. It covers architecture, components, schemas,
> model pipeline, known issues, and development status.

---

## 1. What Is SAGE?

SAGE is a **local, multi-capability AI agent** designed to run entirely on a single developer machine
(or optionally offload heavy inference to a remote Kaggle GPU). It is not a typical chatbot or simple
RAG pipeline. The core design principle is:

> **Gemma 4B is the sole semantic controller. Every deterministic capability (documents, code
> execution, vision, math) is an external tool that Gemma explicitly calls by name.**

```
User prompt
    |
    v
FastAPI (app.py)     <- upload handling, streaming, RunState tracking
    |
    v
Orchestrator (orchestrator.py)    <- history management, JSON parsing, loop control
    |
    v
Gemma 4B (via llama.cpp server)   <- semantic reasoning, tool call decisions
    |  (emits JSON: {"type": "tool_calls", "calls": [...]})
    v
Generic Tool Dispatcher (core/dispatcher.py)
    |        |         |          |
    v        v         v          v
SAGE DB   Vision    Coder      Math
(Chroma+  (Qwen3-   (Qwen2.5-  (AST
MiniLM)   VL 4B)    Coder 7B   calculator)
                    + Docker)
```

---

## 2. File & Directory Layout

```
SAGE/
|-- app.py                      # FastAPI server, /api/chat, /api/chat/stream, /api/upload
|-- config.py                   # All paths, ports, model configs, env-var overrides
|-- db_service.py               # Process-level SageDocumentDB singleton
|-- manual_db_test.py           # Standalone CLI test harness for the DB
|-- model_client.py             # HTTP client for llama.cpp REST API
|-- model_manager.py            # Sequential model lifecycle (load/unload GGUF models)
|-- orchestrator.py             # Main agent loop, JSON repair, history, dispatcher wiring
|-- requirements.txt            # Unified Python dependencies
|-- .env / .env.example         # Runtime secrets and overrides (never committed)
|
|-- core/
|   |-- dispatcher.py           # ToolRegistry, ToolResult, latency tracking, error wrapping
|   |-- json_repair.py          # Robust JSON cleanup and corrective retry prompt
|   |-- model_runtime_cache.py  # Per-model request caching and request-ID tracking
|   |-- remote_model_transport.py # HTTP bridge to remote Kaggle GPU
|   |-- run_state.py            # RunState, RegisteredDocument tracking per request
|   `-- sockets.py              # ALL deterministic socket builder functions (FROZEN)
|
|-- tools/
|   |-- registry.py             # Central factory wiring all tool adapters
|   |-- document_database.py    # RAG, exact search, fetch, list adapters
|   |-- vision.py               # Qwen3-VL targeted image inspection adapter
|   |-- coder.py                # Qwen-Coder + Docker sandbox adapter
|   |-- math_tool.py            # Safe AST arithmetic evaluator
|   `-- general_knowledge.py    # General-purpose knowledge placeholder
|
|-- sage_document_db/           # CANONICAL DOCUMENT DATABASE PACKAGE
|   |-- __init__.py             # SageDocumentDB facade (public API)
|   |-- models.py               # NormalizedDocument, NormalizedElement, RagResult, etc.
|   |-- artifact_store.py       # Filesystem: manifest, text, images, tables, code
|   |-- chroma_store.py         # Chroma vector DB (sage_source + sage_derived)
|   |-- chunker.py              # Structure-aware chunking (token-aware, context-tagged)
|   |-- config.py               # DB-specific paths and CUDA/CPU auto-detect
|   |-- derived.py              # Vision model output persistence hook
|   |-- docling_parser.py       # IBM Docling: heavy PDF/DOCX/PPTX structure parsing
|   |-- embeddings.py           # all-MiniLM-L6-v2 embedding service (CUDA/CPU fallback)
|   |-- exact_search.py         # Literal and regex search across canonical artifacts
|   |-- pipeline.py             # Ingestion orchestration pipeline
|   |-- router.py               # File extension -> parser routing
|   |-- simple_parsers.py       # Fast parsers for TXT, MD, CSV, JSON, images
|   `-- utils.py                # sha256, iso_now, safe_mkdir helpers
|
|-- code_executor/              # Docker sandbox pipeline
|   |-- pipeline.py             # Main: generate -> execute -> repair loop
|   |-- sandbox.py              # Docker container management
|   |-- extractor.py            # Code block extraction from LLM output
|   `-- fixer.py                # Error-driven repair prompt generation
|
|-- prompts/                    # FROZEN - do not edit before Stage 3
|   |-- agent_system.txt        # Gemma system prompt
|   |-- tools.json              # Tool signatures Gemma sees
|   `-- abilities.json          # Delegation boundaries
|
|-- schemas/deterministic/      # 25 frozen socket JSON schemas (NEVER MODIFY)
|-- static/                     # Frontend HTML/CSS/JS
|-- temp/                       # Per-request temp files (auto-created, auto-cleaned)
|-- artifacts/                  # Canonical document artifacts (not committed to git)
|-- chroma_db/                  # Chroma vector index (not committed to git)
`-- tests/                      # Full test suite (mock mode, no GPU/Docker required)
```

---

## 3. Model Pipeline

SAGE runs all models via llama-server.exe REST API. Only ONE model is in VRAM at a time.

### Models

| Key | Name | Role | ~VRAM |
|-----|------|------|-------|
| agent | Gemma 4B Instruct Q4_K_M | Semantic controller - sole decision maker | 3.5 GB |
| coder | Qwen2.5-Coder 7B Instruct Q4_K_M | Code generation specialist | 5 GB |
| document_analyzer | Qwen3-VL 4B Instruct Q4_K_M + mmproj | Vision/OCR specialist | 3.5 GB |
| final_synthesizer | Qwen3.5 2B Instruct Q4_K_M | Final answer synthesis | 2 GB |

Target GPU: RTX 4060 (8 GB VRAM). Models are hot-swapped via SIGTERM + relaunch.

### Local vs Remote Inference

- SAGE_MODEL_BACKEND=local  -> llama-server.exe on local machine (default)
- SAGE_MODEL_BACKEND=remote -> HTTP to Kaggle GPU worker via ngrok tunnel

Remote transport: core/remote_model_transport.py
API key: SAGE_REMOTE_GPU_API_KEY environment variable

---

## 4. SAGE Document Database (sage_document_db)

The permanent source-of-truth storage layer.

### Core Philosophy

```
CHROMA:    WHERE is relevant content likely to be? (fuzzy, disposable, rebuildable)
ARTIFACTS: WHAT EXACTLY was in the source?        (canonical, permanent, never modified)
```

Deleting chroma_db/ loses NOTHING. Rebuild from artifacts/ with one command.

### Supported File Types

| Extension | Parser |
|-----------|--------|
| .pdf .docx .pptx .xlsx | IBM Docling (layout-aware) |
| .txt .md | Simple text parser |
| .json | Simple JSON parser |
| .csv | Simple CSV parser |
| .png .jpg .jpeg .webp | Simple image parser |

### On-Disk Artifact Layout

```
artifacts/
`-- doc_<sha10>/
    |-- manifest.json          # Master index
    |-- text/
    |   |-- page_0001.json     # Text elements, paginated (PDF/PPTX)
    |   `-- document.json      # Single-file (TXT/MD/JSON/CSV)
    |-- images/
    |   `-- img_000001.png     # Extracted binary images
    |-- tables/
    |   `-- table_000001.json  # Structured rows + markdown
    `-- derived/
        `-- vision/
            `-- img_000001.json  # Vision model analysis history (append-only)
```

### Public API (SageDocumentDB)

```python
db = SageDocumentDB()

db.ingest_document("file.pdf", index_in_chroma=True, debug=True)
# -> {"doc_id", "artifact_dir", "counts", "index", "warnings"}

db.rag_search("query", top_k=5, doc_ids=None)
# -> list[RagResult]

db.exact_search("AUTH_V2", regex=True, case_sensitive=False)
# -> list[ExactSearchResult]

db.artifact_fetch("doc_27d0685bb3", "img_000001")
# -> dict: {id, type, page, local_path, exists} (image)
# -> dict: {id, type, text, page} (text)
# -> dict: {id, type, rows, markdown} (table)

db.list_doc_ids()
# -> list[str]

db.add_image_analysis("doc_id", "img_000001", model="qwen3-vl",
    task_type="ocr", raw_output="...", description="...", ocr_text="...")
# -> DerivedInsertResult

db.rebuild_all_indexes()
# -> {"source": {...}, "derived": {...}}
```

### Chroma Collections

| Collection | Record ID Format | Purpose |
|-----------|-----------------|---------|
| sage_source | {doc_id}:chunk:{N:06d} | Canonical text chunks, 384-dim MiniLM |
| sage_derived | {doc_id}:derived:image:{image_id} | Vision model analysis text (separate!) |

---

## 5. Agent Reasoning Loop

```
User message arrives
    |
    v
[app.py] _prepare_chat_request()
    - Registers uploaded files as RegisteredDocuments in RunState
    - Initializes RunState for this request
    |
    v
[orchestrator.run()] up to MAX_AGENT_LOOPS=8:

    Loop N:
    |-- Build Gemma prompt: system_prompt + history + tool_results (if N > 1)
    |-- Call Gemma via model_client / remote_model_transport
    |-- json_repair.py: strip <think> tags, extract JSON, one-shot repair if malformed
    |-- Parse response:
    |       {"type": "tool_calls", "calls": [...]}
    |    or {"type": "final", "answer": "..."}
    |
    |-- If tool_calls:
    |       dispatcher.dispatch_all(calls)
    |           |-- document_database: rag/exact/fetch/list
    |           |-- vision: resolve image -> Qwen3-VL -> save derived result
    |           |-- coder: Qwen-Coder -> Docker execute -> repair if failed
    |           `-- math: safe AST eval
    |       Each tool returns rich socket dict
    |       Socket-to-plug mapper strips internal fields -> compact Gemma result
    |       Append to history as {"type": "tool_results", "results": [...]}
    |
    `-- If final:
            Return answer to user
```

---

## 6. Socket -> Plug Architecture

Every internal component produces a RICH SOCKET (detailed record with timestamps, paths,
debug data, telemetry). A MAPPER converts this to a COMPACT PLUG for Gemma.

Gemma never sees raw internal data.

Example (RAG):
```
Internal socket (rich):          Gemma-facing plug (compact):
{                                {
  record_id: "doc_x:chunk:4",     "doc": "doc_x",
  distance: 0.23,                 "text": "...",
  local_path: "/abs/artifacts",   "page": 1,
  embedding: [0.12, ...384...],   "element_ids": ["txt_000005"]
  chroma_internal: {...},        }
  ...
}
```

All socket schemas frozen in: schemas/deterministic/*.schema.json
All socket builders in: core/sockets.py (DO NOT MODIFY)

Fields Gemma must NEVER see:
  local_path, traceback, pid, port, embedding vectors,
  Docker container IDs, cache file paths, raw Chroma responses

---

## 7. Configuration & Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| SAGE_ARTIFACTS_ROOT | ./artifacts | Canonical document artifact storage |
| SAGE_CHROMA_ROOT | ./chroma_db | Chroma vector database directory |
| SAGE_MAX_UPLOAD_SIZE_BYTES | 52428800 | Max upload size (50MB) |
| LLAMA_SERVER_PATH | auto-discovered | Path to llama-server.exe |
| MODEL_DIR | auto ./models | Directory containing GGUF model files |
| SAGE_MODEL_BACKEND | local | "local" or "remote" |
| SAGE_REMOTE_GPU_URL | (empty) | ngrok tunnel URL for Kaggle GPU |
| SAGE_REMOTE_GPU_API_KEY | (empty) | API key for remote GPU worker |
| SAGE_MOCK_MODE | 0 | Set to 1 for CPU-only test mode |
| GEMMA_CONTEXT | 16384 | Gemma context window in tokens |

---

## 8. Running SAGE

Install dependencies:
  pip install -r requirements.txt

Run full test suite (no GPU/Docker needed):
  set SAGE_MOCK_MODE=1
  pytest tests/ -v

Test document DB via CLI:
  python manual_db_test.py ingest "path/to/file.pdf" --debug
  python manual_db_test.py rag "quarterly results" --top-k 5
  python manual_db_test.py exact "192.168.1.1"
  python manual_db_test.py exact "AUTH_.*_V2" --regex
  python manual_db_test.py fetch doc_27d0685bb3 img_000001
  python manual_db_test.py rebuild-all

Start web application:
  python app.py
  # Open: http://127.0.0.1:8899

---

## 9. Development Status

### Stage 1: Security & Code Quality Audit [COMPLETE] - 211/211 tests passing

Security fixes applied:
  SEC-01: Image reference path validation (prevent arbitrary file reads)
  SEC-02: doc_id/image_id sanitized against path traversal
  SEC-04: Math exponentiation bounded against CPU/DoS
  SEC-07: request_id sanitized in ModelRuntimeCache

Correctness fixes applied:
  CORR-01: .env loading order fixed
  CORR-02: _dataclass_to_dict no longer drops similarity metrics
  CORR-03: clean_json_string regex fixed for nested JSON
  CORR-05: referenced_images field added to RunState

Code quality fixes applied:
  DUP-01: app.py upload logic deduplicated via _prepare_chat_request
  DUP-02: run_state.py tool result recording deduplicated
  DUP-03: coder.py LLM query helper extracted
  DUP-04: list_doc_ids() added as facade method (additive only)

Dead code removed:
  manual_test.py (root scratch script)
  Unused _find_blocks in code_executor/extractor.py
  Unused self.document_system_prompt in orchestrator.py

### Stage 2A: Performance Measurement [COMPLETE] (no code changes, measurement only)

Key bottlenecks measured:

  BOT-01: Model switching / GGUF reload     4,500-7,000 ms per switch    CRITICAL
  BOT-02: Cold system prompt KV eval        1,587-3,000 ms on Loop 1     HIGH
  BOT-03: HTTP client instantiation churn   ~200 ms per inference call   MEDIUM-HIGH
  BOT-04: Context explosion across loops    +460 ms per 1,000 tokens     HIGH
  BOT-05: temp/req_* dirs never cleaned     unbounded disk growth         HIGH
  BOT-06: No concurrency lock on model sw.  crash if 2 users overlap      CRITICAL

Proposed Stage 2B (NOT YET APPLIED):
  OPT-01: Persistent httpx.Client pooling   -150-300 ms per call          Very Low risk
  OPT-02: Health poll 500ms -> 100ms        -200-400 ms per switch        Very Low risk
  OPT-03: temp/req_* cleanup background     eliminates disk exhaustion    Very Low risk
  OPT-04: threading.Lock around ensure_model prevents crash on concurrency Low risk

### Not Started Yet

  Stage 2B: Apply performance optimizations
  Stage 3:  Prompt/context audit
  Stage 4:  High-level architecture documentation
  Stage 5:  Low-level socket-plug interface documentation
  Phase 17 hard stop: tools.json, abilities.json, agent_system.md (external finalization)

---

## 10. Files That Must NOT Be Modified

  schemas/deterministic/*.schema.json   - 25 frozen socket contracts
  schemas/DETERMINISTIC_SOCKETS.md     - Authoritative contract specification
  core/sockets.py                      - All socket builder functions
  prompts/abilities.json               - Frozen until Stage 3
  prompts/agent_system.txt             - Frozen until Stage 3
  prompts/tools.json                   - Frozen until Stage 3 / Phase 17

---

## 11. Architecture Invariants (NEVER VIOLATE)

  1. Gemma 4B is the ONLY semantic decision maker.
  2. Canonical artifacts (artifacts/) are write-once after ingestion.
  3. Chroma is disposable - always rebuildable from artifacts/.
  4. sage_source and sage_derived Chroma collections must remain separate.
  5. The dispatcher makes no semantic decisions.
  6. SAGE_MOCK_MODE=1 must allow 100% of tests to pass on any CPU.
  7. Gemma-facing compact plugs must NEVER contain internal fields.

---

## 12. Branch History

  SARTHAK2427/SAGE:main             - Original SAGE agent runtime (Sarthak base)
  RakshitJain-py/SAGE:rakshit_docdb_layer - Rakshit DB layer standalone
  USB snapshot (merged runtime)     - Agent + DB + dispatcher + tools (Stage 1+2A done)
