# SAGE — Unified Multi-Model Agent Orchestrator & Document Database

SAGE is a clean, local multi-model agent orchestrator running offline on GPU/CPU with sequential inference and a structure-aware document database.

---

## System Architecture

```text
                     ┌─────────────────────────────┐
                     │          USER / UI          │
                     └──────────────┬──────────────┘
                                    │ (Uploads & Objectives)
                                    ▼
                     ┌─────────────────────────────┐
                     │      FastAPI (app.py)       │
                     └──────────────┬──────────────┘
                                    │ (RunState + RegisteredDocument)
                                    ▼
                     ┌─────────────────────────────┐
                     │   GEMMA 4B (Semantic Brain) │
                     └──────────────┬──────────────┘
                                    │ (JSON Tool Call Requests)
                                    ▼
                     ┌─────────────────────────────┐
                     │ Generic Tool Dispatcher     │
                     │ (core/dispatcher.py)        │
                     └──────┬─────┬─────┬─────┬────┘
                            │     │     │     │
         ┌──────────────────┘     │     │     └──────────────────┐
         ▼                        ▼     ▼                        ▼
┌──────────────────┐   ┌────────────┐ ┌───────────────┐   ┌────────────┐
│ SAGE Document DB │   │  Vision    │ │ Coder Sandbox │   │  Safe Math │
│ (Chroma + MiniLM │   │ (Qwen3-VL) │ │ (Qwen2.5-Coder│   │ Calculator │
│ + ArtifactStore) │   └────────────┘ │  + Docker)    │   └────────────┘
└──────────────────┘                  └───────────────┘
```

---

## Features

- **Gemma 4B Semantic Controller**: Sole agentic decision maker — never bypassed by background heuristics.
- **Unified Document Database (`sage_document_db`)**:
  - Structure-aware chunking & canonical artifact storage (`artifacts/`).
  - Semantic RAG with `all-MiniLM-L6-v2` (`chroma_db/`) with automatic CUDA/CPU fallback.
  - Literal and regex exact search on canonical text, tables, and code.
  - Direct artifact element fetch by ID.
  - Derived visual analysis hooks with save-vs-index separation.
- **Decoupled Generic Dispatcher (`core/dispatcher.py`)**:
  - Independent capability registry ensuring tool-to-tool isolation.
  - Standardized JSON execution records, error handling, and latency tracking.
- **Coder Specialist (`tools/coder.py`)**:
  - Qwen2.5-Coder 7B code generation with Docker sandbox execution and automated multi-attempt repair loop.
- **Vision Specialist (`tools/vision.py`)**:
  - Targeted image inspection via Qwen3-VL 4B (never reparses entire PDFs).
- **Safe Math Utility (`tools/math_tool.py`)**:
  - Deterministic AST arithmetic evaluator preventing unsafe code injection.
- **Sequential Model Lifecycle (`model_manager.py`)**:
  - Swaps GGUF models dynamically on demand via `llama-server.exe` to fit within 8GB VRAM (RTX 4060).
- **Deterministic Mock Mode (`SAGE_MOCK_MODE=1`)**:
  - Enables 100% of unit and integration tests to run instantly on any CPU without GPU or Docker dependencies.

---

## Directory Layout

```text
SAGE/
├── app.py                     # FastAPI web application & upload handler
├── config.py                  # Consolidated runtime paths, ports & model configs
├── db_service.py              # Process-level SageDocumentDB singleton
├── manual_db_test.py          # Standalone CLI document database test harness
├── model_client.py            # HTTP client for llama.cpp server
├── model_manager.py           # Sequential model loading and lifecycle management
├── orchestrator.py            # Main agent loop, JSON repair & dispatcher wiring
├── requirements.txt           # Unified dependency specifications
│
├── core/
│   ├── dispatcher.py          # ToolRegistry & ToolResult dispatching infrastructure
│   ├── json_repair.py         # Robust JSON cleanup & corrective retry prompt
│   └── run_state.py           # Backend-owned RunState & RegisteredDocument tracking
│
├── tools/
│   ├── registry.py            # Central tool factory registering all adapters
│   ├── document_database.py   # RAG search, exact search, fetch & list adapters
│   ├── vision.py              # Qwen3-VL targeted image inspection adapter
│   ├── coder.py               # Qwen-Coder + Docker sandbox adapter
│   └── math_tool.py           # Safe AST calculator adapter
│
├── sage_document_db/          # Canonical Document Database package
│   ├── artifact_store.py      # Artifact filesystem store & manifest manager
│   ├── chroma_store.py        # Vector database management (sage_source, sage_derived)
│   ├── chunker.py             # Structure-aware chunking engine
│   ├── config.py              # Anchored absolute database paths & auto device
│   ├── derived.py             # Image analysis persistence (save vs index)
│   ├── docling_parser.py      # Heavy PDF/DOCX structure parser
│   ├── embeddings.py          # MiniLM sentence transformer with CPU fallback
│   ├── exact_search.py        # Literal/regex search on canonical artifacts
│   ├── pipeline.py            # Document ingestion orchestration
│   └── simple_parsers.py      # Fast text, markdown, CSV, JSON parsers
│
├── code_executor/             # Docker sandbox pipeline & code fixer
├── prompts/                   # System prompt definitions
├── static/                    # Frontend Web UI (HTML, CSS, JS)
└── tests/                     # Automated test suite (mock mode, no GPU needed)
    ├── conftest.py
    ├── test_app_upload.py
    ├── test_coder.py
    ├── test_db_adapters.py
    ├── test_dispatcher.py
    ├── test_math_tool.py
    ├── test_orchestrator_integration.py
    ├── test_run_state.py
    └── test_vision.py
```

---

## Setup & Running

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Test Suite (CPU / Mock Mode)
```bash
pytest tests/
```

### 3. Test the Document Database Directly
```bash
# Ingest a document
python manual_db_test.py ingest "sample.pdf" --debug

# Semantic RAG search
python manual_db_test.py rag "quarterly performance summary" --top-k 3

# Exact search
python manual_db_test.py exact "INV-2026-004"

# Direct element fetch
python manual_db_test.py fetch doc_a81f42c91e txt_000001
```

### 4. Start the SAGE Web Application
```bash
python app.py
```
Open your browser at `http://127.0.0.1:8899`.
