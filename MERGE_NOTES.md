# SAGE Merge Notes
## integration/sage-db-agent-merge

Tracking file for the merge of the SAGE Document Database with the old SAGE Agent Runtime.

---

## Migration Log

| Old Path | New Path | Status | Notes |
|----------|----------|--------|-------|
| `SAGE-rakshit_docdb_layer/sage_document_db/` | `SAGE/sage_document_db/` | COMPLETE | Copied into unified runtime, absolute paths & auto-device added |
| `SAGE-rakshit_docdb_layer/manual_db_test.py` | `SAGE/manual_db_test.py` | COMPLETE | Copied CLI DB testing utility into unified runtime |
| `SAGE/document_processor.py` | `SAGE/document_processor.py` | DEPRECATED | Deprecation notice added; active calls disconnected from orchestrator |
| `SAGE/app.py` | `SAGE/app.py` | UPDATED | Upload flow ingests into document_db and registers in RunState |
| `SAGE/orchestrator.py` | `SAGE/orchestrator.py` | UPDATED | Dispatches all tools via generic ToolRegistry and records in RunState |
| `SAGE/model_manager.py` | `SAGE/model_manager.py` | KEPT | Sequential model lifecycle preserved |
| `SAGE/model_client.py` | `SAGE/model_client.py` | KEPT | HTTP client preserved |
| `SAGE/code_executor/` | `SAGE/code_executor/` | KEPT | Exposed cleanly through coder adapter |
| `SAGE/prompts/` | `SAGE/prompts/` | KEPT | Not frozen — waiting for Phase 17 final protocol files |
| `SAGE/static/` | `SAGE/static/` | KEPT | Frontend contract preserved |

## New Modular Files

| Path | Purpose | Phase |
|------|---------|-------|
| `SAGE/db_service.py` | Process-level SageDocumentDB singleton | 3 |
| `SAGE/core/dispatcher.py` | Generic tool registry + dispatch + timing | 5 |
| `SAGE/core/run_state.py` | Runtime execution state + RegisteredDocument | 9 |
| `SAGE/core/json_repair.py` | Decoupled JSON cleanup and 1-turn repair prompt | 15 |
| `SAGE/tools/document_database.py` | DB capability adapters (rag, exact, fetch, list) | 6 |
| `SAGE/tools/vision.py` | Vision/OCR adapter scaffold with mock mode | 12 |
| `SAGE/tools/coder.py` | Coder adapter wrapping sandbox execution pipeline | 13 |
| `SAGE/tools/math_tool.py` | Safe deterministic AST calculator | 10 |
| `SAGE/tools/registry.py` | Central tool registry wiring factory | 6/15 |
| `SAGE/manual_db_test.py` | Standalone CLI DB test harness | 2 |
| `SAGE/tests/` | Comprehensive test suite (CPU / mock mode) | 7, 9, 10, 12, 13, 15, 16 |

## Pending / Future (Phase 17 Hard Stop)

Waiting for external finalization of:
- `tools.json`
- `abilities.json`
- `agent_system.md`
- Gemma JSON schema
