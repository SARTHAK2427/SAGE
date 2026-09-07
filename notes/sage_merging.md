# SAGE Merge & Compatibility Refactor Plan
## SIH Hackathon 2026 — Problem Statement 117
## Purpose: safely merge the old SAGE agent runtime with the new SAGE document database WITHOUT prematurely freezing the final agent/tool JSON protocol

> **Audience:** IDE coding agent working directly inside the SAGE repository.
>
> **Important:** Follow this document literally. Do not redesign the architecture, do not invent new frameworks, and do not replace working modules unless this plan explicitly says to.
>
> **Current priority:** perform all merge/refactor work that is safe to do **before** the final `tools.json`, `abilities.json`, Gemma agent JSON protocol, system prompt, and tool-result schemas are finalized.

---

# 0. Executive Summary

There are currently **two major SAGE codebases/subsystems** that must become one:

1. **The newer SAGE Document Database subsystem**
   - This is the current source of truth for document ingestion, parsing, artifact storage, semantic retrieval, exact retrieval, and direct artifact retrieval.
   - It already uses:
     - file routing
     - Docling for heavy documents
     - simple parsers for simple formats
     - normalized document representation
     - canonical artifact storage
     - structure-aware chunking
     - `all-MiniLM-L6-v2` embeddings
     - ChromaDB
     - semantic RAG
     - exact/literal/regex search
     - direct artifact fetch
     - derived vision-analysis storage hooks
   - **Treat this subsystem as functionally complete unless a real integration bug is discovered.**

2. **The older pulled SAGE agent/runtime subsystem (`SAGE_PULL`)**
   - This contains useful runtime infrastructure that should be preserved:
     - FastAPI backend
     - Gemma agent loop
     - JSON parsing / repair behavior
     - llama.cpp server management
     - model loading/unloading
     - model HTTP client
     - Qwen coder integration
     - Docker sandbox execution
     - UI/static frontend
   - It also contains a **legacy document-processing path** that must no longer be used for normal document queries.

The task is **not to rewrite SAGE**.

The task is to:

```text
NEW DOCUMENT DB
        +
OLD AGENT / MODEL RUNTIME
        ↓
ONE CLEAN SAGE APPLICATION
```

The merged architecture must preserve a strict principle:

> **Gemma 4B is the only strategic controller / brain.**
>
> Registered tools, database subsystems, and specialist models must not autonomously make cross-tool decisions.

However, deterministic internal implementation operations are allowed without another Gemma round-trip.

Examples:

```text
Gemma authorizes vision on img_000017
        ↓
backend resolves img_000017 → local file path
        ↓
backend loads Qwen3-VL
        ↓
backend runs inference
        ↓
backend saves the raw result
```

That is allowed.

What is not allowed:

```text
RAG decides to call vision
Vision decides to call coder
Coder decides to search Chroma
DB decides to invoke another registered Sage tool
```

Those decisions belong to Gemma.

---

# 1. DO NOT IMPLEMENT THE FINAL AGENT PROTOCOL YET

The final design of the following is still being discussed externally and **must not be invented by the IDE**:

- final Gemma outer JSON schema
- final tool-call JSON schema
- final tool-result JSON schema
- final error schema
- final retry/acceptance protocol
- final `tools.json`
- final `abilities.json`
- final `agent_system.md`
- final specialist prompts
- final runtime context exposed to Gemma
- final names/arguments of every future tool

Therefore:

## Safe to implement now

- repository restructuring
- compatibility cleanup
- import cleanup
- preservation of working old runtime
- disconnection of legacy document parser
- upload → new DB ingestion integration
- document registration / `doc_id` plumbing
- generic tool dispatcher infrastructure
- tool registry infrastructure
- adapters/wrappers around existing DB capabilities
- mock tool support
- model-service abstraction cleanup
- derived vision persistence scaffolding
- device auto-detection for embedding model
- tests for all of the above
- logging, validation, tracing, deterministic state
- runtime-owned reference resolution
- backward-compatible adapters if required

## Not safe to freeze yet

- final JSON contracts Gemma will emit
- final names of every tool/function
- final prompt wording
- final abilities restrictions
- final acceptance policy for vision results
- final UI trace representation if it depends on the new JSON schema

If a later phase in this document depends on those unresolved schemas, implement it behind a neutral adapter/interface and leave a clearly marked TODO.

---

# 2. First Task: Inspect the Repository and Identify the Two Roots

Before editing anything:

1. Inspect the repository tree.
2. Locate the **new SAGE root** containing the document DB, expected to include something close to:

```text
sage_document_db/
manual_db_test.py
artifacts/
chroma_db/
README_DOCUMENT_DB.md
```

3. Locate the **old pulled SAGE runtime**, expected to be under something close to:

```text
SAGE_PULL/
└── SAGE_github/
    └── SAGE/
```

with files similar to:

```text
app.py
config.py
orchestrator.py
model_manager.py
model_client.py
document_processor.py
code_executor/
prompts/
static/
tests/
```

4. Do not assume these exact paths if the repository has changed.
5. Print/log the discovered paths before modifying code.
6. If either major subsystem cannot be found, stop and report exactly what is missing.

---

# 3. Create a Safe Working Baseline Before Refactoring

Before moving files or changing imports:

1. Run `git status`.
2. Ensure current work is not accidentally overwritten.
3. Create a temporary merge/refactor branch if the repository is under Git, for example:

```text
integration/sage-db-agent-merge
```

4. Do **not** delete old source files during the first pass.
5. If files must be moved:
   - prefer `git mv`
   - preserve history
   - update imports systematically
6. Keep a migration note, for example:

```text
MERGE_NOTES.md
```

Record:
- old path
- new path
- whether file was kept / moved / disconnected / deprecated
- tests performed
- unresolved TODOs

---

# 4. Current Target System Architecture

The final merged system should structurally converge toward:

```text
                              USER
                               │
                    upload + query
                               │
                               ▼
                            BACKEND
                               │
              ┌────────────────┴────────────────┐
              │                                 │
            FILES                             QUERY
              │                                 │
              ▼                                 │
      SAGE DOCUMENT DATABASE                    │
              │                                 │
       FILE TYPE ROUTER                         │
              │                                 │
       ┌──────┴──────┐                          │
       ▼             ▼                          │
    DOCLING      SIMPLE PARSERS                 │
       └──────┬──────┘                          │
              ▼                                 │
       NORMALIZED DOCUMENT                      │
              │                                 │
       ┌──────┴───────────────┐                 │
       ▼                      ▼                 │
  CANONICAL ARTIFACTS       CHUNKER             │
       │                      │                 │
       │                 EMBEDDINGS             │
       │                      │                 │
       │                   CHROMA               │
       │                      │                 │
       └──────────┬───────────┘                 │
                  │ doc_id(s)                   │
                  └─────────────┬───────────────┘
                                ▼
                             GEMMA 4B
                        ONLY STRATEGIC BRAIN
                                │
                        structured decisions
                                │
                                ▼
                         TOOL DISPATCH LAYER
                                │
          ┌─────────────────────┼──────────────────────┐
          │                     │                      │
          ▼                     ▼                      ▼
   DOCUMENT DATABASE       VISION / OCR             CODER
      capabilities            specialist             specialist
          │                     │                      │
          └─────────────────────┼──────────────────────┘
                                │
                         structured results
                                │
                                ▼
                             GEMMA 4B
                                │
                         repeat / complete
                                │
                                ▼
                               USER
```

Important terminology:

- Upload time performs **RAG indexing**, not RAG querying.
- Query-time semantic retrieval happens only when Gemma authorizes it.
- Canonical artifacts are the source of truth.
- Chroma is a searchable index.
- Model-derived vision/OCR output is not canonical source truth.

---

# 5. Document Database: Current Contract

The new document DB is the canonical document subsystem.

Expected public facade:

```python
from sage_document_db import SageDocumentDB

db = SageDocumentDB()
```

Expected operations already provided:

```python
db.ingest_document(...)
db.rag_search(...)
db.exact_search(...)
db.artifact_fetch(...)
db.add_image_analysis(...)
db.rebuild_all_indexes()
```

Expected supported formats:

```text
.pdf
.docx
.pptx
.xlsx
.md
.txt
.json
.csv
.png
.jpg
.jpeg
.webp
```

The DB architecture is:

```text
NEW FILE
   ↓
FILE TYPE ROUTER
   ↓
┌───────────────────┐
│                   │
DOCLING        SIMPLE PARSERS
│                   │
└─────────┬─────────┘
          ↓
 NORMALIZED DOCUMENT
          ↓
┌───────────────────┐
│                   │
ARTIFACTS         CHUNKER
│                   ↓
│              EMBEDDINGS
│                   ↓
│                CHROMA
│
├─ exact search
└─ artifact fetch
```

Critical DB rule:

```text
CHROMA
= where relevant information is likely to be

ARTIFACTS
= what exactly existed in the source
```

Do not violate this distinction.

---

# 6. Canonical Artifact Rules

The artifact store is the source of truth.

Expected structure conceptually:

```text
artifacts/
└── doc_<id>/
    ├── manifest.json
    ├── text/
    ├── images/
    ├── tables/
    ├── code/
    └── derived/
        └── vision/
```

Rules:

1. Canonical source content must not be overwritten by model conclusions.
2. Extracted image files remain unchanged.
3. Vision/OCR outputs belong under `derived/`.
4. Chroma can be rebuilt.
5. Original source document does not need to remain forever after successful ingestion if current DB semantics already support rebuild from artifacts.
6. Do not introduce a second competing document cache.
7. Do not allow the old `document_processor.py` to create a parallel document truth.

---

# 7. Old Runtime Components: KEEP vs DEPRECATE

## KEEP and integrate

The old pulled runtime contains valuable components.

### Keep

```text
app.py
orchestrator.py
model_manager.py
model_client.py
code_executor/
static/
existing JSON cleanup/repair logic
existing model switching logic
existing health polling logic
existing Docker code execution/repair logic
```

These may be reorganized, but preserve working behavior.

## Deprecate / disconnect

### Legacy document processing path

The old:

```text
document_processor.py
```

or equivalent legacy code that:

- reparses PDF/DOCX on demand
- uses PyMuPDF/python-docx independently
- renders pages based on text length
- sends whole pages/images directly to Qwen as part of document analysis

must no longer be the default document path.

Do not necessarily delete it immediately.

Instead:

1. mark it deprecated
2. remove its use from active orchestration
3. ensure uploads and document queries flow through `sage_document_db`
4. add a clear comment such as:

```python
# DEPRECATED:
# Legacy on-demand document parsing path.
# SAGE document access now goes through sage_document_db.
```

5. add tests proving normal document queries cannot accidentally invoke it

---

# 8. Repository Unification Strategy

Aim for one clean application root.

A possible target layout is:

```text
SAGE/
├── app.py
│
├── core/
│   ├── orchestrator.py
│   ├── model_manager.py
│   ├── model_client.py
│   ├── run_state.py
│   └── dispatcher.py
│
├── tools/
│   ├── __init__.py
│   ├── registry.py
│   ├── document_database.py
│   ├── vision.py
│   ├── coder.py
│   └── math_tool.py
│
├── sage_document_db/
│   ├── ...
│
├── code_executor/
│   ├── ...
│
├── prompts/
│   ├── ...
│
├── static/
│   ├── ...
│
├── config.py
├── requirements.txt
├── tools.json              # final content NOT frozen yet
├── abilities.json          # final content NOT frozen yet
└── README...
```

This exact directory layout is not mandatory.

The important requirements are:

- one runtime
- one DB
- one model manager
- one dispatcher
- one canonical artifact store
- no duplicate document-processing pipeline
- no duplicate app entrypoint in production

If moving files would create unnecessary risk, keep paths and use compatibility imports first.

---

# 9. Phase A — Make the New DB Importable from the Old Runtime

Goal:

The old runtime should be able to instantiate the new DB.

Implement a single DB service instance per backend process.

Example concept:

```python
from sage_document_db import SageDocumentDB

document_db = SageDocumentDB()
```

Do not create a new `SageDocumentDB()` for every tool call unless the existing class is explicitly designed for that.

Requirements:

1. one reusable process-level instance
2. embedding model should not be repeatedly reloaded
3. Chroma client should be reused if supported
4. no circular imports
5. path resolution must work regardless of current working directory
6. add startup diagnostics:
   - artifact root
   - Chroma root
   - embedding model
   - embedding device

Add a smoke test:

```text
backend import
→ construct DB
→ no exception
```

---

# 10. Phase B — Change Upload Handling

Current old behavior likely resembles:

```text
upload
→ temporary file
→ file_map / attachment manifest
→ orchestrator sees raw path
→ legacy document processor
```

Replace this for supported documents with:

```text
upload
→ temporary source path
→ SageDocumentDB.ingest_document(...)
→ ingestion result
→ doc_id
→ runtime document registration
```

Detailed steps:

1. Preserve the current FastAPI upload endpoint contract as much as possible.
2. Save the received file temporarily.
3. Call:

```python
result = document_db.ingest_document(temp_path, ...)
```

4. Extract at minimum:

```text
doc_id
original filename
file type
ingestion status
```

5. Store this in a backend runtime structure associated with the request/session.
6. Do not send the raw file path to Gemma as the primary document identity.
7. Gemma should eventually reason about a registered document by stable `doc_id`.
8. The raw temporary file may be removed only after successful ingestion if safe under the current DB implementation.
9. On ingestion failure:
   - preserve useful error message
   - do not pretend document is available
   - do not create fake `doc_id`
10. Existing UI upload behavior should continue to work.

Create tests for:

```text
upload TXT
upload PDF
upload unsupported extension
duplicate upload
ingestion failure
```

No live LLM is required for these tests.

---

# 11. Phase C — Introduce a Neutral Runtime Document Reference

Do not tie agent logic to absolute filesystem paths.

Create a small runtime representation, for example:

```python
@dataclass
class RegisteredDocument:
    doc_id: str
    display_name: str
    file_type: str | None = None
    source_name: str | None = None
```

This object is runtime metadata only.

Do not duplicate the full canonical manifest.

The goal is to allow runtime state to say:

```text
available documents:
- doc_a81f42c91e : report.pdf
- doc_b2c3d4e5f6 : employees.docx
```

without forcing the agent to know storage paths.

---

# 12. Phase D — Build a Generic Tool Registry Infrastructure

The final `tools.json` is not frozen yet, but the runtime infrastructure can be built now.

Do NOT hardcode logic like:

```python
if tool == "document_analyzer":
    ...
elif tool == "coder":
    ...
```

Create a generic registry.

Conceptual interface:

```python
register_tool(
    tool_name,
    function_name,
    callable,
)
```

or:

```python
registry[(tool_name, function_name)] = callable
```

Requirements:

1. tool lookup by stable string identifier
2. function lookup
3. validation that requested tool exists
4. validation that requested function exists
5. structured success result
6. structured failure result
7. exception capture
8. timing capture
9. call ID capture
10. ability to swap real implementation for mock implementation
11. registry must not depend on Gemma prompt wording
12. registry must not itself decide what tool should run

Create tests:

```text
known tool/function executes
unknown tool fails cleanly
unknown function fails cleanly
tool exception becomes structured error
mock implementation can replace real implementation
```

---

# 13. Phase E — Add Database Tool Adapters

Create wrappers around the new DB.

Do not expose Chroma internals directly to the agent runtime.

At minimum create neutral wrappers for the already-existing DB operations:

```text
semantic retrieval
exact/literal retrieval
direct artifact fetch
```

These may temporarily use placeholder internal names until final `tools.json` is frozen.

Suggested internal Python names:

```python
tool_rag_search(...)
tool_exact_search(...)
tool_artifact_fetch(...)
```

Do not assume these are the final external agent function names.

Each wrapper should:

1. accept plain serializable arguments
2. call `SageDocumentDB`
3. convert dataclasses/objects into JSON-safe dictionaries
4. return bounded output
5. preserve:
   - `doc_id`
   - element ID
   - page/slide/sheet info when available
   - text/snippet
   - distance/score where relevant
   - image refs where relevant
6. never expose unsafe arbitrary filesystem operations
7. never autonomously call another registered Sage tool

---

# 14. Phase F — Implement `list_artifacts` at the DB/Adapter Boundary

The current DB public README does not clearly expose a `list_artifacts()` facade method, but the agent will likely need a deterministic way to enumerate artifacts in a document.

Implement this carefully.

Preferred behavior:

```python
list_artifacts(
    doc_id,
    artifact_type=None,
)
```

Possible artifact types:

```text
text
image
table
code
```

Return only metadata required for discovery.

Example shape:

```json
{
  "doc_id": "doc_abc",
  "artifacts": [
    {
      "element_id": "img_000001",
      "type": "image",
      "page": 3
    }
  ]
}
```

Rules:

1. derive from canonical artifact manifest/store
2. do not infer nonexistent artifacts
3. do not invoke vision
4. do not load image pixels unless needed
5. do not dump entire document content
6. support filters
7. ensure deterministic ordering where possible

If the artifact store already has an internal enumeration API, reuse it.

Do not duplicate manifest parsing logic unnecessarily.

---

# 15. Phase G — Tool-to-Tool Isolation Rule

Enforce the following architectural constraint in code comments and module design:

> Registered Sage tools may not autonomously invoke other registered Sage tools.

Examples:

## Forbidden

```text
rag_search wrapper
→ vision tool
```

```text
vision tool
→ coder tool
```

```text
coder tool
→ exact search tool
```

## Allowed internal implementation

```text
rag_search
→ embedding service
→ Chroma
```

```text
vision
→ resolve artifact path
→ model manager
→ Qwen3-VL
```

```text
coder
→ model manager
→ Qwen Coder
→ Docker sandbox
```

These are implementation dependencies, not agentic cross-tool delegation.

Add a comment in the dispatcher or architecture documentation explaining this rule.

---

# 16. Phase H — Runtime-Owned Reference Resolution

To avoid useless Gemma round-trips, implement deterministic reference resolution.

Example:

Gemma later authorizes vision for:

```text
doc_id = doc_abc
image_id = img_000017
```

The vision tool implementation is allowed to internally resolve:

```text
doc_abc + img_000017
→ canonical image path
```

without requiring a second explicit `artifact_fetch` tool call.

This is allowed because:

- Gemma already authorized access to that artifact
- path resolution is deterministic plumbing
- no new reasoning decision is made

Similarly:

- coder may resolve explicitly supplied code artifact references
- tool runtime may resolve doc IDs to manifests
- dispatcher may create call IDs
- model manager may load/unload models automatically
- trace logger may persist execution records automatically

Do not classify these as autonomous tool calls.

---

# 17. Phase I — Preserve the Existing Model Manager

Do not rewrite the model lifecycle unless necessary.

The old system already manages sequential llama.cpp model loading.

Keep the behavior:

```text
need model A
→ stop current model if required
→ start llama-server
→ wait for health
→ call model
→ later swap as required
```

Expected specialist models:

```text
Gemma 4B
Qwen3-VL 4B + mmproj
Qwen2.5-Coder 7B
```

Hardware target:

```text
RTX 4060 Laptop GPU
8 GB VRAM
one heavy model inference at a time
```

Therefore:

- do not try to keep all large models resident
- do not implement parallel heavy inference
- do not introduce a multi-model server unless explicitly requested later
- preserve sequential model swapping

---

# 18. Phase J — Embedding Device Auto-Selection

The document DB currently uses `all-MiniLM-L6-v2`.

The deployment RTX 4060 should use CUDA.

Development machines without CUDA should still work.

Implement config behavior similar to:

```text
EMBEDDING_DEVICE = auto
```

Resolution:

```python
if configured == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = configured
```

Requirements:

1. CPU fallback must remain functional
2. no hardcoded `"cuda"` that breaks local development
3. log selected embedding device
4. embedding model should be reused, not reloaded per call
5. CUDA embedding should use SentenceTransformers/PyTorch directly
6. do **not** route MiniLM through llama.cpp
7. preserve current 384-dimension vector compatibility
8. do not change the embedding model for MVP
9. if switching device does not change embedding values materially, Chroma schema should not need redesign
10. test CPU mode locally even if CUDA unavailable

Add config comments explaining:

```text
Large LLMs/VLMs:
    llama.cpp + CUDA

MiniLM embeddings:
    SentenceTransformers/PyTorch + CUDA when available
```

---

# 19. Phase K — Vision Tool Scaffolding

Do not freeze the final external tool schema yet.

Build an internal vision adapter capable of:

```text
input:
- doc_id
- one or more image artifact IDs
- textual task/instruction

process:
- resolve canonical image path
- invoke Qwen3-VL through existing model runtime
- collect raw output
- return structured result
```

Important:

1. Qwen3-VL is a targeted visual specialist.
2. It must not reparse whole PDFs.
3. It must only inspect explicitly selected image artifacts.
4. It may perform:
   - OCR
   - visual understanding
   - diagram interpretation
   - chart interpretation
   - screenshot/image reading
5. The agent does not need to know Qwen is a model.
6. The future registered capability may be called something like vision/OCR, but do not freeze naming unless final schema exists.

Support a `MOCK_MODE` or dependency injection so this path can be tested without GPU.

Example mock:

```python
return {
    "raw_output": "MOCK: image contains invoice number A9821",
    "model": "mock",
}
```

---

# 20. Phase L — Vision Result Persistence

The current document DB supports derived image analysis.

Use that capability, but preserve source-vs-derived separation.

Target flow:

```text
Gemma authorizes vision
        ↓
Qwen result
        ↓
save raw derived attempt
        ↓
return result to Gemma
```

Do not modify:

```text
artifacts/doc_x/images/img_x.png
```

Store analysis under:

```text
artifacts/doc_x/derived/vision/
```

The DB already conceptually supports:

```python
db.add_image_analysis(...)
```

However, there is an important future acceptance concern.

A model may be retried on the same image.

Therefore the merge should prepare for:

```text
attempt 1
attempt 2
attempt 3
```

without destroying earlier output.

Preferred behavior:

```json
{
  "image_id": "img_000017",
  "analyses": [
    {
      "attempt": 1,
      "raw_output": "...",
      "status": "pending_or_superseded"
    },
    {
      "attempt": 2,
      "raw_output": "...",
      "status": "accepted"
    }
  ]
}
```

Do not force this exact JSON schema yet if current DB format differs.

Instead:

1. inspect existing `add_image_analysis()`
2. preserve its current compatibility
3. if it immediately indexes every attempt into `sage_derived`, add a safe optional flag or internal pathway such as:

```python
index_in_chroma=False
```

for a nonaccepted attempt

4. avoid breaking existing tests
5. leave acceptance policy configurable

The eventual intended policy is:

```text
raw attempt always preserved
accepted/latest trusted result becomes searchable derived knowledge
```

If this change is too invasive for the existing DB, isolate it behind the vision adapter and document the limitation rather than rewriting the DB.

---

# 21. Phase M — Derived Vision Search Semantics

Maintain strict distinction:

```text
sage_source
= canonical source chunks

sage_derived
= model-derived vision conclusions
```

Do not silently merge derived output into canonical artifacts.

Do not overwrite source chunks.

If a derived analysis is accepted/indexed, it should be searchable as derived knowledge with metadata linking back to:

```text
doc_id
image_id
model
task type
analysis ID / attempt
```

If current DB already does this, preserve behavior.

---

# 22. Phase N — Preserve Existing Coder Pipeline

The old coder subsystem is valuable and should be retained.

Expected flow:

```text
Gemma authorizes coding task
        ↓
Qwen2.5-Coder
        ↓
extract code
        ↓
Docker sandbox
        ↓
execution result
        ↓
optional repair loop
        ↓
structured result
```

Do not break:

- code fence extraction
- Python extraction logic
- sandbox isolation
- network disable
- resource limits
- timeout
- auto-repair loop
- stdout/stderr capture

Refactor only enough to expose it through the generic tool dispatcher.

Add mock mode for local non-GPU integration tests.

---

# 23. Phase O — Add a Deterministic Math Utility

A lightweight arithmetic capability can be implemented now because it does not depend on the final LLM schema.

Create a safe calculator backend.

Requirements:

1. no unrestricted `eval()`
2. support basic arithmetic
3. support parentheses
4. support percentages/formulas
5. return structured success/error
6. do not require loading Qwen Coder
7. do not autonomously invoke other tools

Potential implementation:
- AST-based safe expression evaluator
- whitelisted operators/functions

Do not expose arbitrary Python execution.

---

# 24. Phase P — Introduce Runtime State Without Final Agent Schema

Create a backend-owned state object.

This is not Gemma's private reasoning.

Suggested runtime fields:

```text
run_id
request_id
loop_index
user_text
registered_documents
tool_calls
tool_results
errors
timings
current_model
created_at
updated_at
```

Potential dataclasses:

```python
RunState
ToolCallRecord
ToolResultRecord
RegisteredDocument
```

Rules:

- Python owns factual execution state.
- Gemma owns decisions.
- Do not let Gemma invent:
  - actual file paths
  - actual doc IDs
  - call IDs
  - timestamps
  - execution status
  - model process IDs
- runtime state should be serializable for debugging
- do not expose every internal field to Gemma by default

---

# 25. Phase Q — Keep JSON Repair Logic, but Decouple It

The old orchestrator already has JSON cleanup / corrective retry behavior.

Preserve it.

Refactor it into a reusable component if reasonable:

```text
parse model response
→ validate outer structure
→ attempt cleanup
→ one corrective retry
→ structured failure if still invalid
```

Do not finalize the new Pydantic schemas until the external protocol is frozen.

Instead:

- preserve existing schema support
- isolate validation in one module
- make it easy to swap to final schemas later

---

# 26. Phase R — Mock Mode

This is mandatory because much development is happening without the final RTX 4060 machine.

Support configuration such as:

```text
SAGE_MOCK_MODE=1
```

or per-specialist mocks.

At minimum support:

```text
mock Gemma
mock vision
mock coder
real DB
real math
```

Mock responses should be deterministic.

Purpose:

```text
user request
→ orchestrator
→ mocked structured tool decision
→ real dispatcher
→ real RAG/exact/fetch/math
→ mocked specialist if required
→ mocked next Gemma turn
→ final
```

This allows complete plumbing tests before GPU deployment.

Do not mix mock results into persistent production derived caches unless explicitly marked as mock.

---

# 27. Phase S — Tests to Complete Before GPU Handoff

## DB integration

- DB instantiates
- upload ingests document
- doc ID returned
- artifact directory created
- Chroma populated
- RAG works
- exact search works
- artifact fetch works
- list artifacts works

## Runtime

- run state created
- tool registry resolves valid call
- invalid tool rejected
- invalid function rejected
- tool exceptions captured
- timings recorded

## Vision mock

- image ref resolves
- mock vision runs
- raw derived output can be stored
- canonical image unchanged
- mock outputs are not accidentally indexed as production knowledge

## Coder mock

- dispatcher reaches coder adapter
- mock result returns
- no model required

## Math

- valid expression
- invalid expression
- forbidden syntax
- divide-by-zero

## Upload-to-agent plumbing

- uploaded file becomes doc ID
- agent runtime sees doc reference
- raw path is not used as semantic document identity

## Legacy path guard

- no normal document query invokes legacy `document_processor`

---

# 28. Phase T — Do Not Break the Existing UI

Preserve the frontend unless required.

Existing useful behavior includes:
- chat interface
- upload
- status
- stop
- agent trace drawer

If backend result objects change internally, add compatibility conversion rather than redesigning UI now.

UI redesign is not part of this merge phase.

---

# 29. Phase U — Model/Hardware Context for Deployment

Target GPU machine:

```text
NVIDIA RTX 4060 Laptop GPU
8 GB VRAM
```

Important constraints:

- heavy LLM/VLM inference is sequential
- one heavy model at a time
- Gemma 4B is the central controller
- Qwen3-VL 4B is targeted vision/OCR specialist
- Qwen2.5-Coder 7B is coding specialist
- MiniLM embeddings are small and may use CUDA directly through PyTorch
- do not design around simultaneous residency of all large models

The model manager should therefore remain capable of:

```text
Gemma
→ unload
→ specialist
→ unload
→ Gemma
```

---

# 30. Global Controller Rule

This rule must appear in code comments / architecture docs.

> **Gemma is the sole semantic controller.**
>
> A registered Sage capability executes only the task it was explicitly given.
> It does not decide that another registered capability should be invoked.

Allowed backend automation:

```text
argument validation
file/path resolution
manifest lookup
model loading
model health checks
logging
timing
cache reads
cache writes
persistence
JSON parsing
retry of deterministic I/O
```

Not allowed without Gemma:

```text
choose another tool
change user objective
expand the task semantically
search another data source
ask another specialist for interpretation
decide that more evidence is required
```

---

# 31. Future Agent Context — For Awareness Only

The eventual Gemma prompt will conceptually be built from:

```text
agent_system.md
+
abilities.json
+
tools.json
+
small runtime context
+
user text
```

Do not implement the final content yet.

The conceptual roles are:

```text
agent_system.md
= controller behavior and operating rules

abilities.json
= what Gemma may do itself vs must delegate

tools.json
= exact executable capabilities and argument contracts
```

The user message itself should remain mostly unmodified for MVP.

---

# 32. Abilities Context — Current Direction

The existing abilities design identifies Gemma as a foreground orchestrator with native capabilities such as:

```text
intent understanding
reasoning
planning
task decomposition
dependency reasoning
tool selection
result interpretation
replanning
goal tracking
structured JSON output
```

Delegated areas include:

```text
arithmetic execution
code generation/review
OCR
image understanding
knowledge-base retrieval
file operations
code execution
external information access
```

However, the current abilities file contains old assumptions such as a generic document specialist.

Do not freeze or rewrite it yet.

Future update will align it with the actual final tool surface.

---

# 33. Tentative Tool Surface — Context Only, NOT FINAL CONTRACT

For implementation scaffolding only, the current likely capability groups are:

```text
document database
├── semantic RAG
├── exact search
├── artifact fetch
└── artifact listing

vision / OCR
└── analyze selected image artifact(s)

code specialist
└── solve coding task

math
└── calculate
```

Do not make external prompt contracts irreversible around these names.

Internally, adapters may be created now.

---

# 34. Error Handling Requirements

Every integration boundary should fail explicitly.

Do not silently return empty success.

Examples:

## DB errors

```text
document not found
artifact not found
invalid doc ID
invalid element ID
Chroma unavailable
embedding model unavailable
```

## Vision errors

```text
image artifact missing
invalid image type
model failed to load
model server health timeout
inference HTTP error
```

## Coder errors

```text
model failure
code extraction failure
sandbox timeout
sandbox runtime error
```

## Dispatcher errors

```text
unknown tool
unknown function
invalid arguments
tool exception
```

Error objects must preserve enough information for Gemma to replan later.

Do not expose secrets or arbitrary full stack traces to the final user.

Keep full diagnostics in logs.

---

# 35. Logging / Trace Requirements

Preserve or improve the existing trace drawer backend support.

Each tool execution should have:

```text
call_id
tool/function identifier
start time
end time
duration
status
sanitized arguments
result summary
error if any
```

Model events should record:

```text
model requested
model loaded
health status
inference start/end
token/timing metrics when available
```

Do not store giant binary images or full document blobs in logs.

---

# 36. Configuration Consolidation

Avoid scattered constants.

Create/maintain one clear config layer for:

```text
artifact path
Chroma path
embedding model
embedding device
embedding batch size
model paths
mmproj path
llama-server executable
port
context size
GPU layers
model startup timeout
tool timeout
mock mode
Docker settings
```

Environment-variable overrides are preferred where practical.

Do not hardcode friend-specific absolute paths into source.

---

# 37. Dependency Hygiene

Unify requirements carefully.

Current likely dependencies include:

```text
fastapi
uvicorn
requests/httpx
docling
chromadb
sentence-transformers
pillow
torch
pydantic
python-multipart
python-docx / pymupdf only if still required elsewhere
```

Rules:

1. do not remove a dependency until confirmed unused
2. mark legacy dependencies only used by deprecated path
3. CUDA-enabled PyTorch installation may differ on GPU machine
4. do not pin a CUDA wheel that breaks CPU development unless deployment setup explicitly requires it

---

# 38. Backward Compatibility During Merge

While restructuring:

- preserve old API routes where possible
- preserve UI contract
- preserve old agent JSON parser until final schema arrives
- preserve model manager commands
- preserve coder sandbox behavior
- isolate deprecations behind adapters

Goal:

```text
merge first
protocol redesign second
```

Do not combine both into one risky rewrite.

---

# 39. Stop Conditions

The IDE must STOP and report rather than guessing if:

1. new DB facade differs materially from this document
2. old runtime cannot be located
3. model manager behavior differs from assumed sequential llama.cpp architecture
4. moving files would break unresolved imports extensively
5. DB lacks a safe way to enumerate artifacts
6. derived vision storage semantics are incompatible with preserving attempts
7. upload endpoint contract is significantly different
8. a final tool/agent JSON schema is required to proceed

When stopped, report:

```text
what was expected
what was found
exact files involved
recommended smallest next decision
```

Do not invent architecture.

---

# 40. Required Implementation Order

Follow this sequence.

## Phase 1 — Inspection / baseline

- inspect both code trees
- identify overlapping files
- run existing tests
- document baseline
- create merge notes

## Phase 2 — Import unification

- make new DB importable from runtime
- instantiate shared DB service
- resolve paths/config

## Phase 3 — Upload integration

- route uploads into `ingest_document`
- return/register `doc_id`
- stop treating raw path as primary identity

## Phase 4 — Disable legacy document pipeline

- remove active calls to old `document_processor`
- preserve file as deprecated if needed

## Phase 5 — Generic dispatcher skeleton

- registry
- function resolution
- structured error handling
- trace/timing
- mock injection

## Phase 6 — DB adapters

- semantic retrieval wrapper
- exact retrieval wrapper
- artifact fetch wrapper
- artifact listing wrapper

## Phase 7 — Runtime state

- run/document/tool records
- serialization
- trace

## Phase 8 — Math utility

- safe deterministic evaluator
- tests

## Phase 9 — Vision adapter scaffold

- artifact resolution
- Qwen call boundary
- mock mode
- raw-result persistence hook

## Phase 10 — Coder adapter

- connect existing coder pipeline to dispatcher
- mock mode
- preserve sandbox

## Phase 11 — Derived vision persistence refinement

- preserve raw attempts
- prepare acceptance/indexing separation
- do not contaminate canonical artifacts

## Phase 12 — Embedding device auto mode

- CUDA when available
- CPU fallback
- log selection

## Phase 13 — Local integration tests

- run entire plumbing with mocks
- verify no legacy document path
- verify DB remains source of truth

## Phase 14 — Wait for final protocol files

At this point stop before irreversible prompt/protocol integration and wait for finalized:

```text
tools.json
abilities.json
agent_system.md
Gemma JSON schemas
tool-result schemas
retry/acceptance rules
```

---

# 41. Definition of "Done" for This Merge Stage

This pre-protocol merge is complete when all of the following are true:

```text
[ ] There is one effective SAGE runtime.

[ ] The new document DB is the only active document-ingestion/retrieval source.

[ ] Uploads produce stable doc IDs.

[ ] The old document parser is disconnected.

[ ] The old model manager still works.

[ ] The old coder pipeline still works through an adapter.

[ ] A generic dispatcher exists.

[ ] DB capabilities are callable through wrappers.

[ ] Artifact enumeration exists.

[ ] Vision has a wrapper and mock mode.

[ ] Derived vision output can be preserved without changing canonical image data.

[ ] Math capability exists.

[ ] Runtime execution state is tracked in Python.

[ ] Embedding device can auto-select CUDA/CPU.

[ ] Local tests can run without an RTX 4060.

[ ] No final Gemma schema was invented by the IDE.

[ ] No final tools.json was invented by the IDE.

[ ] No final abilities.json was invented by the IDE.

[ ] No final system prompt was invented by the IDE.
```

---

# 42. Final Architectural Reminder

Do not reinterpret this project as a conventional RAG chatbot.

SAGE is an **agent-controlled local multi-capability system**.

The intended reasoning loop is:

```text
USER
 ↓
GEMMA 4B
 ↓
decide what capability is needed
 ↓
BACKEND EXECUTES EXACT AUTHORIZED CAPABILITY
 ↓
RESULT RETURNS TO GEMMA
 ↓
Gemma interprets
 ↓
more capability calls OR final answer
```

Document storage is not the brain.

Qwen-VL is not the brain.

Qwen-Coder is not the brain.

Chroma is not the brain.

The dispatcher is not the brain.

The backend is not the brain.

**Gemma 4B is the sole semantic controller.**

Everything implemented in this merge must preserve that property.

---

# 43. Immediate IDE Action

Start with **Phase 1 only**.

After inspecting both folders:

1. report the actual discovered tree
2. identify which files correspond to the old runtime and new DB
3. identify import/path conflicts
4. identify the current app entrypoint
5. identify every active reference to the legacy `document_processor`
6. identify the current upload flow
7. identify how `SageDocumentDB` is instantiated
8. identify the current model-manager call sites
9. identify the current coder call sites
10. propose the exact minimal set of file moves/import edits needed

Then proceed to Phase 2 onward only if those findings match the architecture above.

Do not ask for final `tools.json` or agent schema during these early phases; those are intentionally being finalized separately.
