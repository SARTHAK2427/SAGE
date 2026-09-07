# SAGE SIH 2026 — Point 6 Implementation Handoff
## Socket → Plug Mappers + Model-Facing Contracts

## Current Repository State

The deterministic/internal socket layer is already hardened and should be treated as **FROZEN**.

The five model-facing files have already been replaced with the finalized versions and are now located together inside:

```text
prompts/
├── agent_system.txt
├── coder_system.txt
├── vision_system.txt
├── tools.json
└── abilities.json
```

Do not recreate old prompt files or assume `tools.json` / `abilities.json` are at repository root.

Before modifying anything, inspect the current repository and the actual deterministic socket producers/schemas. Do not guess field names when the real producer or schema can be read.

---

# 1. Objective

Implement **Point 6** of the SAGE architecture:

```text
DETERMINISTIC / INTERNAL PRODUCER
        ↓
RICH INTERNAL SOCKET
        ↓
PYTHON MAPPER
        ↓
MINIMAL CONSUMER INPUT PLUG
```

The deterministic sockets intentionally preserve maximum useful information.

The model-facing plugs should do the opposite:

> Send only the minimum information required for the receiving model to make the next correct semantic decision.

This task includes:

1. rich deterministic socket → Gemma-facing tool-result mappers;
2. Gemma tool-result packet construction;
3. Gemma-facing error mapping;
4. Gemma tool-result return contracts in `prompts/tools.json`;
5. a compact Gemma input-plug/output-socket description in `prompts/agent_system.txt`;
6. Gemma coder-call arguments → Qwen-Coder input mapper;
7. Gemma vision-call arguments → Qwen-VL input mapper;
8. tests for all mappings and the complete tool-result loop.

This task does **NOT** redesign the models, deterministic sockets, tool inventory, orchestration philosophy, or prompts.

---

# 2. Hard Architecture Rules

## Gemma is the sole semantic controller

No registered tool may autonomously invoke another registered tool.

Examples that remain forbidden:

```text
RAG → Vision
Vision → Coder
Coder → DB
DB → specialist
```

Python may perform deterministic internal mechanics already authorized by Gemma, such as:

```text
doc_id/image_id → canonical path resolution
argument validation
manifest lookup
model loading
Docker execution
persistence
timing/logging
```

## Multi-call batches remain supported

Gemma may emit:

```text
1 tool call
5 tool calls
25 tool calls
or more
```

in a single loop.

Python executes calls sequentially in the exact listed order.

Do not change this to a one-tool-per-turn design.

## Multiple Gemma loops remain supported

```text
Gemma
→ batch of calls
→ results
→ Gemma
→ another batch
→ results
→ ...
→ final
```

If a later call requires a value discoverable only from an earlier result in the same batch, Gemma must wait for the next loop rather than inventing the value.

---

# 3. Tool Inventory — DO NOT CHANGE

Exactly 4 groups / 7 callable functions:

```text
document_database
├── rag_search
├── exact_search
├── artifact_fetch
└── list_artifacts

vision_ocr
└── analyze_image

code_specialist
└── solve_code_task

math
└── calculate
```

Do not add:

```text
filesystem
Docling
embedding
Chroma
ingestion
generic document analyzer
large_document_analysis
reranker
```

The future small RAG reranker/selector model is **not part of this task**.

---

# 4. Gemma-Facing Tool Result Outer Contract

Every executed tool call should be mapped into the same small outer shell.

Success:

```json
{
  "call_index": 0,
  "tool": "document_database",
  "function": "rag_search",
  "status": "success",
  "result": {}
}
```

Optional `metadata` is allowed only when it can affect Gemma's next semantic decision.

Error:

```json
{
  "call_index": 0,
  "tool": "vision_ocr",
  "function": "analyze_image",
  "status": "error",
  "error": {
    "code": "IMAGE_NOT_FOUND",
    "message": "The requested image could not be resolved.",
    "retryable": false
  }
}
```

The complete packet fed back into Gemma is:

```json
{
  "type": "tool_results",
  "results": [
    {
      "call_index": 0,
      "tool": "...",
      "function": "...",
      "status": "success",
      "result": {}
    }
  ]
}
```

## Important simplicity rule

Prefer `result` to remain an object for every successful tool.

Do not make Gemma handle a mixture of:

```text
raw number
raw string
raw array
sometimes object
```

when a tiny object wrapper gives a more stable contract.

## Do NOT send Gemma

Unless a concrete semantic need exists, strip:

```text
local filesystem paths
temporary paths
PIDs
ports
process IDs
Docker container IDs
tracebacks
raw HTTP payloads
raw Chroma payloads
embedding vectors
embedding telemetry
stage timings
model throughput telemetry
hashes/checksums
storage implementation details
sandbox security telemetry
repair prompts/history
internal cache paths
```

Those remain preserved in the rich deterministic sockets.

---

# 5. Build a Dedicated Mapper Layer

Prefer a small explicit mapper package, for example:

```text
core/
└── mappers/
    ├── __init__.py
    ├── gemma_results.py
    ├── coder_input.py
    └── vision_input.py
```

Exact filenames may be adapted to the repository if there is already a cleaner equivalent location, but do not create an unnecessary framework.

Recommended high-level functions:

```python
map_rag_result(...)
map_exact_result(...)
map_artifact_fetch_result(...)
map_list_artifacts_result(...)
map_vision_result(...)
map_coder_result(...)
map_math_result(...)

map_tool_result_for_gemma(tool, function, rich_socket, ...)
build_tool_results_packet(...)

build_coder_input(...)
build_vision_input(...)
```

Use straightforward Python.

Do not introduce LangChain, LangGraph, event buses, plugin frameworks, or unnecessary abstraction hierarchies.

---

# 6. Mapper: `document_database.rag_search`

Inspect the actual frozen `sage.rag_search` socket before implementing.

## Critical requirement

RAG returns **multiple ranked matches**.

For the current MVP:

> Forward ALL returned matches to Gemma.

Do not select only the top one.

Do not summarize them into one merged answer.

Do not add a reranker.

Gemma-facing success result should be approximately:

```json
{
  "matches": [
    {
      "text": "retrieved text",
      "doc_id": "doc_x",
      "element_id": "text_42",
      "element_type": "text",
      "page": 7,
      "slide": null,
      "sheet": null,
      "raw_distance": 0.21,
      "cosine_similarity": 0.79,
      "image_refs": ["img_12"],
      "source": "canonical"
    },
    {
      "...": "second ranked match"
    }
  ]
}
```

Only include page/slide/sheet when actually available.

Preserve references Gemma may need for later calls:

```text
doc_id
element_id
element_type
image_refs
source/canonical-vs-derived distinction
```

Useful optional metadata:

```json
{
  "returned": 5,
  "top_k": 5
}
```

If the rich socket exposes both the authoritative:

```text
raw_distance
derived_cosine_similarity
```

and a legacy clamped similarity field, new mapper code should prefer the authoritative unclipped cosine similarity.

Do not forward the legacy compatibility score unless required by existing runtime compatibility.

Strip:

```text
embedding vectors
embedding model/device telemetry
raw Chroma response
collection internals
local paths
storage paths
timings
hashes
debug data
```

---

# 7. Mapper: `document_database.exact_search`

Inspect the actual frozen exact-search socket.

Gemma-facing result:

```json
{
  "matches": [
    {
      "doc_id": "doc_x",
      "element_id": "text_17",
      "element_type": "text",
      "matched_text": "AUTH_V2",
      "context": "small useful surrounding context",
      "page": 4,
      "slide": null,
      "sheet": null,
      "start_offset": 132,
      "end_offset": 139
    }
  ]
}
```

For table matches, preserve row/column/cell coordinates if the frozen socket actually exposes them.

For code/text matches, preserve exact useful context and provenance.

Important:

> `exact_search` remains canonical-source-only for the MVP.

Do not silently search or mix `derived/vision` OCR text into canonical exact-search results.

Strip:

```text
local paths
manifest internals
scan timings
debug tracebacks
filesystem metadata
```

---

# 8. Mapper: `document_database.artifact_fetch`

Inspect the actual `artifact_fetch` variants in the frozen schema.

Return one `artifact` object.

## Text

```json
{
  "artifact": {
    "doc_id": "doc_x",
    "element_id": "text_42",
    "type": "text",
    "text": "exact canonical text",
    "page": 7
  }
}
```

## Table

```json
{
  "artifact": {
    "doc_id": "doc_x",
    "element_id": "table_3",
    "type": "table",
    "rows": [],
    "page": 8
  }
}
```

Preserve exact table structure already available in the canonical artifact.

## Code

```json
{
  "artifact": {
    "doc_id": "doc_x",
    "element_id": "code_2",
    "type": "code",
    "code": "exact source code",
    "language": "python",
    "page": 5
  }
}
```

## Image

```json
{
  "artifact": {
    "doc_id": "doc_x",
    "element_id": "img_12",
    "type": "image",
    "image_id": "img_12",
    "image_ref": "stable registered image reference",
    "page": 6,
    "caption": "optional caption"
  }
}
```

Gemma must receive a stable registered image/artifact reference.

Gemma must **not** receive the local filesystem path.

The deterministic runtime may resolve that reference later when Gemma explicitly calls Vision.

Strip:

```text
local path
hashes
file timestamps
storage implementation data
raw manifest structure
debug telemetry
```

---

# 9. Mapper: `document_database.list_artifacts`

This result may become large, so keep each entry lightweight.

Gemma-facing result:

```json
{
  "artifacts": [
    {
      "element_id": "img_12",
      "type": "image",
      "order": 14,
      "page": 6,
      "slide": null,
      "sheet": null,
      "caption": "optional",
      "heading": "optional",
      "derived_available": true
    }
  ]
}
```

Preserve only fields useful for deciding what artifact to inspect next.

This tool is especially useful when:
- RAG does not surface every image/table;
- a standalone image has little/no searchable text;
- Gemma wants to enumerate a document's visual/table/code artifacts.

Do not include binary data.

Normally strip:

```text
local paths
byte sizes
checksums
manifest implementation details
storage existence/debug fields
bbox
```

Only retain bbox if the actual downstream semantic selection genuinely needs it.

---

# 10. Mapper: `vision_ocr.analyze_image`

Qwen-VL itself returns plain task-focused text.

The rich runtime/persistence socket may contain much more.

Gemma-facing success result should be small:

```json
{
  "doc_id": "doc_x",
  "image_id": "img_12",
  "analysis": "Qwen-VL OCR/visual answer"
}
```

If a stable derived-analysis ID is already useful for future references, it may be exposed:

```json
{
  "derived_analysis_id": "analysis_x"
}
```

but do not include it merely because it exists internally.

Strip:

```text
local image path
cache path
raw llama.cpp payload
HTTP telemetry
token timing
PID/port
full persistence metadata
transport diagnostics
```

Do not resend the original image to Gemma.

Gemma already reasons through the textual analysis and registered image identity.

---

# 11. Mapper: `code_specialist.solve_code_task`

Qwen-Coder itself returns raw code only.

The high-level code tool may then deterministically perform:

```text
extraction
sandbox execution
repair attempts
final selection
```

Gemma should receive only what is useful for deciding whether the coding task succeeded or what to do next.

Example success:

```json
{
  "code": "final selected code",
  "language": "python",
  "execution_status": "success",
  "stdout": "optional output",
  "stderr": "",
  "exit_code": 0,
  "attempts_used": 1
}
```

If execution was never performed:

```json
{
  "code": "generated code",
  "language": "python",
  "execution_status": "not_executed"
}
```

Do not fabricate:

```text
exit_code = 0
stdout = ""
```

if no process ever ran.

Example failed execution:

```json
{
  "code": "best/final code if available",
  "language": "python",
  "execution_status": "error",
  "stderr": "decision-useful error output",
  "exit_code": 1,
  "attempts_used": 3
}
```

Strip:

```text
raw Qwen response history
repair prompts
all intermediate code unless needed
Docker/container IDs
sandbox security telemetry
temp paths
hashes
PIDs
detailed timing
internal traceback
```

If the final code tool has a status distinction already defined in its frozen socket, map honestly instead of inventing a parallel status system.

---

# 12. Mapper: `math.calculate`

Keep this tiny.

Success:

```json
{
  "expression": "17*23",
  "value": 391
}
```

Do not send AST provenance, whitelisted operator lists, or evaluator internals.

Errors use the common Gemma-facing error shell.

---

# 13. Common Error Mapping

Every tool has rich deterministic error data.

Map it down to:

```json
{
  "code": "STABLE_ERROR_CODE",
  "message": "Short useful explanation",
  "retryable": false
}
```

Only add extra error information if Gemma genuinely needs it to choose a recovery action.

Do not send:

```text
traceback
source_file
source_function
internal exception object
local path
HTTP raw body
Docker diagnostic internals
```

The rich socket/logs retain those.

---

# 14. Qwen-Coder Input Mapper

Current model contract is frozen:

```text
Gemma JSON tool call
        ↓
Python mapper
        ↓
minimal plain-text Qwen input
```

Gemma tool arguments:

```json
{
  "instruction": "string",
  "code": "string or null",
  "language": "string or null"
}
```

Do not send this JSON wrapper directly to Qwen.

Construct:

## No existing code

```text
TASK:
<instruction>
```

## Target language present

```text
TASK:
<instruction>

TARGET LANGUAGE:
<language>
```

## Existing code present

```text
TASK:
<instruction>

TARGET LANGUAGE:
<language if present>

CODE:
<code exactly, indentation preserved>
```

Requirements:

- preserve code text and indentation exactly;
- do not escape/reformat the source unnecessarily;
- do not inject SAGE runtime metadata;
- do not inject tool names;
- do not inject JSON syntax;
- do not add conversation history unless explicitly required by the coding task.

Qwen-Coder's system prompt already requires raw code-only output.

Do not change that semantic output contract.

---

# 15. Qwen-VL Input Mapper

Gemma tool arguments:

```json
{
  "instruction": "string",
  "doc_id": "string or null",
  "image_id": "string",
  "image_ref": "string or null"
}
```

The mapper/runtime should:

1. validate the authorized reference;
2. deterministically resolve it to the canonical image using existing DB/artifact infrastructure;
3. provide that actual image through the existing multimodal model transport;
4. send only the semantic text:

```text
TASK:
<instruction>
```

Do not send Gemma's JSON wrapper to Qwen-VL.

Do not send unnecessary:

```text
doc manifest
Chroma results
local storage explanation
tool registry info
SAGE architecture
```

The local path may be used internally by Python/model transport but must never become part of the Gemma-facing result contract.

Qwen-VL's output remains plain task-focused text.

Do not force JSON output on Qwen-VL.

---

# 16. Update `prompts/tools.json`

The current tool definitions and arguments are already approved.

Do not alter tool names or invent arguments merely for convenience.

Add **compact return/result descriptions** so Gemma understands what each function will give back.

The exact representation can be concise, for example:

```json
{
  "name": "rag_search",
  "description": "...",
  "arguments": {...},
  "returns": {
    "matches": "array of ranked document matches with text, provenance, scores and artifact refs"
  }
}
```

Or a slightly more structured form if still concise.

The purpose is NOT to reproduce the deterministic JSON Schemas.

The purpose is to teach Gemma:

```text
what kind of result comes back
which IDs/references may be reused in later calls
which functions return multiple items
```

Important points `tools.json` must communicate:

### RAG
Returns multiple ranked matches and ALL are currently supplied.

### Exact Search
Returns one or more literal/regex matches with exact provenance.

### Artifact Fetch
Returns the requested canonical text/table/code/image artifact.

### List Artifacts
Returns an artifact inventory.

### Vision
Returns task-focused textual analysis for the referenced image.

### Coder
Returns final code plus decision-useful execution outcome when available.

### Math
Returns expression + numeric value.

Keep return descriptions compact to save Gemma context tokens.

---

# 17. Update `prompts/agent_system.txt`

Do not rewrite the prompt from scratch.

Keep the current concise system prompt and add only what is necessary to explicitly explain Gemma's own **input plug** and **output socket**.

Gemma should clearly understand:

## Input plug

Initial turn conceptually contains:

```text
user request
+ available registered document/context references
+ tools.json
+ abilities.json
```

Subsequent loops may include:

```json
{
  "type": "tool_results",
  "results": [...]
}
```

Tool-result objects follow the compact result contracts described in `tools.json`.

Gemma should treat those results as evidence for the next decision.

## Output socket

Exactly:

```json
{
  "type": "tool_calls",
  "calls": [...]
}
```

or:

```json
{
  "type": "final",
  "answer": "..."
}
```

Keep the existing no-chain-of-thought rule.

Do not expose deterministic rich-socket vocabulary to Gemma.

Gemma does not need to know that internal objects are called:

```text
sage.rag_search
sage.chroma_query
sage.model_transport
```

It only needs the registered tool interface and its compact returned evidence.

---

# 18. `prompts/abilities.json`

Do not redesign this file.

Only update it if required to remain exactly consistent with `tools.json` after adding return descriptions.

There should still be exactly the approved delegation boundaries.

No new capabilities.

---

# 19. Integrate the Mapping Layer Into the Orchestrator

Inspect the current orchestrator/dispatcher flow.

The desired flow is:

```text
Gemma output
    ↓
JSON parse/validation
    ↓
dispatcher executes each call in listed order
    ↓
each tool returns its rich internal result/socket
    ↓
tool-specific mapper
    ↓
compact Gemma-facing result object
    ↓
append to results[]
    ↓
build {"type":"tool_results","results":[...]}
    ↓
feed to Gemma on next loop
```

Do not bypass the rich deterministic socket layer.

Do not make individual tools manually craft Gemma prompts if a centralized mapper is more appropriate.

Do not let Gemma see raw deterministic sockets.

Preserve the current mock/live architecture.

---

# 20. Compatibility

Preserve:

- existing public deterministic APIs;
- frozen deterministic socket schemas;
- current dispatcher behavior;
- current mock mode;
- current coder Docker pipeline;
- current vision wrapper/model manager;
- current sequential model switching;
- existing upload/document DB flow;
- current multi-tool orchestration;
- existing JSON repair behavior unless a mapper integration requires a tiny compatible hook.

Do not delete rich fields just because Gemma no longer receives them.

The mapper is a projection, not a mutation of source data.

---

# 21. Tests Required

Add unit tests for every mapper.

At minimum:

## RAG

- several matches in → same number of matches out;
- ranking/order preserved;
- all matches forwarded;
- useful provenance/refs preserved;
- raw Chroma/embedding/internal telemetry absent;
- unclipped cosine similarity preferred where available.

## Exact Search

- multiple matches preserved;
- exact matched text preserved;
- table coordinates preserved when present;
- internal paths/debug data removed.

## Artifact Fetch

Test text/image/table/code separately.

For image:
- stable image ref present;
- local filesystem path absent.

## List Artifacts

- multiple artifacts preserved;
- lightweight identity/provenance preserved;
- heavy/internal fields removed.

## Vision

- textual analysis preserved;
- image identity preserved;
- local path/transport telemetry removed.

## Coder

Test:
- success;
- failed execution;
- not executed;
- nullable exit code;
- final code preserved;
- repair history/Docker internals removed.

## Math

- expression/value preserved;
- AST/debug internals removed.

## Error mapping

- stable code/message/retryable preserved;
- traceback/internal debug removed.

## Coder input

- TASK formatting correct;
- target language optional;
- code indentation preserved exactly;
- no JSON wrapper injected.

## Vision input

- correct image is deterministically resolved;
- textual input is only TASK content;
- local path remains internal;
- no Gemma JSON wrapper reaches Qwen-VL.

## Packet

Given N executed calls:

```text
N mapped result objects
```

must appear in:

```json
{
  "type": "tool_results",
  "results": [...]
}
```

in the exact original call order.

## Multi-call integration

Test at least one loop containing 5+ heterogeneous tool calls.

## Multi-loop integration

Test:

```text
Gemma → tools → results → Gemma → tools → results → Gemma → final
```

using mock models where necessary.

## JSON serializability

Every Gemma-facing result and packet must serialize cleanly with standard `json.dumps`.

## Leakage test

Explicitly verify representative internal-only fields do NOT appear in Gemma-facing packets:

```text
local_path
traceback
pid
port
raw_chroma_responses
embedding vector
Docker/container identifiers
cache file path
```

---

# 22. Prompt / Contract Tests

Add lightweight tests that verify:

1. `prompts/tools.json` parses as JSON.
2. `prompts/abilities.json` parses as JSON.
3. Exactly 7 callable functions are present.
4. Every function has a compact return contract.
5. `agent_system.txt` documents:
   - `tools.json` as authoritative;
   - `abilities.json` as authoritative;
   - `tool_results` input;
   - `tool_calls` output;
   - `final` output;
   - multi-call batches.
6. Coder prompt still requires raw code only.
7. Vision prompt still requires plain task-focused text.
8. No unapproved tool was added.

---

# 23. Do Not Implement Yet

Do NOT implement:

- future RAG reranker/selector model;
- memory system redesign;
- autonomous background agents;
- arbitrary filesystem access;
- new tools;
- model-to-model autonomous calls;
- UI redesign;
- a new orchestration framework;
- new deterministic sockets;
- semantic JSON output for Qwen-Coder;
- semantic JSON output for Qwen-VL.

---

# 24. Stop Condition

When Point 6 is complete:

```text
Rich deterministic sockets       ✅ frozen
        ↓
Python socket→plug mappers        ✅ implemented
        ↓
Gemma compact tool results        ✅ implemented
        ↓
Coder minimal input plug          ✅ implemented
Vision minimal input plug         ✅ implemented
        ↓
Gemma / Coder / Vision contracts  ✅ synchronized
```

STOP after this.

Do not proceed into unrelated new architecture work.

---

# 25. Final Report Required

Return a concise implementation report containing:

1. files created;
2. files modified;
3. mapper functions added;
4. exact Gemma-facing result shape for each of the 7 functions;
5. fields deliberately stripped for each tool;
6. coder input mapping final behavior;
7. vision input mapping final behavior;
8. changes made to `prompts/tools.json`;
9. changes made to `prompts/agent_system.txt`;
10. whether `prompts/abilities.json` changed and why;
11. orchestrator integration points;
12. full test count/result;
13. multi-call test result;
14. multi-loop test result;
15. any remaining place where a raw rich socket reaches Gemma;
16. any ambiguity requiring a design decision.

If a raw deterministic socket is still being passed directly to Gemma anywhere, explicitly flag it rather than hiding it.

The desired endpoint is:

> Every deterministic producer remains rich internally, while every model receives only a deliberate minimal plug.
