# SAGE — Current Architecture Reference

> **Last updated:** 28 September 2026
>
> This is the local handoff for the current SAGE implementation. It supersedes
> earlier descriptions of a single, hot-swapped model agent. The legacy agent
> remains in the repository, but the browser UI's default chat path is now
> **SAGE Flash**.

---

## 1. System purpose and active modes

SAGE is a local-first FastAPI application with a browser UI, durable chat
ledger, scoped memory, document RAG, optional image inspection, and a
host-agnostic inference runtime.

### Active browser mode: Flash

Flash is the shipped/default chat path (`POST /api/flash`). It is designed to
work with either:

- a local RTX 4060-class machine (Gemma and Qwen on GPU; memory curator on
  CPU/RAM), or
- remote OpenAI-compatible GPU bridges such as a Kaggle/cloudflared bridge,
  private server, or another machine.

The browser never needs to know whether inference is local or remote. Runtime
bindings, URLs, API keys, and model IDs are set in the Flash Runtime UI and
exist only for the current SAGE server session. They are never written to
`.env`, the database, or browser persistence.

### Placeholder browser mode: Reasoning

The composer exposes a Reasoning switch as a deliberate "coming soon" surface.
It is not wired as a second inference graph. Gemma Flash itself currently has
its catalog reasoning mode enabled; this is a runtime generation setting, not
the separate future UI mode.

### Legacy mode

`/api/chat`, `orchestrator.py`, the dispatcher, and the old sequential
model-manager/tool loop are retained for compatibility. They are not the
architecture to extend for Flash features unless explicitly required.

---

## 2. High-level Flash flow

```
Browser composer
  ├─ creates a browser-side observer run id
  ├─ opens SSE: /api/observer/stream/{run_id}
  └─ POST /api/flash (prompt, attachments, chat id, run id)
          |
          v
FastAPI intake (app.py)
  ├─ validates/stages uploads
  ├─ ingests non-direct documents into the document database
  └─ runs FlashService in a worker thread
          |
          v
FlashService (flash/service.py)
  ├─ reads recent turns, current-chat cold memory, and global personal memory
  ├─ loads the current chat's attachments and resolves attachment proximity
  ├─ sends fixed system/abilities, callable actions, JSON context, then latest message
  ├─ Gemma returns a final response or one typed action request
  ├─ action results/errors return in the next JSON context packet
  ├─ Qwen inspects chat images or a selected embedded document image
  ├─ commits the completed turn to the canonical SQLite ledger
  └─ queues memory curation/indexing asynchronously
          |
          v
ObserverStore → SSE → compact progress panel + full Execution Inspector
```

### Flash action protocol

The model sees no A/B/C/D labels. It returns either:

- `final=true` with the user-facing response, or
- `final=false` with one typed action request.

Available actions are `chat_ledger.search`, `document.search`,
`document.image.inspect`, and `vision.inspect`. Global personal memory,
current-chat cold memory, and recent conversation are already supplied to
Gemma; the active Flash graph has no global/cold memory-search action.

The system message contains only action contracts callable on that turn. The
JSON context repeats availability and allowed attachment IDs. This avoids
advertising an unavailable action and relying on a small controller model to
resolve contradictory instructions.

Unavailable, invalid, repeated, or failed actions return structured results to
Gemma for recovery. They do not become user-visible Python errors. Document
search is always constrained to document IDs registered to the current chat.

---

## 3. Inference runtime

### Roles

| Role | Default responsibility | Binding options |
|---|---|---|
| `gemma` | routing, direct answers, synthesis; catalog reasoning enabled | local GPU or remote bridge |
| `qwen` | same-chat image and embedded document-image inspection | local GPU or remote bridge |
| `memory` | 2B memory curation/compression | local CPU, local GPU, remote bridge, or disabled |

`flash/runtime_config.py` owns ephemeral role bindings. `flash/transport.py`
provides connection pooling, API-key headers, retries, model-ID probing, and
OpenAI-compatible `/v1/chat/completions` calls. It forces IPv4 for tunnel
reliability and keeps TLS clients alive per origin.

`flash/local_manager.py` launches/reconciles local llama-server processes when
a role is bound locally. Remote bridge deployment is supported through the
runtime UI, but SAGE does not assume a specific provider or tunnel.

### Runtime diagnostics behaviour

Applying a remote runtime is normally quick: configure bindings, then warm
connections in a daemon thread. “Test connection” is deliberately heavier: it
probes **Gemma, Qwen, then Memory sequentially**, querying `/v1/models` and
making a tiny completion for each enabled role. Slow or unavailable tunnels
therefore accumulate delays. This is a known UX/performance limitation, not
evidence that the remote models are running on the user's laptop.

---

## 4. Context and memory currently implemented

### Canonical ledger

`sage_memory.py` stores every persisted turn in SQLite. It is the literal,
chat-scoped conversation record and carries sequence/time metadata. It is not
discarded merely because a turn leaves recent context.

### Hot/recent context

`memory_system/coordinator.py` reads recent turns subject to both:

- a turn cap (`SAGE_RECENT_CHAT_MAX_TURNS`, currently 5), and
- a token budget (`SAGE_RECENT_CHAT_BUDGET_TOKENS`, currently 2400).

The budget prevents five extremely large pairs from consuming the whole model
context. Flash normally gives this recent context to Gemma directly.

### Durable personal memory (global)

Global memory is deliberately narrow. It stores only stable **personal
information** such as a name or age, scoped to the user rather than a chat.
It is injected as authoritative known user information into Gemma's system
context for every chat, without needing a BGE search. Identity normalization
and deduplication live in `memory_system/global_identity.py`; old values can
be superseded and duplicates are soft-deleted.

Do not use global memory for preferences, generic facts, project details, or
chat events. Those are chat-scoped memory.

### Cold/chat memory and curation

`flash/memory_worker.py` performs 2B curation in background jobs after a
conversation commit. It can create compressed chat memories and extract
eligible personal facts. `memory_system/job_store.py` persists job state;
`memory_system/index_outbox.py` ensures vector-index operations can recover
after restart.

The completed user reply is not blocked on curation. The observer records
queued, started, model-input, model-output, and stored/failed phases.

### Retrieval

Global personal memory, bounded current-chat cold memory, and bounded recent
conversation are pre-fed in the JSON context on every Flash turn. Gemma does
not call a memory search tool for them. `chat_ledger.search` remains available
only for explicit older-history requests that cannot be answered by the recent
window, such as a date, an earlier-message count, or an old topic. Ordinary
greetings and general questions do not run BGE retrieval.

Memory indexing is decoupled from the response path.
`memory_system/index_worker.py` processes the durable index outbox in a
separate low-priority process, preventing a cold BGE load from delaying the
next chat request.

### Planned, not yet implemented

`notes/New_memory system.md` contains the next-generation proposal: bounded
global/cold buckets, Smart Memory Compression at capacity, ten-or-token-budget
hot context, a complete SQL message ledger, cross-chat RAG with provenance,
and artifact-aware retrieval. It is design work, not current runtime
behaviour. Do not describe it as implemented.

---

## 5. Document RAG and attachments

### Data model

`sage_document_db/` is the document subsystem. Its canonical artifacts and
rebuildable vector index live under one generated data root:

```
data/
├─ attachments/            # durable chat attachment instances
├─ documents/
│  ├─ artifacts/          # source-derived canonical document artifacts
│  └─ chroma_bge_m3/      # disposable document vector index
└─ memory/
   ├─ sage_memory.db      # conversation ledger + memory jobs
   └─ chroma/             # memory vector index
```

`SAGE_DATA_ROOT` changes this root; granular path overrides remain available.
Deleting `data/` is a hard local reset. It is safe from a code-import
perspective: directories are recreated on demand, but all local chats,
memories, ingested source artifacts, and indexes are lost.

### Document pipeline

1. App intake stores an upload in a temporary request directory.
2. Text-style files can be used directly; other documents are ingested.
3. Docling is optional; `sage_document_db/pdf_fallback.py` uses PyMuPDF to
   recover PDF text/tables when Docling is unavailable/fails.
4. `Chunker` builds structure-aware chunks.
5. BGE-M3 embeds and Chroma stores 1024-dimensional vectors.
6. The chat attachment registry supplies an allow-list of document IDs.
7. Gemma may request `document.search`; the action calls
   `document_db.rag_search_socket` within that allow-list.
8. Retrieved records return in the next JSON context packet with provenance.
9. Gemma may request `document.image.inspect` with one document attachment ID
   and a one-based image number. SAGE enumerates the document's canonical image
   artifacts in reading order and sends only that child image to Qwen, with
   its page, caption, element ID, and image-count metadata.

Each upload has a durable attachment instance linked to chat, message, turn,
timestamp, media type, processing status, and content/artifact ID. Stored
vision evidence supports same-chat follow-ups. Any still-readable image owned
by that same chat may be inspected again by Qwen when stored evidence is too
shallow for a later request. Cross-chat attachments never enter the context or
action allow-list.

### Attachment reference resolution

Attachments are represented as roots rather than one flat pool:

- uploaded images are `chat_image` roots;
- uploaded documents are `document` roots;
- extracted document images are children of their owning document and are
  addressed only through `document.image.inspect`.

For unnamed references such as “this”, “it”, or “what is up with this”, SAGE
computes `attachment_reference` before invoking Gemma. A unique attachment on
the current message wins; otherwise the unique attachment from the nearest
prior attachment-bearing turn wins. Attachments tied to the same relevant
message remain explicitly ambiguous and Gemma may ask the user to choose.
Separately uploaded screenshots are never counted as images inside a PDF.

### Embeddings and performance

The canonical document embedding model is `BAAI/bge-m3`, CPU by default
(`EMBEDDING_DEVICE=cpu`) so an 8 GB GPU remains free for Flash. This is
substantially heavier than MiniLM. First load/download, document ingestion,
index recovery, and a fresh BGE search can saturate CPU on weaker laptops.
Ordinary warm text chat should not invoke document RAG by default.

The optional reranker is disabled by default (`SAGE_RERANKER_ENABLED=false`).

---

## 6. Observer and UI

`core/observer.py` is an in-memory, thread-safe per-run event ledger. It
sanitizes payloads and assigns ordered event sequences. FastAPI exposes:

- `GET /api/observer/runs`
- `GET /api/observer/runs/{run_id}` (snapshot/fallback)
- `GET /api/observer/stream/{run_id}` (SSE live stream)

`static/app.js` opens the SSE stream before it posts Flash. The normal right
panel presents a user-friendly, top-to-bottom flow; the expanded Execution
Inspector shows sanitized inputs, outputs, routing, timings, retrieval, and
background memory work. Tiles are driven by real backend events, not fake UI
timers.

Local context-packet assembly remains visible in the detailed Inspector but is
not shown as a separate compact progress tile. It normally takes only a few
milliseconds and is not a model or retrieval task.

If tiles never appear on another machine, investigate the browser's EventSource
request, stale UI/server versions, local proxy/antivirus interference, and any
pre-Flash document ingestion delay. SSE failure is distinct from remote model
inference latency.

---

## 7. Key files

```
app.py                         FastAPI routes, uploads, Flash endpoints, SSE
flash/service.py               Flash graph and assembled model context
flash/protocol.py              Three-message Gemma protocol and action parsing
flash/actions.py               Ledger, document, document-image, and vision actions
flash/transport.py             OpenAI-compatible local/remote transport
flash/runtime_config.py        Ephemeral runtime role/connection configuration
flash/local_manager.py         Local llama-server lifecycle
flash/memory_worker.py         Background 2B curation
memory_system/                 Ledger coordination, jobs, quality, identity, index outbox
memory_system/index_worker.py  Separate low-priority BGE/index-outbox process
sage_memory.py                 SQLite canonical conversation/memory store
sage_document_db/              Artifacts, BGE RAG, chunking, PDF fallback, reranking
core/observer.py               Execution event store
static/app.js                  Chat, Runtime UI, Observer UI
prompts/flash_system.txt       Gemma Flash prompt and routing contract
prompts/qwen_flash_system.txt  Qwen visual specialist prompt
tests/flash/                   Flash runtime/service/document tests
tests/test_memory_system_v2.py Memory-system integration coverage
```

Scripts previously scattered at repository root have been organized under
`scripts/`; the friend's former standalone memory project is retained under
`reference/standalone_memory/` for reference only.

---

## 8. Configuration essentials

See `.env.example` for the full documented list. The main values are:

| Variable | Purpose |
|---|---|
| `SAGE_DATA_ROOT` | root for all generated document/memory data |
| `SAGE_DOCUMENT_CHROMA_ROOT` | optional BGE document index override |
| `SAGE_MEMORY_CHROMA_ROOT` | optional memory index override |
| `SAGE_SQLITE_PATH` | optional SQLite ledger override |
| `SAGE_RAG_EMBEDDING_MODEL` | BGE model (default `BAAI/bge-m3`) |
| `EMBEDDING_DEVICE` | CPU by default; CUDA is explicit opt-in |
| `SAGE_RERANKER_ENABLED` | optional reranker, default false |
| `SAGE_MOCK_MODE` | mock local inference for tests/dev |
| `SAGE_MEMORY_CURATOR_*` | curation enablement and size limits |
| `LLAMA_SERVER_PATH`, `MODEL_DIR` | local model runtime paths |

Remote URLs and keys are session-only Runtime UI inputs. Do not add real
credentials to `.env.example`, prompts, test fixtures, or Git history.

---

## 9. Validation and repository hygiene

Focused verification for the current Flash runtime and attachment protocol:

```powershell
python -m pytest tests/flash/test_runtime_config.py tests/flash/test_transport.py tests/flash/test_attachment_resolution.py -q
```

Latest result: **15 passed**. `tests/flash/test_service.py` still contains
assertions for the retired Case A/B/C/D and semantic-memory-request protocol;
it must be rewritten before it can serve as a current full-Flash regression
suite.

`.gitignore` excludes `data/`, historical artifact/index locations, local
databases, model weights, `.env*` (except `.env.example`), logs, and local
notes. `notes/` is intended to remain local; files already tracked by Git must
be explicitly untracked or reverted if they must disappear from a future
commit.

---

## 10. Current known limitations / next work

1. Runtime diagnostics probe roles serially; parallel, timed diagnostics would
   be better for remote users.
2. BGE-M3 can temporarily use most CPU during cold initialization, document
   ingestion, or index recovery. CPU-thread limits and deferred recovery are
   desirable before broad distribution.
3. Observer SSE should gain a polling fallback for environments that block
   EventSource.
4. The implemented memory system is deliberately conservative. The richer
   bucket/SMC/ledger/artifact-aware design remains future work.
5. Document-image selection depends on successful parser extraction of
   canonical image artifacts. A missing/unreadable image returns a structured
   action error for Gemma to explain.

## 11. Core invariants

1. Gemma is the semantic controller and final text writer.
2. Gemma returns a final response or one named action; it never selects a case.
3. Qwen is only for readable same-chat images or a specifically selected
   embedded image belonging to a same-chat document.
4. Document text RAG and document-image inspection are separate grounded
   sources; an uploaded screenshot cannot substitute for an image in a PDF.
5. Canonical document artifacts and SQLite ledger are authoritative local data;
   Chroma indexes are rebuildable accelerators.
6. Runtime credentials are ephemeral and never persisted by SAGE.
7. Background memory work must never hold the user response hostage.
8. Observer telemetry is diagnostic only; it must not alter model decisions.
9. `SAGE_MOCK_MODE=1` must keep tests runnable without GPU models or Docker.
