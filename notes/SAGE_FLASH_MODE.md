# SAGE — Flash Mode Architecture

## 1. Purpose

**SAGE Flash Mode** is the lightweight, fast, self-reliant operating mode of SAGE.

It is designed for common workbench tasks such as:

- general conversation and reasoning
- summarisation
- light coding
- quick document understanding
- OCR / simple visual understanding
- mixed lightweight requests containing both text and media

Flash Mode keeps its **core interactive inference path entirely on the laptop**: Gemma and Qwen remain resident locally and can handle the normal lightweight workbench path without Kaggle or government GPU infrastructure. In the complete SAGE deployment, retrieval, RAG-processing, and memory-support services run on Kaggle as an auxiliary lane so they do not consume the laptop GPU budget or unnecessarily delay foreground generation.

The design goal is:

> **Keep two small models resident in VRAM at all times, let one model reason and route, invoke the second only when needed, and avoid unnecessary model switching or final re-processing.**

---

## 2. Target Hardware

Primary target:

- **GPU:** NVIDIA RTX 4060 Laptop GPU
- **Physical VRAM:** 8 GB
- **Flash Mode VRAM budget:** 8 GB

Flash Mode is designed so that both models remain loaded simultaneously.

### Estimated VRAM allocation

| Component | Assumed VRAM |
|---|---:|
| Gemma 4 E2B | ~3.0 GB |
| Qwen3-VL 2B | ~1.5 GB |
| Context / KV / runtime headroom | ~1.5 GB |
| **Estimated total** | **~6.0 GB** |

With ~4.5 GB assumed for the two resident models and ~1.5 GB reserved for context / KV / runtime usage, the working estimate is ~6.0 GB total, leaving roughly ~2 GB of headroom within the 8 GB RTX 4060 Laptop GPU. This avoids repeated model unload/reload latency.

---

## 3. Flash Mode Models

### Model 1 — Main Reasoning + Routing Model

**Unsloth Gemma 4 E2B QAT IQ4-NL GGUF**

- GGUF file size: ~2.6 GB
- Architecture assumption: ~3.0 GB runtime VRAM
- Role:
  - primary conversational model
  - general reasoning
  - summarisation
  - light coding
  - request decomposition
  - routing
  - deciding whether Qwen is required
  - deciding whether a Qwen result is final or must return to Gemma

Gemma is the **entry model for every Flash Mode request**.

There is no separate router model and no separate routing inference.

Gemma performs routing and any work it can already complete in the **same inference**.

**Gemma thinking mode is enabled by default in Flash Mode.** The purpose is to preserve stronger reasoning quality while still keeping the overall architecture lightweight by avoiding extra routing and final-rewrite model calls.

---

### Model 2 — Visual / Multimodal Worker

**Unsloth Qwen3-VL 2B Instruct IQ4-NL GGUF**

- GGUF file size: ~1.05 GB
- Architecture assumption: ~1.5 GB runtime VRAM
- Role:
  - image understanding
  - OCR-style reading
  - visual information extraction
  - lightweight multimodal responses
  - visual subtasks requested by Gemma

Qwen is invoked only when the request contains a visual requirement that Gemma should delegate.

---

## 4. Core Runtime Principle

Flash Mode uses only two active intelligence components:

```text
Gemma 4 E2B
    |
    | reasoning + routing
    |
    +------> Qwen3-VL 2B, when required
```

Both models remain loaded in the RTX 4060 Laptop GPU.

The main optimization is to avoid this old pattern:

```text
Model A
  -> switch
Model B
  -> switch
Model A
  -> switch
Final model
```

Flash Mode instead keeps both models resident and calls only the minimum sequence required by the task.

---

## 5. Data Path

User input may contain:

```text
message
+
zero or more files / images
```

### Simple files and images

Simple files and images can be placed directly into the active **shared memory pool** for the current request.

### Complex documents

Complex documents continue through the document-processing path:

```text
USER
  |
  +--> complex file --> Docling pipeline --> Shared Memory Pool
  |
  +--> simple file / image -------------> Shared Memory Pool
```

Gemma and Qwen can read from and write to the shared working state as required by the runtime.

The shared pool is also the temporary working area for **mid-generation state**. This may include short-lived cache entries, intermediate model results, retrieved evidence, temporary task memory, partial summaries, references to generated outputs, and other information that later stages of the same request may need. Useful information should not be thrown away merely because it was produced mid-run; the backend may later promote selected items into the appropriate persistent memory or object metadata layer.

Gemma may also call retrieval / memory tools when the current request depends on information that is not present in the current message or currently attached files. Examples include:

- retrieving older chat information
- RAG over previously ingested documents
- fetching stored memory
- retrieving previously generated or derived information
- resolving a user reference such as "the file I shared before" or "continue what we discussed earlier"

The exact memory implementation, persistence mechanism, promotion rules, and LangGraph state schema are intentionally left to the backend / memory implementation.

---

## 6. Auxiliary Retrieval and Memory Lane — Kaggle

The **interactive Flash inference pair remains local on the RTX 4060 Laptop GPU**:

```text
Gemma 4 E2B + Qwen3-VL 2B
```

All retrieval and memory-support work for the complete Flash deployment is assigned to the **Kaggle GPU backend only**. The auxiliary model(s) and services required for RAG chunk processing, memory summarisation, memory categorisation / fact extraction, and related retrieval work remain loaded on Kaggle rather than competing with the two local Flash models for laptop VRAM. The exact auxiliary model choice is intentionally not frozen in this document.

This Kaggle-side lane hosts:

- RAG retrieval support / chunk processing
- memory summarisation
- memory categorisation / fact extraction
- retrieval over older chat or persistent memory
- other background memory-processing operations required by the memory subsystem

These services should run **in parallel with the local generation path whenever they are not a hard dependency**. Their work should not block the user's interaction merely because background memory maintenance or indexing is still running.

Where a result is actually required before generation can continue, the dependency is respected:

```text
Gemma requests older information
        |
        v
Kaggle retrieval / memory service
        |
        v
required result returned
        |
        v
Gemma continues
```

Where no hard dependency exists, the work remains asynchronous / parallel:

```text
local user response generation  --------------------->

background memory summarise / classify / persist --->
```

Therefore, Flash Mode has two deployment layers:

1. **Local core path** — Gemma + Qwen, always resident on the 8 GB RTX 4060 Laptop GPU.
2. **Kaggle auxiliary path** — retrieval, RAG-processing, memory summarisation/categorisation and related persistent-context services, kept resident remotely and synchronized with the local pipeline when needed.

The local core remains the fast lightweight workbench path. Kaggle is used to keep persistent-context and retrieval work from consuming the laptop's latency and VRAM budget.

---

## 7. Flash Mode Execution Algorithm

### Step 1 — User request arrives

The runtime receives:

- user message
- attachment metadata
- available attachment content / references

### Step 2 — Gemma always runs first

Gemma performs all of the following in one inference:

1. Understand the complete request.
2. Determine whether the request contains one or multiple subtasks.
3. Identify which parts it can answer itself immediately.
4. Determine whether Qwen is required.
5. If Qwen is required, generate a task/instruction for Qwen.
6. Decide whether Qwen's output is already suitable for the user or must return to Gemma for further reasoning.
7. Produce any Gemma-side final answer that can already be completed.

There is no separate classifier/router inference.

---

## 8. The Four Atomic Flash Cases

### Case A — Gemma only

Example:

> Why is sunrise red?

```text
USER
  |
  v
GEMMA
  |
  v
USER
```

Gemma answers directly.

Qwen is omitted completely.

---

### Case B — Qwen only

Example:

> What is written in this image?

Gemma determines that Qwen can directly satisfy the visual request.

```text
USER
  |
  v
GEMMA
  |
  | route visual task
  v
QWEN
  |
  v
USER
```

Gemma is not called again.

Qwen's result is marked as final.

---

### Case C — Qwen -> Gemma

Example:

> Read this screenshot and explain why this error happened.

Gemma cannot complete the reasoning until Qwen first extracts the visual information.

```text
USER
  |
  v
GEMMA
  |
  v
QWEN
  |
  | intermediate result
  v
GEMMA
  |
  v
USER
```

The Qwen output is not final. It becomes an input for Gemma's second inference.

---

### Case D — Gemma + Qwen independently

Example:

> Why is sunrise red, and what is written in this image?

The tasks are independent.

Gemma answers its part during the first inference while also creating the Qwen task.

```text
                 +--> GEMMA answer --------+
USER --> GEMMA --|                          +--> response collection --> USER
                 +--> QWEN task --> QWEN ---+
```

Gemma does not wait for Qwen to finish before producing its own independent answer.

Both outputs are collected and returned in the same logical order as the user requested them.

---

## 9. Mixed Requests

A request may contain any combination of the four atomic cases.

Example:

> Why is sunrise red, and inspect this screenshot and explain why the error occurred.

This contains:

```text
Task 1: Gemma only
Task 2: Qwen -> Gemma
```

Flow:

```text
                    +--> Gemma completes Task 1 -------------------+
USER --> GEMMA -----|                                               |
                    +--> Qwen Task 2 --> QWEN --> Gemma Task 2 -----+
                                                                    |
                                                           response collection
                                                                    |
                                                                    v
                                                                   USER
```

The already completed Gemma answer should be preserved.

When Qwen returns an intermediate result, Gemma should continue only the unfinished dependent portion instead of regenerating completed work.

---

## 10. Dependency Rule

The runtime must distinguish between **independent** and **dependent** work.

### Independent

```text
Gemma task
Qwen task
```

Both may proceed without waiting for each other where hardware/runtime permits.

### Dependent

```text
Qwen
  |
  v
Gemma
```

Gemma's second stage must wait for Qwen because the required information does not yet exist.

Flash Mode should never add an extra model hop merely to forward a result.

---

## 11. `final = true / false`

A useful control cue for the eventual JSON / socket-plug design is a boolean `final` field.

### `final = true`

The worker's output completes that branch of the user's request.

```text
worker output
   |
   v
response collection / user
```

No further reasoning model needs to inspect it.

### `final = false`

The worker's output is intermediate and is required by another model/task.

```text
worker output
   |
   v
dependent model/task
```

For Flash Mode, the most common case is:

```text
Qwen output
final = false
   |
   v
Gemma
```

### Important simplification

If Qwen is not required, its field/task should simply be omitted.

There is no need for redundant fields such as:

```json
"needed": false
```

---

## 12. JSON Guidance

The final JSON schema is **not frozen here**.

The implementation should preserve these ideas:

- Gemma may return its own completed answer during the same inference in which it routes work.
- Qwen tasks should contain the exact instruction required for that branch.
- Dependencies must be explicit enough for the runtime to know whether a result can go directly to the user or must return to Gemma.
- `final=true/false` is recommended as the simplest branch-completion signal.
- Completed independent branches should not be regenerated during later dependent inference calls.
- If a model/component is not required, omit it rather than representing unnecessary empty state.

An illustrative shape could be:

```json
{
  "gemma_response": "...",
  "qwen": {
    "request": "...",
    "final": true
  }
}
```

or, for a dependent Qwen stage:

```json
{
  "qwen": {
    "request": "...",
    "final": false
  }
}
```

These examples are guidance only, not the final interface contract.

---

## 13. Socket-Plug Architecture Guidance

Flash Mode should continue SAGE's **socket -> plug** philosophy, but the exact contracts should be designed by the LangGraph/backend implementation.

Conceptually:

- **Socket:** rich internal result produced by a component.
- **Plug:** only the subset of that result required by the next component.

Example:

```text
Qwen internal result
      |
      | mapper / state adapter
      v
Gemma-facing input
```

The purpose is to prevent unnecessary internal data, debug information, paths, telemetry, or large irrelevant payloads from being injected into another model's context.

However, the full underlying result should remain available in shared state/storage when needed.

The Flash architecture defines **who should receive a result and whether it is final**. The LangGraph/backend layer should define the exact socket, plug, state, and transport schemas.

---

## 14. Response Collection

When all requested branches have reached a final result:

```text
completed branch outputs
        |
        v
rule-based response collection
        |
        v
restore original request order
        |
        v
USER
```

No dedicated final-response LLM is required in Flash Mode.

This avoids:

- another inference
- another model switch
- re-reading long worker outputs
- accidental modification of correct code or extracted content
- unnecessary latency

---

## 15. Final Flash Mode Architecture

The local interactive path and the remote auxiliary path coexist: local Gemma/Qwen handle the fast user-facing inference, while Kaggle-hosted retrieval/memory services operate in parallel or are awaited only when they are a required dependency.

```text
                              USER
                                |
                   +------------+-------------+
                   |                          |
             complex files             simple files/images
                   |                          |
                   v                          v
            DOCLING PIPELINE --------> SHARED MEMORY POOL
                                           ^      ^
                                           |      |
                                      read/write read/write
                                           |      |
                                     GEMMA 4 E2B <-----> QWEN3-VL 2B
                                     (thinking on)
                                           |
                                   reasoning + routing
                                           |
                         +-----------------+-----------------+
                         |                                   |
                    direct final                     dependent branch
                         |                                   |
                         |                            Qwen -> Gemma
                         |                                   |
                         +-----------------+-----------------+
                                           |
                                    response collection
                                           |
                                           v
                                          USER

               KAGGLE AUXILIARY LANE (resident / parallel)
               --------------------------------------------
               retrieval / RAG chunk processing / memory
               summarisation / categorisation / fact extraction
                               ^              |
                               |              v
                         shared state / retrieval requests
```

---

## 16. Positioning of Flash Mode Within SAGE

Flash Mode is SAGE's **fast local workbench mode**.

It is intentionally designed to be:

- lightweight
- fast
- self-contained
- locally self-reliant for the core interactive Gemma/Qwen path
- able to perform basic workbench tasks without Kaggle / government GPU infrastructure
- continuously resident on a consumer RTX 4060 Laptop GPU
- suitable for frequent everyday workloads

Typical workloads include:

```text
summarisation
light coding
casual/general reasoning
quick document understanding
simple OCR / visual questions
mixed lightweight text + media tasks
```

More demanding long-horizon, specialist-heavy, or high-compute workloads can be handled by SAGE's heavier modes/architecture separately.

---

## 17. Final Design Principle

> **Gemma always starts, with thinking enabled by default, and performs routing plus reasoning in the same inference. Qwen is called only when its multimodal capability is required. If Qwen's answer is sufficient, it goes directly to the user; if Gemma needs Qwen's information to continue reasoning, Qwen returns an intermediate result to Gemma. Both interactive models remain resident locally on the 8 GB RTX 4060 Laptop GPU. Retrieval, RAG-processing, and memory-support services remain resident on Kaggle and operate in parallel unless their result is a hard dependency for the current generation. No extra model is used merely to route or rewrite completed answers.**
