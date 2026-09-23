# SAGE Standalone Multi-Tier Memory Engine

A lightweight, deterministic, multi-tier semantic memory subsystem designed for production AI agents and LLM applications.

Supports **Hot Working Memory**, **Cold Archival Memory**, **Global Directives**, and **Chronological Chat Continuity** using SQLite (WAL mode) / PostgreSQL and ChromaDB.

---

## Features

- **4 Distinct Memory Tiers**:
  - **Recent Chat**: FIFO conversation ledger for immediate conversational continuity.
  - **Global Directives**: Pinned persona and system instructions injected on every turn within a strict token budget.
  - **Hot Memory**: Fast session-scoped scratchpad for active tasks and working context.
  - **Cold Memory**: Long-term semantic knowledge repository for cross-session facts and preferences.
- **Dual Persistence Architecture**:
  - Relational database (SQLite with WAL mode or PostgreSQL via psycopg3) as the canonical ledger.
  - ChromaDB vector collections for semantic similarity search (`all-MiniLM-L6-v2`).
- **Deterministic Offline Testing**:
  - Built-in fallback embeddings allow testing the entire pipeline without GPU hardware, PyTorch, or live model downloads.
- **Agent-Ready Tools**:
  - Exposes 8 standard tools with standardized JSON schemas compatible with OpenAI, Anthropic, Ollama, LangChain, and AutoGen.
- **Lifecycle Management**:
  - Memory promotion (Hot $\to$ Cold), compaction (summarizing multiple memories), forgetting (decay score), and full audit logging.

---

## Directory Structure

```text
standalone_memory/
├── __init__.py           # Clean package exports
├── config.py             # Environment configuration with safe defaults
├── memory_engine.py      # Relational persistence (SQLite / PostgreSQL)
├── vector_index.py       # ChromaDB vector index & embedding manager
├── tools.py              # 8 agent-callable tools with function schemas
├── demo.py               # Complete end-to-end runnable demo
├── test_standalone.py    # Deterministic pytest verification suite
├── requirements.txt      # Minimal dependencies
├── pyproject.toml        # Pip-installable package setup
└── README.md             # This documentation
```

---

## Installation

### 1. Install Dependencies
```bash
pip install -r standalone_memory/requirements.txt
```

*(Optional) If using PostgreSQL instead of the default SQLite:*
```bash
pip install psycopg[binary]
```

### 2. Install Package Locally (Optional)
```bash
pip install -e .
```

---

## 30-Second Quickstart

```python
from standalone_memory import (
    memory_store_hot,
    memory_store_cold,
    memory_get_context,
    memory_promote_to_cold,
    memory_engine,
)

# 1. Store a session task fact in Hot Memory
hot_record = memory_store_hot(
    user_id="alice",
    chat_id="session_42",
    content="Alice is debugging network latency on switch S-4",
    category="task",
    importance=0.9,
)
mem_id = hot_record["memory"]["memory_id"]

# 2. Store permanent knowledge in Cold Memory
memory_store_cold(
    user_id="alice",
    content="Company policy requires 2 approvals for production deploys",
    category="instruction",
    is_global=True,  # Injected on every turn
)

# 3. Log a conversation turn to the ledger
memory_engine.add_message(
    chat_id="session_42",
    role="user",
    content="What switch was experiencing latency?",
    user_id="alice",
)

# 4. Assemble composite prompt context within token budget
context = memory_get_context(
    query="What switch was experiencing latency?",
    chat_id="session_42",
    user_id="alice",
    max_tokens=1200,
)

print(context["context_text"])
```

**Output:**
```text
=== GLOBAL DIRECTIVES ===
• [GLOBAL RULE] Company policy requires 2 approvals for production deploys

=== RECENT CONVERSATION ===
[USER]: What switch was experiencing latency?

=== RELEVANT MEMORIES ===
• [HOT MEMORY - task] Alice is debugging network latency on switch S-4 (relevance: 0.94)
```

---

## Integrating with AI Agents

### OpenAI Function Calling Integration

Pass `TOOL_DEFINITIONS` directly into OpenAI's client:

```python
from openai import OpenAI
from standalone_memory import TOOL_DEFINITIONS, tools

client = OpenAI()

# 1. Map tool names to execution functions
TOOL_MAP = {
    "memory_store_hot": tools.memory_store_hot,
    "memory_search_hot": tools.memory_search_hot,
    "memory_store_cold": tools.memory_store_cold,
    "memory_search_cold": tools.memory_search_cold,
    "memory_promote_to_cold": tools.memory_promote_to_cold,
    "memory_get_context": tools.memory_get_context,
    "memory_decay_check": tools.memory_decay_check,
    "memory_delete": tools.memory_delete,
}

# 2. Call OpenAI model with memory tools
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant with multi-tier memory."},
        {"role": "user", "content": "Remember that our Kubernetes cluster is hosted on AWS us-east-1."},
    ],
    tools=TOOL_DEFINITIONS,
)

# 3. Execute tool if chosen by LLM
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        fn_name = tool_call.function.name
        fn_args = json.loads(tool_call.function.arguments)
        result = TOOL_MAP[fn_name](**fn_args)
        print("Executed:", fn_name, "Result:", result)
```

---

## Configuration Reference

Override any configuration parameter via environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `SAGE_MEMORY_DIR` | `./sage_memory_data` | Root directory for SQLite DB and Chroma data |
| `SAGE_MEMORY_DB` | `sqlite` | Relational backend (`sqlite` or `postgres`) |
| `SAGE_SQLITE_PATH` | `./sage_memory_data/sage_memory.db` | Exact path to SQLite database file |
| `CHROMA_PERSIST_DIR` | `./sage_memory_data/chroma_db` | Path to persistent ChromaDB folder |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | SentenceTransformer model name |
| `SAGE_RECENT_CHAT_MAX_MESSAGES`| `5` | Maximum number of turns in chronological window |
| `SAGE_GLOBAL_MEMORY_BUDGET_TOKENS` | `600` | Token limit for pinned global directives |
| `SAGE_CONTEXT_MEMORY_BUDGET_TOKENS` | `1200`| Token limit for vector-retrieved context |

---

## Running the Verification Suite

Run the deterministic test suite to verify database integrity, tier isolation, promotion, and context assembly:

```bash
pytest standalone_memory/test_standalone.py -v
```

All tests execute without requiring GPU or external API keys.
