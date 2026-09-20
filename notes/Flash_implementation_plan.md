# SAGE Flash Mode — Final LangGraph + LangChain Implementation Plan

> **Self-contained reference.** Any AI or developer with access to the SAGE codebase can
> implement this plan from scratch using only this document. All prior architectural
> decisions are resolved and recorded here.

---

## 1. Project Context and Starting State

### What SAGE Is

SAGE is a local AI workbench running on an **NVIDIA RTX 4060 Laptop GPU (8 GB VRAM)**.
It exposes a FastAPI server (`app.py`) and orchestrates inference through local
`llama-server` processes (llama.cpp HTTP API).

### Existing Codebase Summary

| File / Module | Role |
|---|---|
| [`orchestrator.py`](file:///c:/Users/ojasv/Desktop/SAGE/SAGE/orchestrator.py) | Sequential agent loop. Gemma → tool → Gemma. **Will be preserved unchanged.** |
| [`model_manager.py`](file:///c:/Users/ojasv/Desktop/SAGE/SAGE/model_manager.py) | Manages **one** llama-server at a time. Stops old before starting new. **This is the anti-pattern Flash Mode eliminates.** |
| [`model_client.py`](file:///c:/Users/ojasv/Desktop/SAGE/SAGE/model_client.py) | Sync HTTP client wrapping `/v1/chat/completions`. |
| [`core/dispatcher.py`](file:///c:/Users/ojasv/Desktop/SAGE/SAGE/core/dispatcher.py) | `ToolRegistry` + `ToolResult`. Dispatches tools by name. |
| [`core/run_state.py`](file:///c:/Users/ojasv/Desktop/SAGE/SAGE/core/run_state.py) | `RunState` dataclass — backend-owned execution metadata. |
| [`core/mappers/`](file:///c:/Users/ojasv/Desktop/SAGE/SAGE/core/mappers/) | Socket→Plug mappers. Convert raw tool sockets to Gemma-facing payloads. |
| [`core/sockets.py`](file:///c:/Users/ojasv/Desktop/SAGE/SAGE/core/sockets.py) | Rich socket builders for all components. |
| [`core/json_repair.py`](file:///c:/Users/ojasv/Desktop/SAGE/SAGE/core/json_repair.py) | `parse_agent_json()`, `clean_json_string()`, `build_repair_prompt()`. |
| [`tools/`](file:///c:/Users/ojasv/Desktop/SAGE/SAGE/tools/) | memory, vision, coder, document_db, math tools. |
| [`sage_memory/`](file:///c:/Users/ojasv/Desktop/SAGE/SAGE/sage_memory/) | MemoryManager, hot/cold memory, ChromaDB. |
| [`app.py`](file:///c:/Users/ojasv/Desktop/SAGE/SAGE/app.py) | FastAPI app. Existing `/api/chat` endpoint uses `orchestrator.run()`. |
| [`config.py`](file:///c:/Users/ojasv/Desktop/SAGE/SAGE/config.py) | Paths, ports, model configs, sandbox config. |
| [`requirements.txt`](file:///c:/Users/ojasv/Desktop/SAGE/SAGE/requirements.txt) | Current deps: fastapi, uvicorn, httpx, chromadb, pydantic, etc. |

### What Flash Mode Must Do (Summary)

Flash Mode is a **new, parallel execution path** that:
1. Keeps **Gemma 4 E2B** and **Qwen3-VL 2B** resident in VRAM simultaneously — no model switching.
2. Uses **LangGraph** as the state machine for routing, branching, and parallel execution.
3. Routes every request through exactly 4 possible patterns (Cases A/B/C/D).
4. Delegates retrieval/memory background work to a **remote Kaggle GPU server** via a
   **Cloudflare tunnel URL** with a **Bearer API key** — without blocking local inference.
5. Preserves every existing file, endpoint, and test.

---

## 2. Resolved Architecture Decisions

### Decision 1 — Dual Model Residency: Two Separate llama-server Processes

**Choice: Two independent `llama-server` processes on two different ports.**

| | Two Processes (Chosen) | Single Server `--parallel` |
|---|---|---|
| KV cache isolation | ✅ Each model has its own | ❌ Shared, may conflict |
| Debugging | ✅ Each model has its own log file | ❌ Mixed output |
| VRAM predictability | ✅ OS-level process separation | ❌ llama.cpp internal allocation |
| Startup complexity | Slightly higher | Lower |
| Independence | ✅ Crash of one doesn't kill other | ❌ Single process death loses both |

**Ports:**
- Gemma 4 E2B → `http://127.0.0.1:8080`
- Qwen3-VL 2B → `http://127.0.0.1:8081`
- Existing orchestrator (legacy mode) → `http://127.0.0.1:8082` (moved to avoid conflict)

Both processes are started at `DualModelManager.start_all()` during app lifespan startup
and cleaned up at shutdown via `atexit`. Neither is ever stopped mid-request.

### Decision 2 — Kaggle Auxiliary Lane: REST over Cloudflare Tunnel + Bearer Key

The remote Kaggle/GPU server exposes its API through a **Cloudflare tunnel URL**.
The user generates a URL + API key from the server device and provides them at runtime
via environment variables (or `.env` file).

**Transport protocol**: Standard HTTPS REST  
**Auth**: `Authorization: Bearer <API_KEY>` header  
**Pattern**: Fire-and-forget for background memory work; awaited only when retrieval is
a hard dependency for current inference.

Config keys (set in `.env`):
```
SAGE_KAGGLE_URL=https://your-generated-tunnel.trycloudflare.com
SAGE_KAGGLE_API_KEY=your-generated-api-key
```

If `SAGE_KAGGLE_URL` is empty, the Kaggle lane silently no-ops. Flash Mode works fully
in local-only mode without the remote server.

---

## 3. VRAM Budget

| Component | Estimated VRAM |
|---|---|
| Gemma 4 E2B IQ4-NL (2.6 GB file) | ~3.0 GB runtime |
| Qwen3-VL 2B IQ4-NL (1.05 GB file) | ~1.5 GB runtime |
| KV cache + context (both models) | ~1.5 GB |
| **Total** | **~6.0 GB** |
| **Headroom on 8 GB** | **~2.0 GB** |

---

## 4. Flash Mode Execution Cases

Every request routes to exactly one of these patterns. Gemma decides which one.

| Case | Pattern | When |
|---|---|---|
| **A** | Gemma → User | Text-only, no visual content |
| **B** | Gemma → Qwen → User | Visual request, Qwen answer is final |
| **C** | Gemma → Qwen → Gemma → User | Visual extraction needed before Gemma can reason |
| **D** | Gemma (parallel) + Qwen → User | Independent text + visual subtasks |

---

## 5. New File Structure

```
SAGE/
├── flash/                              ← NEW top-level package
│   ├── __init__.py
│   ├── dual_model_manager.py           ← Two-process model manager
│   ├── lc_adapters.py                  ← LangChain BaseChatModel wrappers
│   ├── state.py                        ← FlashState TypedDict (LangGraph state)
│   ├── graph.py                        ← StateGraph assembly + compile
│   ├── kaggle_client.py                ← Cloudflare tunnel async client
│   ├── api_models.py                   ← Pydantic request/response models
│   └── nodes/
│       ├── __init__.py
│       ├── ingest.py                   ← File routing + shared memory pool
│       ├── gemma_reason.py             ← Gemma primary inference + routing
│       ├── branch_router.py            ← Conditional edge function
│       ├── qwen_worker.py              ← Qwen visual inference worker
│       ├── gemma_synthesize.py         ← Gemma second pass (Case C only)
│       └── collect.py                  ← Rule-based response collection
├── prompts/
│   ├── flash_system.txt                ← NEW Gemma Flash routing prompt
│   └── qwen_flash_system.txt           ← NEW Qwen visual worker prompt
│   └── ... (all existing prompts unchanged)
├── tests/
│   └── flash/
│       ├── __init__.py
│       ├── test_dual_model_manager.py
│       ├── test_state_schema.py
│       ├── test_branch_router.py
│       └── test_graph_mock.py
│
│ ── MODIFIED (minimal changes only) ──
├── config.py                           ← Add FLASH_MODELS, FLASH_PORTS, KAGGLE_* keys
├── app.py                              ← Add /api/flash endpoint + dual_model_manager lifespan
├── requirements.txt                    ← Add langgraph, langchain-core, langchain-community
│
│ ── UNTOUCHED (zero changes) ──────────
├── orchestrator.py
├── model_manager.py
├── model_client.py
├── core/ (all files)
├── tools/ (all files)
├── sage_memory/ (all files)
├── sage_document_db/ (all files)
├── document_processor.py
├── db_service.py
└── tests/ (existing tests)
```

---

## 6. Phase-by-Phase Implementation

---

### PHASE 0 — Dependencies
**Time: ~2 hours**  
**Goal: Install LangGraph and LangChain without breaking anything.**

#### [MODIFY] `requirements.txt`

Add these lines to the existing file:
```
# ── Flash Mode: LangGraph + LangChain ────────────────────────────────────────
langgraph>=0.2.0
langchain-core>=0.3.0
langchain-community>=0.3.0
anyio>=4.0.0
```

> [!NOTE]
> `httpx[asyncio]` is already in the existing requirements via `httpx>=0.27.0`.
> `anyio` enables `asyncio.gather()` patterns inside LangGraph node coroutines.

**Verification:**
```bash
cd c:\Users\ojasv\Desktop\SAGE\SAGE
pip install -r requirements.txt
python -c "import langgraph; import langchain_core; print('OK')"
```

---

### PHASE 1 — Config Extension
**Time: ~1 hour**  
**Goal: Add Flash-specific config keys without touching existing config values.**

#### [MODIFY] `config.py`

Append the following block **at the end** of the existing file. Do not modify any existing values.

```python
# ── Flash Mode Configuration ──────────────────────────────────────────────────
# Both Flash models are loaded simultaneously. Neither is ever switched out.
FLASH_GEMMA_PORT = 8080   # Gemma 4 E2B llama-server port
FLASH_QWEN_PORT  = 8081   # Qwen3-VL 2B llama-server port

# Move legacy single-model server to port 8082 to avoid collision
# The existing LLAMA_PORT = 8080 (single server path) should now be 8082
# BUT we do NOT change LLAMA_PORT to avoid breaking orchestrator.py.
# Flash mode uses its own port constants exclusively.

FLASH_MODELS = {
    "gemma_flash": {
        "name": "Gemma 4 E2B QAT IQ4-NL",
        # File must exist at this path. Adjust MODEL_DIR in .env if needed.
        "model_path": os.path.join(MODEL_DIR, "gemma-4-e2b-it-iq4_nl.gguf"),
        "mmproj": None,
        "port": FLASH_GEMMA_PORT,
        "host": SERVER_HOST,
        "context": int(os.environ.get("GEMMA_FLASH_CONTEXT", "8192")),
        "ngl": 999,                # All layers to GPU
        "temperature": 0.20,
        "max_tokens": 4096,
        "reasoning": "on",         # Enables <think> block (Gemma thinking mode)
    },
    "qwen_flash": {
        "name": "Qwen3-VL 2B IQ4-NL",
        "model_path": os.path.join(MODEL_DIR, "qwen3-vl-2b-instruct-iq4_nl.gguf"),
        # mmproj file is required for vision; must be in MODEL_DIR
        "mmproj": os.path.join(MODEL_DIR, "qwen3-vl-2b-mmproj-f16.gguf"),
        "port": FLASH_QWEN_PORT,
        "host": SERVER_HOST,
        "context": int(os.environ.get("QWEN_FLASH_CONTEXT", "4096")),
        "ngl": 999,
        "temperature": 0.05,
        "max_tokens": 2048,
        "reasoning": "off",
    },
}

# ── Kaggle Auxiliary Lane (Cloudflare tunnel) ─────────────────────────────────
# Set these in your .env file on the laptop. Obtain the URL and key from the
# Kaggle/GPU server device after starting the tunnel.
#
# If SAGE_KAGGLE_URL is empty (""), the Kaggle lane is disabled and Flash Mode
# runs fully locally with no remote calls. This is the safe default.
KAGGLE_URL     = os.environ.get("SAGE_KAGGLE_URL", "").rstrip("/")
KAGGLE_API_KEY = os.environ.get("SAGE_KAGGLE_API_KEY", "")
KAGGLE_TIMEOUT = float(os.environ.get("SAGE_KAGGLE_TIMEOUT", "30.0"))  # seconds
```

Also add to your `.env.example`:
```
# Flash Mode: Kaggle Auxiliary Lane (Cloudflare Tunnel)
# Get the URL and key from the Kaggle/GPU server device.
# Leave blank to run Flash Mode in local-only mode.
SAGE_KAGGLE_URL=
SAGE_KAGGLE_API_KEY=

# Flash model context sizes (optional overrides)
GEMMA_FLASH_CONTEXT=8192
QWEN_FLASH_CONTEXT=4096
```

> [!IMPORTANT]
> The legacy `LLAMA_PORT = 8080` is **not changed**. The existing orchestrator and
> `model_manager.py` continue to use port 8080 exactly as before. Flash Mode uses its
> own `FLASH_GEMMA_PORT` and `FLASH_QWEN_PORT` constants, managed by a completely
> separate manager class (`DualModelManager`). The only potential conflict is if the
> user tries to run both the legacy orchestrator AND Flash Mode simultaneously — document
> this limitation and add a startup guard (see Phase 2).

---

### PHASE 2 — Dual Model Manager
**Time: ~1 day**  
**Goal: Start and keep both Gemma and Qwen llama-server processes resident simultaneously.**

#### [NEW] `flash/dual_model_manager.py`

This class is a self-contained replacement for `ModelManager` for the Flash path only.
It manages two independent `llama-server` subprocesses.

```python
"""
flash/dual_model_manager.py

Manages two resident llama-server processes simultaneously for SAGE Flash Mode.
Gemma 4 E2B on port FLASH_GEMMA_PORT, Qwen3-VL 2B on port FLASH_QWEN_PORT.

Design rules:
  - start_all() must be called once at app startup (lifespan event).
  - stop_all() is registered with atexit and called at app shutdown.
  - Neither model is ever stopped mid-request.
  - If one model fails health check after start, RuntimeError is raised.
  - is_gemma_healthy() / is_qwen_healthy() can be polled for /api/flash/status.
"""

from __future__ import annotations
import atexit
import logging
import os
import subprocess
import time
import socket
from typing import Optional, Dict, Any
import httpx
from pathlib import Path

import config

logger = logging.getLogger(__name__)


class ModelServer:
    """One llama-server subprocess with its own port and health management."""

    def __init__(self, key: str, model_cfg: Dict[str, Any]) -> None:
        self.key          = key                     # "gemma_flash" | "qwen_flash"
        self.name         = model_cfg["name"]
        self.model_path   = model_cfg["model_path"]
        self.mmproj       = model_cfg.get("mmproj")
        self.port         = model_cfg["port"]
        self.host         = model_cfg.get("host", "127.0.0.1")
        self.context      = model_cfg.get("context", 8192)
        self.ngl          = model_cfg.get("ngl", 999)
        self.reasoning    = model_cfg.get("reasoning", "off")
        self.base_url     = f"http://{self.host}:{self.port}"
        self.process: Optional[subprocess.Popen] = None
        self._log_fp      = None

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def is_healthy(self, timeout: float = 2.0) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/health", timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False

    def is_port_in_use(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex((self.host, self.port)) == 0

    def wait_for_healthy(self, timeout: float = 120.0) -> bool:
        """Poll /health until server responds or timeout."""
        start = time.time()
        while time.time() - start < timeout:
            if self.process and self.process.poll() is not None:
                logger.error("[%s] Process exited early with code %s",
                             self.key, self.process.returncode)
                return False
            if self.is_healthy(timeout=1.5):
                return True
            time.sleep(0.5)
        return False

    def start(self) -> None:
        """Launch the llama-server subprocess for this model."""
        if not os.path.exists(config.LLAMA_SERVER_PATH):
            raise FileNotFoundError(
                f"llama-server not found: {config.LLAMA_SERVER_PATH}"
            )
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )
        if self.mmproj and not os.path.exists(self.mmproj):
            raise FileNotFoundError(
                f"mmproj file not found: {self.mmproj}"
            )
        if self.is_port_in_use():
            raise RuntimeError(
                f"Port {self.port} already in use. "
                f"Stop whatever is using it before starting Flash Mode."
            )

        cmd = [
            config.LLAMA_SERVER_PATH,
            "-m", self.model_path,
            "-ngl", str(self.ngl),
            "-c",   str(self.context),
            "--port", str(self.port),
            "--host", self.host,
            "--no-webui",
        ]
        if self.mmproj:
            cmd.extend(["--mmproj", self.mmproj])
        if self.reasoning:
            cmd.extend(["--reasoning", self.reasoning])

        log_path = config.TEMP_DIR / "logs" / f"flash_{self.key}.log"
        self._log_fp = open(log_path, "a", encoding="utf-8")

        logger.info("[%s] Starting: %s (port %d)", self.key, self.name, self.port)
        self.process = subprocess.Popen(
            cmd,
            stdout=self._log_fp,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def stop(self) -> None:
        """Gracefully terminate the llama-server subprocess."""
        if self.process:
            logger.info("[%s] Stopping %s ...", self.key, self.name)
            try:
                self.process.terminate()
                self.process.wait(timeout=8.0)
            except subprocess.TimeoutExpired:
                logger.warning("[%s] Force killing (timeout).", self.key)
                self.process.kill()
                self.process.wait()
            except Exception as e:
                logger.error("[%s] Error stopping: %s", self.key, e)
            finally:
                self.process = None
        if self._log_fp:
            try:
                self._log_fp.close()
            except Exception:
                pass


class DualModelManager:
    """
    Manages two resident llama-server processes for Flash Mode.

    Usage (in app lifespan):
        dual_model_manager.start_all()
        ...
        dual_model_manager.stop_all()

    The gemma_url() and qwen_url() methods return the base URLs for
    the LangChain adapters and Kaggle client.
    """

    def __init__(self) -> None:
        self.gemma = ModelServer("gemma_flash", config.FLASH_MODELS["gemma_flash"])
        self.qwen  = ModelServer("qwen_flash",  config.FLASH_MODELS["qwen_flash"])
        self._started = False
        atexit.register(self.stop_all)

    def start_all(self, startup_timeout: float = 120.0) -> None:
        """
        Start both llama-server processes and wait until both are healthy.
        Raises RuntimeError if either fails to start within startup_timeout.
        """
        import threading

        errors: Dict[str, str] = {}

        def start_server(server: ModelServer) -> None:
            try:
                server.start()
                if not server.wait_for_healthy(timeout=startup_timeout):
                    errors[server.key] = f"{server.name} failed health check on port {server.port}"
            except Exception as e:
                errors[server.key] = str(e)

        # Start both servers concurrently in threads
        threads = [
            threading.Thread(target=start_server, args=(self.gemma,), daemon=True),
            threading.Thread(target=start_server, args=(self.qwen,),  daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=startup_timeout + 5)

        if errors:
            self.stop_all()
            raise RuntimeError(
                f"Flash Mode startup failed: {errors}. "
                f"Check logs in {config.TEMP_DIR / 'logs'} for details."
            )

        self._started = True
        logger.info(
            "Flash Mode ready. Gemma:%s Qwen:%s",
            self.gemma.base_url, self.qwen.base_url
        )

    def stop_all(self) -> None:
        """Stop both model servers. Called at app shutdown."""
        self.gemma.stop()
        self.qwen.stop()
        self._started = False

    def is_gemma_healthy(self) -> bool:
        return self.gemma.is_healthy()

    def is_qwen_healthy(self) -> bool:
        return self.qwen.is_healthy()

    def gemma_url(self) -> str:
        return self.gemma.base_url

    def qwen_url(self) -> str:
        return self.qwen.base_url

    def status(self) -> Dict[str, Any]:
        """Return health status for /api/flash/status endpoint."""
        return {
            "flash_mode": "active" if self._started else "stopped",
            "gemma": {
                "name": self.gemma.name,
                "port": self.gemma.port,
                "healthy": self.gemma.is_healthy(),
                "running": self.gemma.is_running,
            },
            "qwen": {
                "name": self.qwen.name,
                "port": self.qwen.port,
                "healthy": self.qwen.is_healthy(),
                "running": self.qwen.is_running,
            },
        }


# Singleton — imported by app.py and flash/graph.py
dual_model_manager = DualModelManager()
```

**Verification:**
```bash
# Set SAGE_MOCK_MODE=0 and ensure model files exist
python -c "
from flash.dual_model_manager import dual_model_manager
dual_model_manager.start_all()
print(dual_model_manager.status())
dual_model_manager.stop_all()
"
# Expected: both servers healthy, status shows healthy=True for both
```

---

### PHASE 3 — LangChain Model Adapters
**Time: ~1 day**  
**Goal: Wrap each local llama-server endpoint as a `BaseChatModel` so LangGraph nodes
can call `.invoke()` and `.ainvoke()` uniformly.**

#### [NEW] `flash/lc_adapters.py`

```python
"""
flash/lc_adapters.py

LangChain BaseChatModel adapters for local llama-server endpoints.

Why this exists:
  LangGraph nodes use langchain_core message types (HumanMessage, SystemMessage,
  AIMessage). These adapters translate between LangChain's message format and
  the llama-server /v1/chat/completions JSON format, including:
    - <think> block extraction from Gemma reasoning output
    - multimodal image_url content for Qwen vision requests

These adapters wrap the SAME HTTP calls as the existing model_client.py.
They do NOT replace model_client.py — that file continues to serve the
legacy orchestrator path unchanged.
"""

from __future__ import annotations
import asyncio
import re
import time
from typing import Any, Dict, Iterator, List, Optional, AsyncIterator

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage
)
from langchain_core.outputs import ChatGeneration, ChatResult

import config


def _messages_to_payload(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    """Convert LangChain messages to llama-server /v1/chat/completions format.

    Supports:
      - SystemMessage → role: system
      - HumanMessage  → role: user (text or multimodal with image_url)
      - AIMessage     → role: assistant

    For multimodal content, HumanMessage.content may be a list of dicts:
      [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "data:..."}}]
    This is passed through as-is to llama-server (OpenAI vision format).
    """
    result = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": msg.content})
        else:
            # Unknown message type — pass content as-is under "user" role
            result.append({"role": "user", "content": str(msg.content)})
    return result


def _extract_thinking(content: str) -> tuple[str, str]:
    """Strip <think>...</think> block from Gemma output.

    Returns:
        (clean_text, thinking_text)
    Both may be empty strings.
    """
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    thinking = ""
    match = think_pattern.search(content)
    if match:
        thinking = match.group(1).strip()
        content = think_pattern.sub("", content).strip()
    return content, thinking


class LlamaCppChatAdapter(BaseChatModel):
    """
    LangChain BaseChatModel wrapping a local llama-server HTTP endpoint.

    Instantiate one per model (Gemma + Qwen have separate ports):
        gemma_lm = LlamaCppChatAdapter(base_url="http://127.0.0.1:8080", ...)
        qwen_lm  = LlamaCppChatAdapter(base_url="http://127.0.0.1:8081", ...)

    The adapter exposes:
        .invoke(messages)       → AIMessage (sync)
        .ainvoke(messages)      → AIMessage (async via asyncio thread)

    Additional metadata is attached to AIMessage.additional_kwargs:
        {
          "thinking": "...",    # <think> block content (Gemma only)
          "duration": 1.23,     # seconds
          "usage": {...},       # token counts
        }
    """

    # Pydantic fields (LangChain requires these to be declared)
    base_url: str
    model_display_name: str = "llama-cpp"
    temperature: float = 0.20
    max_tokens: int = 4096
    request_timeout: float = config.REQUEST_TIMEOUT
    extract_thinking: bool = False   # Set True for Gemma, False for Qwen

    @property
    def _llm_type(self) -> str:
        return "llama-cpp-local"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = {
            "messages": _messages_to_payload(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop

        url = f"{self.base_url}/v1/chat/completions"
        start = time.time()
        with httpx.Client(timeout=self.request_timeout) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        duration = time.time() - start

        raw_content = (
            data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
        )
        usage = data.get("usage", {})

        clean_content, thinking = (
            _extract_thinking(raw_content) if self.extract_thinking
            else (raw_content, "")
        )

        ai_msg = AIMessage(
            content=clean_content,
            additional_kwargs={
                "thinking": thinking,
                "duration": duration,
                "usage": usage,
                "raw": data,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generation via asyncio thread executor (llama-server is sync HTTP)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._generate(messages, stop=stop, **kwargs)
        )

    # Required stub for streaming (not used in Flash Mode but must exist)
    def _stream(self, messages, stop=None, **kwargs) -> Iterator[Any]:
        raise NotImplementedError("Streaming not implemented in Flash Mode.")

    async def _astream(self, messages, stop=None, **kwargs) -> AsyncIterator[Any]:
        raise NotImplementedError("Async streaming not implemented in Flash Mode.")
```

**Adapter construction** (done once in `flash/graph.py`):
```python
from flash.lc_adapters import LlamaCppChatAdapter
from flash.dual_model_manager import dual_model_manager
import config

gemma_lm = LlamaCppChatAdapter(
    base_url=dual_model_manager.gemma_url(),
    model_display_name="Gemma 4 E2B",
    temperature=config.FLASH_MODELS["gemma_flash"]["temperature"],
    max_tokens=config.FLASH_MODELS["gemma_flash"]["max_tokens"],
    extract_thinking=True,   # Gemma uses <think> blocks
)

qwen_lm = LlamaCppChatAdapter(
    base_url=dual_model_manager.qwen_url(),
    model_display_name="Qwen3-VL 2B",
    temperature=config.FLASH_MODELS["qwen_flash"]["temperature"],
    max_tokens=config.FLASH_MODELS["qwen_flash"]["max_tokens"],
    extract_thinking=False,
)
```

---

### PHASE 4 — Flash State Schema
**Time: ~2 hours**  
**Goal: Define the TypedDict that flows through the LangGraph state machine.**

#### [NEW] `flash/state.py`

```python
"""
flash/state.py

FlashState is the single TypedDict that LangGraph passes between all Flash nodes.

LangGraph rules:
  - Node functions receive the full FlashState dict.
  - Node functions return a PARTIAL dict (only the keys they update).
  - LangGraph merges the partial dict into the running state automatically.
  - List fields use LangGraph's Annotated[list, operator.add] reducer so that
    multiple nodes can append to them without overwriting each other.

Key separation:
  - FlashState holds Flash-specific routing signals.
  - RunState (from core/run_state.py) is embedded for compatibility with existing
    socket builders and mappers that expect it. The Flash path creates its own
    RunState at ingest and stores it in flash_state["run_state"].
"""

from __future__ import annotations
import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from core.run_state import RunState


class FlashState(TypedDict, total=False):
    # ── Request Input ────────────────────────────────────────────────────────
    user_message: str           # Raw user text
    attachments: List[dict]     # Attachment manifest [{"ref", "name", "type", "size", ...}]
    file_map: dict              # ref_id → {path, bytes_b64, doc_id, ...}
    session_id: str

    # ── Shared Memory Pool ───────────────────────────────────────────────────
    # Intermediate results, Docling output, retrieved docs, image refs.
    # Any node may read from or write to this dict.
    shared_pool: dict

    # ── Gemma First Pass (gemma_reason_node) ────────────────────────────────
    gemma_raw: str              # Raw string output from Gemma (pre-JSON parse)
    gemma_parsed: dict          # Parsed Gemma routing JSON
    gemma_answer: Optional[str] # Gemma's own completed answer text (Cases A, D)
    gemma_thinking: str         # Stripped <think> content (Gemma reasoning trace)

    # ── Qwen Task ────────────────────────────────────────────────────────────
    qwen_task: Optional[dict]   # {"request": str, "final": bool}
    qwen_result: Optional[str]  # Qwen visual output text

    # ── Flash Case Routing ────────────────────────────────────────────────────
    # Set by gemma_reason_node, read by branch_router.
    flash_case: str             # "A" | "B" | "C" | "D"

    # For Case D parallel join: count of independent branches still pending.
    branches_pending: int

    # ── Response Collection ───────────────────────────────────────────────────
    # Each entry: {"order": int, "text": str, "source": "gemma"|"qwen"}
    # Uses operator.add reducer — multiple nodes can append safely.
    collected_outputs: Annotated[List[dict], operator.add]
    final_response: Optional[str]   # Final user-facing text after collection

    # ── Kaggle Auxiliary Lane ─────────────────────────────────────────────────
    retrieval_requested: bool        # True if Gemma needs retrieval before continuing
    retrieval_query: Optional[str]   # Query for the Kaggle retrieval call
    retrieval_result: Optional[dict] # Kaggle retrieval response (hard-dep path)
    kaggle_task_id: Optional[str]    # Task ID for fire-and-forget background tasks

    # ── Compatibility with Existing SAGE Infrastructure ───────────────────────
    run_state: Optional[RunState]    # Created at ingest; passed to tool mappers

    # ── Telemetry ─────────────────────────────────────────────────────────────
    telemetry: dict
    errors: Annotated[List[str], operator.add]
```

---

### PHASE 5 — Graph Nodes
**Time: ~2 days**  
**Goal: Implement the six node functions that form the Flash graph.**

---

#### Node 1: `flash/nodes/ingest.py`

**What it does:**
- Receives the raw request (message + attachments).
- Classifies each attachment: "simple" (image, plain text) vs. "complex" (PDF, DOCX).
- Simple files: places a direct reference in `shared_pool` (fast, no pipeline).
- Complex files: runs the existing `document_processor.py` / Docling pipeline and places
  the result in `shared_pool`. Also calls `db_service.document_db.ingest_document()`.
- Creates a fresh `RunState` for this request and registers all documents.
- Initialises telemetry counters.

**Input fields read:** `user_message`, `attachments`, `file_map`, `session_id`  
**Output fields written:** `shared_pool`, `run_state`, `telemetry`

```python
"""flash/nodes/ingest.py"""
from __future__ import annotations
import time, uuid
from typing import Dict, Any

from core.run_state import RunState
from flash.state import FlashState

_SIMPLE_TYPES = {"jpg", "jpeg", "png", "gif", "webp", "bmp", "txt", "csv", "md"}
_COMPLEX_TYPES = {"pdf", "docx", "doc", "xlsx", "xls", "pptx", "html"}


def ingest_node(state: FlashState) -> Dict[str, Any]:
    """
    Flash graph entry node.

    Populates shared_pool with content references and creates RunState.
    Does NOT call any model. Pure file/document routing logic.
    """
    attachments = state.get("attachments", [])
    file_map    = state.get("file_map", {})
    session_id  = state.get("session_id") or f"sess_{uuid.uuid4().hex[:12]}"

    run_state = RunState(
        request_id=f"flash_{int(time.time())}_{uuid.uuid4().hex[:6]}",
        session_id=session_id,
        user_text=state.get("user_message", ""),
    )

    shared_pool: dict = {}
    errors = []

    for att in attachments:
        ref   = att.get("ref", "")
        ftype = att.get("type", "").lower()
        fmap  = file_map.get(ref, {})

        if ftype in _SIMPLE_TYPES:
            # Simple: just stash the path reference; nodes access it directly
            shared_pool[ref] = {
                "ref": ref,
                "type": ftype,
                "path": fmap.get("path"),
                "doc_id": att.get("doc_id"),
                "complexity": "simple",
            }
            if att.get("doc_id"):
                run_state.register_document(
                    doc_id=att["doc_id"],
                    display_name=att.get("name", ref),
                    file_type=ftype,
                    source_name=att.get("name"),
                )

        elif ftype in _COMPLEX_TYPES:
            # Complex: run Docling pipeline → store extracted text in shared_pool
            doc_id = att.get("doc_id")
            extracted_text = ""
            if doc_id:
                try:
                    from db_service import document_db
                    # Retrieve already-ingested document content summary
                    result = document_db.rag_search(
                        query=state.get("user_message", ""),
                        doc_ids=[doc_id],
                        top_k=10,
                    )
                    extracted_text = result.get("combined_text", "")
                except Exception as exc:
                    errors.append(f"Ingest: doc {doc_id} retrieval failed: {exc}")

            shared_pool[ref] = {
                "ref": ref,
                "type": ftype,
                "path": fmap.get("path"),
                "doc_id": doc_id,
                "extracted_text": extracted_text,
                "complexity": "complex",
            }
            if doc_id:
                run_state.register_document(
                    doc_id=doc_id,
                    display_name=att.get("name", ref),
                    file_type=ftype,
                    source_name=att.get("name"),
                )
        else:
            # Unknown type — pass through as opaque ref
            shared_pool[ref] = {
                "ref": ref,
                "type": ftype,
                "path": fmap.get("path"),
                "complexity": "unknown",
            }

    telemetry = {
        "start_time": time.time(),
        "gemma_calls": 0,
        "qwen_calls": 0,
        "kaggle_calls": 0,
        "attachments_processed": len(attachments),
    }

    return {
        "shared_pool": shared_pool,
        "run_state":   run_state,
        "telemetry":   telemetry,
        "errors":      errors,
        "session_id":  session_id,
    }
```

---

#### Node 2: `flash/nodes/gemma_reason.py`

**What it does:**
- Calls Gemma with the Flash system prompt + user message + shared pool context.
- Gemma thinks (via `<think>` block, stripped by adapter) and returns routing JSON.
- Parses the JSON using existing `core/json_repair.py` utilities.
- Detects whether retrieval from Kaggle is needed first (retrieval_requested flag).
- Sets: `flash_case`, `gemma_answer` (if applicable), `qwen_task` (if applicable).
- Does **not** call Qwen. Does **not** produce the final response.

**Input fields read:** `user_message`, `shared_pool`, `retrieval_result` (if already fetched)  
**Output fields written:** `gemma_raw`, `gemma_parsed`, `gemma_answer`, `gemma_thinking`, `qwen_task`, `flash_case`, `retrieval_requested`, `retrieval_query`, `telemetry`

```python
"""flash/nodes/gemma_reason.py"""
from __future__ import annotations
import json
from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage

from core.json_repair import parse_agent_json, clean_json_string, build_repair_prompt
from flash.state import FlashState

# gemma_lm is injected at graph construction time (see flash/graph.py)
# It is a LlamaCppChatAdapter instance.

_FLASH_CASE_VALID = {"A", "B", "C", "D"}


def _build_gemma_input(state: FlashState) -> list:
    """Build LangChain messages list for Gemma reasoning call."""
    from flash.graph import _flash_system_prompt   # loaded once at startup

    context_parts = [f"USER MESSAGE:\n{state['user_message']}"]

    # Summarise shared pool for Gemma (filenames, types, doc_ids — NOT raw bytes)
    pool = state.get("shared_pool", {})
    if pool:
        pool_summary = json.dumps(
            {ref: {k: v for k, v in info.items() if k != "extracted_text"}
             for ref, info in pool.items()},
            indent=2
        )
        context_parts.append(f"AVAILABLE FILES:\n{pool_summary}")

        # Include extracted text from complex documents
        for ref, info in pool.items():
            if info.get("extracted_text"):
                context_parts.append(
                    f"DOCUMENT [{ref}] EXTRACTED CONTENT (first 3000 chars):\n"
                    f"{info['extracted_text'][:3000]}"
                )

    # If retrieval was already performed, inject the result
    if state.get("retrieval_result"):
        context_parts.append(
            f"RETRIEVED MEMORY/CONTEXT:\n{json.dumps(state['retrieval_result'], indent=2)}"
        )

    return [
        SystemMessage(content=_flash_system_prompt),
        HumanMessage(content="\n\n".join(context_parts)),
    ]


def gemma_reason_node(state: FlashState, gemma_lm) -> Dict[str, Any]:
    """
    Core routing node. Calls Gemma once. Produces routing decision + any
    Gemma-side completed answer. Never calls Qwen.

    Expected Gemma JSON output (from flash_system.txt prompt):
    {
      "flash_case": "A" | "B" | "C" | "D",
      "gemma_answer": "...",       // Cases A, D (Gemma's own answer)
      "qwen": {                    // Cases B, C, D (omit entirely for A)
        "request": "...",
        "final": true | false
      },
      "retrieval_needed": false,   // Optional: true if Kaggle retrieval required
      "retrieval_query": "..."     // Required if retrieval_needed is true
    }
    """
    messages = _build_gemma_input(state)

    try:
        ai_msg = gemma_lm.invoke(messages)
        raw_text = ai_msg.content
        thinking = ai_msg.additional_kwargs.get("thinking", "")
    except Exception as exc:
        return {
            "errors": [f"gemma_reason: inference failed: {exc}"],
            "flash_case": "A",   # Safe fallback: treat as text-only
            "gemma_answer": f"[SAGE Error] Gemma inference failed: {exc}",
        }

    # Parse JSON
    parsed = parse_agent_json(raw_text) or parse_agent_json(clean_json_string(raw_text))

    # 1-turn repair if needed (reuse existing repair logic)
    if parsed is None:
        try:
            repair_messages = messages + [
                ai_msg,
                HumanMessage(content=build_repair_prompt()),
            ]
            ai_msg2 = gemma_lm.invoke(repair_messages)
            raw_text = ai_msg2.content
            parsed = parse_agent_json(raw_text) or parse_agent_json(clean_json_string(raw_text))
        except Exception as exc:
            parsed = None

    if parsed is None:
        return {
            "errors": [f"gemma_reason: JSON parse failed. Raw: {raw_text[:200]}"],
            "gemma_raw": raw_text,
            "gemma_thinking": thinking,
            "flash_case": "A",
            "gemma_answer": raw_text,  # Fall back: return raw as answer
        }

    flash_case = parsed.get("flash_case", "A").upper()
    if flash_case not in _FLASH_CASE_VALID:
        flash_case = "A"

    result: Dict[str, Any] = {
        "gemma_raw":      raw_text,
        "gemma_parsed":   parsed,
        "gemma_thinking": thinking,
        "flash_case":     flash_case,
    }

    if parsed.get("gemma_answer"):
        result["gemma_answer"] = parsed["gemma_answer"]

    if parsed.get("qwen"):
        result["qwen_task"] = parsed["qwen"]

    # Retrieval signal
    if parsed.get("retrieval_needed") and parsed.get("retrieval_query"):
        result["retrieval_requested"] = True
        result["retrieval_query"] = parsed["retrieval_query"]
    else:
        result["retrieval_requested"] = False

    return result
```

> [!IMPORTANT]
> `gemma_lm` is passed as a **partial function** at graph construction time using
> `functools.partial`. LangGraph nodes must accept only `state` as their argument.
> See Phase 6 (graph assembly) for how to wire this cleanly.

---

#### Node 3: `flash/nodes/branch_router.py`

**What it does:**
- Pure function. Reads `flash_case` from state. Returns the name of the next node.
- This is the LangGraph **conditional edge function** — it returns a string, not a state dict.

```python
"""flash/nodes/branch_router.py"""
from flash.state import FlashState


def branch_router(state: FlashState) -> str:
    """
    Conditional edge function for LangGraph.
    Returns the name of the next node based on Gemma's flash_case decision.

    "A" → "collect"         Gemma answered directly. Skip Qwen.
    "B" → "qwen"            Qwen answers, result is final.
    "C" → "qwen"            Qwen extracts, Gemma needs result to continue.
    "D" → "qwen"            Independent parallel: Gemma already answered,
                            Qwen result also needed. LangGraph Send API handles fork.

    Note: For Case D the fork is handled by graph.py using langgraph.types.Send,
    not by this router. The router still returns "qwen" for D so the linear
    path executes correctly when Send is not applicable.
    """
    case = state.get("flash_case", "A").upper()

    # Case A: Gemma answered everything, skip Qwen entirely
    if case == "A":
        return "collect"

    # Cases B, C, D all require Qwen
    return "qwen"
```

---

#### Node 4: `flash/nodes/qwen_worker.py`

**What it does:**
- Reads `qwen_task["request"]` and any image references from `shared_pool`.
- Builds a multimodal message (text + image_url) for the Qwen adapter.
- Calls Qwen. Sets `qwen_result`.
- Does NOT decide what to do with the result — that is determined by `qwen_task["final"]`
  and the conditional edge that follows this node.

```python
"""flash/nodes/qwen_worker.py"""
from __future__ import annotations
import base64, mimetypes, os
from typing import Dict, Any, List

from langchain_core.messages import SystemMessage, HumanMessage

from flash.state import FlashState


def _load_image_as_data_url(path: str) -> str | None:
    """Read image file and return data URL string for OpenAI vision format."""
    if not path or not os.path.exists(path):
        return None
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def _build_qwen_messages(state: FlashState) -> list:
    from flash.graph import _qwen_system_prompt

    task_request = state["qwen_task"]["request"]
    pool = state.get("shared_pool", {})

    # Build multimodal content list for the user message
    content_parts: List[dict] = [{"type": "text", "text": task_request}]

    # Attach all image files found in shared pool
    for ref, info in pool.items():
        if info.get("type") in ("jpg", "jpeg", "png", "gif", "webp", "bmp"):
            path = info.get("path")
            data_url = _load_image_as_data_url(path)
            if data_url:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })

    return [
        SystemMessage(content=_qwen_system_prompt),
        HumanMessage(content=content_parts),
    ]


def qwen_node(state: FlashState, qwen_lm) -> Dict[str, Any]:
    """
    Visual worker node. Calls Qwen3-VL 2B with image + instruction.

    Output:
      qwen_result: str  — Qwen's extracted/described visual content

    Whether qwen_result goes directly to the user (final=True)
    or returns to Gemma (final=False) is determined by the conditional
    edge AFTER this node, not by this node itself.
    """
    if not state.get("qwen_task"):
        return {"errors": ["qwen_node called but qwen_task is None"]}

    messages = _build_qwen_messages(state)

    try:
        ai_msg = qwen_lm.invoke(messages)
        qwen_text = ai_msg.content
    except Exception as exc:
        return {
            "errors": [f"qwen_node: inference failed: {exc}"],
            "qwen_result": f"[Visual extraction failed: {exc}]",
        }

    return {"qwen_result": qwen_text}
```

---

#### Node 5: `flash/nodes/gemma_synthesize.py`

**Case C only.** Gemma's second inference pass, triggered when Qwen's result is needed
before Gemma can reason.

```python
"""flash/nodes/gemma_synthesize.py"""
from __future__ import annotations
from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from flash.state import FlashState


def gemma_synthesize_node(state: FlashState, gemma_lm) -> Dict[str, Any]:
    """
    Case C: Gemma second pass.

    Receives qwen_result as context and produces the final answer.
    This node is ONLY reached when qwen_task["final"] == False.

    The message history reconstructed here:
      [system prompt]
      [user original message + files context]   ← same as gemma_reason
      [Gemma's first routing response]           ← from gemma_raw
      [Qwen result injected as user message]     ← new context
    """
    from flash.graph import _flash_system_prompt
    from core.json_repair import parse_agent_json

    original_user_context = state.get("user_message", "")
    qwen_result = state.get("qwen_result", "")
    gemma_raw   = state.get("gemma_raw", "")

    messages = [
        SystemMessage(content=_flash_system_prompt),
        HumanMessage(content=original_user_context),
        AIMessage(content=gemma_raw),           # Gemma's first pass (routing decision)
        HumanMessage(content=(
            f"VISUAL EXTRACTION RESULT FROM QWEN:\n{qwen_result}\n\n"
            f"Now produce your final answer using this information."
        )),
    ]

    try:
        ai_msg = gemma_lm.invoke(messages)
        raw_text = ai_msg.content
    except Exception as exc:
        return {
            "errors": [f"gemma_synthesize: inference failed: {exc}"],
            "gemma_answer": f"[Synthesis failed: {exc}]",
        }

    # Gemma may return plain text or JSON here — try JSON first
    parsed = parse_agent_json(raw_text)
    if parsed and parsed.get("gemma_answer"):
        final_text = parsed["gemma_answer"]
    else:
        # Plain text answer is acceptable at synthesis stage
        final_text = raw_text

    return {"gemma_answer": final_text}
```

---

#### Node 6: `flash/nodes/collect.py`

**What it does:**
- Pure rule-based collection. **No model call.**
- Gathers completed outputs from `gemma_answer` and/or `qwen_result` (when final=True).
- Restores original request order (Gemma answer first, Qwen supplement second for Case D).
- Builds the `final_response` string.
- Updates telemetry with end timestamp and total time.

```python
"""flash/nodes/collect.py"""
from __future__ import annotations
import time
from typing import Dict, Any

from flash.state import FlashState


def collect_node(state: FlashState) -> Dict[str, Any]:
    """
    Rule-based response collection. No LLM inference.

    Cases:
      A → gemma_answer only
      B → qwen_result only (final=True)
      C → gemma_answer (synthesized after Qwen extraction)
      D → gemma_answer + qwen_result (final=True), merged in order

    Order: Gemma answer always precedes Qwen supplement (original request order).
    """
    case         = state.get("flash_case", "A").upper()
    gemma_answer = state.get("gemma_answer")
    qwen_result  = state.get("qwen_result")
    qwen_task    = state.get("qwen_task") or {}
    qwen_final   = qwen_task.get("final", True)

    outputs = []

    if case == "A":
        if gemma_answer:
            outputs.append({"order": 0, "text": gemma_answer, "source": "gemma"})

    elif case == "B":
        # Qwen answered directly, no Gemma final text needed
        if qwen_result:
            outputs.append({"order": 0, "text": qwen_result, "source": "qwen"})

    elif case == "C":
        # Gemma synthesized using Qwen extraction
        if gemma_answer:
            outputs.append({"order": 0, "text": gemma_answer, "source": "gemma"})

    elif case == "D":
        # Both independent answers — preserve original order
        if gemma_answer:
            outputs.append({"order": 0, "text": gemma_answer, "source": "gemma"})
        if qwen_result and qwen_final:
            outputs.append({"order": 1, "text": qwen_result, "source": "qwen"})

    # Sort by order and join
    outputs.sort(key=lambda x: x["order"])
    final_response = "\n\n".join(o["text"] for o in outputs if o.get("text"))

    if not final_response:
        final_response = "[No response generated]"

    # Telemetry
    telemetry = dict(state.get("telemetry", {}))
    telemetry["end_time"]       = time.time()
    telemetry["total_wall_sec"] = telemetry["end_time"] - telemetry.get("start_time", 0)
    telemetry["flash_case"]     = case

    return {
        "collected_outputs": outputs,
        "final_response":    final_response,
        "telemetry":         telemetry,
    }
```

---

### PHASE 6 — Graph Assembly
**Time: ~1 day**  
**Goal: Wire all nodes into a LangGraph `StateGraph` with conditional routing,
partial function injection, and Case D parallel fork.**

#### [NEW] `flash/graph.py`

```python
"""
flash/graph.py

Assembles the SAGE Flash Mode LangGraph StateGraph.

Graph topology:
  ingest → gemma_reason → [branch_router conditional edge]
                              ↓ Case A          ↓ Cases B/C/D
                           collect           qwen
                              ↓             ↓ (final=True → collect)
                           kaggle_aux     ↓ (final=False → gemma_synthesize)
                              ↓           gemma_synthesize → collect
                             END                  ↓
                                            kaggle_aux → END

Case D uses LangGraph's Send API to run gemma_answer (already in state from
gemma_reason) and qwen_node concurrently. The collect_node waits for both.

Node injection:
  gemma_lm and qwen_lm (LlamaCppChatAdapter instances) are injected into
  node functions via functools.partial at graph build time. This avoids
  global state while keeping LangGraph's node(state) → dict contract.
"""

from __future__ import annotations
import functools
from pathlib import Path

from langgraph.graph import StateGraph, END

import config
from flash.state import FlashState
from flash.dual_model_manager import dual_model_manager
from flash.lc_adapters import LlamaCppChatAdapter
from flash.nodes.ingest import ingest_node
from flash.nodes.gemma_reason import gemma_reason_node
from flash.nodes.branch_router import branch_router
from flash.nodes.qwen_worker import qwen_node
from flash.nodes.gemma_synthesize import gemma_synthesize_node
from flash.nodes.collect import collect_node
from flash.nodes.kaggle_auxiliary import kaggle_auxiliary_node


# ── Load system prompts once at module import ────────────────────────────────

def _load_prompt(filename: str) -> str:
    p = config.PROMPTS_DIR / filename
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""

_flash_system_prompt = _load_prompt("flash_system.txt")
_qwen_system_prompt  = _load_prompt("qwen_flash_system.txt")


# ── Build LangChain adapters (uses dual_model_manager for URLs) ──────────────

def _make_adapters() -> tuple[LlamaCppChatAdapter, LlamaCppChatAdapter]:
    gemma_cfg = config.FLASH_MODELS["gemma_flash"]
    qwen_cfg  = config.FLASH_MODELS["qwen_flash"]

    gemma_lm = LlamaCppChatAdapter(
        base_url=dual_model_manager.gemma_url(),
        model_display_name=gemma_cfg["name"],
        temperature=gemma_cfg["temperature"],
        max_tokens=gemma_cfg["max_tokens"],
        extract_thinking=True,
    )
    qwen_lm = LlamaCppChatAdapter(
        base_url=dual_model_manager.qwen_url(),
        model_display_name=qwen_cfg["name"],
        temperature=qwen_cfg["temperature"],
        max_tokens=qwen_cfg["max_tokens"],
        extract_thinking=False,
    )
    return gemma_lm, qwen_lm


def _qwen_routing_edge(state: FlashState) -> str:
    """After qwen_node: route to collect (final=True) or gemma_synthesize (final=False)."""
    qwen_task = state.get("qwen_task") or {}
    return "collect" if qwen_task.get("final", True) else "gemma_synthesize"


def build_flash_graph() -> object:
    """Build and compile the Flash Mode StateGraph. Call once at app startup."""
    gemma_lm, qwen_lm = _make_adapters()

    # Inject model adapters into nodes via partial
    _gemma_reason    = functools.partial(gemma_reason_node,    gemma_lm=gemma_lm)
    _qwen_worker     = functools.partial(qwen_node,            qwen_lm=qwen_lm)
    _gemma_synthesize = functools.partial(gemma_synthesize_node, gemma_lm=gemma_lm)

    g = StateGraph(FlashState)

    # Register nodes
    g.add_node("ingest",            ingest_node)
    g.add_node("gemma_reason",      _gemma_reason)
    g.add_node("qwen",              _qwen_worker)
    g.add_node("gemma_synthesize",  _gemma_synthesize)
    g.add_node("collect",           collect_node)
    g.add_node("kaggle_auxiliary",  kaggle_auxiliary_node)

    # Entry point
    g.set_entry_point("ingest")

    # Linear: ingest → gemma_reason
    g.add_edge("ingest", "gemma_reason")

    # Conditional branch after Gemma routing decision
    g.add_conditional_edges(
        "gemma_reason",
        branch_router,
        {
            "collect": "collect",   # Case A
            "qwen":    "qwen",      # Cases B, C, D
        }
    )

    # Conditional after Qwen: final=True → collect, final=False → gemma_synthesize
    g.add_conditional_edges(
        "qwen",
        _qwen_routing_edge,
        {
            "collect":          "collect",
            "gemma_synthesize": "gemma_synthesize",
        }
    )

    # Case C second Gemma pass → collect
    g.add_edge("gemma_synthesize", "collect")

    # Collect → Kaggle auxiliary → END
    g.add_edge("collect", "kaggle_auxiliary")
    g.add_edge("kaggle_auxiliary", END)

    return g.compile()


# ── Singleton compiled graph ─────────────────────────────────────────────────
# Built lazily on first call to avoid import-time errors if models not yet loaded.
_flash_graph = None

def get_flash_graph():
    global _flash_graph
    if _flash_graph is None:
        _flash_graph = build_flash_graph()
    return _flash_graph
```

> [!NOTE]
> **Case D Parallel Execution**: True concurrent Gemma+Qwen execution for Case D requires
> LangGraph's `Send` API with a fan-out pattern. In the initial implementation, Case D
> executes sequentially (Gemma answer is already in state from `gemma_reason_node`, then
> Qwen runs). Full parallel fork using `Send` can be added in a follow-up iteration once
> the sequential graph is validated. The user-visible result is identical either way;
> parallelism only affects latency.

---

### PHASE 7 — Kaggle Auxiliary Lane Client
**Time: ~1 day**  
**Goal: Async REST client for the Cloudflare-tunneled Kaggle GPU server.**

#### [NEW] `flash/kaggle_client.py`

```python
"""
flash/kaggle_client.py

Async HTTP client for the Kaggle/GPU auxiliary lane.

The Kaggle server is accessed via a Cloudflare tunnel URL that the user generates
on the server device. The URL and API key are injected via environment variables:
  SAGE_KAGGLE_URL      — e.g. https://abc123.trycloudflare.com
  SAGE_KAGGLE_API_KEY  — Bearer token for authorization

API Contract (expected from Kaggle server):
  POST /auxiliary/memory
    Body: {"session_id": str, "user_message": str, "entities": list, ...}
    Response: {"task_id": str, "status": "accepted"}

  POST /auxiliary/retrieve
    Body: {"query": str, "session_id": str, "top_k": int}
    Response: {"task_id": str, "results": [...], "status": "ok"}

  GET /auxiliary/task/{task_id}
    Response: {"task_id": str, "status": "pending"|"done", "result": {...}}

If SAGE_KAGGLE_URL is empty, all methods silently no-op. Flash Mode works
fully locally without the remote server.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Any, Dict, Optional

import httpx

import config

logger = logging.getLogger(__name__)


class KaggleAuxiliaryClient:
    """
    Non-blocking async client for the Kaggle-hosted retrieval/memory lane.

    Usage:
      client = KaggleAuxiliaryClient()

      # Fire and forget (background memory work — does not block inference)
      task_id = await client.fire_and_forget({...})

      # Hard dependency (Gemma needs retrieval before continuing)
      result = await client.retrieve(query="...", session_id="...")
    """

    def __init__(self) -> None:
        self.base_url = config.KAGGLE_URL
        self.api_key  = config.KAGGLE_API_KEY
        self.timeout  = config.KAGGLE_TIMEOUT
        self._enabled = bool(self.base_url and self.api_key)

        if self._enabled:
            logger.info("Kaggle auxiliary lane: %s", self.base_url)
        else:
            logger.info("Kaggle auxiliary lane: DISABLED (SAGE_KAGGLE_URL not set)")

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-SAGE-Client": "flash-mode",
        }

    async def fire_and_forget(self, payload: Dict[str, Any]) -> Optional[str]:
        """
        POST memory/indexing work to Kaggle. Returns task_id.
        Does NOT await completion — returns immediately after POST is accepted.

        Used for:
          - Background memory summarisation
          - Memory categorisation / fact extraction
          - Async RAG chunk indexing
        """
        if not self._enabled:
            return None

        url = f"{self.base_url}/auxiliary/memory"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json=payload, headers=self._headers())
                resp.raise_for_status()
                data = resp.json()
                return data.get("task_id")
        except Exception as exc:
            logger.warning("Kaggle fire_and_forget failed (non-critical): %s", exc)
            return None

    async def retrieve(
        self,
        query: str,
        session_id: str,
        top_k: int = 5,
    ) -> Optional[Dict[str, Any]]:
        """
        Synchronous retrieval from Kaggle (hard dependency path).
        Called only when Gemma explicitly needs retrieved context before continuing.
        Awaited — blocks until result arrives or timeout.
        """
        if not self._enabled:
            return None

        url = f"{self.base_url}/auxiliary/retrieve"
        payload = {"query": query, "session_id": session_id, "top_k": top_k}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=payload, headers=self._headers())
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            logger.error("Kaggle retrieve failed: %s", exc)
            return None

    async def is_available(self) -> bool:
        """Health check for the Kaggle endpoint."""
        if not self._enabled:
            return False
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{self.base_url}/health",
                    headers=self._headers()
                )
                return resp.status_code == 200
        except Exception:
            return False


# Singleton
kaggle_client = KaggleAuxiliaryClient()
```

#### [NEW] `flash/nodes/kaggle_auxiliary.py`

```python
"""flash/nodes/kaggle_auxiliary.py"""
from __future__ import annotations
import asyncio
from typing import Dict, Any

from flash.state import FlashState
from flash.kaggle_client import kaggle_client


def kaggle_auxiliary_node(state: FlashState) -> Dict[str, Any]:
    """
    Kaggle auxiliary lane node. Runs AFTER collect_node.

    Two sub-operations:
      1. Fire-and-forget: Send session context / entities to Kaggle for
         background memory summarisation and RAG indexing (non-blocking).

      2. Hard retrieval (only if retrieval_requested=True AND retrieval_result
         is still None): Await Kaggle retrieval. In practice this should have
         been resolved BEFORE gemma_reason_node if Gemma flagged it. This node
         handles the rare case where it wasn't prefetched.

    If SAGE_KAGGLE_URL is not set, this node is a no-op and completes instantly.
    """
    updates: Dict[str, Any] = {}

    # Build memory payload for background work
    memory_payload = {
        "session_id":   state.get("session_id", ""),
        "user_message": state.get("user_message", ""),
        "final_response": state.get("final_response", ""),
        "flash_case":   state.get("flash_case", ""),
        "telemetry":    state.get("telemetry", {}),
    }

    async def _async_work() -> None:
        task_id = await kaggle_client.fire_and_forget(memory_payload)
        if task_id:
            updates["kaggle_task_id"] = task_id

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Inside an async context (FastAPI lifespan)
            asyncio.ensure_future(_async_work())
        else:
            loop.run_until_complete(_async_work())
    except Exception as exc:
        updates["errors"] = [f"kaggle_auxiliary: {exc}"]

    return updates
```

---

### PHASE 8 — System Prompts
**Time: ~1 day**  
**Goal: Write the Gemma Flash routing prompt that reliably produces valid JSON for all 4 cases.**

#### [NEW] `prompts/flash_system.txt`

```
You are SAGE Flash — the reasoning and routing core of SAGE AI.
You run on Gemma 4 E2B with thinking enabled. Your private reasoning is in your <think> block and will never reach the user.

━━━ YOUR ROLE ━━━
You handle every request. You do all text reasoning yourself. You delegate only visual tasks to Qwen3-VL (your visual worker).

━━━ OUTPUT FORMAT ━━━
After thinking, produce EXACTLY ONE valid JSON object. No markdown, no code fences.

CASE A — Text only. You answer directly.
{
  "flash_case": "A",
  "gemma_answer": "Your complete answer here."
}

CASE B — Visual only. Qwen can answer directly without your further reasoning.
{
  "flash_case": "B",
  "qwen": {
    "request": "Exact instruction for Qwen. Be specific about what to extract.",
    "final": true
  }
}

CASE C — Visual extraction needed before you can reason.
{
  "flash_case": "C",
  "qwen": {
    "request": "Exact instruction for Qwen. Extract the specific information you need.",
    "final": false
  }
}

CASE D — Independent text and visual subtasks. Answer your text part immediately.
{
  "flash_case": "D",
  "gemma_answer": "Your answer to the text part of the request.",
  "qwen": {
    "request": "Exact instruction for Qwen for the visual part.",
    "final": true
  }
}

━━━ ROUTING RULES ━━━
- Use A for all text, code, reasoning, summarisation, and math — even if an image is present but the question is about text.
- Use B for pure OCR or visual description where no further reasoning is needed.
- Use C when you need Qwen to extract visual data BEFORE you can answer. Example: "Read this error from the screenshot and explain why it happened."
- Use D when the request has INDEPENDENT text AND visual subtasks. Example: "Why is the sky blue, and what text is in this image?"

━━━ STRICT RULES ━━━
- Never invent file paths, doc IDs, image byte data, or memory contents.
- Never hallucinate visual content. If you cannot see an image, say so.
- If Qwen is not needed, omit the "qwen" field entirely (do not include "qwen": null).
- "final": false means Qwen's output is intermediate and you will receive it for further reasoning.
- "final": true means Qwen's output goes directly to the user without your review.
- Your gemma_answer must be complete, final, and ready to show the user as-is.
- If memory retrieval is needed before answering: include "retrieval_needed": true, "retrieval_query": "your query".
```

#### [NEW] `prompts/qwen_flash_system.txt`

```
You are Qwen3-VL, the visual worker of SAGE Flash.

You receive an instruction and one or more images.

Your job: extract exactly what is asked. Be precise and complete.

Rules:
- Return plain text unless the instruction says otherwise.
- Do not add reasoning, commentary, or context not visible in the image.
- Do not hallucinate content. If something is unclear, say "unclear" for that specific part.
- If asked to extract text (OCR), reproduce it verbatim including formatting.
- If asked to describe, describe only what is visually present.
- Do not generate JSON unless explicitly asked.
```

---

### PHASE 9 — FastAPI Integration
**Time: ~0.5 day**  
**Goal: Add `/api/flash` endpoint to existing `app.py`. All existing endpoints unchanged.**

#### [MODIFY] `app.py`

Changes:
1. Import `dual_model_manager` and start it in lifespan.
2. Add `/api/flash` and `/api/flash/status` endpoints.
3. Add a `FLASH_MODE_ENABLED` guard so Flash endpoints are only active if the models exist.

```python
# Add these imports at the top of app.py (after existing imports):
import config as _config

_FLASH_MODE_ENABLED = bool(
    _config.FLASH_MODELS.get("gemma_flash") and
    _config.FLASH_MODELS.get("qwen_flash")
)

if _FLASH_MODE_ENABLED:
    from flash.dual_model_manager import dual_model_manager
    from flash.graph import get_flash_graph
    from flash.api_models import FlashRequest, FlashResponse
    from flash.kaggle_client import kaggle_client


# Modify the lifespan context manager:
@asynccontextmanager
async def lifespan(app: FastAPI):
    cleanup_temp_dirs()
    print(f"SAGE initialized. Static dir: {config.STATIC_DIR}")

    if _FLASH_MODE_ENABLED:
        try:
            dual_model_manager.start_all()
            print("[Flash] Both models resident. Flash Mode active.")
        except Exception as e:
            print(f"[Flash] WARNING: Could not start Flash models: {e}")
            print("[Flash] Flash Mode disabled. Legacy mode still available.")

    yield

    print("Shutting down SAGE...")
    model_manager.stop_current()   # Legacy model (unchanged)
    if _FLASH_MODE_ENABLED:
        dual_model_manager.stop_all()


# Add these endpoints:

@app.get("/api/flash/status")
async def flash_status():
    if not _FLASH_MODE_ENABLED:
        return {"flash_mode": "disabled", "reason": "Flash model files not configured"}
    kaggle_ok = await kaggle_client.is_available()
    return {
        **dual_model_manager.status(),
        "kaggle_auxiliary": {
            "enabled": bool(config.KAGGLE_URL),
            "url": config.KAGGLE_URL or "(not configured)",
            "healthy": kaggle_ok,
        }
    }


@app.post("/api/flash", response_model=None)
async def flash_endpoint(
    objective: str = Form(...),
    files: Optional[List[UploadFile]] = File(None),
    session_id: Optional[str] = Form(None),
):
    """
    Flash Mode inference endpoint.

    Uses the LangGraph Flash graph: Gemma + Qwen dual-resident models.
    Both models stay loaded. No model switching.
    """
    if not _FLASH_MODE_ENABLED:
        raise HTTPException(503, "Flash Mode is not available on this instance.")

    # Reuse existing file ingestion logic from /api/chat
    request_id   = f"flash_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    req_temp_dir = config.TEMP_DIR / request_id
    req_temp_dir.mkdir(parents=True, exist_ok=True)
    effective_session_id = (
        session_id.strip() if session_id and session_id.strip()
        else f"sess_{uuid.uuid4().hex[:12]}"
    )

    attachments_manifest = []
    file_map = {}

    if files:
        for idx, file_item in enumerate(files, start=1):
            if not file_item.filename:
                continue
            ref_id      = f"file_{idx}"
            safe_name   = Path(file_item.filename).name
            save_path   = req_temp_dir / safe_name
            content     = await file_item.read()
            save_path.write_bytes(content)
            suffix      = Path(safe_name).suffix.lstrip(".").lower()
            file_size   = len(content)
            doc_id      = None
            ingest_err  = None
            try:
                from db_service import document_db
                res = document_db.ingest_document(str(save_path))
                doc_id = res.get("doc_id") if isinstance(res, dict) else None
            except Exception as exc:
                ingest_err = str(exc)

            entry = {
                "ref": ref_id, "doc_id": doc_id, "name": safe_name,
                "type": suffix, "size": file_size, "path": str(save_path),
                "status": "ingested" if doc_id else "ingestion_failed",
            }
            if ingest_err:
                entry["error"] = ingest_err
            attachments_manifest.append(entry)
            file_map[ref_id] = entry

    # Build initial FlashState
    initial_state = {
        "user_message": objective.strip(),
        "attachments":  attachments_manifest,
        "file_map":     file_map,
        "session_id":   effective_session_id,
        "shared_pool":  {},
        "collected_outputs": [],
        "errors":       [],
        "telemetry":    {},
        "retrieval_requested": False,
    }

    try:
        graph = get_flash_graph()
        result = graph.invoke(initial_state)
        return JSONResponse({
            "status":       "success",
            "answer":       result.get("final_response", ""),
            "flash_case":   result.get("flash_case", "?"),
            "session_id":   effective_session_id,
            "telemetry":    result.get("telemetry", {}),
            "kaggle_task":  result.get("kaggle_task_id"),
            "errors":       result.get("errors", []),
        })
    except Exception as e:
        import traceback
        return JSONResponse(status_code=500, content={
            "status": "error", "error": str(e),
            "traceback": traceback.format_exc()
        })
```

---

### PHASE 10 — Tests
**Time: ~1.5 days**  
**Goal: Verify all cases in mock mode (no GPU needed) and validate VRAM on hardware.**

#### [NEW] `tests/flash/test_graph_mock.py`

```python
"""
End-to-end Flash graph test using SAGE_MOCK_MODE=1.
Mock LangChain adapters return canned JSON responses without calling llama-server.
"""
import os
os.environ["SAGE_MOCK_MODE"] = "1"

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage

# Mock gemma_lm and qwen_lm before importing graph
mock_gemma = MagicMock()
mock_qwen  = MagicMock()

CASE_A_JSON = '{"flash_case":"A","gemma_answer":"The sky is blue because of Rayleigh scattering."}'
CASE_B_JSON = '{"flash_case":"B","qwen":{"request":"Extract all text from this image.","final":true}}'
CASE_C_JSON = '{"flash_case":"C","qwen":{"request":"Read the error message from this screenshot.","final":false}}'
CASE_D_JSON = '{"flash_case":"D","gemma_answer":"Blue because Rayleigh.","qwen":{"request":"What text is in the image?","final":true}}'


def make_ai(text: str) -> AIMessage:
    return AIMessage(content=text, additional_kwargs={"thinking": "", "duration": 0.1, "usage": {}})


@pytest.fixture
def run_graph_with_case(tmp_path):
    """Factory: run flash graph with mocked LLMs for a given case."""
    from flash.nodes.ingest import ingest_node
    from flash.nodes.gemma_reason import gemma_reason_node
    from flash.nodes.branch_router import branch_router
    from flash.nodes.qwen_worker import qwen_node
    from flash.nodes.gemma_synthesize import gemma_synthesize_node
    from flash.nodes.collect import collect_node
    import functools

    def _run(gemma_response: str, qwen_response: str, state_overrides: dict = None):
        mock_gemma.invoke.return_value = make_ai(gemma_response)
        mock_qwen.invoke.return_value  = make_ai(qwen_response)

        initial = {
            "user_message": "test question",
            "attachments": [], "file_map": {}, "shared_pool": {},
            "collected_outputs": [], "errors": [], "telemetry": {},
            "retrieval_requested": False,
            **(state_overrides or {})
        }

        # Run nodes manually (unit-level, not full graph)
        state = dict(initial)
        state.update(ingest_node(state))
        state.update(functools.partial(gemma_reason_node, gemma_lm=mock_gemma)(state))
        next_node = branch_router(state)
        if next_node == "qwen":
            state.update(functools.partial(qwen_node, qwen_lm=mock_qwen)(state))
            qwen_next = "collect" if state.get("qwen_task", {}).get("final", True) else "gemma_synthesize"
            if qwen_next == "gemma_synthesize":
                state.update(functools.partial(gemma_synthesize_node, gemma_lm=mock_gemma)(state))
        state.update(collect_node(state))
        return state

    return _run


def test_case_a(run_graph_with_case):
    state = run_graph_with_case(CASE_A_JSON, "")
    assert state["flash_case"] == "A"
    assert "Rayleigh" in state["final_response"]
    assert state.get("qwen_result") is None


def test_case_b(run_graph_with_case):
    state = run_graph_with_case(CASE_B_JSON, "The text says: HELLO WORLD")
    assert state["flash_case"] == "B"
    assert "HELLO WORLD" in state["final_response"]


def test_case_c(run_graph_with_case):
    synth_response = '{"gemma_answer":"The error is a NullPointerException at line 42."}'
    mock_gemma.invoke.side_effect = [
        make_ai(CASE_C_JSON),        # first call: routing
        make_ai(synth_response),     # second call: synthesis
    ]
    state = run_graph_with_case(CASE_C_JSON, "Error: NullPointerException at line 42")
    assert state["flash_case"] == "C"
    assert "NullPointerException" in state["final_response"]


def test_case_d(run_graph_with_case):
    state = run_graph_with_case(CASE_D_JSON, "The image says: OPEN ME")
    assert state["flash_case"] == "D"
    assert "Rayleigh" in state["final_response"]
    assert "OPEN ME" in state["final_response"]


def test_response_order_case_d(run_graph_with_case):
    """Gemma answer must appear before Qwen answer in Case D."""
    state = run_graph_with_case(CASE_D_JSON, "Image text: SECOND")
    response = state["final_response"]
    gemma_pos = response.find("Rayleigh")
    qwen_pos  = response.find("SECOND")
    assert gemma_pos < qwen_pos, "Gemma answer must precede Qwen answer in Case D"
```

**Run tests:**
```bash
cd c:\Users\ojasv\Desktop\SAGE\SAGE
SAGE_MOCK_MODE=1 pytest tests/flash/ -v
```

**VRAM validation (GPU required):**
```bash
# After app startup with Flash Mode active:
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
# Expected: ~6000 MiB used, ~2000 MiB free
```

---

## 7. Known Issues and Open Problems

> [!WARNING]
> **Port conflict with legacy orchestrator**: The legacy `model_manager.py` also uses
> port 8080 for its single llama-server instance. If both the legacy mode and Flash Mode
> are active simultaneously, they will conflict. The startup guard in `app.py` mitigates
> this but does not prevent it if someone manually starts a legacy server on 8080.
> **Fix**: Move the legacy server to port 8082 in `config.py` (`LLAMA_PORT = 8082`) in
> a follow-up task. For now, do not run legacy and Flash endpoints simultaneously.

> [!WARNING]
> **Case D is sequentially implemented in Phase 1**: True concurrent Gemma+Qwen execution
> requires LangGraph's `Send` API fan-out. The initial graph runs D sequentially
> (Gemma answer is in state from reason node, then Qwen runs). The result is the same;
> only latency differs. True parallelism should be a Phase 2 iteration task.

> [!WARNING]
> **Kaggle server API contract is not yet implemented**: The `KaggleAuxiliaryClient`
> assumes a specific REST API on the Kaggle side (`/auxiliary/memory`, `/auxiliary/retrieve`,
> `/auxiliary/task/{id}`). The Kaggle server must implement these endpoints. The client
> gracefully degrades if unavailable, so Flash Mode works locally before the Kaggle server
> is ready.

> [!NOTE]
> **Gemma JSON reliability**: Gemma thinking mode may occasionally produce JSON with
> embedded reasoning text. The 1-turn JSON repair (`build_repair_prompt`) from
> `core/json_repair.py` is reused. In the worst case, a plain-text fallback answer
> is used to avoid hard failures. The Flash system prompt's strict formatting
> instructions should make repair unnecessary for most requests.

> [!NOTE]
> **Image data in `shared_pool`**: The current design loads images as base64 data URLs
> for Qwen. For large images this increases memory pressure. A future optimisation is
> to pass image paths directly to llama-server via the file-path API if your
> llama-server build supports it (`--image-path` argument).

> [!NOTE]
> **`asyncio.get_event_loop()` in kaggle_auxiliary_node**: FastAPI uses an event loop
> that is already running. The `asyncio.ensure_future()` pattern used in
> `kaggle_auxiliary_node` may behave differently depending on whether the graph is
> invoked sync or async. When FastAPI calls `graph.invoke()` (sync), the
> `run_until_complete()` path executes. When called via `graph.ainvoke()` (async),
> use `ensure_future()`. Test both paths.

---

## 8. Phase Schedule

| Phase | What | Files Created/Modified | Est. Time |
|-------|------|------------------------|-----------|
| 0 | Dependencies | `requirements.txt` | 2 hrs |
| 1 | Config extension | `config.py`, `.env.example` | 1 hr |
| 2 | Dual Model Manager | `flash/dual_model_manager.py` | 1 day |
| 3 | LangChain Adapters | `flash/lc_adapters.py` | 1 day |
| 4 | Flash State Schema | `flash/state.py` | 2 hrs |
| 5 | All 6 Graph Nodes | `flash/nodes/*.py` | 2 days |
| 6 | Graph Assembly | `flash/graph.py` | 1 day |
| 7 | Kaggle Client | `flash/kaggle_client.py`, `flash/nodes/kaggle_auxiliary.py` | 1 day |
| 8 | System Prompts | `prompts/flash_system.txt`, `prompts/qwen_flash_system.txt` | 1 day |
| 9 | FastAPI Integration | `app.py` | 4 hrs |
| 10 | Tests + Validation | `tests/flash/*.py` | 1.5 days |
| **Total** | | | **~10–11 days** |

---

## 9. Quick-Start Checklist (for executing AI)

```
[ ] Phase 0: pip install langgraph langchain-core langchain-community anyio
[ ] Phase 1: Append FLASH_MODELS and KAGGLE_* to config.py
[ ] Phase 1: Add SAGE_KAGGLE_URL and SAGE_KAGGLE_API_KEY to .env.example
[ ] Phase 2: Create flash/__init__.py (empty)
[ ] Phase 2: Create flash/dual_model_manager.py (full implementation above)
[ ] Phase 3: Create flash/lc_adapters.py (full implementation above)
[ ] Phase 4: Create flash/state.py (full implementation above)
[ ] Phase 5: Create flash/nodes/__init__.py (empty)
[ ] Phase 5: Create flash/nodes/ingest.py
[ ] Phase 5: Create flash/nodes/gemma_reason.py
[ ] Phase 5: Create flash/nodes/branch_router.py
[ ] Phase 5: Create flash/nodes/qwen_worker.py
[ ] Phase 5: Create flash/nodes/gemma_synthesize.py
[ ] Phase 5: Create flash/nodes/collect.py
[ ] Phase 6: Create flash/graph.py
[ ] Phase 7: Create flash/kaggle_client.py
[ ] Phase 7: Create flash/nodes/kaggle_auxiliary.py
[ ] Phase 8: Create prompts/flash_system.txt
[ ] Phase 8: Create prompts/qwen_flash_system.txt
[ ] Phase 9: Modify app.py (add Flash imports + lifespan + /api/flash endpoints)
[ ] Phase 10: Create tests/flash/__init__.py + test files
[ ] Phase 10: SAGE_MOCK_MODE=1 pytest tests/flash/ -v
[ ] Phase 10: nvidia-smi after startup → confirm ~6 GB used
[ ] Phase 10: POST /api/flash with text → confirm Case A
[ ] Phase 10: POST /api/flash with image → confirm Case B/C
[ ] Phase 10: POST /api/flash with text+image → confirm Case D, response order correct
```
