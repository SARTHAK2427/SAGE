import os
import shutil
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp"
PROMPTS_DIR = BASE_DIR / "prompts"
STATIC_DIR = BASE_DIR / "static"
ARTIFACTS_ROOT = Path(os.environ.get("SAGE_ARTIFACTS_ROOT", str(BASE_DIR / "artifacts")))
MODEL_RUNTIME_ROOT = ARTIFACTS_ROOT / "model_runtime"
CHROMA_ROOT = Path(os.environ.get("SAGE_CHROMA_ROOT", str(BASE_DIR / "chroma_db")))

# .env loading (prioritizes local .env file)
_env_file = BASE_DIR / ".env"
if _env_file.is_file():
    try:
        import dotenv
        dotenv.load_dotenv(dotenv_path=_env_file, override=True)
    except Exception:
        try:
            with open(_env_file, "r", encoding="utf-8") as _f:
                for _line in _f:
                    _line = _line.strip()
                    if not _line or _line.startswith("#") or "=" not in _line:
                        continue
                    _k, _v = _line.split("=", 1)
                    _k = _k.strip()
                    _v = _v.strip().strip("'\"")
                    if _k:
                        os.environ[_k] = _v
        except Exception:
            pass

# Gemma Context Window
GEMMA_CONTEXT = int(os.environ.get("GEMMA_CONTEXT", "16384"))

# Single-User Identity Namespace & Memory DB Config
DEFAULT_USER_ID = os.environ.get("SAGE_DEFAULT_USER_ID", "local_user")
SAGE_MEMORY_DB = os.environ.get("SAGE_MEMORY_DB", "sqlite").lower().strip()
SAGE_MEMORY_COLLECTION = os.environ.get("SAGE_MEMORY_COLLECTION", "sage_memory")
SAGE_MEMORY_HOT_COLLECTION = os.environ.get("SAGE_MEMORY_HOT_COLLECTION", "sage_memory_hot")
SAGE_MEMORY_COLD_COLLECTION = os.environ.get("SAGE_MEMORY_COLD_COLLECTION", "sage_memory_cold")
SAGE_DATABASE_URL = (
    os.environ.get("SAGE_DATABASE_URL")
    or os.environ.get("DATABASE_URL")
    or ""
)

# Ensure directories exist
TEMP_DIR.mkdir(parents=True, exist_ok=True)
(TEMP_DIR / "logs").mkdir(parents=True, exist_ok=True)
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
MODEL_RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
CHROMA_ROOT.mkdir(parents=True, exist_ok=True)

# Upload and Security Boundaries
MAX_UPLOAD_SIZE_BYTES = int(os.environ.get("SAGE_MAX_UPLOAD_SIZE_BYTES", str(50 * 1024 * 1024)))  # 50 MB
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "SAGE_CORS_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000,http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]


# ── Portable Executable & Model Discovery ─────────────────────────────────────

def _discover_llama_server() -> str:
    """Discover llama-server executable path using priority order:
    1. Explicit LLAMA_SERVER_PATH environment variable (if non-empty)
    2. Automatic discovery via system PATH (llama-server, llama-server.exe)
    3. Sensible repo-relative locations if they exist (bin/ or repo root)
    4. Unresolved ("") - component raises clear configuration error when needed
    """
    env_path = os.environ.get("LLAMA_SERVER_PATH", "").strip()
    if env_path:
        return env_path

    # PATH discovery
    for binary_name in ("llama-server", "llama-server.exe"):
        found = shutil.which(binary_name)
        if found:
            return str(Path(found).resolve())

    # Sensible repo-relative candidates
    repo_candidates = [
        BASE_DIR / "bin" / "llama-server.exe",
        BASE_DIR / "bin" / "llama-server",
        BASE_DIR / "llama-server.exe",
        BASE_DIR / "llama-server",
    ]
    for candidate in repo_candidates:
        if candidate.is_file():
            return str(candidate.resolve())

    return ""


LLAMA_SERVER_PATH = _discover_llama_server()


def _discover_model_dir() -> str:
    """Discover model directory using priority order:
    1. Explicit MODEL_DIR environment variable (if non-empty)
    2. Sensible repo-relative locations if they exist (models/ or model/)
    3. Unresolved ("") - component raises clear configuration error when needed
    """
    env_dir = os.environ.get("MODEL_DIR", "").strip()
    if env_dir:
        return env_dir

    repo_candidates = [
        BASE_DIR / "models",
        BASE_DIR / "model",
    ]
    for candidate in repo_candidates:
        if candidate.is_dir():
            return str(candidate.resolve())

    return ""


MODEL_DIR = _discover_model_dir()

# Server Ports and Host
SERVER_HOST = "127.0.0.1"
LLAMA_PORT = 8080
APP_PORT = 8899
LLAMA_BASE_URL = f"http://{SERVER_HOST}:{LLAMA_PORT}"

# Orchestrator Configuration
MAX_AGENT_LOOPS = 8
MODEL_START_TIMEOUT = 90  # Seconds to wait for /health
REQUEST_TIMEOUT = 180.0   # HTTP timeout for inference calls

# Phase 8: bounded automatic context injected before Gemma's first turn.
# These limits apply only to persisted chat context and semantic memories; the
# current user request is always included in full.
SAGE_CONTEXT_MEMORY_BUDGET_TOKENS = int(
    os.environ.get("SAGE_CONTEXT_MEMORY_BUDGET_TOKENS", "1200")
)
SAGE_GLOBAL_MEMORY_BUDGET_TOKENS = int(
    os.environ.get("SAGE_GLOBAL_MEMORY_BUDGET_TOKENS", "600")
)
SAGE_GLOBAL_MEMORY_MAX_ITEMS = int(
    os.environ.get("SAGE_GLOBAL_MEMORY_MAX_ITEMS", "8")
)
SAGE_RECENT_CHAT_MAX_MESSAGES = int(
    os.environ.get("SAGE_RECENT_CHAT_MAX_MESSAGES", "5")
)

# ── Code Execution Sandbox ────────────────────────────────────────────────────
# Docker is the ONLY backend. No subprocess/local-execution fallback.
# If Docker is unavailable the pipeline returns a clear infrastructure error.
SANDBOX = {
    "image": "python:3.12-slim",  # stdlib-only; extend to a custom image later
    "timeout_seconds": 15,        # wall-clock kill limit per execution
    "memory_mb": 256,             # container memory cap (swap disabled)
    "cpu_cores": 1.0,             # nano_cpus quota
    "max_fix_attempts": 2,        # LLM repair cycles after initial failure
}

# Per-Model Configurations
MODELS = {
    "agent": {
        "key": "agent",
        "name": "Gemma 4B Instruct",
        "model_path": os.path.join(MODEL_DIR, "gemma-4-E4B-it-Q4_K_M.gguf"),
        "mmproj": None,
        "context": GEMMA_CONTEXT,
        "ngl": 999,
        "temperature": 0.20,
        "max_tokens": 2048,
        "reasoning": "on",
    },
    "coder": {
        "key": "coder",
        "name": "Qwen2.5-Coder 7B Instruct",
        "model_path": os.path.join(MODEL_DIR, "qwen2.5-coder-7b-instruct-q4_k_m.gguf"),
        "mmproj": None,
        "context": 8192,
        "ngl": 999,
        "temperature": 0.10,
        "max_tokens": 4096,
        "reasoning": "off",
    },
    "document_analyzer": {
        "key": "document_analyzer",
        "name": "Qwen3-VL 4B Document/OCR",
        "model_path": os.path.join(MODEL_DIR, "Qwen3-VL-4B-Instruct-Q4_K_M.gguf"),
        "mmproj": os.path.join(MODEL_DIR, "mmproj-F16.gguf"),
        "context": 4096,
        "ngl": 999,
        "temperature": 0.05,
        "max_tokens": 2048,
        "reasoning": "off",
    },
    "final_synthesizer": {
        "key": "final_synthesizer",
        "name": "Qwen3.5 2B Final Synthesizer",
        "model_path": os.path.join(MODEL_DIR, "qwen3.5-2b-instruct-q4_k_m.gguf"),
        "mmproj": None,
        "context": 8192,
        "ngl": 999,
        "temperature": 0.20,
        "max_tokens": 2048,
        "reasoning": "off",
    },
}

# ── Remote GPU Model Transport Configuration ───────────────────────────────────
SAGE_MODEL_BACKEND = os.environ.get("SAGE_MODEL_BACKEND", "local").lower().strip()
SAGE_REMOTE_GPU_URL = os.environ.get("SAGE_REMOTE_GPU_URL", "").rstrip("/")
SAGE_REMOTE_GPU_API_KEY = os.environ.get("SAGE_REMOTE_GPU_API_KEY", "")
REMOTE_CONNECT_TIMEOUT = float(os.environ.get("SAGE_REMOTE_CONNECT_TIMEOUT", "10.0"))
REMOTE_REQUEST_TIMEOUT = float(os.environ.get("SAGE_REMOTE_REQUEST_TIMEOUT", "180.0"))


def is_remote_backend() -> bool:
    """Return True if SAGE is configured to offload inference to the remote GPU worker."""
    return SAGE_MODEL_BACKEND == "remote"


# ── Flash Mode Runtime ────────────────────────────────────────────────────────
# Flash uses independent ports and never takes ownership of the legacy 8080
# model-manager port. Remote URLs and keys are supplied by the UI and live only
# in process memory; they are deliberately absent from environment defaults.
FLASH_DEFAULT_ENABLED = os.environ.get("SAGE_FLASH_DEFAULT", "1").strip().lower() not in {"0", "false", "no"}
FLASH_GEMMA_PORT = int(os.environ.get("SAGE_FLASH_GEMMA_PORT", "8090"))
FLASH_QWEN_PORT = int(os.environ.get("SAGE_FLASH_QWEN_PORT", "8091"))
FLASH_MEMORY_PORT = int(os.environ.get("SAGE_FLASH_MEMORY_PORT", "8092"))
FLASH_MODEL_START_TIMEOUT = float(os.environ.get("SAGE_FLASH_START_TIMEOUT", "120"))
FLASH_REQUEST_TIMEOUT = float(os.environ.get("SAGE_FLASH_REQUEST_TIMEOUT", "300"))
FLASH_DEPLOY_TIMEOUT = float(os.environ.get("SAGE_FLASH_DEPLOY_TIMEOUT", "900"))
FLASH_VRAM_RESERVE_GIB = float(os.environ.get("SAGE_FLASH_VRAM_RESERVE_GIB", "0.75"))
FLASH_MEMORY_CPU_THREADS = int(os.environ.get("SAGE_FLASH_MEMORY_CPU_THREADS", "4"))
FLASH_MEMORY_QUEUE_SIZE = int(os.environ.get("SAGE_FLASH_MEMORY_QUEUE_SIZE", "32"))

