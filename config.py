import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp"
PROMPTS_DIR = BASE_DIR / "prompts"
STATIC_DIR = BASE_DIR / "static"

# Ensure directories exist
TEMP_DIR.mkdir(parents=True, exist_ok=True)
(TEMP_DIR / "logs").mkdir(parents=True, exist_ok=True)
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Llama Server Executable
LLAMA_SERVER_PATH = r"C:\Users\sarth\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe"

# Model Directory
MODEL_DIR = r"C:\Users\sarth\OneDrive\Desktop\model"

# Server Ports and Host
SERVER_HOST = "127.0.0.1"
LLAMA_PORT = 8080
APP_PORT = 8899
LLAMA_BASE_URL = f"http://{SERVER_HOST}:{LLAMA_PORT}"

# Orchestrator Configuration
MAX_AGENT_LOOPS = 8
MODEL_START_TIMEOUT = 90  # Seconds to wait for /health
REQUEST_TIMEOUT = 180.0   # HTTP timeout for inference calls

# Per-Model Configurations
MODELS = {
    "agent": {
        "key": "agent",
        "name": "Gemma 4B Instruct",
        "model_path": os.path.join(MODEL_DIR, "gemma-4-E4B-it-Q4_K_M.gguf"),
        "mmproj": None,
        "context": 8192,
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
    }
}
