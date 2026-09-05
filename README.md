# SAGE — Minimal Real Multi-Model Agent Orchestrator

SAGE is a clean, minimal local multi-model agent orchestrator running entirely offline on your GPU/CPU using `llama.cpp` (`llama-server.exe`) and sequential inference.

## Architecture

```
USER
  ↓
PYTHON RUNTIME (FastAPI)
  ↓
GEMMA 4B (Agent / Controller)
  ↓ (JSON Tool Request)
PYTHON
  ↓ (Sequential Model Switch)
SPECIALIST MODEL:
  • Qwen3-VL 4B (Document Analyzer / OCR)
  • Qwen2.5-Coder 7B (Code Specialist)
  ↓
RESULT
  ↓
GEMMA 4B
  ↓
(Repeat until final answer)
```

## Setup & Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configuration:
Edit `config.py` to change model paths, context size, GPU offload layers (`-ngl`), ports, or sampling parameters:
- `LLAMA_SERVER_PATH`: Path to `llama-server.exe`
- `MODELS["agent"]`: Path and parameters for Gemma 4B
- `MODELS["coder"]`: Path and parameters for Qwen2.5-Coder 7B
- `MODELS["document_analyzer"]`: Path and parameters for Qwen3-VL 4B with `mmproj-F16.gguf`

## Running SAGE

Run the application with:
```bash
python app.py
```
or
```bash
uvicorn app:app --host 127.0.0.1 --port 8899
```

Then open your browser at:
```
http://127.0.0.1:8899
```

## Features

- **Sequential GPU Inference**: Runs 1 model at a time to stay strictly within 8GB VRAM budgets.
- **Multimodal Document Processing**: Extracts text and embedded images from PDFs (PyMuPDF) and DOCX (python-docx), and transcribes images via Qwen3-VL.
- **Rich Terminal Trace**: Complete visibility with Rich panels, input/output inspection, timings, and token metrics.
- **Web UI**: Dark developer-oriented interface with file attachments, image thumbnail previews, clipboard pasting, and a real-time Agent Trace sidebar.
