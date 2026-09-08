"""
verify_remote_transport.py

Live remote transport verification script.
Runs against the actual Cloudflare GPU worker using config from .env

Does NOT run document E2E tests.
Does NOT test EXPERIMENT 3.
Does NOT modify any source files.
Does NOT commit/push.

Results are printed to stdout. Exit code 0 = all passed, 1 = any failure.
"""

import io
import os
import sys
import subprocess
import traceback

# Load .env via config module (stdlib-only parser already handles this)
sys.path.insert(0, os.path.dirname(__file__))
import config

PASS = "[PASS]"
FAIL = "[FAIL]"
results = {}


def check(name: str, fn):
    try:
        fn()
        results[name] = (True, None)
        print(f"  {PASS}  {name}")
    except Exception as exc:
        results[name] = (False, exc)
        print(f"  {FAIL}  {name}")
        print(f"      ERROR: {exc}")


print()
print("=" * 65)
print(" SAGE Remote Model Transport - Live Verification")
print("=" * 65)
print(f"  Backend  : {config.SAGE_MODEL_BACKEND}")
print(f"  URL      : {config.SAGE_REMOTE_GPU_URL or '(not set)'}")
print(f"  API Key  : {'(set)' if config.SAGE_REMOTE_GPU_API_KEY else '(NOT SET)'}")
print()

if not config.is_remote_backend():
    print("SAGE_MODEL_BACKEND is not 'remote'.  Set it to 'remote' in .env or environment.")
    sys.exit(1)

if not config.SAGE_REMOTE_GPU_URL:
    print("SAGE_REMOTE_GPU_URL is not set.  Cannot verify remote transport.")
    sys.exit(1)

from core.remote_model_transport import remote_model_transport

# ── 1. Health check ────────────────────────────────────────────────────────────
print("1. GET /health")
def check_health():
    health = remote_model_transport.get_health()
    assert health.get("status") == "ok", f"Unexpected health body: {health}"
    gpu = health.get("gpu", "unknown")
    print(f"       Worker: {health.get('worker')} | GPU: {gpu}")
    vram_free = health.get("vram_free_mb", "?")
    print(f"       VRAM free: {vram_free} MB | Current model: {health.get('current_model')}")
check("GET /health -> status=ok", check_health)

# ── 2. Gemma ───────────────────────────────────────────────────────────────────
print()
print("2. POST /infer/gemma (via remote_model_transport.dispatch)")
def check_gemma():
    result = remote_model_transport.dispatch(
        model_key="agent",
        messages=[{"role": "user", "content": "Reply with exactly three words."}],
        temperature=0.1,
        max_tokens=512,
    )
    assert isinstance(result["content"], str) and result["content"].strip(), \
        f"Empty content from Gemma: {result}"
    print(f"       Content preview: {result['content'][:80]!r}")
    print(f"       Duration: {result['duration']:.2f}s  |  Tokens: {result['usage']}")
check("Gemma via /infer/gemma", check_gemma)

# ── 3. Coder ───────────────────────────────────────────────────────────────────
print()
print("3. POST /infer/coder (via remote_model_transport.dispatch)")
def check_coder():
    messages = [
        {"role": "system", "content": "You are a coding assistant. Return only code."},
        {"role": "user", "content": "Write a Python function named add(a, b) that returns a+b."},
    ]
    result = remote_model_transport.dispatch(
        model_key="coder",
        messages=messages,
        temperature=0.1,
        max_tokens=200,
    )
    assert isinstance(result["content"], str) and result["content"].strip(), \
        f"Empty content from Coder: {result}"
    print(f"       Content preview: {result['content'][:120]!r}")
    print(f"       Duration: {result['duration']:.2f}s  |  Tokens: {result['usage']}")
check("Coder via /infer/coder", check_coder)

# ── 4. Vision ──────────────────────────────────────────────────────────────────
print()
print("4. POST /infer/vision (image read locally, transmitted as base64)")
def check_vision():
    import base64
    from PIL import Image

    # Create a small test image in memory - no filesystem path is transmitted
    img = Image.new("RGB", (64, 64), color="green")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()
    b64_str = base64.b64encode(raw_bytes).decode("utf-8")
    data_uri = f"data:image/png;base64,{b64_str}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What color is the dominant color in this image?"},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }
    ]
    result = remote_model_transport.dispatch(
        model_key="document_analyzer",
        messages=messages,
        temperature=0.05,
        max_tokens=100,
    )
    assert isinstance(result["content"], str) and result["content"].strip(), \
        f"Empty content from Vision: {result}"
    print(f"       Content preview: {result['content'][:120]!r}")
    print(f"       Duration: {result['duration']:.2f}s  |  Tokens: {result['usage']}")
check("Vision via /infer/vision (base64 bytes, not path)", check_vision)

# ── 5. No local llama-server process ──────────────────────────────────────────
print()
print("5. Verifying zero local llama-server processes on this machine")
def check_no_local_process():
    result = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq llama-server.exe", "/FO", "CSV", "/NH"],
        capture_output=True, text=True, timeout=5
    )
    lines = [l for l in result.stdout.strip().splitlines() if "llama-server" in l.lower()]
    print(f"       llama-server processes found: {len(lines)}")
    if lines:
        print(f"       WARNING: {lines}")
    assert len(lines) == 0, f"Local llama-server processes detected: {lines}"
check("Zero local llama-server.exe processes", check_no_local_process)

# ── Summary ────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
passed = sum(1 for ok, _ in results.values() if ok)
failed = sum(1 for ok, _ in results.values() if not ok)
print(f" Results: {passed} passed, {failed} failed")
print("=" * 65)
print()

if failed > 0:
    print("Failed tests:")
    for name, (ok, exc) in results.items():
        if not ok:
            print(f"  - {name}: {exc}")
    sys.exit(1)
else:
    print("All remote transport checks passed.")
    sys.exit(0)
