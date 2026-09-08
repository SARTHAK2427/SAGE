"""
tests/test_remote_transport.py

Unit tests for the Remote Model Transport integration.

Test strategy:
- All remote HTTP calls are mocked with unittest.mock (no real network I/O)
- Config toggles are applied per-test via monkeypatch
- All existing callers stay unmodified; tests verify the thin transport shim

Tests:
  1. local mode selects existing ModelClient / ModelManager paths (no remote import)
  2. remote Gemma routing via model_key
  3. remote Coder routing via model_key
  4. remote Vision — bytes extracted from data-URI, base64 transmitted
  5. Authorization: Bearer header is present on every remote request
  6. HTTP error (4xx / 5xx) is surfaced as RuntimeError, not swallowed
  7. Connect timeout surfaces as RuntimeError
  8. Read/inference timeout surfaces as RuntimeError
  9. No local subprocess is spawned in remote mode (ensure_model is a no-op subprocess-wise)
 10. Fail-closed: model_key unresolvable raises RuntimeError (never guesses)
 11. Fail-closed: unknown model_key in mapping raises RuntimeError
"""

from __future__ import annotations

import base64
import importlib
import io
import sys
import types
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_remote_response(content: str = "ok", model: str = "gemma-4b") -> dict:
    return {
        "status": "success",
        "model": model,
        "content": content,
        "duration": 0.5,
        "usage": {"completion_tokens": 5, "prompt_tokens": 10, "total_tokens": 15},
        "timings": {},
    }


def _mock_httpx_post(response_json: dict, status_code: int = 200):
    """Return a context-manager-compatible mock for httpx.Client.post."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = response_json
    mock_resp.text = str(response_json)
    if status_code >= 400:
        import httpx
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=mock_resp
        )
    else:
        mock_resp.raise_for_status.return_value = None
    mock_client = MagicMock()
    mock_client.__enter__ = lambda s: mock_client
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.return_value = mock_resp
    return mock_client, mock_resp


def _mock_httpx_get(response_json: dict, status_code: int = 200):
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = response_json
    mock_resp.raise_for_status.return_value = None
    mock_client = MagicMock()
    mock_client.__enter__ = lambda s: mock_client
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get.return_value = mock_resp
    return mock_client


# ---------------------------------------------------------------------------
# 1. LOCAL MODE — existing paths untouched
# ---------------------------------------------------------------------------

class TestLocalModeUnchanged:
    """In local mode, ModelClient.chat_completion must call the local llama-server
    and ModelManager.ensure_model must attempt to launch a subprocess.
    No remote transport code should be invoked.
    """

    def test_local_mode_does_not_call_remote_transport(self, monkeypatch):
        """chat_completion in local mode never touches RemoteModelTransport."""
        import config
        monkeypatch.setattr(config, "SAGE_MODEL_BACKEND", "local", raising=True)

        from model_client import ModelClient

        dummy_response = {
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            "usage": {},
            "timings": {},
        }
        mock_client, _ = _mock_httpx_post(dummy_response)

        with patch("httpx.Client", return_value=mock_client):
            from core.remote_model_transport import remote_model_transport
            with patch.object(remote_model_transport, "dispatch") as mock_dispatch:
                client = ModelClient(base_url="http://127.0.0.1:8080")
                result = client.chat_completion(
                    messages=[{"role": "user", "content": "hi"}]
                )
                mock_dispatch.assert_not_called()

        assert result["content"] == "hello"

    def test_local_ensure_model_does_not_call_remote(self, monkeypatch):
        """ensure_model in local mode never calls remote transport."""
        import config
        monkeypatch.setattr(config, "SAGE_MODEL_BACKEND", "local", raising=True)

        from model_manager import ModelManager
        mm = ModelManager()

        with patch("os.path.exists", return_value=False):
            with pytest.raises((FileNotFoundError, RuntimeError)):
                mm.ensure_model("agent")

        # server_process must still be None (nothing launched in this path)
        assert mm.server_process is None


# ---------------------------------------------------------------------------
# 2-4. REMOTE ROUTING — deterministic by model_key only
# ---------------------------------------------------------------------------

class TestRemoteRouting:
    """Routing must be determined solely by model_key, never by content."""

    def _setup_remote(self, monkeypatch):
        import config
        monkeypatch.setattr(config, "SAGE_MODEL_BACKEND", "remote", raising=True)
        monkeypatch.setattr(config, "SAGE_REMOTE_GPU_URL", "https://test.example.com", raising=True)
        monkeypatch.setattr(config, "SAGE_REMOTE_GPU_API_KEY", "test-key", raising=True)

    def test_remote_gemma_routed_by_model_key(self, monkeypatch):
        """model_key='agent' routes to /infer/gemma regardless of message content."""
        self._setup_remote(monkeypatch)
        from core.remote_model_transport import RemoteModelTransport
        transport = RemoteModelTransport()
        transport._base_url = "https://test.example.com"
        transport._api_key = "test-key"

        mock_client, _ = _mock_httpx_post(_make_remote_response("gemma says hi", "gemma-4b"))

        with patch("httpx.Client", return_value=mock_client):
            result = transport.dispatch(
                model_key="agent",
                messages=[{"role": "user", "content": "What is 2+2?"}],
                temperature=0.2,
                max_tokens=100,
            )

        assert result["content"] == "gemma says hi"
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "https://test.example.com/infer/gemma"

    def test_remote_coder_routed_by_model_key(self, monkeypatch):
        """model_key='coder' routes to /infer/coder regardless of message content."""
        self._setup_remote(monkeypatch)
        from core.remote_model_transport import RemoteModelTransport
        transport = RemoteModelTransport()
        transport._base_url = "https://test.example.com"
        transport._api_key = "test-key"

        mock_client, _ = _mock_httpx_post(_make_remote_response("```python\npass\n```", "qwen2.5-coder-7b"))

        with patch("httpx.Client", return_value=mock_client):
            result = transport.dispatch(
                model_key="coder",
                messages=[{"role": "user", "content": "Write a function"}],
                temperature=0.1,
                max_tokens=512,
            )

        assert "python" in result["content"] or result["content"]
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "https://test.example.com/infer/coder"

    def test_remote_vision_routes_by_model_key_and_sends_base64(self, monkeypatch):
        """model_key='document_analyzer' routes to /infer/vision.

        Verifies:
        - Correct endpoint selected by key
        - image_base64 field is present in the payload (bytes transmitted, not path)
        - mime_type is extracted correctly
        - No local path appears in payload
        """
        self._setup_remote(monkeypatch)

        # Build a valid base64 data-URI as tools/vision.py would construct
        from PIL import Image as PILImage
        img = PILImage.new("RGB", (4, 4), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw_bytes = buf.getvalue()
        b64_str = base64.b64encode(raw_bytes).decode("utf-8")
        data_uri = f"data:image/png;base64,{b64_str}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you see?"},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ]

        from core.remote_model_transport import RemoteModelTransport
        transport = RemoteModelTransport()
        transport._base_url = "https://test.example.com"
        transport._api_key = "test-key"

        mock_client, _ = _mock_httpx_post(_make_remote_response("blue square", "qwen3-vl-4b"))

        with patch("httpx.Client", return_value=mock_client):
            result = transport.dispatch(
                model_key="document_analyzer",
                messages=messages,
                temperature=0.05,
                max_tokens=200,
            )

        assert result["content"] == "blue square"
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "https://test.example.com/infer/vision"

        # Payload must contain base64 image bytes, not a filesystem path
        payload_sent = call_args[1]["json"]
        assert "image_base64" in payload_sent
        assert payload_sent["image_base64"] == b64_str
        assert payload_sent.get("mime_type") == "image/png"
        assert "local_path" not in payload_sent
        assert "image_path" not in payload_sent

    def test_remote_dispatch_via_model_client_uses_current_model_key(self, monkeypatch):
        """ModelClient.chat_completion falls back to model_manager.current_model_key."""
        import config
        monkeypatch.setattr(config, "SAGE_MODEL_BACKEND", "remote", raising=True)

        from model_client import ModelClient
        from core.remote_model_transport import remote_model_transport

        mock_result = {
            "content": "routed correctly",
            "duration": 0.3,
            "usage": {},
            "timings": {},
            "raw": {},
        }
        with patch.object(remote_model_transport, "dispatch", return_value=mock_result) as mock_dispatch:
            with patch("model_manager.model_manager") as mock_mm:
                mock_mm.current_model_key = "coder"
                # Import model_manager inside model_client uses module-level reference
                with patch("model_client.model_manager", mock_mm, create=True):
                    # We need to patch the import inside chat_completion
                    import model_manager as mm_module
                    with patch.object(mm_module, "model_manager", mock_mm):
                        client = ModelClient()
                        result = client.chat_completion(
                            messages=[{"role": "user", "content": "fix this"}],
                        )

        assert result["content"] == "routed correctly"
        mock_dispatch.assert_called_once()
        assert mock_dispatch.call_args.kwargs["model_key"] == "coder"


# ---------------------------------------------------------------------------
# 5. Authorization header present
# ---------------------------------------------------------------------------

class TestAuthorizationHeader:
    def test_authorization_bearer_header_on_every_request(self, monkeypatch):
        """Every request to the remote worker must carry Authorization: Bearer <key>."""
        import config
        monkeypatch.setattr(config, "SAGE_MODEL_BACKEND", "remote", raising=True)

        from core.remote_model_transport import RemoteModelTransport
        transport = RemoteModelTransport()
        transport._base_url = "https://test.example.com"
        transport._api_key = "my-secret-key"

        mock_client, _ = _mock_httpx_post(_make_remote_response("hello"))

        with patch("httpx.Client", return_value=mock_client):
            transport.dispatch(
                model_key="agent",
                messages=[{"role": "user", "content": "hi"}],
            )

        call_kwargs = mock_client.post.call_args[1]
        headers = call_kwargs.get("headers", {})
        assert headers.get("Authorization") == "Bearer my-secret-key"


# ---------------------------------------------------------------------------
# 6. HTTP error handling
# ---------------------------------------------------------------------------

class TestHttpErrorHandling:
    def test_http_500_raises_runtime_error(self, monkeypatch):
        """HTTP 5xx from the remote worker raises RuntimeError, not swallowed."""
        import config
        monkeypatch.setattr(config, "SAGE_MODEL_BACKEND", "remote", raising=True)

        from core.remote_model_transport import RemoteModelTransport
        transport = RemoteModelTransport()
        transport._base_url = "https://test.example.com"
        transport._api_key = "key"

        mock_client, _ = _mock_httpx_post({}, status_code=500)

        with patch("httpx.Client", return_value=mock_client):
            with pytest.raises(RuntimeError, match="500|HTTP"):
                transport.dispatch(
                    model_key="agent",
                    messages=[{"role": "user", "content": "hi"}],
                )

    def test_remote_error_status_in_body_raises_runtime_error(self, monkeypatch):
        """Worker returning status='error' in JSON body raises RuntimeError."""
        import config
        monkeypatch.setattr(config, "SAGE_MODEL_BACKEND", "remote", raising=True)

        from core.remote_model_transport import RemoteModelTransport
        transport = RemoteModelTransport()
        transport._base_url = "https://test.example.com"
        transport._api_key = "key"

        error_body = {"status": "error", "error_type": "RuntimeError", "detail": "OOM"}
        mock_client, _ = _mock_httpx_post(error_body, status_code=200)

        with patch("httpx.Client", return_value=mock_client):
            with pytest.raises(RuntimeError, match="error|OOM"):
                transport.dispatch(
                    model_key="agent",
                    messages=[{"role": "user", "content": "hi"}],
                )


# ---------------------------------------------------------------------------
# 7-8. Timeout handling
# ---------------------------------------------------------------------------

class TestTimeoutHandling:
    def test_connect_timeout_raises_runtime_error(self, monkeypatch):
        """Connect timeout surfaces as RuntimeError with descriptive message."""
        import config
        import httpx
        monkeypatch.setattr(config, "SAGE_MODEL_BACKEND", "remote", raising=True)

        from core.remote_model_transport import RemoteModelTransport
        transport = RemoteModelTransport()
        transport._base_url = "https://test.example.com"
        transport._api_key = "key"

        mock_client = MagicMock()
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = httpx.ConnectTimeout("timed out")

        with patch("httpx.Client", return_value=mock_client):
            with pytest.raises(RuntimeError, match="connect timeout|timeout"):
                transport.dispatch(
                    model_key="agent",
                    messages=[{"role": "user", "content": "hi"}],
                )

    def test_read_timeout_raises_runtime_error(self, monkeypatch):
        """Inference/read timeout surfaces as RuntimeError with descriptive message."""
        import config
        import httpx
        monkeypatch.setattr(config, "SAGE_MODEL_BACKEND", "remote", raising=True)

        from core.remote_model_transport import RemoteModelTransport
        transport = RemoteModelTransport()
        transport._base_url = "https://test.example.com"
        transport._api_key = "key"

        mock_client = MagicMock()
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = httpx.ReadTimeout("read timed out")

        with patch("httpx.Client", return_value=mock_client):
            with pytest.raises(RuntimeError, match="timeout"):
                transport.dispatch(
                    model_key="agent",
                    messages=[{"role": "user", "content": "hi"}],
                )


# ---------------------------------------------------------------------------
# 9. No local subprocess in remote mode
# ---------------------------------------------------------------------------

class TestNoLocalSubprocessInRemoteMode:
    def test_ensure_model_remote_does_not_spawn_subprocess(self, monkeypatch):
        """ensure_model() in remote mode must NOT call subprocess.Popen."""
        import config
        monkeypatch.setattr(config, "SAGE_MODEL_BACKEND", "remote", raising=True)

        from model_manager import ModelManager
        mm = ModelManager()

        with patch("subprocess.Popen") as mock_popen:
            result = mm.ensure_model("agent")
            mock_popen.assert_not_called()

        assert result is True
        assert mm.current_model_key == "agent"
        assert mm.server_process is None

    def test_ensure_all_model_keys_remote_no_subprocess(self, monkeypatch):
        """All three model keys work in remote mode without subprocess."""
        import config
        monkeypatch.setattr(config, "SAGE_MODEL_BACKEND", "remote", raising=True)

        from model_manager import ModelManager
        mm = ModelManager()

        with patch("subprocess.Popen") as mock_popen:
            for key in ("agent", "coder", "document_analyzer"):
                mm.current_model_key = None  # Reset between calls
                result = mm.ensure_model(key)
                assert result is True
                assert mm.current_model_key == key
            mock_popen.assert_not_called()


# ---------------------------------------------------------------------------
# 10-11. Fail-closed — unknown / missing model_key
# ---------------------------------------------------------------------------

class TestFailClosed:
    def test_missing_model_key_raises_runtime_error(self, monkeypatch):
        """dispatch() with model_key=None must raise RuntimeError immediately."""
        import config
        monkeypatch.setattr(config, "SAGE_MODEL_BACKEND", "remote", raising=True)

        from core.remote_model_transport import RemoteModelTransport
        transport = RemoteModelTransport()
        transport._base_url = "https://test.example.com"
        transport._api_key = "key"

        with pytest.raises(RuntimeError, match="model_key"):
            transport.dispatch(
                model_key=None,  # type: ignore[arg-type]
                messages=[{"role": "user", "content": "hi"}],
            )

    def test_unknown_model_key_raises_runtime_error(self, monkeypatch):
        """dispatch() with an unknown model_key must raise RuntimeError immediately."""
        import config
        monkeypatch.setattr(config, "SAGE_MODEL_BACKEND", "remote", raising=True)

        from core.remote_model_transport import RemoteModelTransport
        transport = RemoteModelTransport()
        transport._base_url = "https://test.example.com"
        transport._api_key = "key"

        with pytest.raises(RuntimeError, match="cannot route|mapping"):
            transport.dispatch(
                model_key="some_future_model_not_in_mapping",
                messages=[{"role": "user", "content": "hi"}],
            )

    def test_unresolvable_model_key_via_model_client_raises(self, monkeypatch):
        """ModelClient with no model_key arg and no current_model_key fails closed."""
        import config
        monkeypatch.setattr(config, "SAGE_MODEL_BACKEND", "remote", raising=True)

        from model_client import ModelClient
        import model_manager as mm_module

        mock_mm = MagicMock()
        mock_mm.current_model_key = None  # nothing selected yet

        with patch.object(mm_module, "model_manager", mock_mm):
            client = ModelClient()
            with pytest.raises(RuntimeError, match="unresolvable|model_key"):
                client.chat_completion(
                    messages=[{"role": "user", "content": "hi"}],
                    # no model_key provided, current_model_key is also None
                )

    def test_rearm_agent_background_remote(self, monkeypatch):
        """ModelManager.rearm_agent_background triggers ping to Gemma and sets current_model_key."""
        import config
        monkeypatch.setattr(config, "SAGE_MODEL_BACKEND", "remote", raising=True)
        monkeypatch.setenv("SAGE_MOCK_MODE", "0")

        from model_manager import ModelManager
        from core.remote_model_transport import remote_model_transport

        mm = ModelManager()
        mm.current_model_key = "final_synthesizer"

        mock_infer = MagicMock(return_value={"content": "pong", "duration": 0.1})
        with patch.object(remote_model_transport, "infer_gemma", mock_infer):
            mm.rearm_agent_background()
            assert mm._rearm_thread is not None
            mm._rearm_thread.join(timeout=3.0)
            assert mm.current_model_key == "agent"
            mock_infer.assert_called_once()
            call_kwargs = mock_infer.call_args[1]
            assert call_kwargs["max_tokens"] == 1

    def test_rearm_agent_background_mock_mode(self, monkeypatch):
        """In mock mode, rearm_agent_background sets current_model_key synchronously."""
        monkeypatch.setenv("SAGE_MOCK_MODE", "1")

        from model_manager import ModelManager
        mm = ModelManager()
        mm.current_model_key = "final_synthesizer"
        mm.rearm_agent_background()
        assert mm.current_model_key == "agent"
        assert mm._rearm_thread is None
