"""Unified local/remote inference transport for all Flash roles."""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

import config
from flash.catalog import load_catalog
from flash.local_manager import local_flash_manager
from flash.runtime_config import runtime_config

_RETRYABLE_STATUS = {502, 503, 504, 522, 524, 530}


def _origin(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return url.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}"


class FlashTransport:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clients: Dict[str, httpx.Client] = {}

    @staticmethod
    def _mock_enabled() -> bool:
        return os.environ.get("SAGE_MOCK_MODE", "0") == "1"

    @staticmethod
    def _suggest_model_id(role: str, models: List[str]) -> Optional[str]:
        """Find a unique, human-readable bridge id matching a Flash role."""
        lowered = [(model_id, str(model_id).lower()) for model_id in models if model_id]
        if role == "gemma":
            matches = [model_id for model_id, value in lowered if "gemma" in value]
        elif role == "qwen":
            matches = [
                model_id for model_id, value in lowered
                if "qwen" in value and any(marker in value for marker in ("vl", "vision", "visual"))
            ]
        else:
            matches = [
                model_id for model_id, value in lowered
                if any(marker in value for marker in ("memory", "compress", "summar"))
            ]
        return matches[0] if len(matches) == 1 else None

    def _headers(self, connection: Optional[Dict[str, Any]], json_body: bool = True) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if connection and connection.get("api_key"):
            headers["Authorization"] = f"Bearer {connection['api_key']}"
        if json_body:
            headers["Content-Type"] = "application/json"
        return headers

    @staticmethod
    def _timeout(read: float) -> httpx.Timeout:
        return httpx.Timeout(connect=20.0, read=read, write=30.0, pool=20.0)

    def _build_client(self) -> httpx.Client:
        # Keepalive reuses TLS. Force IPv4: many home routers advertise IPv6
        # Cloudflare addresses that black-hole, and Happy Eyeballs then burns
        # ~40s before falling back to working IPv4.
        return httpx.Client(
            timeout=self._timeout(config.FLASH_REQUEST_TIMEOUT),
            limits=httpx.Limits(
                max_keepalive_connections=8,
                max_connections=16,
                keepalive_expiry=300.0,
            ),
            transport=httpx.HTTPTransport(local_address="0.0.0.0"),
            follow_redirects=True,
        )

    def _client_for(self, url: str) -> httpx.Client:
        origin = _origin(url)
        with self._lock:
            client = self._clients.get(origin)
            if client is None or client.is_closed:
                client = self._build_client()
                self._clients[origin] = client
            return client

    def _drop_client(self, url: str) -> None:
        origin = _origin(url)
        with self._lock:
            client = self._clients.pop(origin, None)
        if client is not None and not client.is_closed:
            client.close()

    def close(self) -> None:
        with self._lock:
            clients = list(self._clients.values())
            self._clients.clear()
        for client in clients:
            if not client.is_closed:
                client.close()

    def _request(
        self,
        method: str,
        url: str,
        *,
        headers: Dict[str, str],
        timeout: httpx.Timeout,
        json: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        last_error: Optional[BaseException] = None
        response: Optional[httpx.Response] = None
        for attempt in range(2):
            client = self._client_for(url)
            try:
                response = client.request(
                    method,
                    url,
                    headers=headers,
                    json=json,
                    timeout=timeout,
                )
            except httpx.RequestError as exc:
                self._drop_client(url)
                last_error = exc
                continue
            if response.status_code in _RETRYABLE_STATUS and attempt == 0:
                self._drop_client(url)
                continue
            return response
        if last_error is not None:
            raise last_error
        assert response is not None
        return response

    def warm(self, base_url: str, connection: Optional[Dict[str, Any]] = None) -> None:
        """Open the pooled TLS session so the first chat is not the cold connect."""
        url = f"{base_url.rstrip('/')}/health"
        try:
            self._request(
                "GET",
                url,
                headers=self._headers(connection, False),
                timeout=self._timeout(20.0),
            )
        except Exception:
            try:
                self._request(
                    "GET",
                    f"{base_url.rstrip('/')}/v1/models",
                    headers=self._headers(connection, False),
                    timeout=self._timeout(20.0),
                )
            except Exception:
                return

    def warm_configured_connections(self) -> None:
        snapshot = runtime_config.public_snapshot()
        seen: set[str] = set()
        for connection in (snapshot.get("connections") or {}).values():
            live = runtime_config.connection(str(connection.get("id") or ""))
            base_url = str((live or connection).get("base_url") or "").rstrip("/")
            if not base_url or base_url in seen:
                continue
            seen.add(base_url)
            self.warm(base_url, live)

    def _resolve(self, role: str) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
        binding = runtime_config.role(role)
        provider = binding["provider"]
        if provider == "disabled":
            raise RuntimeError(f"Flash role {role!r} is disabled")
        if provider in {"local_gpu", "local_cpu"}:
            base_url = local_flash_manager.ensure(role, provider)
        else:
            connection = binding.get("connection")
            if not connection:
                raise RuntimeError(f"Flash role {role!r} has no remote connection")
            base_url = connection["base_url"]
        return base_url.rstrip("/"), binding, load_catalog()[role]

    @staticmethod
    def _apply_generation_policy(payload: Dict[str, Any], model: Dict[str, Any]) -> None:
        reasoning = str(model.get("reasoning", "off")).strip().lower()
        if reasoning in {"", "off", "false", "0", "none"}:
            # Already-running bridges may have been started with reasoning on.
            # llama.cpp ignores unknown fields; these disable thinking when supported.
            payload["reasoning_budget"] = 0
            payload["chat_template_kwargs"] = {"enable_thinking": False}

    def invoke(
        self,
        role: str,
        messages: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> Dict[str, Any]:
        # Mock only fills in for local roles. A configured remote endpoint must
        # always be used, otherwise remote GPU testing looks "broken".
        binding = runtime_config.role(role)
        if self._mock_enabled() and binding.get("provider") != "remote":
            if role == "gemma":
                content = (
                    '{"flash_case":"A","gemma_answer":'
                    '"[Mock mode] SAGE Flash is running without local GGUF models. '
                    'Configure a remote GPU in Settings → Flash runtime, or install llama-server and the catalog models."}'
                )
            else:
                content = "Mock worker response."
            return {"content": content, "duration": 0.001, "usage": {}, "timings": {}, "raw": {}}

        base_url, binding, model = self._resolve(role)
        payload: Dict[str, Any] = {
            "model": binding.get("model_id") or role,
            "messages": messages,
            "stream": False,
            "temperature": model.get("temperature", 0.2) if temperature is None else temperature,
            "max_tokens": model.get("max_tokens", 2048) if max_tokens is None else max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        self._apply_generation_policy(payload, model)
        connection = binding.get("connection")
        started = time.perf_counter()
        try:
            response = self._request(
                "POST",
                f"{base_url}/v1/chat/completions",
                headers=self._headers(connection),
                json=payload,
                timeout=self._timeout(config.FLASH_REQUEST_TIMEOUT),
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"Flash {role} endpoint returned HTTP {exc.response.status_code}: {exc.response.text[:500]}") from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"Flash {role} endpoint unavailable at {base_url}: {exc}") from exc

        message = (data.get("choices") or [{}])[0].get("message") or {}
        return {
            "content": message.get("content") or "",
            "reasoning": message.get("reasoning_content") or "",
            "duration": time.perf_counter() - started,
            "usage": data.get("usage") or {},
            "timings": data.get("timings") or {},
            "raw": data,
        }

    def _probe_completion(
        self,
        role: str,
        base_url: str,
        binding: Dict[str, Any],
        connection: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run a tiny chat completion to prove the endpoint can actually infer."""
        if self._mock_enabled() and binding.get("provider") != "remote":
            return {"ok": True, "preview": "mock", "mock": True}
        payload = {
            "model": binding.get("model_id") or role,
            "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
            "stream": False,
            "temperature": 0.0,
            "max_tokens": 8,
            "reasoning_budget": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        response = self._request(
            "POST",
            f"{base_url.rstrip('/')}/v1/chat/completions",
            headers=self._headers(connection),
            json=payload,
            timeout=self._timeout(20.0),
        )
        response.raise_for_status()
        data = response.json()
        message = (data.get("choices") or [{}])[0].get("message") or {}
        preview = str(message.get("content") or "").strip()[:120]
        if not preview:
            return {"ok": False, "preview": "", "error": "Endpoint returned an empty completion"}
        return {"ok": True, "preview": preview, "mock": False}

    def test_role(self, role: str) -> Dict[str, Any]:
        binding = runtime_config.role(role)
        if binding["provider"] == "disabled":
            return {"role": role, "enabled": False, "healthy": True, "message": "disabled"}
        if self._mock_enabled() and binding.get("provider") != "remote":
            return {
                "role": role,
                "enabled": True,
                "healthy": True,
                "provider": binding["provider"],
                "model_id": binding.get("model_id"),
                "message": "SAGE_MOCK_MODE is active — inference is simulated",
                "probe": "mock",
            }
        connection = binding.get("connection")
        if binding["provider"] in {"local_gpu", "local_cpu"}:
            try:
                base_url = local_flash_manager.ensure(role, binding["provider"])
            except Exception as exc:
                return {
                    "role": role, "enabled": True, "healthy": False,
                    "provider": binding["provider"], "model_id": binding.get("model_id"),
                    "error": str(exc),
                }
            try:
                health = self._request(
                    "GET",
                    f"{base_url.rstrip('/')}/health",
                    headers=self._headers(None, False),
                    timeout=self._timeout(3.0),
                )
                if health.status_code != 200:
                    return {
                        "role": role, "enabled": True, "healthy": False,
                        "provider": binding["provider"], "model_id": binding.get("model_id"),
                        "endpoint": base_url,
                        "error": f"Local /health returned HTTP {health.status_code}",
                    }
                probe = self._probe_completion(role, base_url, binding)
                return {
                    "role": role, "enabled": True, "healthy": bool(probe.get("ok")),
                    "provider": binding["provider"], "model_id": binding.get("model_id"),
                    "endpoint": base_url,
                    "probe": probe.get("preview") or "",
                    "error": None if probe.get("ok") else probe.get("error"),
                }
            except httpx.HTTPStatusError as exc:
                return {
                    "role": role, "enabled": True, "healthy": False,
                    "provider": binding["provider"], "model_id": binding.get("model_id"),
                    "endpoint": base_url,
                    "error": f"Inference probe failed (HTTP {exc.response.status_code}): {exc.response.text[:400]}",
                }
            except Exception as exc:
                return {
                    "role": role, "enabled": True, "healthy": False,
                    "provider": binding["provider"], "model_id": binding.get("model_id"),
                    "endpoint": base_url,
                    "error": f"Inference probe failed: {exc}",
                }

        base_url = binding["connection"]["base_url"]
        connection = binding["connection"]
        try:
            response = self._request(
                "GET",
                f"{base_url.rstrip('/')}/v1/models",
                headers=self._headers(connection, False),
                timeout=self._timeout(8.0),
            )
            response.raise_for_status()
            models = [item.get("id") for item in response.json().get("data", [])]
            wanted = binding.get("model_id")
            model_available = bool(wanted and wanted in models)
            suggested = None if model_available else self._suggest_model_id(role, models)
            if not model_available:
                return {
                    "role": role, "enabled": True, "healthy": False, "provider": binding["provider"],
                    "model_id": wanted, "model_available": False, "models": models,
                    "error": f"Model id {wanted!r} was not listed by the endpoint",
                    "suggested_model_id": suggested,
                }
            probe = self._probe_completion(role, base_url, binding, connection)
            return {
                "role": role, "enabled": True, "healthy": bool(probe.get("ok")),
                "provider": binding["provider"],
                "model_id": wanted, "model_available": True, "models": models,
                "probe": probe.get("preview") or "",
                "error": None if probe.get("ok") else probe.get("error"),
                "suggested_model_id": suggested,
            }
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            if code in {401, 403}:
                message = f"Authentication failed (HTTP {code}). Use the API KEY printed by Kaggle Cell 3, not the Hugging Face token."
            elif code == 404:
                message = "The endpoint has no /v1/models route (HTTP 404). Enter the bridge root URL, not a model-specific URL."
            else:
                message = f"Endpoint returned HTTP {code}: {exc.response.text[:500]}"
            return {"role": role, "enabled": True, "healthy": False, "error": message, "models": []}
        except httpx.RequestError as exc:
            return {
                "role": role, "enabled": True, "healthy": False,
                "error": f"Could not reach {base_url}: {exc}", "models": [],
            }
        except Exception as exc:
            return {"role": role, "enabled": True, "healthy": False, "error": str(exc)}

    def deploy(self, role: str, connection_id: str, gpu: int = 0) -> Dict[str, Any]:
        binding = runtime_config.role(role)
        # Deployment can target a configured endpoint before a role is assigned
        # to it, which is useful while preparing Kaggle/remote hardware.
        connection = runtime_config.connection(connection_id)
        if not connection:
            raise ValueError("Unknown deploy target connection")
        model = load_catalog()[role]
        payload = {
            "id": binding.get("model_id") or role,
            "repo": model.get("repo"), "file": model.get("file"), "mmproj": model.get("mmproj"),
            "gpu": int(gpu), "context": int(model.get("context", 4096)),
            "reasoning": model.get("reasoning", "off"),
            "est_size_gib": float(model.get("estimated_vram_gib", 0)),
        }
        response = self._request(
            "POST",
            f"{connection['base_url'].rstrip('/')}/control/models/start",
            headers=self._headers(connection),
            json=payload,
            timeout=self._timeout(config.FLASH_DEPLOY_TIMEOUT),
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            try:
                detail = response.json().get("error", {}).get("message") or response.text
            except Exception:
                detail = response.text
            raise RuntimeError(
                f"Bridge rejected {role} deployment (HTTP {response.status_code}): {str(detail)[:1200]}"
            ) from exc
        return response.json()


flash_transport = FlashTransport()
