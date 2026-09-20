"""Unified local/remote inference transport for all Flash roles."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import httpx

import config
from flash.catalog import load_catalog
from flash.local_manager import local_flash_manager
from flash.runtime_config import runtime_config


class FlashTransport:
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

    def invoke(
        self,
        role: str,
        messages: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> Dict[str, Any]:
        if os.environ.get("SAGE_MOCK_MODE", "0") == "1":
            content = '{"flash_case":"A","gemma_answer":"Mock Flash response."}' if role == "gemma" else "Mock worker response."
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
        connection = binding.get("connection")
        started = time.perf_counter()
        try:
            with httpx.Client(timeout=config.FLASH_REQUEST_TIMEOUT) as client:
                response = client.post(
                    f"{base_url}/v1/chat/completions",
                    headers=self._headers(connection),
                    json=payload,
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

    def test_role(self, role: str) -> Dict[str, Any]:
        binding = runtime_config.role(role)
        if binding["provider"] == "disabled":
            return {"role": role, "enabled": False, "healthy": True, "message": "disabled"}
        if binding["provider"] in {"local_gpu", "local_cpu"}:
            base_url = local_flash_manager.ensure(role, binding["provider"])
            return {
                "role": role, "enabled": True, "healthy": True,
                "provider": binding["provider"], "model_id": binding.get("model_id"),
                "endpoint": base_url,
            }
        else:
            base_url = binding["connection"]["base_url"]
            connection = binding["connection"]
        try:
            with httpx.Client(timeout=8.0) as client:
                response = client.get(f"{base_url.rstrip('/')}/v1/models", headers=self._headers(connection, False))
                response.raise_for_status()
                models = [item.get("id") for item in response.json().get("data", [])]
            wanted = binding.get("model_id")
            model_available = bool(wanted and wanted in models)
            suggested = None if model_available else self._suggest_model_id(role, models)
            return {
                "role": role, "enabled": True, "healthy": model_available, "provider": binding["provider"],
                "model_id": wanted, "model_available": model_available, "models": models,
                "error": None if model_available else f"Model id {wanted!r} was not listed by the endpoint",
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
        with httpx.Client(timeout=config.FLASH_DEPLOY_TIMEOUT) as client:
            response = client.post(
                f"{connection['base_url'].rstrip('/')}/control/models/start",
                headers=self._headers(connection), json=payload,
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
