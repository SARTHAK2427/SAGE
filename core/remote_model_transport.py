"""
SAGE/core/remote_model_transport.py

Minimal transport shim that executes already-selected model inference on the
remote RTX 4060 GPU worker instead of a local llama-server.exe instance.

CRITICAL DESIGN CONSTRAINT
───────────────────────────
This module answers ONE question only:
    "WHERE does the already-selected model execute?"

It must NEVER answer:
    "WHICH model should execute?"

Routing is determined exclusively by the model_key already chosen by the
existing SAGE execution flow, using the canonical mapping:

    "agent"             -> POST /infer/gemma
    "coder"             -> POST /infer/coder
    "document_analyzer" -> POST /infer/vision

If model_key is absent or unknown, this module FAILS CLOSED with a clear
RuntimeError — it does not guess based on message contents or task type.

Return contract
───────────────
All public inference methods return the identical dict shape produced by
ModelClient.chat_completion():

    {
        "content":  str,   # generated text
        "duration": float, # wall-clock seconds
        "usage":    dict,  # prompt/completion/total token counts
        "timings":  dict,  # server-side timings (if any)
        "raw":      dict,  # verbatim response body
    }

Vision images
─────────────
The remote GPU worker cannot access this machine's filesystem.
Images are always read locally and transmitted as base64 bytes.
Local filesystem paths are never sent to the remote endpoint.
"""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

import config

logger = logging.getLogger(__name__)

# ── Canonical model_key → remote endpoint mapping ─────────────────────────────
# This is the ONLY place routing is decided. No content inspection anywhere.

_REMOTE_ENDPOINT_MAP: Dict[str, str] = {
    "agent":             "/infer/gemma",
    "coder":             "/infer/coder",
    "document_analyzer": "/infer/vision",
    "final_synthesizer": "/infer/fast",
}


def _resolve_endpoint(model_key: Optional[str]) -> str:
    """Return the remote endpoint path for *model_key*.

    Fails closed with RuntimeError if model_key is None, empty, or not in the
    canonical mapping.  No guessing, no content inspection.
    """
    if not model_key:
        raise RuntimeError(
            "RemoteModelTransport: model_key is required for remote routing but was "
            "not supplied.  Ensure ensure_model() is called before chat_completion()."
        )
    endpoint = _REMOTE_ENDPOINT_MAP.get(model_key)
    if endpoint is None:
        known = list(_REMOTE_ENDPOINT_MAP.keys())
        raise RuntimeError(
            f"RemoteModelTransport: cannot route model_key={model_key!r} — "
            f"not in canonical mapping {known}.  "
            f"Verify ensure_model() was called with a valid model key."
        )
    return endpoint


class RemoteModelTransport:
    """HTTP transport to the remote RTX 4060 GPU worker.

    Instantiated once as a module-level singleton; callers never construct this
    directly — they use the `remote_model_transport` instance at the bottom of
    this file.
    """

    def __init__(self) -> None:
        # Resolved lazily so config is already fully loaded when first used.
        self._base_url: str = ""
        self._api_key: str = ""
        self._connect_timeout: float = config.REMOTE_CONNECT_TIMEOUT
        self._request_timeout: float = config.REMOTE_REQUEST_TIMEOUT

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_config(self) -> None:
        """Validate that remote configuration has been provided."""
        if not self._base_url:
            self._base_url = config.SAGE_REMOTE_GPU_URL
        if not self._api_key:
            self._api_key = config.SAGE_REMOTE_GPU_API_KEY
        if not self._base_url:
            raise RuntimeError(
                "RemoteModelTransport: SAGE_REMOTE_GPU_URL is not set.  "
                "Add it to .env or export it before starting SAGE."
            )
        if not self._api_key:
            raise RuntimeError(
                "RemoteModelTransport: SAGE_REMOTE_GPU_API_KEY is not set.  "
                "Add it to .env or export it before starting SAGE."
            )

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def _httpx_timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            connect=self._connect_timeout,
            read=self._request_timeout,
            write=self._request_timeout,
            pool=self._connect_timeout,
        )

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST *payload* to *endpoint* and return the parsed JSON body.

        Raises RuntimeError on HTTP errors, network failures, or malformed JSON.
        """
        self._ensure_config()
        url = f"{self._base_url}{endpoint}"
        logger.debug("RemoteModelTransport POST %s", url)
        try:
            with httpx.Client(timeout=self._httpx_timeout()) as client:
                response = client.post(url, json=payload, headers=self._headers())
        except httpx.ConnectTimeout as exc:
            raise RuntimeError(
                f"RemoteModelTransport: connect timeout after {self._connect_timeout}s "
                f"reaching {url}"
            ) from exc
        except httpx.ReadTimeout as exc:
            raise RuntimeError(
                f"RemoteModelTransport: read/inference timeout after {self._request_timeout}s "
                f"waiting for {url}"
            ) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(
                f"RemoteModelTransport: network error reaching {url}: {exc}"
            ) from exc

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"RemoteModelTransport: HTTP {exc.response.status_code} from {url}: "
                f"{exc.response.text}"
            ) from exc

        try:
            data = response.json()
        except Exception as exc:
            raise RuntimeError(
                f"RemoteModelTransport: malformed JSON response from {url}: {exc}"
            ) from exc

        return data

    @staticmethod
    def _to_standard_result(data: Dict[str, Any], duration: float) -> Dict[str, Any]:
        """Convert a remote worker response to the standard ModelClient dict shape."""
        status = data.get("status", "")
        if status == "error":
            detail = data.get("detail") or data.get("error_type") or "unknown"
            raise RuntimeError(
                f"RemoteModelTransport: remote worker reported error — {detail}"
            )
        content = data.get("content", "")
        usage = data.get("usage") or {}
        timings = data.get("timings") or {}
        return {
            "content": content,
            "duration": duration,
            "usage": usage,
            "timings": timings,
            "raw": data,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_healthy(self, timeout: float = 5.0) -> bool:
        """Return True if the remote GPU worker is reachable and healthy."""
        try:
            self._ensure_config()
        except RuntimeError:
            return False
        url = f"{self._base_url}/health"
        try:
            with httpx.Client(timeout=httpx.Timeout(connect=timeout, read=timeout, write=timeout, pool=timeout)) as client:
                r = client.get(url, headers=self._headers())
                return r.status_code == 200
        except Exception:
            return False

    def get_health(self) -> Dict[str, Any]:
        """Return raw health telemetry from the remote worker."""
        self._ensure_config()
        url = f"{self._base_url}/health"
        try:
            with httpx.Client(timeout=self._httpx_timeout()) as client:
                r = client.get(url, headers=self._headers())
                r.raise_for_status()
                return r.json()
        except Exception as exc:
            raise RuntimeError(f"RemoteModelTransport: health check failed: {exc}") from exc

    # ── Gemma ──────────────────────────────────────────────────────────

    def infer_gemma(
        self,
        *,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Route a Gemma inference request to POST /infer/gemma.

        Accepts *messages* in chat format (the existing SAGE call pattern)
        and also forwards temperature/max_tokens when provided.
        """
        payload: Dict[str, Any] = {"messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        t0 = time.time()
        data = self._post("/infer/gemma", payload)
        duration = time.time() - t0
        return self._to_standard_result(data, duration)

    # ── Coder ──────────────────────────────────────────────────────────

    def infer_coder(
        self,
        *,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Route a coder inference request to POST /infer/coder.

        Passes messages directly (the existing SAGE call pattern).
        """
        payload: Dict[str, Any] = {"messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        t0 = time.time()
        data = self._post("/infer/coder", payload)
        duration = time.time() - t0
        return self._to_standard_result(data, duration)

    # ── Vision ─────────────────────────────────────────────────────────

    def infer_vision(
        self,
        *,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Route a vision inference request to POST /infer/vision.

        The existing SAGE vision tool (tools/vision.py) already:
          1. Reads the local image file into bytes
          2. Encodes those bytes as a base64 data-URI embedded in *messages*

        This method extracts the image bytes / mime-type from that data-URI
        message content so the remote worker receives raw base64 + mime_type,
        never a local filesystem path.

        If no image can be extracted from messages, raises RuntimeError (fail-closed).
        """
        # Extract image bytes from the data-URI inside messages
        image_b64: Optional[str] = None
        mime_type: str = "image/png"
        instruction: str = ""

        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            instruction = part.get("text", instruction)
                        elif part.get("type") == "image_url":
                            url_val = part.get("image_url", {}).get("url", "")
                            if url_val.startswith("data:"):
                                # data:<mime>;base64,<b64>
                                try:
                                    header, b64_data = url_val.split(",", 1)
                                    mime_type = header.split(":")[1].split(";")[0]
                                    image_b64 = b64_data
                                except Exception:
                                    pass
            elif isinstance(content, str) and not instruction:
                instruction = content

        if image_b64 is None:
            raise RuntimeError(
                "RemoteModelTransport: could not extract image data from messages for "
                "/infer/vision.  Ensure tools/vision.py embeds a data-URI before calling "
                "chat_completion with model_key='document_analyzer'."
            )

        payload: Dict[str, Any] = {
            "image_base64": image_b64,
            "mime_type": mime_type,
            "instruction": instruction,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        t0 = time.time()
        data = self._post("/infer/vision", payload)
        duration = time.time() - t0
        return self._to_standard_result(data, duration)

    # ── Final Synthesizer (Qwen3.5 2B) ──────────────────────────────────

    def infer_final_synthesizer(
        self,
        *,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Route a final synthesis request to the dedicated Qwen3.5 2B synthesizer."""
        endpoint = _resolve_endpoint("final_synthesizer")
        t0 = time.time()
        payload: Dict[str, Any] = {"messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        data = self._post(endpoint, payload)
        duration = time.time() - t0
        return self._to_standard_result(data, duration)

    # ── Dispatcher — called by ModelClient.chat_completion() ───────────

    def dispatch(
        self,
        *,
        model_key: str,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Route inference to the correct remote endpoint using *model_key* only.

        This is the single entry-point called by ModelClient when
        SAGE_MODEL_BACKEND=remote.  Routing is deterministic — it uses
        model_key exclusively and never inspects message contents.

        Args:
            model_key:   Key already selected by SAGE execution flow
                         ("agent" | "coder" | "document_analyzer").
            messages:    Chat messages as built by the existing callers.
            temperature: Forwarded verbatim when provided.
            max_tokens:  Forwarded verbatim when provided.

        Returns:
            Standard ModelClient result dict:
            {"content", "duration", "usage", "timings", "raw"}

        Raises:
            RuntimeError if model_key is unresolvable or the remote call fails.
        """
        endpoint = _resolve_endpoint(model_key)
        logger.info(
            "RemoteModelTransport.dispatch: model_key=%s -> %s", model_key, endpoint
        )

        if endpoint == "/infer/gemma":
            return self.infer_gemma(
                messages=messages, temperature=temperature, max_tokens=max_tokens
            )
        elif endpoint == "/infer/coder":
            return self.infer_coder(
                messages=messages, temperature=temperature, max_tokens=max_tokens
            )
        elif endpoint == "/infer/vision":
            return self.infer_vision(
                messages=messages, temperature=temperature, max_tokens=max_tokens
            )
        elif endpoint == "/infer/fast":
            return self.infer_final_synthesizer(
                messages=messages, temperature=temperature, max_tokens=max_tokens
            )
        else:  # pragma: no cover — _resolve_endpoint already guards this
            raise RuntimeError(
                f"RemoteModelTransport: internal error — unexpected endpoint {endpoint!r}"
            )


# Module-level singleton — import and use directly.
remote_model_transport = RemoteModelTransport()
