"""Ephemeral, host-agnostic runtime bindings for SAGE Flash.

Nothing in this module is persisted. Tunnel URLs and bearer keys live only in
the SAGE server process and disappear when it restarts.
"""

from __future__ import annotations

import copy
import threading
from typing import Any, Dict
from urllib.parse import urlparse


ROLE_NAMES = ("gemma", "qwen", "memory")
PROVIDERS = {"local_gpu", "local_cpu", "remote", "disabled"}


def _default_profile() -> Dict[str, Any]:
    return {
        "connections": {},
        "roles": {
            "gemma": {"provider": "local_gpu", "connection_id": None, "model_id": "gemma"},
            "qwen": {"provider": "local_gpu", "connection_id": None, "model_id": "qwen"},
            "memory": {"provider": "disabled", "connection_id": None, "model_id": "memory"},
        },
        "configured": False,
        "revision": 0,
    }


class RuntimeConfigStore:
    """Thread-safe in-memory configuration shared by API requests."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._profile = _default_profile()

    @staticmethod
    def _validate_url(value: str) -> str:
        value = (value or "").strip().rstrip("/")
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Remote endpoints must be complete http:// or https:// URLs.")
        if parsed.username or parsed.password:
            raise ValueError("Credentials must not be embedded in the endpoint URL.")
        if parsed.query or parsed.fragment:
            raise ValueError("Remote endpoint URLs cannot include a query string or fragment.")
        if parsed.path.rstrip("/").endswith("/v1"):
            # Users commonly paste an OpenAI base URL. Internally we retain the
            # server root because both /v1/* and /control/* are used.
            value = value[:-3].rstrip("/")
        return value

    def configure(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        connections: Dict[str, Dict[str, str]] = {}
        raw_connections = payload.get("connections") or []
        if not isinstance(raw_connections, list):
            raise ValueError("connections must be a list")

        for raw in raw_connections:
            connection_id = str(raw.get("id") or "").strip()
            if not connection_id or not connection_id.replace("_", "").replace("-", "").isalnum():
                raise ValueError("Each connection needs an alphanumeric id.")
            if connection_id in connections:
                raise ValueError(f"Duplicate connection id: {connection_id}")
            connections[connection_id] = {
                "id": connection_id,
                "label": str(raw.get("label") or connection_id).strip()[:80],
                "base_url": self._validate_url(str(raw.get("base_url") or "")),
                "api_key": str(raw.get("api_key") or "").strip(),
            }

        incoming_roles = payload.get("roles") or {}
        roles: Dict[str, Dict[str, Any]] = {}
        for role in ROLE_NAMES:
            raw = incoming_roles.get(role) or {}
            provider = str(raw.get("provider") or ("disabled" if role == "memory" else "local_gpu"))
            if provider not in PROVIDERS:
                raise ValueError(f"Unsupported provider for {role}: {provider}")
            if role in {"gemma", "qwen"} and provider in {"disabled", "local_cpu"}:
                raise ValueError(f"{role} must use local_gpu or remote")
            connection_id = raw.get("connection_id")
            if provider == "remote" and connection_id not in connections:
                raise ValueError(f"{role} refers to an unknown remote connection")
            roles[role] = {
                "provider": provider,
                "connection_id": connection_id if provider == "remote" else None,
                "model_id": str(raw.get("model_id") or role).strip(),
            }

        with self._lock:
            revision = int(self._profile.get("revision", 0)) + 1
            self._profile = {
                "connections": connections,
                "roles": roles,
                "configured": True,
                "revision": revision,
            }
            return self.public_snapshot()

    def role(self, role: str) -> Dict[str, Any]:
        if role not in ROLE_NAMES:
            raise KeyError(f"Unknown Flash role: {role}")
        with self._lock:
            binding = copy.deepcopy(self._profile["roles"][role])
            connection_id = binding.get("connection_id")
            binding["connection"] = copy.deepcopy(self._profile["connections"].get(connection_id))
            return binding

    def connection(self, connection_id: str) -> Dict[str, str] | None:
        """Return one private connection, including its key, to trusted backend code."""
        with self._lock:
            connection = self._profile["connections"].get(connection_id)
            return copy.deepcopy(connection) if connection else None

    def public_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            snapshot = copy.deepcopy(self._profile)
        for connection in snapshot["connections"].values():
            connection["has_api_key"] = bool(connection.pop("api_key", ""))
        return snapshot

    def reset(self) -> Dict[str, Any]:
        with self._lock:
            self._profile = _default_profile()
            return self.public_snapshot()


runtime_config = RuntimeConfigStore()
