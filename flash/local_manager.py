"""Local Flash llama-server lifecycle for GPU and CPU roles."""

from __future__ import annotations

import atexit
import os
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

import config
from flash.catalog import load_catalog


ROLE_PORTS = {
    "gemma": config.FLASH_GEMMA_PORT,
    "qwen": config.FLASH_QWEN_PORT,
    "memory": config.FLASH_MEMORY_PORT,
}


class LocalModelProcess:
    def __init__(self, role: str) -> None:
        self.role = role
        self.process: Optional[subprocess.Popen] = None
        self.log_fp = None

    @property
    def port(self) -> int:
        return ROLE_PORTS[self.role]

    @property
    def base_url(self) -> str:
        return f"http://{config.SERVER_HOST}:{self.port}"

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def healthy(self, timeout: float = 1.5) -> bool:
        try:
            return httpx.get(f"{self.base_url}/health", timeout=timeout).status_code == 200
        except Exception:
            return False

    def start(self, provider: str) -> None:
        if os.environ.get("SAGE_MOCK_MODE", "0") == "1":
            return
        if self.running and self.healthy():
            return
        if self.process is not None or self.log_fp is not None:
            self.stop()

        catalog = load_catalog()[self.role]
        model_dir = Path(config.MODEL_DIR or config.BASE_DIR / "models")
        model_path = model_dir / catalog["file"]
        mmproj_path = model_dir / catalog["mmproj"] if catalog.get("mmproj") else None
        if not config.LLAMA_SERVER_PATH or not Path(config.LLAMA_SERVER_PATH).is_file():
            raise FileNotFoundError("llama-server is not configured. Set LLAMA_SERVER_PATH.")
        if not model_path.is_file():
            raise FileNotFoundError(f"Flash {self.role} model not found: {model_path}")
        if mmproj_path and not mmproj_path.is_file():
            raise FileNotFoundError(f"Flash {self.role} mmproj not found: {mmproj_path}")

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.3)
            if sock.connect_ex((config.SERVER_HOST, self.port)) == 0:
                raise RuntimeError(f"Flash {self.role} port {self.port} is already in use")

        ngl = 0 if provider == "local_cpu" else int(catalog.get("ngl", 999))
        cmd = [
            config.LLAMA_SERVER_PATH,
            "-m", str(model_path),
            "-ngl", str(ngl),
            "-c", str(catalog.get("context", 4096)),
            "--port", str(self.port),
            "--host", config.SERVER_HOST,
            "--no-webui",
        ]
        if mmproj_path:
            cmd.extend(["--mmproj", str(mmproj_path)])
        if catalog.get("reasoning"):
            cmd.extend(["--reasoning", str(catalog["reasoning"])])
        if provider == "local_cpu":
            cmd.extend(["-t", str(config.FLASH_MEMORY_CPU_THREADS)])

        log_path = config.TEMP_DIR / "logs" / f"flash_{self.role}.log"
        self.log_fp = open(log_path, "a", encoding="utf-8")
        self.process = subprocess.Popen(cmd, stdout=self.log_fp, stderr=subprocess.STDOUT, text=True)

        deadline = time.time() + config.FLASH_MODEL_START_TIMEOUT
        while time.time() < deadline:
            if self.process.poll() is not None:
                exit_code = self.process.returncode
                self.stop()
                raise RuntimeError(f"Flash {self.role} server exited with code {exit_code}; see {log_path}")
            if self.healthy():
                return
            time.sleep(0.25)
        self.stop()
        raise RuntimeError(f"Flash {self.role} did not become healthy within {config.FLASH_MODEL_START_TIMEOUT}s")

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        self.process = None
        if self.log_fp:
            self.log_fp.close()
            self.log_fp = None


class LocalFlashManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._servers = {role: LocalModelProcess(role) for role in ROLE_PORTS}
        atexit.register(self.stop_all)

    def _gpu_free_gib(self) -> Optional[float]:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=4, check=True,
            )
            values = [float(line.strip()) / 1024 for line in result.stdout.splitlines() if line.strip()]
            return max(values) if values else None
        except Exception:
            return None

    def ensure(self, role: str, provider: str) -> str:
        if role not in self._servers:
            raise KeyError(f"Unknown local Flash role: {role}")
        with self._lock:
            if provider == "local_gpu" and not self._servers[role].running:
                required = float(load_catalog()[role].get("estimated_vram_gib", 0)) + config.FLASH_VRAM_RESERVE_GIB
                free = self._gpu_free_gib()
                if free is not None and free < required:
                    raise RuntimeError(
                        f"Insufficient free VRAM for {role}: {free:.2f} GiB free, {required:.2f} GiB required including reserve"
                    )
            self._servers[role].start(provider)
            return self._servers[role].base_url

    def stop_all(self) -> None:
        with self._lock:
            for server in self._servers.values():
                server.stop()

    def reconcile(self, runtime_snapshot: Dict[str, Any]) -> None:
        """Release local processes for roles that were rebound remotely/disabled."""
        roles = runtime_snapshot.get("roles") or {}
        with self._lock:
            for role, server in self._servers.items():
                if (roles.get(role) or {}).get("provider") not in {"local_gpu", "local_cpu"}:
                    server.stop()

    def status(self) -> Dict[str, Any]:
        mock = os.environ.get("SAGE_MOCK_MODE", "0") == "1"
        result: Dict[str, Any] = {}
        for role, server in self._servers.items():
            running = mock or server.running
            result[role] = {
                "port": server.port,
                "running": running,
                # A stopped managed process cannot be healthy. Avoid three
                # sequential network timeouts during every UI status refresh.
                "healthy": mock or (running and server.healthy()),
            }
        return result


local_flash_manager = LocalFlashManager()
