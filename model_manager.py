import os
import sys
import time
import socket
import atexit
import subprocess
import httpx
from typing import Optional, Dict, Any
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

import config

# Initialize UTF-8 safe Rich Console
console = Console(highlight=False, legacy_windows=False)

class ModelManager:
    def __init__(self):
        self.current_model_key: Optional[str] = None
        self.server_process: Optional[subprocess.Popen] = None
        self.switch_count: int = 0
        self.log_files: Dict[str, Any] = {}

    def is_port_in_use(self, port: int, host: str = "127.0.0.1") -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex((host, port)) == 0

    def wait_for_port_free(self, port: int, timeout: float = 10.0) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            if not self.is_port_in_use(port, config.SERVER_HOST):
                return True
            time.sleep(0.3)
        return False

    def is_healthy(self, timeout: float = 2.0) -> bool:
        try:
            r = httpx.get(f"{config.LLAMA_BASE_URL}/health", timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False

    def wait_for_healthy(self, timeout: float = 90.0) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            if self.server_process and self.server_process.poll() is not None:
                console.print(f"[bold red][ERROR] Server process exited unexpectedly with code {self.server_process.returncode}[/bold red]")
                return False
            if self.is_healthy(timeout=1.5):
                return True
            time.sleep(0.5)
        return False

    def stop_current(self) -> None:
        if self.server_process:
            model_name = self.current_model_key or "unknown"
            console.print(f"[yellow][STOP] Stopping llama-server ({model_name})...[/yellow]")
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=8.0)
            except subprocess.TimeoutExpired:
                console.print("[red][WARN] Server did not exit gracefully, force killing...[/red]")
                self.server_process.kill()
                self.server_process.wait()
            except Exception as e:
                console.print(f"[red]Error stopping server: {e}[/red]")
            finally:
                self.server_process = None
                self.current_model_key = None
        
        self.wait_for_port_free(config.LLAMA_PORT, timeout=5.0)

    def ensure_model(self, model_key: str) -> bool:
        if model_key not in config.MODELS:
            raise ValueError(f"Unknown model key: {model_key}. Available: {list(config.MODELS.keys())}")

        if self.current_model_key == model_key and self.is_healthy():
            return True

        model_cfg = config.MODELS[model_key]
        model_path = model_cfg["model_path"]

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        if model_cfg.get("mmproj") and not os.path.exists(model_cfg["mmproj"]):
            raise FileNotFoundError(f"MMPROJ file not found at: {model_cfg['mmproj']}")

        if not os.path.exists(config.LLAMA_SERVER_PATH):
            raise FileNotFoundError(f"llama-server.exe not found at: {config.LLAMA_SERVER_PATH}")

        # Stop existing model if running
        if self.current_model_key is not None or self.server_process is not None or self.is_port_in_use(config.LLAMA_PORT):
            prev = self.current_model_key or "previous instance"
            console.print(Panel(f"[bold cyan]MODEL SWITCH[/bold cyan]\n[dim]{prev}[/dim] -> [bold green]{model_cfg['name']}[/bold green]", border_style="cyan"))
            self.stop_current()
            self.switch_count += 1

        # Prepare launch command
        cmd = [
            config.LLAMA_SERVER_PATH,
            "-m", model_path,
            "-ngl", str(model_cfg.get("ngl", 999)),
            "-c", str(model_cfg.get("context", 8192)),
            "--port", str(config.LLAMA_PORT),
            "--host", config.SERVER_HOST,
            "--no-webui"
        ]

        if model_cfg.get("mmproj"):
            cmd.extend(["--mmproj", model_cfg["mmproj"]])

        if model_cfg.get("reasoning"):
            cmd.extend(["--reasoning", str(model_cfg["reasoning"])])

        # Setup log file
        log_file_path = config.TEMP_DIR / "logs" / f"{model_key}.log"
        log_fp = open(log_file_path, "a", encoding="utf-8")

        console.print(f"[bold blue][LAUNCH] Starting llama-server:[/bold blue] {model_cfg['name']}")
        console.print(f"   [dim]Path: {model_path}[/dim]")
        console.print(f"   [dim]Context: {model_cfg.get('context')} | NGL: {model_cfg.get('ngl')} | Port: {config.LLAMA_PORT}[/dim]")

        start_time = time.time()
        self.server_process = subprocess.Popen(
            cmd,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            text=True
        )

        healthy = self.wait_for_healthy(timeout=config.MODEL_START_TIMEOUT)
        load_time = time.time() - start_time

        if not healthy:
            console.print(f"[bold red][ERROR] Failed to start {model_cfg['name']} within {config.MODEL_START_TIMEOUT}s[/bold red]")
            try:
                log_fp.flush()
                with open(log_file_path, "r", encoding="utf-8") as f:
                    recent_logs = f.readlines()[-20:]
                console.print(Panel("".join(recent_logs), title="Startup Log Tail", border_style="red"))
            except Exception:
                pass
            self.stop_current()
            raise RuntimeError(f"Server for {model_cfg['name']} failed health check.")

        self.current_model_key = model_key
        console.print(f"[bold green][OK] Model ready in {load_time:.2f}s[/bold green] ({model_cfg['name']})\n")
        return True

model_manager = ModelManager()
atexit.register(model_manager.stop_current)
