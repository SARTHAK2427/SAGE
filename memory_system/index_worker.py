"""Low-priority process for derived memory-vector indexing.

The canonical SQL memory write must never wait for BGE or Chroma.  This worker
consumes the durable outbox from a separate process, so heavy model loading and
CPU embedding cannot block SAGE's HTTP/EventSource response lane.
"""

from __future__ import annotations

import multiprocessing
import os
import threading
from typing import Optional

import config


def _lower_process_priority() -> None:
    """Best-effort CPU de-prioritisation; failure must not stop indexing."""
    if os.name != "nt":
        return
    try:
        import ctypes
        BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS,
        )
    except Exception:
        pass


def index_worker_loop(stop_event: multiprocessing.synchronize.Event) -> None:
    _lower_process_priority()
    # BGE remains CPU-only and is restricted to one compute thread here. The
    # main process stays responsive even on a low-core laptop.
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    from memory_system.index_outbox import index_outbox

    poll_seconds = max(0.25, float(getattr(config, "SAGE_INDEX_WORKER_POLL_SECONDS", 1.0)))
    while not stop_event.is_set():
        processed = index_outbox.recover(limit=1)
        stop_event.wait(0.05 if processed else poll_seconds)


class IndexWorkerManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._process: Optional[multiprocessing.Process] = None
        self._stop_event: Optional[multiprocessing.synchronize.Event] = None

    def start(self) -> None:
        if not getattr(config, "SAGE_INDEX_WORKER_ENABLED", True):
            return
        with self._lock:
            if self._process is not None and self._process.is_alive():
                return
            context = multiprocessing.get_context("spawn")
            self._stop_event = context.Event()
            self._process = context.Process(
                target=index_worker_loop,
                args=(self._stop_event,),
                name="sage-memory-indexer",
                daemon=True,
            )
            self._process.start()

    def stop(self) -> None:
        with self._lock:
            process = self._process
            stop_event = self._stop_event
            self._process = None
            self._stop_event = None
        if stop_event is not None:
            stop_event.set()
        if process is not None:
            process.join(timeout=4)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)


index_worker_manager = IndexWorkerManager()
