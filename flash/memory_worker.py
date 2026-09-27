"""Non-blocking CPU/remote 2B curator backed by the canonical memory ledger."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
from core.json_repair import clean_json_string
from core.observer import observer
from flash.runtime_config import runtime_config
from flash.transport import flash_transport
from memory_system.job_store import memory_job_store
from memory_system.global_identity import global_memory_identity, same_global_memory
from memory_system.quality import is_valid_memory, memory_rejection_reason
from sage_memory import ALLOWED_CATEGORIES, sage_memory


class FlashMemoryWorker:
    GLOBAL_EXTRACT = "global_extract"
    COLD_COMPRESS = "cold_compress"

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sage-memory-curator")
        self._slots = threading.BoundedSemaphore(config.FLASH_MEMORY_QUEUE_SIZE)
        self._cleared_sessions: set[str] = set()
        self._lock = threading.RLock()

    def schedule(self, session_id: str, user_message: str, answer: str, *, user_id: Optional[str] = None) -> Optional[str]:
        return self.schedule_turn(
            session_id=session_id,
            user_id=user_id or config.DEFAULT_USER_ID,
            user_message={"msg_id": "legacy_user", "content": user_message},
            assistant_message={"msg_id": "legacy_assistant", "content": answer},
            run_id=session_id,
            job_type=self.GLOBAL_EXTRACT,
        )

    def schedule_turn(self, *, session_id: str, user_id: str, user_message: Dict[str, Any],
                      assistant_message: Dict[str, Any], run_id: str,
                      job_type: str = COLD_COMPRESS) -> Optional[str]:
        if job_type not in {self.GLOBAL_EXTRACT, self.COLD_COMPRESS}:
            raise ValueError(f"Unsupported memory job type: {job_type}")
        if not config.SAGE_MEMORY_CURATOR_ENABLED or runtime_config.role("memory")["provider"] == "disabled":
            observer.emit(run_id, "memory-2b", "curation_queue", "failed", "Memory curator is disabled",
                          {"job_type": job_type, "provider": runtime_config.role("memory")["provider"]})
            return None
        with self._lock:
            if session_id in self._cleared_sessions:
                return None
        user_id_value = str(user_message.get("msg_id") or "")
        assistant_id = str(assistant_message.get("msg_id") or "")
        key = f"{job_type}:{session_id}:{user_id_value}:{assistant_id}"
        payload = {
            "session_id": session_id,
            "user_id": user_id,
            "run_id": run_id,
            "user_message": user_message,
            "assistant_message": assistant_message,
            "job_type": job_type,
        }
        job_id = memory_job_store.create(key=key, user_id=user_id, chat_id=session_id, payload=payload)
        if not job_id:
            return None
        summary = "Queued immediate global memory extraction" if job_type == self.GLOBAL_EXTRACT else "Queued evicted turn for cold-memory compression"
        observer.emit(run_id, "memory-2b", "curation_queue", "queued", summary,
                      {"job_id": job_id, "job_type": job_type, "source_message_ids": [user_id_value, assistant_id],
                       "provider": runtime_config.role("memory").get("provider")})
        if not self._slots.acquire(blocking=False):
            memory_job_store.update(job_id, "queued", error="Worker queue is full; awaiting retry")
            return job_id
        memory_job_store.update(job_id, "scheduled")
        self._executor.submit(self._run, job_id, payload)
        return job_id

    @staticmethod
    def _parse(raw: str) -> List[Dict[str, Any]]:
        parsed: Any = None
        for candidate in (raw, clean_json_string(raw)):
            try:
                parsed = json.loads(candidate, strict=False)
                break
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        if not isinstance(parsed, dict) or not isinstance(parsed.get("memories"), list):
            return []
        validated: List[Dict[str, Any]] = []
        for item in parsed["memories"][:12]:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            scope = str(item.get("scope") or "cold").lower()
            category = str(item.get("category") or "summary").lower()
            if not content or scope not in {"cold", "global"} or category not in ALLOWED_CATEGORIES:
                continue
            # Global memory is a small, stable personal profile. Preferences,
            # projects, instructions, and topical facts remain chat-scoped Cold.
            if scope == "global" and category != "personal":
                continue
            if not is_valid_memory(content, scope):
                continue
            validated.append({
                "content": content[:8000], "scope": scope, "category": category,
                "importance": max(0.0, min(1.0, float(item.get("importance", 0.6)))),
                "confidence": max(0.0, min(1.0, float(item.get("confidence", 0.8)))),
            })
        return validated

    @staticmethod
    def _chunks(user_text: str, assistant_text: str) -> List[str]:
        combined = f"USER:\n{user_text}\n\nASSISTANT:\n{assistant_text}"
        size = max(2000, config.SAGE_MEMORY_CURATOR_CHUNK_CHARS)
        return [combined[i:i + size] for i in range(0, len(combined), size)] or [combined]

    def _run(self, job_id: str, payload: Dict[str, Any]) -> None:
        run_id = str(payload.get("run_id") or payload["session_id"])
        job_type = str(payload.get("job_type") or self.COLD_COMPRESS)
        try:
            with self._lock:
                if payload["session_id"] in self._cleared_sessions:
                    memory_job_store.update(job_id, "cancelled", error="Chat was cleared before curation")
                    return
            memory_job_store.update(job_id, "running")
            user_text = str(payload["user_message"].get("content") or "")
            assistant_text = str(payload["assistant_message"].get("content") or "")
            chunks = self._chunks(user_text, assistant_text)
            task_label = "global fact extraction" if job_type == self.GLOBAL_EXTRACT else "cold-memory compression"
            observer.emit(run_id, "memory-2b", "curation", "started", f"Background {task_label} started",
                          {"job_id": job_id, "job_type": job_type, "chunk_count": len(chunks),
                           "input_chars": len(user_text) + len(assistant_text),
                           "provider": runtime_config.role("memory").get("provider")})
            prompt = (config.PROMPTS_DIR / "memory_curator_system.txt").read_text(encoding="utf-8")
            mode_instruction = (
                "TASK MODE: GLOBAL EXTRACTION. This turn is still in Hot memory. Return only category=personal, scope=global records for stable personal identity or biographical information explicitly stated by the user (for example name, age, pronouns, or location). Do not store preferences, projects, instructions, device setup, events, or topical facts globally; they belong to chat-scoped Cold memory after eviction. Do not create a cold summary in this mode."
                if job_type == self.GLOBAL_EXTRACT else
                "TASK MODE: COLD COMPRESSION. This turn has just left the five-turn Hot window. Return only scope=cold records that preserve useful chat-specific episode context."
            )
            proposed: List[Dict[str, Any]] = []
            for index, chunk in enumerate(chunks):
                observer.emit(run_id, "memory-2b", "model_input", "started", f"Curating chunk {index + 1}/{len(chunks)}",
                              {"job_id": job_id, "job_type": job_type, "chunk_index": index, "content": chunk,
                               "provider": runtime_config.role("memory").get("provider")})
                response = flash_transport.invoke("memory", [
                    {"role": "system", "content": f"{prompt}\n\n{mode_instruction}"},
                    {"role": "user", "content": chunk},
                ], json_mode=True, temperature=0.0)
                observer.emit(run_id, "memory-2b", "model_output", "completed", f"Curator returned chunk {index + 1}",
                              {"job_id": job_id, "job_type": job_type, "content": response.get("content", ""),
                               "provider": runtime_config.role("memory").get("provider")},
                              duration_ms=float(response.get("duration", 0)) * 1000)
                allowed_scope = "global" if job_type == self.GLOBAL_EXTRACT else "cold"
                proposed.extend(item for item in self._parse(str(response.get("content") or "")) if item["scope"] == allowed_scope)

            stored: List[Dict[str, Any]] = []
            with self._lock:
                if payload["session_id"] in self._cleared_sessions:
                    memory_job_store.update(job_id, "cancelled", error="Chat was cleared during curation")
                    return
            seen = set()
            existing = sage_memory.list_memories(user_id=payload["user_id"], memory_tier="cold", limit=500)
            removed: List[Dict[str, str]] = []
            for memory in existing:
                if memory.get("source_chat_id"):
                    continue
                reason = memory_rejection_reason(str(memory.get("content") or ""), "global")
                if reason and sage_memory.delete_memory(str(memory["memory_id"])):
                    removed.append({"memory_id": str(memory["memory_id"]), "reason": reason})
            if removed:
                observer.emit(run_id, "memory-2b", "memory_quality_cleanup", "completed",
                              f"Quarantined {len(removed)} invalid global memories",
                              {"job_id": job_id, "removed": removed})
            existing = [
                memory for memory in existing
                if memory.get("source_chat_id") or is_valid_memory(str(memory.get("content") or ""), "global")
            ]
            existing_global = [memory for memory in existing if not memory.get("source_chat_id")]
            recategorized_personal: List[str] = []
            # Preserve valid identity records from earlier releases while
            # moving them to the sole permitted Global category.
            for index, memory in enumerate(existing_global):
                if memory.get("category") == "personal":
                    continue
                if not global_memory_identity(str(memory.get("content") or "")):
                    continue
                updated = sage_memory.update_memory(str(memory["memory_id"]), category="personal")
                if updated:
                    existing_global[index] = updated
                    recategorized_personal.append(str(updated["memory_id"]))
            if recategorized_personal:
                observer.emit(run_id, "memory-2b", "global_category_cleanup", "completed",
                              f"Moved {len(recategorized_personal)} identity records to personal Global memory",
                              {"job_id": job_id, "memory_ids": recategorized_personal})
            merged_duplicates: List[Dict[str, str]] = []
            # Repair prior duplicates as soon as the curator sees a user's
            # Global set. Keep the highest-ranked canonical record and mark
            # exact stable-slot/value duplicates deleted.
            seen_global_identities: dict[tuple[str, str], Dict[str, Any]] = {}
            for memory in existing_global:
                identity = global_memory_identity(str(memory.get("content") or ""))
                if not identity:
                    continue
                keeper = seen_global_identities.get(identity)
                if keeper is None:
                    seen_global_identities[identity] = memory
                    continue
                if sage_memory.delete_memory(str(memory["memory_id"])):
                    merged_duplicates.append({
                        "deleted_memory_id": str(memory["memory_id"]),
                        "kept_memory_id": str(keeper["memory_id"]),
                        "slot": identity[0],
                    })
            if merged_duplicates:
                observer.emit(run_id, "memory-2b", "global_deduplication", "completed",
                              f"Merged {len(merged_duplicates)} duplicate global memories",
                              {"job_id": job_id, "merged": merged_duplicates})
                deleted_ids = {item["deleted_memory_id"] for item in merged_duplicates}
                existing = [memory for memory in existing if str(memory.get("memory_id")) not in deleted_ids]
                existing_global = [memory for memory in existing if not memory.get("source_chat_id")]

            existing_keys = {(str(m.get("source_chat_id") or "global"), " ".join(str(m.get("content") or "").lower().split())) for m in existing}
            # PostgreSQL stores source_msg_id as UUID.  Keep the canonical
            # user message as the source pointer; the complete pair remains in
            # the durable job payload/result and Observer event.
            source_msg_id = str(payload["user_message"].get("msg_id") or "") or None
            for item in proposed:
                normalized = " ".join(item["content"].lower().split())
                scope_key = payload["session_id"] if item["scope"] == "cold" else "global"
                key = (scope_key, normalized)
                if key in seen or key in existing_keys:
                    continue
                seen.add(key)
                if item["scope"] == "global":
                    candidate_identity = global_memory_identity(item["content"])
                    same_identity = None
                    same_slot = None
                    for memory in existing_global:
                        existing_identity = global_memory_identity(str(memory.get("content") or ""))
                        if candidate_identity and existing_identity:
                            if existing_identity == candidate_identity:
                                same_identity = memory
                                break
                            if existing_identity[0] == candidate_identity[0]:
                                same_slot = memory
                        elif same_global_memory(item["content"], str(memory.get("content") or "")):
                            same_identity = memory
                            break
                    if same_identity:
                        observer.emit(run_id, "memory-2b", "global_deduplication", "completed",
                                      "Skipped a duplicate global memory candidate",
                                      {"job_id": job_id, "candidate": item["content"],
                                       "existing_memory_id": str(same_identity["memory_id"]),
                                       "slot": candidate_identity[0] if candidate_identity else None})
                        continue
                    if same_slot:
                        record = sage_memory.supersede_memory(
                            str(same_slot["memory_id"]), item["content"],
                            user_id=payload["user_id"], category=item["category"],
                            importance=item["importance"], confidence=item["confidence"],
                            source_chat_id=None, source_msg_id=source_msg_id, memory_tier="cold",
                        )
                        record["scope"] = "global"
                        record["dedup_action"] = "superseded_same_slot"
                        stored.append(record)
                        existing_global = [memory for memory in existing_global if str(memory["memory_id"]) != str(same_slot["memory_id"])]
                        existing_global.append(record)
                        existing_keys.add(("global", normalized))
                        continue
                record = sage_memory.store_memory(
                    user_id=payload["user_id"], content=item["content"], category=item["category"],
                    importance=item["importance"], confidence=item["confidence"],
                    source_chat_id=payload["session_id"] if item["scope"] == "cold" else None,
                    source_msg_id=source_msg_id, memory_tier="cold",
                )
                record["scope"] = item["scope"]
                stored.append(record)
                if item["scope"] == "global":
                    existing_global.append(record)
                    existing_keys.add(("global", normalized))
            result = {"job_type": job_type, "proposed": proposed, "stored": stored,
                      "removed_invalid_global": removed, "merged_duplicate_globals": merged_duplicates,
                      "recategorized_personal_globals": recategorized_personal,
                      "chunk_count": len(chunks)}
            memory_job_store.update(job_id, "completed", result=result)
            observer.emit(run_id, "memory-2b", "curation", "completed", f"Stored {len(stored)} curated memories",
                          {"job_id": job_id, "provider": runtime_config.role("memory").get("provider"), **result})
        except Exception as exc:
            memory_job_store.update(job_id, "failed", error=str(exc))
            observer.emit(run_id, "memory-2b", "curation", "failed", "Memory curation failed",
                          {"job_id": job_id, "job_type": job_type,
                           "provider": runtime_config.role("memory").get("provider"), "error": str(exc)})
        finally:
            self._slots.release()
            self._drain_queued(limit=1)

    def _drain_queued(self, limit: int | None = None) -> int:
        submitted = 0
        queued = list(reversed(memory_job_store.list(status="queued", limit=config.FLASH_MEMORY_QUEUE_SIZE)))
        for job in queued:
            if limit is not None and submitted >= limit:
                break
            if not self._slots.acquire(blocking=False):
                break
            job_id = str(job["job_id"])
            memory_job_store.update(job_id, "scheduled", error=None)
            self._executor.submit(self._run, job_id, dict(job["payload"]))
            submitted += 1
        return submitted

    def recover_pending(self) -> int:
        recovered = 0
        for job in reversed(memory_job_store.list(limit=config.FLASH_MEMORY_QUEUE_SIZE)):
            if job.get("status") not in {"queued", "scheduled", "running"}:
                continue
            if not self._slots.acquire(blocking=False):
                break
            job_id = str(job["job_id"])
            memory_job_store.update(job_id, "scheduled", error=None)
            self._executor.submit(self._run, job_id, dict(job["payload"]))
            recovered += 1
        return recovered

    def recent(self, session_id: str, limit: int = 4) -> List[Dict[str, Any]]:
        return [job for job in memory_job_store.list(limit=200) if job.get("chat_id") == session_id][:limit]

    def clear_session(self, session_id: str) -> int:
        with self._lock:
            self._cleared_sessions.add(session_id)
        return memory_job_store.cancel_chat(session_id)


memory_worker = FlashMemoryWorker()
