"""Recent-turn context, scoped semantic recall, and curator scheduling."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import config
from core.observer import observer
from memory_system.quality import is_valid_memory
from sage_memory import sage_memory


def _tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


class MemoryCoordinator:
    @staticmethod
    def _completed_turns(messages: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        completed: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        pending_user: Dict[str, Any] | None = None
        for message in messages:
            if message.get("role") == "user":
                pending_user = message
            elif message.get("role") == "assistant" and pending_user:
                completed.append((pending_user, message))
                pending_user = None
        return completed

    def recent_turns(self, chat_id: str, *, max_turns: int = 5, token_budget: Optional[int] = None) -> Dict[str, Any]:
        budget = max(128, int(token_budget or getattr(config, "SAGE_RECENT_CHAT_BUDGET_TOKENS", 2400)))
        messages = sage_memory.get_messages(chat_id)
        turns: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role") or "")
            if role == "user" and current:
                if any(item.get("role") == "assistant" for item in current):
                    turns.append(current)
                current = []
            current.append(message)
            if role == "assistant":
                turns.append(current)
                current = []
        # Hot memory is defined as completed user/assistant pairs.  A partial
        # ledger turn (for example after a crash) is preserved in PostgreSQL
        # but is not injected as if it were a completed exchange.
        selected: List[List[Dict[str, Any]]] = []
        used = 0
        for turn in reversed(turns[-max_turns:]):
            rendered = "\n".join(f"{m.get('role','unknown').upper()}: {m.get('content','')}" for m in turn)
            cost = _tokens(rendered)
            if selected and used + cost > budget:
                break
            if cost > budget:
                rendered = rendered[-budget * 4:]
                turn = [{"role": "context", "content": "[earlier part omitted]\n" + rendered}]
                cost = _tokens(rendered)
            selected.append(turn)
            used += cost
        selected.reverse()
        flat = [message for turn in selected for message in turn]
        text = "\n".join(f"{m.get('role','unknown').upper()}: {m.get('content','')}" for m in flat)
        return {"messages": flat, "text": text, "turn_count": len(selected), "token_estimate": used}

    def search(self, *, query: str, user_id: str, chat_id: str, scope: str = "both", limit: int = 6, run_id: str | None = None) -> Dict[str, Any]:
        observer.emit(run_id or chat_id, "memory", "semantic_search", "started", "Searching scoped memory",
                      {"query": query, "scope": scope, "chat_id": chat_id, "limit": limit})
        from sage_memory_index import memory_vector_index
        candidates: List[Tuple[float, Dict[str, Any]]] = []
        vector_hits: List[Dict[str, Any]] = []
        requested_chat_ids = ([chat_id] if scope == "cold" else [""] if scope == "global" else [chat_id, ""])
        for scoped_chat_id in requested_chat_ids:
            vector_hits.extend(memory_vector_index.search_memory_vectors(
                user_id=user_id, query=query, tier="cold",
                chat_id=scoped_chat_id, limit=max(limit * 2, 12),
            ))
        deduped_hits: List[Dict[str, Any]] = []
        seen_hit_ids = set()
        for hit in sorted(vector_hits, key=lambda item: float(item.get("distance", 1.0))):
            memory_id = str(hit.get("memory_id") or "")
            if memory_id and memory_id not in seen_hit_ids:
                seen_hit_ids.add(memory_id)
                deduped_hits.append(hit)
        for hit in deduped_hits:
            memory = sage_memory.get_memory(str(hit.get("memory_id") or ""), touch_access=True)
            if not memory or memory.get("status") != "active" or memory.get("user_id") != user_id:
                continue
            source_chat = memory.get("source_chat_id")
            is_global = not source_chat and memory.get("category") == "personal"
            is_unscoped_legacy_cold = not source_chat and not is_global
            if not is_valid_memory(str(memory.get("content") or ""), "global" if is_global else "cold"):
                continue
            allowed = (
                (scope == "global" and is_global)
                or (scope == "cold" and (source_chat == chat_id or (not chat_id and is_unscoped_legacy_cold)))
                or (scope == "both" and (is_global or source_chat == chat_id or (not chat_id and is_unscoped_legacy_cold)))
            )
            if not allowed:
                continue
            relevance = max(0.0, 1.0 - float(hit.get("distance", 1.0)))
            score = relevance * 0.65 + float(memory.get("importance", 0.5)) * 0.2 + float(memory.get("confidence", 1.0)) * 0.15
            item = dict(memory)
            item.update(scope="global" if is_global else "cold", relevance=round(relevance, 4), score=round(score, 4))
            candidates.append((score, item))
        if not candidates:
            fallback = sage_memory.list_memories(user_id=user_id, memory_tier="cold", limit=100)
            words = {word for word in query.lower().split() if len(word) > 2}
            for memory in fallback:
                source_chat = memory.get("source_chat_id")
                is_global = not source_chat and memory.get("category") == "personal"
                is_unscoped_legacy_cold = not source_chat and not is_global
                if not is_valid_memory(str(memory.get("content") or ""), "global" if is_global else "cold"):
                    continue
                if scope == "global" and not is_global:
                    continue
                if scope == "cold" and source_chat != chat_id and not (not chat_id and is_unscoped_legacy_cold):
                    continue
                if scope == "both" and not (is_global or source_chat == chat_id or (not chat_id and is_unscoped_legacy_cold)):
                    continue
                overlap = sum(1 for word in words if word in str(memory.get("content", "")).lower())
                if overlap or not words:
                    item = dict(memory)
                    item.update(scope="global" if is_global else "cold", relevance=round(overlap / max(1, len(words)), 4), score=round(overlap / max(1, len(words)), 4))
                    candidates.append((float(item["score"]), item))
        memories = [item for _, item in sorted(candidates, key=lambda pair: pair[0], reverse=True)[:max(1, min(limit, 20))]]
        context = "\n".join(f"[{m['scope'].upper()} MEMORY {m['memory_id']}] {m['content']}" for m in memories)
        observer.emit(run_id or chat_id, "memory", "semantic_search", "completed",
                      f"Hydrated {len(memories)} memories from the canonical store",
                      {"candidate_ids": [m["memory_id"] for m in memories], "memories": memories})
        return {"memories": memories, "context": context, "count": len(memories)}

    def persist_turn(self, *, chat_id: str, user_id: str, user_text: str, answer: str, run_id: str | None = None) -> Dict[str, Any]:
        user_msg_id = sage_memory.write_message(chat_id, user_id, "user", user_text)
        assistant_msg_id = sage_memory.write_message(chat_id, user_id, "assistant", answer)
        observer.emit(run_id or chat_id, "postgres", "conversation_commit", "completed",
                      "Committed completed turn to the canonical ledger",
                      {"chat_id": chat_id, "user_message_id": user_msg_id, "assistant_message_id": assistant_msg_id})
        from flash.memory_worker import memory_worker
        jobs: List[str] = []
        # Examine the current chat's recent backlog. Idempotency makes repeat
        # scheduling a no-op, while this repairs stable facts from turns made
        # before the curator endpoint was available or this feature existed.
        completed = self._completed_turns(sage_memory.get_messages(chat_id))
        for hot_user, hot_assistant in completed[-config.FLASH_MEMORY_QUEUE_SIZE:]:
            global_job = memory_worker.schedule_turn(
                session_id=chat_id, user_id=user_id,
                user_message=hot_user, assistant_message=hot_assistant,
                run_id=run_id or chat_id, job_type=memory_worker.GLOBAL_EXTRACT,
            )
            if global_job:
                jobs.append(global_job)
        jobs.extend(self.schedule_evicted(chat_id=chat_id, user_id=user_id, run_id=run_id or chat_id))
        return {"user_message_id": user_msg_id, "assistant_message_id": assistant_msg_id, "memory_jobs": jobs}

    def schedule_evicted(self, *, chat_id: str, user_id: str, run_id: str) -> List[str]:
        completed = self._completed_turns(sage_memory.get_messages(chat_id))
        from flash.memory_worker import memory_worker
        jobs: List[str] = []
        for user_message, assistant_message in completed[:-5] if len(completed) > 5 else []:
            job_id = memory_worker.schedule_turn(session_id=chat_id, user_id=user_id,
                                                  user_message=user_message, assistant_message=assistant_message,
                                                  run_id=run_id, job_type=memory_worker.COLD_COMPRESS)
            if job_id:
                jobs.append(job_id)
        return jobs

    def schedule_global_backfill(self, *, user_id: str, limit_chats: int = 5,
                                 turns_per_chat: int = 5, run_id: str = "memory-backfill") -> List[str]:
        """Queue missing immediate-global jobs for recent completed turns.

        Idempotency keys make this safe to run after each ephemeral runtime
        configuration. It repairs recent conversations created while the
        curator endpoint was unavailable without touching Hot/Cold boundaries.
        """
        from flash.memory_worker import memory_worker
        jobs: List[str] = []
        for chat in sage_memory.list_chats(user_id, limit=limit_chats):
            chat_id = str(chat.get("chat_id") or "")
            if not chat_id:
                continue
            completed = self._completed_turns(sage_memory.get_messages(chat_id))
            for user_message, assistant_message in completed[-max(1, turns_per_chat):]:
                job_id = memory_worker.schedule_turn(
                    session_id=chat_id, user_id=user_id,
                    user_message=user_message, assistant_message=assistant_message,
                    run_id=run_id, job_type=memory_worker.GLOBAL_EXTRACT,
                )
                if job_id:
                    jobs.append(job_id)
        return jobs


memory_coordinator = MemoryCoordinator()
