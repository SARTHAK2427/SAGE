"""Flash execution flow: Gemma routes, Qwen sees, Gemma optionally synthesizes."""

from __future__ import annotations

import base64
import json
import mimetypes
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import config
from core.json_repair import clean_json_string
from core.observer import observer
from flash.transport import flash_transport
from flash.runtime_config import runtime_config
from memory_system.coordinator import memory_coordinator


VALID_CASES = {"A", "B", "C", "D"}
IMAGE_SUFFIXES = {"png", "jpg", "jpeg", "webp", "gif", "bmp"}
TEXT_SUFFIXES = {"txt", "md", "csv", "json", "log", "py", "js", "html", "xml", "yaml", "yml"}
ATTACHMENT_EVIDENCE_PREFIX = "CHAT ATTACHMENT EVIDENCE"
ATTACHMENT_EVIDENCE_MAX_CHARS = 9000


def _prompt(name: str) -> str:
    path = config.PROMPTS_DIR / name
    return path.read_text(encoding="utf-8").strip()


def _parse_route(raw: str) -> Dict[str, Any]:
    for candidate in (raw, clean_json_string(raw)):
        try:
            parsed = json.loads(candidate, strict=False)
            if isinstance(parsed, dict) and parsed.get("flash_case"):
                return parsed
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    # Keep Flash useful if a model ignores JSON mode: its text becomes Case A.
    return {"flash_case": "A", "gemma_answer": raw.strip()}


def _image_parts(file_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    for info in file_map.values():
        if str(info.get("type", "")).lower() not in IMAGE_SUFFIXES:
            continue
        path = Path(str(info.get("path") or ""))
        if not path.is_file():
            continue
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
    return parts


def _has_readable_images(file_map: Dict[str, Dict[str, Any]]) -> bool:
    """Whether this turn contains an image Qwen can actually receive."""
    for info in file_map.values():
        if str(info.get("type", "")).lower() not in IMAGE_SUFFIXES:
            continue
        if Path(str(info.get("path") or "")).is_file():
            return True
    return False


def _document_context(
    objective: str,
    attachments: List[Dict[str, Any]],
    file_map: Dict[str, Dict[str, Any]],
    max_chars: int = 16000,
    run_id: str | None = None,
) -> str:
    """Build grounded document context for Gemma.

    Plain-text direct attachments remain readable without loading the document
    index. Ingested documents use the rich Daksh RAG socket so retrieval input,
    ranked output, model configuration, reranking, timing, and failures are all
    visible in SAGE Observer instead of being silently discarded.
    """
    chunks: List[str] = []
    remaining = max_chars
    for item in attachments:
        if remaining <= 0:
            break
        ref = str(item.get("ref") or "")
        suffix = str(item.get("type") or "").lower()
        info = file_map.get(ref) or {}
        name = str(item.get("name") or ref)
        text = ""
        if suffix in TEXT_SUFFIXES:
            path = Path(str(info.get("path") or ""))
            if path.is_file():
                text = path.read_text(encoding="utf-8", errors="replace")
        elif item.get("doc_id") and suffix not in IMAGE_SUFFIXES:
            doc_id = str(item["doc_id"])
            query = objective.strip() or "document overview"
            retrieval_started = time.perf_counter()
            if run_id:
                observer.emit(
                    run_id, "document-rag", "semantic_search", "started",
                    f"Searching {name} with BGE document retrieval",
                    {"query": query, "doc_ids": [doc_id], "top_k": 8, "attachment_ref": ref},
                )
            try:
                from db_service import document_db
                socket = document_db.rag_search_socket(
                    query=query,
                    doc_ids=[doc_id],
                    top_k=8,
                )
                if socket.get("status") != "success":
                    error = socket.get("error") or {}
                    raise RuntimeError(str(error.get("message") or "Document RAG search failed"))
                records = (socket.get("result") or {}).get("records") or []
                retrieved: List[str] = []
                for record in records:
                    record_text = str(record.get("text") or "").strip()
                    if not record_text:
                        continue
                    provenance = [
                        f"record={record.get('record_id')}",
                        f"doc={record.get('doc_id') or doc_id}",
                    ]
                    if record.get("page") is not None:
                        provenance.append(f"page={record.get('page')}")
                    if record.get("reranker_score") is not None:
                        provenance.append(f"reranker_score={record.get('reranker_score')}")
                    elif record.get("derived_cosine_similarity") is not None:
                        provenance.append(f"similarity={record.get('derived_cosine_similarity')}")
                    retrieved.append(f"[RAG {'; '.join(provenance)}]\n{record_text}")
                text = "\n\n".join(retrieved)
                if run_id:
                    observer.emit(
                        run_id, "document-rag", "semantic_search", "completed",
                        f"Retrieved {len(retrieved)} grounded passages from {name}",
                        socket,
                        duration_ms=float((socket.get("timing") or {}).get("duration_ms") or 0.0),
                    )
            except Exception as exc:
                # Ingestion metadata still reaches Gemma; retrieval failure should
                # not discard an otherwise answerable request.
                text = ""
                if run_id:
                    observer.emit(
                        run_id, "document-rag", "semantic_search", "failed",
                        f"Could not retrieve passages from {name}",
                        {"query": query, "doc_ids": [doc_id], "attachment_ref": ref, "error": str(exc)},
                        duration_ms=(time.perf_counter() - retrieval_started) * 1000,
                    )
        if text:
            excerpt = text[:remaining]
            chunks.append(f"DOCUMENT [{ref}] {name}:\n{excerpt}")
            remaining -= len(excerpt)
    return "\n\n".join(chunks)


def _recent_attachment_evidence(chat_id: str, user_id: str, max_chars: int = 9000) -> tuple[str, List[str]]:
    """Return prior grounded attachment evidence for this chat only.

    This is deliberately separate from general Cold-memory retrieval: an image
    or document reviewed one turn ago must remain available even when a tiny
    routing model does not formulate the perfect semantic retrieval query.
    """
    try:
        from sage_memory import sage_memory
        records = sage_memory.list_memories(
            user_id=user_id, category="summary", memory_tier="cold", chat_id=chat_id, limit=80,
        )
    except Exception:
        return "", []
    evidence = [record for record in records if str(record.get("content") or "").startswith(ATTACHMENT_EVIDENCE_PREFIX)]
    selected: List[str] = []
    selected_ids: List[str] = []
    remaining = max_chars
    # list_memories is newest-first; retain that ordering for natural follow-up.
    for record in evidence:
        content = str(record.get("content") or "")
        if remaining <= 0:
            break
        excerpt = content[:remaining]
        selected.append(excerpt)
        selected_ids.append(str(record.get("memory_id") or ""))
        remaining -= len(excerpt)
    return "\n\n---\n\n".join(selected), [memory_id for memory_id in selected_ids if memory_id]


def _global_memory_context(user_id: str) -> tuple[str, List[str]]:
    """Load a small, canonical cross-chat profile without semantic retrieval.

    This is intentionally a SQL read, not BGE/Chroma RAG. Stable Global
    records must be reliably available in every chat, including a greeting or
    a direct identity question, while keeping the latency-sensitive Flash path
    free of embedding-model startup.
    """
    try:
        from memory_system.quality import is_valid_memory
        from sage_memory import sage_memory

        records = sage_memory.list_memories(
            user_id=user_id,
            memory_tier="cold",
            status="active",
            limit=max(20, config.SAGE_GLOBAL_MEMORY_MAX_ITEMS * 4),
        )
    except Exception:
        return "", []

    selected: List[str] = []
    selected_ids: List[str] = []
    remaining_tokens = max(0, config.SAGE_GLOBAL_MEMORY_BUDGET_TOKENS)
    for record in records:
        # Global facts deliberately have no source chat. Cold records remain
        # chat-scoped and must not leak into every unrelated conversation.
        if record.get("source_chat_id") or record.get("category") != "personal":
            continue
        content = str(record.get("content") or "").strip()
        if not content or not is_valid_memory(content, "global"):
            continue
        estimated_tokens = max(1, (len(content) + 3) // 4)
        if selected and estimated_tokens > remaining_tokens:
            continue
        if estimated_tokens > remaining_tokens:
            content = content[:remaining_tokens * 4].rstrip()
            estimated_tokens = max(1, (len(content) + 3) // 4)
        if not content:
            break
        memory_id = str(record.get("memory_id") or "")
        selected.append(f"[GLOBAL MEMORY {memory_id}] {content}")
        if memory_id:
            selected_ids.append(memory_id)
        remaining_tokens -= estimated_tokens
        if len(selected) >= config.SAGE_GLOBAL_MEMORY_MAX_ITEMS or remaining_tokens <= 0:
            break
    return "\n".join(selected), selected_ids


def _store_attachment_evidence(
    *, chat_id: str, user_id: str, attachments: List[Dict[str, Any]],
    document_context: str, visual_evidence: str, run_id: str,
) -> List[str]:
    """Persist grounded attachment output as chat-scoped, immediately usable memory."""
    if not attachments:
        return []
    attachment_labels = [
        f"{item.get('name') or item.get('ref') or 'attachment'}"
        f" ({item.get('type') or 'unknown'})"
        for item in attachments
    ]
    parts = [f"{ATTACHMENT_EVIDENCE_PREFIX}\nAttachments: " + ", ".join(attachment_labels)]
    if document_context.strip():
        parts.append("DOCUMENT EXCERPTS / RETRIEVAL:\n" + document_context.strip())
    if visual_evidence.strip():
        parts.append("VISION EVIDENCE (Qwen, grounded in the uploaded image):\n" + visual_evidence.strip())
    if len(parts) == 1:
        return []
    content = "\n\n".join(parts)[:ATTACHMENT_EVIDENCE_MAX_CHARS]
    try:
        from sage_memory import sage_memory
        stored = sage_memory.store_memory(
            user_id=user_id, content=content, category="summary", importance=0.9, confidence=0.95,
            source_chat_id=chat_id, memory_tier="cold",
        )
        memory_id = str(stored.get("memory_id") or "")
        observer.emit(run_id, "memory", "attachment_evidence", "completed", "Stored chat-scoped attachment evidence",
                      {"memory_id": memory_id, "attachments": attachment_labels, "characters": len(content)})
        return [memory_id] if memory_id else []
    except Exception as exc:
        observer.emit(run_id, "memory", "attachment_evidence", "failed", "Attachment evidence was not persisted",
                      {"error": str(exc), "attachments": attachment_labels})
        return []


class FlashService:
    def run(
        self,
        *,
        objective: str,
        attachments: List[Dict[str, Any]],
        file_map: Dict[str, Dict[str, Any]],
        session_id: str | None = None,
        observer_run_id: str | None = None,
        user_id: str | None = None,
        temperature: float | None = None,
        save_history: bool = True,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        session_id = (session_id or "").strip() or f"flash_{uuid.uuid4().hex[:12]}"
        run_id = (observer_run_id or "").strip() or session_id
        user_id = (user_id or config.DEFAULT_USER_ID).strip() or config.DEFAULT_USER_ID
        observer.emit(run_id, "sage", "request", "started", "Flash request accepted",
                      {"objective": objective, "attachments": attachments, "save_history": save_history})
        recent = memory_coordinator.recent_turns(
            session_id,
            max_turns=config.SAGE_RECENT_CHAT_MAX_TURNS,
            token_budget=config.SAGE_RECENT_CHAT_BUDGET_TOKENS,
        ) if save_history else {"text": "", "turn_count": 0, "token_estimate": 0}
        attachment_summary = [
            {k: item.get(k) for k in ("ref", "name", "type", "doc_id", "status")}
            for item in attachments
        ]
        context_sections: List[str] = []
        if recent.get("text"):
            context_sections.append("RECENT CONVERSATION (verbatim, bounded):\n" + str(recent["text"]))
        recalled_memory_ids: List[str] = []
        attachment_memory_ids: List[str] = []
        memory_recall_seconds = 0.0
        global_context = ""
        if save_history:
            attachment_evidence, attachment_memory_ids = _recent_attachment_evidence(session_id, user_id)
            if attachment_evidence:
                context_sections.append("CHAT-SCOPED ATTACHMENT EVIDENCE (grounded prior image/document analysis; use it for follow-ups without requesting re-upload):\n" + attachment_evidence)
            global_context, global_memory_ids = _global_memory_context(user_id)
            if global_context:
                recalled_memory_ids.extend(global_memory_ids)
        current_turn = "CURRENT USER MESSAGE (authoritative):\n" + objective.strip()
        shared_text = ""
        if attachment_summary:
            current_turn += "\n\nCURRENT-TURN ATTACHMENTS:\n" + json.dumps(attachment_summary, ensure_ascii=False)
            shared_text = _document_context(objective, attachments, file_map, run_id=run_id)
            if shared_text:
                current_turn += "\n\nCURRENT-TURN SHARED DOCUMENT CONTENT:\n" + shared_text
        context_sections.append(current_turn)
        user_context = "\n\n---\n\n".join(context_sections)
        observer.emit(run_id, "memory", "recent_context", "completed",
                      f"Selected {recent.get('turn_count', 0)} recent turns",
                      {"turn_count": recent.get("turn_count", 0), "token_estimate": recent.get("token_estimate", 0),
                       "context": recent.get("text", ""), "global_memory_ids": recalled_memory_ids})

        gemma_system = _prompt("flash_system.txt")
        if global_context:
            gemma_system += (
                "\n\nAUTHORITATIVE KNOWN USER PERSONAL INFORMATION:\n"
                + global_context
                + "\nUse applicable facts here as known user information. "
                  "Do not say an applicable supplied fact is unknown."
            )
        gemma_messages = [
            {"role": "system", "content": gemma_system},
            {"role": "user", "content": user_context},
        ]
        observer.emit(run_id, "gemma", "model_input", "started", "Sent routing request to Gemma",
                      {"messages": gemma_messages, "json_mode": True, "temperature": temperature,
                       "provider": runtime_config.role("gemma").get("provider")})
        gemma = flash_transport.invoke(
            "gemma",
            gemma_messages,
            json_mode=True,
            temperature=temperature,
        )
        observer.emit(run_id, "gemma", "model_output", "completed", "Gemma returned a routing decision",
                      {"content": gemma.get("content", ""), "usage": gemma.get("usage", {}), "timings": gemma.get("timings", {}),
                       "provider": runtime_config.role("gemma").get("provider")},
                      duration_ms=float(gemma.get("duration", 0)) * 1000)
        gemma_total_duration = float(gemma.get("duration", 0))
        route = _parse_route(gemma["content"])
        memory_request = route.get("memory") if isinstance(route.get("memory"), dict) else None
        if memory_request:
            recall_started = time.perf_counter()
            query = str(memory_request.get("query") or objective).strip()
            scope = str(memory_request.get("scope") or "both").lower()
            if scope not in {"cold", "global", "both"}:
                scope = "both"
            recalled = memory_coordinator.search(
                query=query, user_id=user_id, chat_id=session_id, scope=scope,
                limit=config.SAGE_MEMORY_RECALL_LIMIT, run_id=run_id,
            )
            recalled_memory_ids = list(dict.fromkeys([
                *recalled_memory_ids,
                *[str(item.get("memory_id")) for item in recalled["memories"]],
            ]))
            recall_messages = [
                {"role": "system", "content": gemma_system},
                {"role": "user", "content": user_context},
                {"role": "user", "content": "MEMORY SEARCH RESULTS (authoritative records):\n" + (recalled["context"] or "No relevant memory was found.") + "\n\nNow answer the current user message. Do not request memory again."},
            ]
            observer.emit(run_id, "gemma", "model_input", "started", "Returned hydrated memory records to Gemma",
                          {"messages": recall_messages, "memory_ids": recalled_memory_ids,
                           "provider": runtime_config.role("gemma").get("provider")})
            gemma = flash_transport.invoke("gemma", recall_messages, json_mode=True, temperature=temperature)
            observer.emit(run_id, "gemma", "model_output", "completed", "Gemma completed routing after memory recall",
                          {"content": gemma.get("content", ""), "usage": gemma.get("usage", {}),
                           "provider": runtime_config.role("gemma").get("provider")},
                          duration_ms=float(gemma.get("duration", 0)) * 1000)
            gemma_total_duration += float(gemma.get("duration", 0))
            route = _parse_route(gemma["content"])
            memory_recall_seconds += time.perf_counter() - recall_started
        case = str(route.get("flash_case") or "A").upper()
        if case not in VALID_CASES:
            case = "A"

        # Qwen receives direct image files only.  A PDF/document may already
        # have supplied grounded text through the document RAG above, but it
        # cannot be handed to Qwen as pixels.  Let Gemma correct an accidental
        # visual route using that supplied evidence instead of failing a valid
        # document request or applying a brittle static route override.
        if case in {"B", "C", "D"} and shared_text and not _has_readable_images(file_map):
            correction = (
                "ROUTE CORRECTION — CAPABILITY BOUNDARY:\n"
                "There is no readable image attachment available to Qwen on this turn. "
                "The current attachment is a document, and CURRENT-TURN SHARED DOCUMENT CONTENT "
                "contains its grounded excerpts. Answer the user's request from that evidence. "
                "Return exactly a Case A JSON response with a complete gemma_answer. Do not delegate to Qwen."
            )
            correction_messages = [
                {"role": "system", "content": gemma_system},
                {"role": "user", "content": user_context},
                {"role": "user", "content": correction},
            ]
            observer.emit(run_id, "gemma", "route_correction_input", "started",
                          "Requested document-compatible route correction from Gemma",
                          {"messages": correction_messages, "previous_route": route,
                           "provider": runtime_config.role("gemma").get("provider")})
            corrected = flash_transport.invoke("gemma", correction_messages, json_mode=True, temperature=temperature)
            observer.emit(run_id, "gemma", "route_correction_output", "completed",
                          "Gemma returned a document-compatible route",
                          {"content": corrected.get("content", ""), "usage": corrected.get("usage", {}),
                           "timings": corrected.get("timings", {}),
                           "provider": runtime_config.role("gemma").get("provider")},
                          duration_ms=float(corrected.get("duration", 0)) * 1000)
            gemma_total_duration += float(corrected.get("duration", 0))
            route = _parse_route(corrected["content"])
            case = str(route.get("flash_case") or "A").upper()
            if case not in VALID_CASES:
                case = "A"
            if case in {"B", "C", "D"}:
                raise RuntimeError("Gemma could not select a document-compatible Flash route")
        observer.emit(run_id, "gemma", "route_decision", "completed", f"Selected Flash Case {case}",
                      {"flash_case": case, "route": route})

        gemma_answer = str(route.get("gemma_answer") or "").strip()
        qwen_result = ""
        qwen_task = route.get("qwen") if isinstance(route.get("qwen"), dict) else None
        qwen_duration = 0.0
        synth_duration = 0.0

        if case in {"B", "C", "D"}:
            if not qwen_task:
                raise RuntimeError(f"Gemma selected Flash Case {case} without a qwen task")
            images = _image_parts(file_map)
            if not images:
                raise RuntimeError(f"Gemma selected Flash Case {case}, but no readable image is attached")
            qwen_content: List[Dict[str, Any]] = [
                {"type": "text", "text": str(qwen_task.get("request") or "Inspect the image precisely.")},
                *images,
            ]
            observer.emit(run_id, "qwen", "model_input", "started", "Sent delegated visual task to Qwen",
                          {"instruction": qwen_task.get("request"), "image_count": len(images),
                           "provider": runtime_config.role("qwen").get("provider")})
            qwen = flash_transport.invoke(
                "qwen",
                [
                    {"role": "system", "content": _prompt("qwen_flash_system.txt")},
                    {"role": "user", "content": qwen_content},
                ],
                temperature=temperature,
            )
            qwen_result = qwen["content"].strip()
            qwen_duration = qwen["duration"]
            observer.emit(run_id, "qwen", "model_output", "completed", "Qwen returned grounded visual evidence",
                          {"content": qwen_result, "usage": qwen.get("usage", {}),
                           "provider": runtime_config.role("qwen").get("provider")}, duration_ms=qwen_duration * 1000)

            if case == "C" or qwen_task.get("final") is False:
                synth_messages = [
                    {"role": "system", "content": "Answer the original request using the visual evidence. Return plain text."},
                    {"role": "user", "content": objective.strip()},
                    {"role": "user", "content": f"QWEN VISUAL EVIDENCE:\n{qwen_result}"},
                ]
                observer.emit(run_id, "gemma", "synthesis_input", "started", "Sent visual evidence back to Gemma",
                              {"messages": synth_messages, "provider": runtime_config.role("gemma").get("provider")})
                synth = flash_transport.invoke(
                    "gemma",
                    synth_messages,
                    temperature=temperature,
                )
                gemma_answer = synth["content"].strip()
                synth_duration = synth["duration"]
                gemma_total_duration += float(synth_duration)
                observer.emit(run_id, "gemma", "synthesis_output", "completed", "Gemma synthesized the final answer",
                              {"content": gemma_answer, "usage": synth.get("usage", {}),
                               "provider": runtime_config.role("gemma").get("provider")},
                              duration_ms=float(synth_duration) * 1000)

        if case == "A":
            answer = gemma_answer or gemma["content"]
        elif case == "B":
            answer = qwen_result
        elif case == "C":
            answer = gemma_answer
        else:
            answer = "\n\n".join(part for part in (gemma_answer, qwen_result) if part)
        if not answer:
            raise RuntimeError("Flash completed without producing an answer")

        if save_history:
            attachment_memory_ids.extend(_store_attachment_evidence(
                chat_id=session_id, user_id=user_id, attachments=attachments,
                document_context=shared_text, visual_evidence=qwen_result, run_id=run_id,
            ))
        if save_history:
            try:
                persisted = memory_coordinator.persist_turn(
                    chat_id=session_id, user_id=user_id, user_text=objective.strip(), answer=answer, run_id=run_id
                )
                memory_jobs = persisted.get("memory_jobs", [])
            except Exception as exc:
                memory_jobs = []
                observer.emit(run_id, "memory", "conversation_commit", "failed", "Chat remained available but persistence failed", {"error": str(exc)})
        else:
            memory_jobs = []
        observer.emit(run_id, "sage", "request", "completed", "Flash request completed",
                      {"flash_case": case, "answer": answer, "recalled_memory_ids": recalled_memory_ids,
                       "attachment_memory_ids": attachment_memory_ids, "memory_jobs": memory_jobs},
                      duration_ms=(time.perf_counter() - started) * 1000)
        return {
            "status": "success",
            "answer": answer,
            "flash_case": case,
            "session_id": session_id,
            "chat_id": session_id,
            "observer_run_id": run_id,
            "memory_job": memory_jobs[0] if memory_jobs else None,
            "memory_jobs": memory_jobs,
            "recalled_memory_ids": recalled_memory_ids,
            "attachment_memory_ids": attachment_memory_ids,
            "history_saved": bool(save_history),
            "telemetry": {
                "mode": "flash",
                "gemma_seconds": round(gemma_total_duration, 4),
                "qwen_seconds": round(qwen_duration, 4),
                "synthesis_seconds": round(synth_duration, 4),
                "memory_recall_seconds": round(memory_recall_seconds, 4),
                "recent_turns": recent.get("turn_count", 0),
                "recent_context_tokens": recent.get("token_estimate", 0),
                "total_wall_time": round(time.perf_counter() - started, 4),
            },
        }


flash_service = FlashService()
