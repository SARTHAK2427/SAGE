"""SAGE Flash: structured context, typed actions, and one final text controller."""

from __future__ import annotations

import base64
import json
import mimetypes
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import config
from core.observer import observer
from flash.actions import IMAGE_TYPES, TEXT_TYPES, flash_action_executor, is_image_media_type
from flash.protocol import ACTION_NAMES, model_messages, parse_decision
from flash.runtime_config import runtime_config
from flash.transport import flash_transport
from memory_system.coordinator import memory_coordinator
from memory_system.quality import is_valid_memory
from sage_memory import sage_memory


MAX_ACTION_STEPS = 3


def _role_temperature(requested: float | None, maximum: float) -> float | None:
    """Keep the structured Flash roles stable despite a legacy UI slider."""
    if requested is None:
        return None
    return max(0.0, min(float(requested), maximum))
ATTACHMENT_EVIDENCE_PREFIX = "CHAT ATTACHMENT EVIDENCE"


def _prompt(name: str) -> str:
    return (config.PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


def _parse_route(raw: str) -> Dict[str, Any]:
    """Compatibility export; the active protocol is final/response/action."""
    return parse_decision(raw)


def _image_parts(file_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compatibility helper for current-turn images."""
    parts: List[Dict[str, Any]] = []
    for info in file_map.values():
        if not is_image_media_type(info.get("type")):
            continue
        path = Path(str(info.get("path") or ""))
        if not path.is_file():
            continue
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"},
        })
    return parts


def _document_context(objective: str, attachments: List[Dict[str, Any]],
                      file_map: Dict[str, Dict[str, Any]], max_chars: int = 16000,
                      run_id: str | None = None) -> str:
    """Compatibility helper for explicit current-document retrieval."""
    remaining = max_chars
    chunks: List[str] = []
    for item in attachments:
        if remaining <= 0:
            break
        ref = str(item.get("ref") or "")
        media_type = str(item.get("type") or "").lower()
        info = file_map.get(ref) or {}
        name = str(item.get("name") or ref)
        text = ""
        if media_type in TEXT_TYPES:
            path = Path(str(info.get("path") or ""))
            if path.is_file():
                text = path.read_text(encoding="utf-8", errors="replace")
        elif item.get("doc_id") and not is_image_media_type(media_type):
            from db_service import document_db
            socket = document_db.rag_search_socket(
                query=objective.strip() or "document overview",
                doc_ids=[str(item["doc_id"])], top_k=8,
            )
            if socket.get("status") == "success":
                text = "\n\n".join(
                    str(record.get("text") or "").strip()
                    for record in (socket.get("result") or {}).get("records") or []
                    if str(record.get("text") or "").strip()
                )
        if text:
            excerpt = text[:remaining]
            chunks.append(f"DOCUMENT [{ref}] {name}:\n{excerpt}")
            remaining -= len(excerpt)
    return "\n\n".join(chunks)


def _recent_attachment_evidence(chat_id: str, user_id: str, max_chars: int = 9000) -> tuple[str, List[str]]:
    """Read legacy attachment summaries created before the registry existed."""
    records = sage_memory.list_memories(
        user_id=user_id, category="summary", memory_tier="cold", chat_id=chat_id, limit=80,
    )
    selected: List[str] = []
    ids: List[str] = []
    remaining = max_chars
    for record in records:
        content = str(record.get("content") or "")
        if not content.startswith(ATTACHMENT_EVIDENCE_PREFIX) or remaining <= 0:
            continue
        selected.append(content[:remaining])
        remaining -= len(selected[-1])
        if record.get("memory_id"):
            ids.append(str(record["memory_id"]))
    return "\n\n".join(selected), ids


def _bounded_memory_items(user_id: str, chat_id: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    global_items: List[Dict[str, Any]] = []
    cold_items: List[Dict[str, Any]] = []
    global_ids: List[str] = []
    global_tokens = max(0, config.SAGE_GLOBAL_MEMORY_BUDGET_TOKENS)
    cold_tokens = max(0, config.SAGE_CONTEXT_MEMORY_BUDGET_TOKENS)
    records = sage_memory.list_memories(user_id=user_id, memory_tier="cold", status="active", limit=200)
    for record in records:
        content = str(record.get("content") or "").strip()
        source_chat = str(record.get("source_chat_id") or "")
        if not content or content.startswith(ATTACHMENT_EVIDENCE_PREFIX):
            continue
        cost = max(1, (len(content) + 3) // 4)
        if not source_chat and record.get("category") == "personal" and is_valid_memory(content, "global"):
            if len(global_items) >= config.SAGE_GLOBAL_MEMORY_MAX_ITEMS or global_tokens <= 0:
                continue
            if cost > global_tokens:
                content = content[:global_tokens * 4].rstrip()
                cost = max(1, (len(content) + 3) // 4)
            if content:
                global_items.append({
                    "memory_id": str(record.get("memory_id") or ""),
                    "category": "personal", "content": content,
                    "created_at": str(record.get("created_at") or ""),
                })
                global_ids.append(str(record.get("memory_id") or ""))
                global_tokens -= cost
        elif source_chat == chat_id and is_valid_memory(content, "cold"):
            if cold_tokens <= 0:
                continue
            if cost > cold_tokens:
                content = content[:cold_tokens * 4].rstrip()
                cost = max(1, (len(content) + 3) // 4)
            if content:
                cold_items.append({
                    "memory_id": str(record.get("memory_id") or ""),
                    "category": str(record.get("category") or "summary"),
                    "content": content, "created_at": str(record.get("created_at") or ""),
                })
                cold_tokens -= cost
    return global_items, cold_items, [value for value in global_ids if value]


def _register_current_attachments(chat_id: str, user_id: str,
                                  attachments: List[Dict[str, Any]],
                                  file_map: Dict[str, Dict[str, Any]],
                                  save_history: bool) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for item in attachments:
        ref = str(item.get("ref") or "")
        merged.append({**item, "path": (file_map.get(ref) or {}).get("path")})
    if save_history and merged:
        return sage_memory.register_attachments(chat_id=chat_id, user_id=user_id, attachments=merged)
    return [{
        "attachment_id": str(item.get("attachment_id") or f"att_{uuid.uuid4().hex}"),
        "chat_id": chat_id, "user_id": user_id, "message_id": None, "turn_number": 1,
        "ref": item.get("ref"), "doc_id": item.get("doc_id"), "image_id": item.get("image_id"),
        "name": item.get("name"), "media_type": item.get("type"), "file_size": item.get("size", 0),
        "storage_path": item.get("path"), "status": item.get("status", "ready"),
        "error": item.get("error"), "vision_evidence": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    } for item in merged]


def _attachment_packet(records: List[Dict[str, Any]], current_ids: set[str], current_turn: int) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    document_ids: List[str] = []
    document_image_ids: List[str] = []
    vision_ids: List[str] = []
    evidence_remaining = 7000
    for record in records:
        attachment_id = str(record.get("attachment_id") or "")
        media_type = str(record.get("media_type") or "").lower()
        turn_number = int(record.get("turn_number") or 0)
        is_image = is_image_media_type(media_type)
        searchable = bool(record.get("doc_id") or media_type in TEXT_TYPES) and not is_image
        # A chat owns its uploaded source files.  Qwen may revisit an earlier
        # image in that same chat when stored evidence is too shallow for the
        # user's follow-up; cross-chat attachments never enter this packet.
        vision_now = is_image and Path(str(record.get("storage_path") or "")).is_file()
        if searchable:
            document_ids.append(attachment_id)
        if searchable and record.get("doc_id"):
            document_image_ids.append(attachment_id)
        if vision_now:
            vision_ids.append(attachment_id)
        evidence = str(record.get("vision_evidence") or "")
        evidence = evidence[:evidence_remaining] if evidence_remaining > 0 else ""
        evidence_remaining -= len(evidence)
        items.append({
            "attachment_id": attachment_id,
            "message_id": str(record.get("message_id") or "") or None,
            "name": str(record.get("name") or attachment_id),
            "media_type": media_type,
            "source_kind": "chat_image" if is_image else "document",
            "status": str(record.get("status") or "unknown"),
            "turn_number": turn_number,
            "turn_distance": max(0, current_turn - turn_number),
            "uploaded_at": str(record.get("created_at") or ""),
            "attached_to_current_message": attachment_id in current_ids,
            "document_searchable": searchable,
            "vision_readable": vision_now,
            "stored_vision_evidence": evidence or None,
        })
    available = {
        "chat_ledger.search": {"available": True, "scope": "current_chat"},
        "document.search": {"available": bool(document_ids), "allowed_attachment_ids": document_ids},
        "document.image.inspect": {
            "available": bool(document_image_ids), "allowed_attachment_ids": document_image_ids,
            "note": "Inspects an embedded image by its one-based order inside one selected document."
        },
        "vision.inspect": {"available": bool(vision_ids), "allowed_attachment_ids": vision_ids,
                           "note": "Only image bytes belonging to this chat may be inspected."},
    }
    return items, available


def _attachment_reference(items: List[Dict[str, Any]], current_ids: set[str]) -> Dict[str, Any]:
    """Resolve an unnamed attachment reference by message proximity.

    A unique current-message attachment wins. Otherwise the unique attachment
    from the nearest earlier turn wins. Equal-distance candidates stay
    explicitly ambiguous so Gemma can ask instead of guessing.
    """
    if not items:
        return {"status": "none", "attachment_id": None, "candidates": []}
    current = [item for item in items if str(item.get("attachment_id") or "") in current_ids]
    if current:
        candidates = current
        reason = "attached_to_current_message"
    else:
        nearest = min(int(item.get("turn_distance") or 0) for item in items)
        candidates = [item for item in items if int(item.get("turn_distance") or 0) == nearest]
        reason = "nearest_prior_attachment_message"
    compact = [{
        "attachment_id": str(item.get("attachment_id") or ""),
        "name": str(item.get("name") or ""),
        "media_type": str(item.get("media_type") or ""),
        "turn_number": int(item.get("turn_number") or 0),
        "uploaded_at": str(item.get("uploaded_at") or ""),
    } for item in candidates]
    if len(compact) == 1:
        return {"status": "resolved", "attachment_id": compact[0]["attachment_id"],
                "reason": reason, "candidate": compact[0]}
    return {"status": "ambiguous", "attachment_id": None, "reason": "same_message_tie",
            "candidates": compact}


def _action_allowed(action: Dict[str, Any], available: Dict[str, Any]) -> tuple[bool, str]:
    name = str(action.get("name") or "")
    if name not in ACTION_NAMES:
        return False, "unknown_action"
    spec = available.get(name) or {}
    if not spec.get("available"):
        return False, "action_unavailable"
    args = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
    requested = args.get("attachment_ids")
    if isinstance(requested, list) and "allowed_attachment_ids" in spec:
        allowed_ids = {str(value) for value in spec.get("allowed_attachment_ids") or []}
        if any(str(value) not in allowed_ids for value in requested):
            return False, "attachment_not_allowed"
    if str(action.get("result_use") or "evidence") == "final" and name not in {"vision.inspect", "document.image.inspect"}:
        return False, "final_result_not_allowed"
    return True, ""


class FlashService:
    def run(self, *, objective: str, attachments: List[Dict[str, Any]],
            file_map: Dict[str, Dict[str, Any]], session_id: str | None = None,
            observer_run_id: str | None = None, user_id: str | None = None,
            temperature: float | None = None, save_history: bool = True,
            interaction_mode: str = "flash") -> Dict[str, Any]:
        started = time.perf_counter()
        session_id = (session_id or "").strip() or f"flash_{uuid.uuid4().hex[:12]}"
        run_id = (observer_run_id or "").strip() or session_id
        user_id = (user_id or config.DEFAULT_USER_ID).strip() or config.DEFAULT_USER_ID
        interaction_mode = "deep_focus" if str(interaction_mode).strip().lower() == "deep_focus" else "flash"
        observer.emit(run_id, "sage", "request", "started", "Flash request accepted",
                      {"objective": objective, "attachments": attachments, "save_history": save_history,
                       "interaction_mode": interaction_mode})
        if interaction_mode == "deep_focus":
            observer.emit(run_id, "sage", "mode", "completed", "Adept response path enabled",
                          {"interaction_mode": interaction_mode})

        current_registered = _register_current_attachments(session_id, user_id, attachments, file_map, save_history)
        current_ids = {str(item.get("attachment_id") or "") for item in current_registered}
        all_attachments = sage_memory.list_attachments(session_id, user_id=user_id, limit=40) if save_history else current_registered
        current_turn = (
            sum(1 for message in sage_memory.get_messages(session_id) if message.get("role") == "user") + 1
            if save_history else 1
        )
        recent = memory_coordinator.recent_turns(
            session_id, max_turns=config.SAGE_RECENT_CHAT_MAX_TURNS,
            token_budget=config.SAGE_RECENT_CHAT_BUDGET_TOKENS,
        ) if save_history else {"messages": [], "turn_count": 0, "token_estimate": 0}
        global_items, cold_items, recalled_memory_ids = _bounded_memory_items(user_id, session_id) if save_history else ([], [], [])
        legacy_evidence, legacy_ids = _recent_attachment_evidence(session_id, user_id) if save_history else ("", [])
        recalled_memory_ids.extend(legacy_ids)
        attachment_items, available_actions = _attachment_packet(all_attachments, current_ids, current_turn)
        attachment_reference = _attachment_reference(attachment_items, current_ids)
        selectable_actions = {
            name: {key: value for key, value in spec.items() if key != "available"}
            for name, spec in available_actions.items()
            if spec.get("available")
        }
        missing_inputs: List[str] = []
        if not available_actions["document.search"].get("available"):
            missing_inputs.append("No searchable document belongs to this chat.")
        if not available_actions["vision.inspect"].get("available"):
            missing_inputs.append("No readable image source belongs to this chat.")

        context: Dict[str, Any] = {
            "run": {
                "chat_id": session_id, "turn_number": current_turn,
                "current_time": datetime.now(timezone.utc).isoformat(),
                "interaction_mode": interaction_mode,
            },
            "memory": {
                "global_personal": global_items,
                "current_chat_cold": cold_items,
                "recent_chat": recent.get("messages", []),
            },
            "attachments": attachment_items,
            "attachment_reference": attachment_reference,
            "legacy_attachment_evidence": legacy_evidence or None,
            "available_actions": selectable_actions,
            "missing_inputs": missing_inputs,
            "action_results": [],
            "execution": {"pass": 1, "actions_used": 0, "action_limit": MAX_ACTION_STEPS},
        }
        observer.emit(run_id, "memory", "context_packet", "completed", "Built bounded chat context",
                      {"recent_turns": recent.get("turn_count", 0), "recent_tokens": recent.get("token_estimate", 0),
                       "global_items": len(global_items), "cold_items": len(cold_items),
                       "attachments": len(attachment_items), "attachment_reference": attachment_reference,
                       "available_actions": selectable_actions,
                       "missing_inputs": missing_inputs})

        gemma_seconds = 0.0
        qwen_seconds = 0.0
        action_seconds = 0.0
        action_count = 0
        gemma_temperature = _role_temperature(temperature, 0.2)
        qwen_temperature = _role_temperature(temperature, 0.1)
        seen_actions: set[str] = set()
        answer = ""
        execution_path = "direct"

        for pass_index in range(MAX_ACTION_STEPS + 1):
            context["execution"] = {
                "pass": pass_index + 1, "actions_used": action_count,
                "action_limit": MAX_ACTION_STEPS,
            }
            messages = model_messages(context, objective)
            observer.emit(run_id, "gemma", "model_input", "started", "Sent context and current message to Gemma",
                          {"messages": messages, "json_mode": True, "temperature": gemma_temperature,
                           "provider": runtime_config.role("gemma").get("provider")})
            gemma = flash_transport.invoke("gemma", messages, json_mode=True, temperature=gemma_temperature)
            gemma_seconds += float(gemma.get("duration", 0))
            decision = parse_decision(str(gemma.get("content") or ""))
            observer.emit(run_id, "gemma", "model_output", "completed", "Gemma returned a decision",
                          {"content": gemma.get("content", ""), "decision": decision,
                           "usage": gemma.get("usage", {}), "reasoning": gemma.get("reasoning", ""),
                           "finish_reason": gemma.get("finish_reason", ""),
                           "provider": runtime_config.role("gemma").get("provider")},
                          duration_ms=float(gemma.get("duration", 0)) * 1000)

            if decision.get("protocol_error"):
                context["action_results"].append({
                    "name": "protocol", "status": "failed",
                    "error": {
                        "code": str(decision.get("protocol_error")),
                        "message": "Return one complete JSON object. Shorten the response if needed.",
                    },
                })
                if pass_index >= MAX_ACTION_STEPS:
                    answer = "I could not produce a complete response within the model output limit."
                    execution_path = "protocol_limit"
                    break
                continue

            if decision.get("final"):
                answer = str(decision.get("response") or "").strip()
                if answer:
                    break
                context["action_results"].append({
                    "name": "protocol", "status": "failed",
                    "error": {"code": "empty_final_response", "message": "Return a non-empty final response."},
                })
                continue

            action = decision.get("action") if isinstance(decision.get("action"), dict) else {}
            signature = json.dumps(action, sort_keys=True, ensure_ascii=False)
            allowed, reason = _action_allowed(action, available_actions)
            if signature in seen_actions:
                allowed, reason = False, "repeated_action"
            if pass_index >= MAX_ACTION_STEPS:
                allowed, reason = False, "action_limit_reached"
            if not allowed:
                context["action_results"].append({
                    "name": str(action.get("name") or "unknown"), "status": "unavailable",
                    "error": {"code": reason, "message": "Choose an available different action or return a final response."},
                })
                # On the final pass, avoid exposing an internal exception.
                if pass_index >= MAX_ACTION_STEPS:
                    answer = "I could not obtain the information needed to complete that request."
                    execution_path = "action_limit"
                    break
                continue

            seen_actions.add(signature)
            action_started = time.perf_counter()
            action_result = flash_action_executor.execute(
                action=action, chat_id=session_id, user_id=user_id,
                latest_user_message=objective, attachments=all_attachments,
                current_file_map=file_map, run_id=run_id, temperature=qwen_temperature,
            )
            action_seconds += time.perf_counter() - action_started
            action_count += 1
            execution_path = str(action.get("name") or "action")
            context["action_results"].append(action_result)
            if action.get("name") in {"vision.inspect", "document.image.inspect"}:
                qwen_seconds += float(((action_result.get("result") or {}).get("model_seconds") or 0))
            if (str(action.get("result_use") or "evidence") == "final"
                    and action.get("name") in {"vision.inspect", "document.image.inspect"}
                    and action_result.get("status") == "completed"):
                answer = str((action_result.get("result") or {}).get("evidence") or "").strip()
                if answer:
                    break

        if not answer:
            answer = "I could not produce a reliable answer from the available information."

        memory_jobs: List[str] = []
        if save_history:
            try:
                persisted = memory_coordinator.persist_turn(
                    chat_id=session_id, user_id=user_id, user_text=objective.strip(), answer=answer, run_id=run_id,
                )
                memory_jobs = persisted.get("memory_jobs", [])
                sage_memory.link_attachments_to_message(
                    [str(item.get("attachment_id") or "") for item in current_registered],
                    persisted.get("user_message_id"),
                )
            except Exception as exc:
                observer.emit(run_id, "memory", "conversation_commit", "failed",
                              "Chat remained available but persistence failed", {"error": str(exc)})

        observer.emit(run_id, "sage", "request", "completed", "Flash request completed",
                      {"answer": answer, "interaction_mode": interaction_mode,
                       "execution_path": execution_path, "action_count": action_count,
                       "recalled_memory_ids": recalled_memory_ids, "memory_jobs": memory_jobs},
                      duration_ms=(time.perf_counter() - started) * 1000)
        return {
            "status": "success", "answer": answer, "interaction_mode": interaction_mode,
            "execution_path": execution_path,
            "session_id": session_id, "chat_id": session_id, "observer_run_id": run_id,
            "memory_job": memory_jobs[0] if memory_jobs else None, "memory_jobs": memory_jobs,
            "recalled_memory_ids": recalled_memory_ids,
            "attachment_ids": [str(item.get("attachment_id") or "") for item in current_registered],
            "history_saved": bool(save_history),
            "telemetry": {
                "mode": "flash", "gemma_seconds": round(gemma_seconds, 4),
                "qwen_seconds": round(qwen_seconds, 4), "action_seconds": round(action_seconds, 4),
                "action_count": action_count, "recent_turns": recent.get("turn_count", 0),
                "recent_context_tokens": recent.get("token_estimate", 0),
                "total_wall_time": round(time.perf_counter() - started, 4),
            },
        }


flash_service = FlashService()
