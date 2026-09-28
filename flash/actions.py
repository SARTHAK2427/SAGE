"""Deterministic, chat-scoped actions available to Gemma Flash."""

from __future__ import annotations

import base64
import mimetypes
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import config
from core.observer import observer
from flash.runtime_config import runtime_config
from flash.transport import flash_transport
from memory_system.quality import is_valid_memory
from sage_memory import sage_memory


IMAGE_TYPES = {"png", "jpg", "jpeg", "webp", "gif", "bmp"}
TEXT_TYPES = {"txt", "md", "csv", "json", "log", "py", "js", "html", "xml", "yaml", "yml"}


def is_image_media_type(value: Any) -> bool:
    """Accept browser MIME types (``image/png``) and older extension-only values."""
    media_type = str(value or "").strip().lower()
    return media_type.startswith("image/") or media_type.split(";", 1)[0] in IMAGE_TYPES


def _words(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9_'-]+", str(value).lower()) if len(token) > 2}


def _local_attachment_path(value: Any) -> Path | None:
    try:
        path = Path(str(value or "")).expanduser().resolve()
        if path.is_file() and path.is_relative_to(config.ATTACHMENTS_ROOT.resolve()):
            return path
    except (OSError, RuntimeError, ValueError):
        pass
    return None


def _local_document_artifact_path(value: Any) -> Path | None:
    try:
        path = Path(str(value or "")).expanduser().resolve()
        if path.is_file() and path.is_relative_to(config.ARTIFACTS_ROOT.resolve()):
            return path
    except (OSError, RuntimeError, ValueError):
        pass
    return None


class FlashActionExecutor:
    def execute(self, *, action: Dict[str, Any], chat_id: str, user_id: str,
                latest_user_message: str, attachments: List[Dict[str, Any]],
                current_file_map: Dict[str, Dict[str, Any]], run_id: str,
                temperature: float | None) -> Dict[str, Any]:
        action_id = f"act_{uuid.uuid4().hex[:12]}"
        name = str(action.get("name") or "").strip()
        arguments = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
        started = time.perf_counter()
        observer.emit(run_id, "gemma", "action_request", "started", f"Requested {name or 'unknown action'}",
                      {"action_id": action_id, "name": name, "arguments": arguments})
        try:
            if name == "global_memory.search":
                result = self._global_memory(arguments, user_id)
            elif name == "chat_ledger.search":
                result = self._chat_ledger(arguments, chat_id)
            elif name == "document.search":
                result = self._document_search(arguments, latest_user_message, attachments, run_id)
            elif name == "document.image.inspect":
                result = self._document_image(arguments, latest_user_message, attachments, run_id, temperature)
            elif name == "vision.inspect":
                result = self._vision(arguments, latest_user_message, attachments, current_file_map, run_id, temperature)
            else:
                return self._error(action_id, name, "unknown_action", "The action is not defined.", started, run_id)
        except Exception as exc:
            return self._error(action_id, name, "action_failed", str(exc), started, run_id)
        payload = {"action_id": action_id, "name": name, "status": "completed", "result": result, "error": None}
        observer.emit(run_id, "sage", "action_result", "completed", f"Completed {name}", payload,
                      duration_ms=(time.perf_counter() - started) * 1000)
        return payload

    @staticmethod
    def _error(action_id: str, name: str, code: str, message: str, started: float, run_id: str) -> Dict[str, Any]:
        payload = {
            "action_id": action_id, "name": name, "status": "unavailable",
            "result": None, "error": {"code": code, "message": message},
        }
        observer.emit(run_id, "sage", "action_result", "failed", f"Could not complete {name or 'action'}", payload,
                      duration_ms=(time.perf_counter() - started) * 1000)
        return payload

    @staticmethod
    def _global_memory(arguments: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        query = str(arguments.get("query") or "").strip()
        query_words = _words(query)
        ranked = []
        for record in sage_memory.list_memories(user_id=user_id, memory_tier="cold", status="active", limit=200):
            if record.get("source_chat_id") or record.get("category") != "personal":
                continue
            content = str(record.get("content") or "").strip()
            if not content or not is_valid_memory(content, "global"):
                continue
            overlap = len(query_words & _words(content))
            score = overlap / max(1, len(query_words)) if query_words else 1.0
            if score or not query_words:
                ranked.append((score, {
                    "memory_id": str(record.get("memory_id") or ""), "category": "personal",
                    "content": content, "created_at": str(record.get("created_at") or ""),
                    "score": round(score, 4),
                }))
        items = [item for _, item in sorted(ranked, key=lambda pair: pair[0], reverse=True)[:12]]
        return {"query": query, "scope": "global_personal", "items": items, "count": len(items)}

    @staticmethod
    def _chat_ledger(arguments: Dict[str, Any], chat_id: str) -> Dict[str, Any]:
        query = str(arguments.get("query") or "").strip()
        query_words = _words(query)
        limit = max(1, min(int(arguments.get("max_results") or 8), 12))
        messages = sage_memory.get_messages(chat_id)
        ranked = []
        for index, message in enumerate(messages):
            content = str(message.get("content") or "")
            overlap = len(query_words & _words(content))
            lexical = overlap / max(1, len(query_words)) if query_words else 0.0
            recency = (index + 1) / max(1, len(messages))
            score = lexical * 0.85 + recency * 0.15
            if overlap or not query_words:
                ranked.append((score, index, {
                    "message_id": str(message.get("msg_id") or ""), "role": str(message.get("role") or ""),
                    "content": content, "created_at": str(message.get("created_at") or ""),
                    "message_index": index + 1, "score": round(score, 4),
                }))
        items = [item for _, _, item in sorted(ranked, key=lambda row: (row[0], row[1]), reverse=True)[:limit]]
        items.sort(key=lambda item: int(item["message_index"]))
        return {"query": query, "scope": "current_chat", "items": items, "count": len(items)}

    @staticmethod
    def _select(arguments: Dict[str, Any], attachments: List[Dict[str, Any]], *, images: bool) -> List[Dict[str, Any]]:
        allowed = {str(item.get("attachment_id") or ""): item for item in attachments if item.get("attachment_id")}
        raw_ids = arguments.get("attachment_ids")
        ids = [str(value) for value in raw_ids] if isinstance(raw_ids, list) else []
        if ids and any(value not in allowed for value in ids):
            raise ValueError("An attachment id is not available in this chat.")
        selected = [allowed[value] for value in ids] if ids else list(allowed.values())
        return [item for item in selected if is_image_media_type(item.get("media_type")) is images]

    def _document_search(self, arguments: Dict[str, Any], latest: str,
                         attachments: List[Dict[str, Any]], run_id: str) -> Dict[str, Any]:
        query = str(arguments.get("query") or latest).strip()
        selected = self._select(arguments, attachments, images=False)
        if not selected:
            raise ValueError("No searchable document attachment is available in this chat.")
        records: List[Dict[str, Any]] = []
        doc_ids: List[str] = []
        by_doc: Dict[str, Dict[str, Any]] = {}
        for item in selected:
            doc_id = str(item.get("doc_id") or "")
            if doc_id:
                doc_ids.append(doc_id)
                by_doc[doc_id] = item
            elif str(item.get("media_type") or "").lower() in TEXT_TYPES:
                path = _local_attachment_path(item.get("storage_path"))
                if path:
                    records.append({
                        "attachment_id": item.get("attachment_id"), "name": item.get("name"),
                        "record_id": None, "page": None,
                        "text": path.read_text(encoding="utf-8", errors="replace")[:8000],
                    })
        if doc_ids:
            observer.emit(run_id, "document-rag", "semantic_search", "started", "Searching chat documents",
                          {"query": query, "doc_ids": doc_ids, "top_k": 10})
            from db_service import document_db
            socket = document_db.rag_search_socket(query=query, doc_ids=doc_ids, top_k=10)
            if socket.get("status") != "success":
                error = socket.get("error") or {}
                raise RuntimeError(str(error.get("message") or "Document search failed"))
            for record in (socket.get("result") or {}).get("records") or []:
                doc_id = str(record.get("doc_id") or "")
                item = by_doc.get(doc_id) or {}
                semantic = record.get("reranker_score")
                if semantic is None:
                    semantic = record.get("derived_cosine_similarity")
                semantic_value = float(semantic) if semantic is not None else 0.0
                turn_distance = int(item.get("turn_distance") or 0)
                records.append({
                    "attachment_id": item.get("attachment_id"), "name": item.get("name"),
                    "doc_id": doc_id, "record_id": record.get("record_id"), "page": record.get("page"),
                    "text": str(record.get("text") or "")[:2500], "semantic_score": round(semantic_value, 4),
                    "turn_distance": turn_distance,
                    "rank_score": round(semantic_value * 0.8 + (1 / (1 + turn_distance)) * 0.2, 4),
                })
            observer.emit(run_id, "document-rag", "semantic_search", "completed",
                          f"Retrieved {len(records)} chat-scoped passages", socket,
                          duration_ms=float((socket.get("timing") or {}).get("duration_ms") or 0.0))
        records.sort(key=lambda item: float(item.get("rank_score") or item.get("semantic_score") or 0), reverse=True)
        bounded: List[Dict[str, Any]] = []
        remaining = 10000
        for record in records:
            text = str(record.get("text") or "")
            if remaining <= 0:
                break
            item = dict(record)
            item["text"] = text[:remaining]
            remaining -= len(item["text"])
            bounded.append(item)
        return {
            "query": query, "scope": "current_chat_attachments",
            "searched_attachment_ids": [str(item.get("attachment_id")) for item in selected],
            "records": bounded[:8], "count": min(len(bounded), 8),
        }

    def _vision(self, arguments: Dict[str, Any], latest: str, attachments: List[Dict[str, Any]],
                current_file_map: Dict[str, Dict[str, Any]], run_id: str,
                temperature: float | None) -> Dict[str, Any]:
        selected = self._select(arguments, attachments, images=True)
        current_by_id = {
            str(item.get("attachment_id") or ""): item for item in current_file_map.values()
            if item.get("attachment_id")
        }
        readable = []
        for item in selected:
            current = current_by_id.get(str(item.get("attachment_id") or ""))
            # Prefer the incoming file map for this turn, then the durable
            # chat-scoped source file for a follow-up on an older image.
            path = _local_attachment_path((current or {}).get("path")) if current else None
            if path is None:
                path = _local_attachment_path(item.get("storage_path"))
            if path:
                readable.append((item, path))
        if not readable:
            raise ValueError("No readable image source remains in this chat. Ask for re-upload.")
        if not isinstance(arguments.get("attachment_ids"), list) and len(readable) > 1:
            raise ValueError("Multiple chat images are available. Select attachment ids.")
        instruction = str(arguments.get("instruction") or latest).strip()
        content: List[Dict[str, Any]] = [{"type": "text", "text": instruction}]
        for item, path in readable:
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append({"type": "text", "text": f"IMAGE {item.get('attachment_id')}: {item.get('name')}"})
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
        messages = [
            {"role": "system", "content": (config.PROMPTS_DIR / "qwen_flash_system.txt").read_text(encoding="utf-8").strip()},
            {"role": "user", "content": content},
        ]
        observer.emit(run_id, "qwen", "model_input", "started", "Sent current images to Qwen",
                      {"instruction": instruction, "attachment_ids": [item.get("attachment_id") for item, _ in readable],
                       "image_count": len(readable), "provider": runtime_config.role("qwen").get("provider")})
        response = flash_transport.invoke("qwen", messages, temperature=temperature)
        evidence = str(response.get("content") or "").strip()
        observer.emit(run_id, "qwen", "model_output", "completed", "Qwen returned visual output",
                      {"content": evidence, "usage": response.get("usage", {}),
                       "provider": runtime_config.role("qwen").get("provider")},
                      duration_ms=float(response.get("duration", 0)) * 1000)
        attachment_ids = [str(item.get("attachment_id")) for item, _ in readable]
        sage_memory.update_attachment_evidence(attachment_ids, evidence)
        return {"instruction": instruction, "attachment_ids": attachment_ids, "evidence": evidence,
                "model_seconds": round(float(response.get("duration", 0)), 4)}

    def _document_image(self, arguments: Dict[str, Any], latest: str,
                        attachments: List[Dict[str, Any]], run_id: str,
                        temperature: float | None) -> Dict[str, Any]:
        selected = [item for item in self._select(arguments, attachments, images=False) if item.get("doc_id")]
        if len(selected) != 1:
            raise ValueError("Select exactly one document attachment before inspecting an embedded image.")
        item = selected[0]
        try:
            image_number = int(arguments.get("image_number") or 1)
        except (TypeError, ValueError) as exc:
            raise ValueError("image_number must be a one-based integer.") from exc
        if image_number < 1:
            raise ValueError("image_number must be at least 1.")

        from db_service import document_db
        doc_id = str(item.get("doc_id") or "")
        images = document_db.list_artifacts(doc_id=doc_id, artifact_type="image")
        if image_number > len(images):
            raise ValueError(f"The selected document contains {len(images)} embedded image(s), not image {image_number}.")
        image_meta = images[image_number - 1]
        element_id = str(image_meta.get("element_id") or "")
        fetched = document_db.artifact_fetch(doc_id, element_id)
        path = _local_document_artifact_path(fetched.get("local_path"))
        if path is None:
            raise ValueError(f"Embedded image {image_number} is not readable from the selected document.")

        instruction = str(arguments.get("instruction") or latest).strip()
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": instruction},
            {"type": "text", "text": (
                f"DOCUMENT {item.get('name')} | EMBEDDED IMAGE {image_number}/{len(images)} | "
                f"PAGE {image_meta.get('page')} | ELEMENT {element_id}"
            )},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
        ]
        messages = [
            {"role": "system", "content": (config.PROMPTS_DIR / "qwen_flash_system.txt").read_text(encoding="utf-8").strip()},
            {"role": "user", "content": content},
        ]
        observer.emit(run_id, "qwen", "model_input", "started", "Sent a document image to Qwen", {
            "instruction": instruction, "attachment_id": item.get("attachment_id"), "doc_id": doc_id,
            "image_number": image_number, "image_count": len(images), "element_id": element_id,
            "page": image_meta.get("page"), "provider": runtime_config.role("qwen").get("provider"),
        })
        response = flash_transport.invoke("qwen", messages, temperature=temperature)
        evidence = str(response.get("content") or "").strip()
        observer.emit(run_id, "qwen", "model_output", "completed", "Qwen returned document-image evidence", {
            "content": evidence, "attachment_id": item.get("attachment_id"), "doc_id": doc_id,
            "image_number": image_number, "element_id": element_id, "page": image_meta.get("page"),
            "usage": response.get("usage", {}), "provider": runtime_config.role("qwen").get("provider"),
        }, duration_ms=float(response.get("duration", 0)) * 1000)
        return {
            "instruction": instruction, "attachment_id": str(item.get("attachment_id") or ""),
            "document_name": str(item.get("name") or ""), "doc_id": doc_id,
            "image_number": image_number, "image_count": len(images), "element_id": element_id,
            "page": image_meta.get("page"), "caption": image_meta.get("caption"),
            "evidence": evidence, "model_seconds": round(float(response.get("duration", 0)), 4),
        }


flash_action_executor = FlashActionExecutor()
