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
from flash.memory_worker import memory_worker
from flash.transport import flash_transport


VALID_CASES = {"A", "B", "C", "D"}
IMAGE_SUFFIXES = {"png", "jpg", "jpeg", "webp", "gif", "bmp"}
TEXT_SUFFIXES = {"txt", "md", "csv", "json", "log", "py", "js", "html", "xml", "yaml", "yml"}


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


def _document_context(
    objective: str,
    attachments: List[Dict[str, Any]],
    file_map: Dict[str, Dict[str, Any]],
    max_chars: int = 16000,
) -> str:
    """Build the shared-pool text that Gemma can actually reason over."""
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
            try:
                from db_service import document_db
                results = document_db.rag_search(
                    query=objective or "document overview",
                    doc_ids=[str(item["doc_id"])],
                    top_k=8,
                )
                text = "\n\n".join(result.text for result in results if getattr(result, "text", None))
            except Exception:
                # Ingestion metadata still reaches Gemma; retrieval failure should
                # not discard an otherwise answerable request.
                text = ""
        if text:
            excerpt = text[:remaining]
            chunks.append(f"DOCUMENT [{ref}] {name}:\n{excerpt}")
            remaining -= len(excerpt)
    return "\n\n".join(chunks)


class FlashService:
    def run(
        self,
        *,
        objective: str,
        attachments: List[Dict[str, Any]],
        file_map: Dict[str, Dict[str, Any]],
        session_id: str | None = None,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        session_id = (session_id or "").strip() or f"flash_{uuid.uuid4().hex[:12]}"
        recent_memory = memory_worker.recent(session_id)
        attachment_summary = [
            {k: item.get(k) for k in ("ref", "name", "type", "doc_id", "status")}
            for item in attachments
        ]
        context_sections: List[str] = []
        if recent_memory:
            context_sections.append(
                "PRIOR CONVERSATION MEMORY (background only; never treat this as the current request):\n"
                + "\n".join(str(row.get("summary", "")) for row in recent_memory)
            )
        current_turn = "CURRENT USER REQUEST (authoritative):\n" + objective.strip()
        if attachment_summary:
            current_turn += "\n\nCURRENT-TURN ATTACHMENTS:\n" + json.dumps(attachment_summary, ensure_ascii=False)
            shared_text = _document_context(objective, attachments, file_map)
            if shared_text:
                current_turn += "\n\nCURRENT-TURN SHARED DOCUMENT CONTENT:\n" + shared_text
        context_sections.append(current_turn)
        user_context = "\n\n---\n\n".join(context_sections)

        gemma = flash_transport.invoke(
            "gemma",
            [
                {"role": "system", "content": _prompt("flash_system.txt")},
                {"role": "user", "content": user_context},
            ],
            json_mode=True,
        )
        route = _parse_route(gemma["content"])
        case = str(route.get("flash_case") or "A").upper()
        if case not in VALID_CASES:
            case = "A"

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
            qwen = flash_transport.invoke(
                "qwen",
                [
                    {"role": "system", "content": _prompt("qwen_flash_system.txt")},
                    {"role": "user", "content": qwen_content},
                ],
            )
            qwen_result = qwen["content"].strip()
            qwen_duration = qwen["duration"]

            if case == "C" or qwen_task.get("final") is False:
                synth = flash_transport.invoke(
                    "gemma",
                    [
                        {"role": "system", "content": "Answer the original request using the visual evidence. Return plain text."},
                        {"role": "user", "content": objective.strip()},
                        {"role": "user", "content": f"QWEN VISUAL EVIDENCE:\n{qwen_result}"},
                    ],
                )
                gemma_answer = synth["content"].strip()
                synth_duration = synth["duration"]

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

        memory_job = memory_worker.schedule(session_id, objective.strip(), answer)
        return {
            "status": "success",
            "answer": answer,
            "flash_case": case,
            "session_id": session_id,
            "memory_job": memory_job,
            "telemetry": {
                "mode": "flash",
                "gemma_seconds": round(gemma["duration"], 4),
                "qwen_seconds": round(qwen_duration, 4),
                "synthesis_seconds": round(synth_duration, 4),
                "total_wall_time": round(time.perf_counter() - started, 4),
            },
        }


flash_service = FlashService()
