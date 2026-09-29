"""Compact Gemma Flash input/output protocol."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import config
from core.json_repair import clean_json_string


ACTION_NAMES = {
    "global_memory.search",
    "chat_ledger.search",
    "document.search",
    "document.image.inspect",
    "vision.inspect",
    "artifact.generate",
}


def _read(name: str) -> str:
    return (config.PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


def system_content(available_action_names: set[str] | None = None) -> str:
    """Return system instructions and only the actions callable this turn."""
    abilities = json.loads(_read("flash_abilities.json"))
    tool_catalog = json.loads(_read("flash_tools.json"))
    tools = {
        name: contract for name, contract in tool_catalog.items()
        if available_action_names is None or name in available_action_names
    }
    return "\n\n".join((
        _read("flash_system.txt"),
        "ABILITIES\n" + json.dumps(abilities, ensure_ascii=False, separators=(",", ":")),
        "ACTIONS AVAILABLE THIS TURN\n" + json.dumps(tools, ensure_ascii=False, separators=(",", ":")),
    ))


def model_messages(context: Dict[str, Any], latest_user_message: str) -> List[Dict[str, str]]:
    """Keep dynamic context separate from the exact latest user message."""
    action_names = {str(name) for name in (context.get("available_actions") or {}).keys()}
    return [
        {"role": "system", "content": system_content(action_names)},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False, separators=(",", ":"), default=str)},
        {"role": "user", "content": latest_user_message.strip()},
    ]


def parse_decision(raw: str) -> Dict[str, Any]:
    """Parse the action protocol, with a safe plain-text final fallback.

    Legacy case output is accepted only as migration tolerance. No case label is
    returned to the active execution graph or described to the model.
    """
    parsed: Dict[str, Any] | None = None
    for candidate in (raw, clean_json_string(raw)):
        try:
            value = json.loads(candidate, strict=False)
            if isinstance(value, dict):
                parsed = value
                break
        except (TypeError, ValueError, json.JSONDecodeError):
            continue

    if parsed is None:
        text = str(raw or "").strip()
        # JSON mode can still be cut off when the model exhausts its output
        # budget. Never expose a partial protocol object as the user answer.
        if text.startswith("{") or text.startswith("["):
            return {
                "final": False,
                "response": None,
                "action": None,
                "protocol_error": "invalid_or_truncated_json",
            }
        return {"final": True, "response": text, "action": None}

    if "final" in parsed:
        final_value = parsed.get("final")
        final = final_value is True or str(final_value).strip().lower() in {"true", "1"}
        response = str(parsed.get("response") or "").strip() or None
        action = parsed.get("action") if isinstance(parsed.get("action"), dict) else None
        if final and response:
            return {"final": True, "response": response, "action": None}
        if not final and action:
            name = str(action.get("name") or "").strip()
            arguments = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
            result_use = str(action.get("result_use") or "evidence").strip().lower()
            if result_use not in {"evidence", "final"}:
                result_use = "evidence"
            return {
                "final": False,
                "response": None,
                "action": {"name": name, "arguments": arguments, "result_use": result_use},
            }

    # Temporary compatibility for a running bridge that retained the previous
    # prompt. This branch can be removed after deployments have restarted.
    legacy_case = str(parsed.get("flash_case") or "").upper()
    if legacy_case == "A":
        return {"final": True, "response": str(parsed.get("gemma_answer") or "").strip(), "action": None}
    legacy_qwen = parsed.get("qwen") if isinstance(parsed.get("qwen"), dict) else None
    if legacy_case in {"B", "C", "D"} and legacy_qwen:
        return {
            "final": False,
            "response": None,
            "action": {
                "name": "vision.inspect",
                "arguments": {"attachment_ids": [], "instruction": str(legacy_qwen.get("request") or "Inspect the relevant image.")},
                "result_use": "final" if legacy_qwen.get("final") is True and legacy_case != "C" else "evidence",
            },
        }

    text = str(parsed.get("response") or parsed.get("gemma_answer") or raw or "").strip()
    return {"final": True, "response": text, "action": None}


# Compatibility name for callers/tests from the previous Flash protocol.
_parse_route = parse_decision
