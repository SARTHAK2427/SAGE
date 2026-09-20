"""Editable Flash model catalog.

Model identities are fixed by role at runtime, while repository and quantized
file names remain editable in ``flash_models.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import config


CATALOG_PATH = config.BASE_DIR / "flash_models.json"


def load_catalog() -> Dict[str, Dict[str, Any]]:
    try:
        data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Flash model catalog is missing: {CATALOG_PATH}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Flash model catalog is invalid JSON: {exc}") from exc

    missing = {"gemma", "qwen", "memory"} - set(data)
    if missing:
        raise RuntimeError(f"Flash model catalog is missing roles: {sorted(missing)}")
    for role, model in data.items():
        if not isinstance(model, dict) or not model.get("file"):
            raise RuntimeError(f"Flash catalog entry {role!r} needs a model file")
    return data


def public_catalog() -> Dict[str, Dict[str, Any]]:
    data = load_catalog()
    for role, model in data.items():
        model["role"] = role
        model["local_path"] = str(Path(config.MODEL_DIR or config.BASE_DIR / "models") / model["file"])
        if model.get("mmproj"):
            model["local_mmproj_path"] = str(Path(config.MODEL_DIR or config.BASE_DIR / "models") / model["mmproj"])
    return data

