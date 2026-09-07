"""
SAGE/tools/registry.py
Central tool registry factory for SAGE.

Constructs and wires all tool adapters into a ToolRegistry:
    - Document Database: rag_search, exact_search, artifact_fetch, list_artifacts
    - Vision / OCR: analyze
    - Coder Specialist: execute
    - Safe Math Utility: calculate
"""

from __future__ import annotations
import logging
from typing import Optional

from core.dispatcher import ToolRegistry
from tools.document_database import register_document_db_tools
from tools.vision import register_vision_tools
from tools.coder import register_coder_tools
from tools.math_tool import register_math_tools

logger = logging.getLogger(__name__)


def create_default_registry(
    db=None,
    model_manager=None,
    model_client=None,
    code_pipeline=None,
) -> ToolRegistry:
    """Create and populate a standard ToolRegistry with all SAGE tools.

    If db is not provided, lazily imports the singleton from db_service.
    """
    registry = ToolRegistry()

    # 1. Document Database tools
    if db is None:
        try:
            from db_service import document_db
            db = document_db
        except Exception as e:
            logger.warning("Could not initialize db_service for registry: %s", e)
            db = None

    if db is not None:
        register_document_db_tools(registry, db)

    # 2. Vision tools
    register_vision_tools(
        registry,
        db=db,
        model_manager=model_manager,
        model_client=model_client,
    )

    # 3. Coder tools
    register_coder_tools(
        registry,
        pipeline=code_pipeline,
        model_manager=model_manager,
        model_client=model_client,
    )

    # 4. Safe Math tool
    register_math_tools(registry)

    logger.info("Initialized ToolRegistry with tools: %s", registry.available_tools)
    return registry
