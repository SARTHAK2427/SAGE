"""
SAGE/core/mappers/__init__.py
Socket → Plug Mapper Package.

Provides all mapping functions that project rich deterministic internal sockets
onto minimal model-facing payloads (input plugs and output result objects).

Architecture:
    DETERMINISTIC PRODUCER
            ↓
    RICH INTERNAL SOCKET (frozen — never mutated here)
            ↓
    PYTHON MAPPER (this package)
            ↓
    MINIMAL CONSUMER INPUT PLUG
            (sent to Gemma / Qwen-Coder / Qwen-VL)

Key rules:
    - Mappers are pure projections. They never mutate source socket data.
    - Local filesystem paths must never appear in Gemma-facing output.
    - All Gemma-facing tool result objects are uniformly wrapped by the outer
      shell defined in gemma_results.map_tool_result_for_gemma().
    - The outer status may be "success", "partial", or "error".
"""

from .gemma_results import (
    map_rag_result,
    map_exact_result,
    map_artifact_fetch_result,
    map_list_artifacts_result,
    map_vision_result,
    map_coder_result,
    map_math_result,
    map_error_for_gemma,
    map_tool_result_for_gemma,
    build_tool_results_packet,
    strip_internal_fields,
    CANONICAL_GEMMA_MAPPERS,
)
from .coder_input import build_coder_input
from .vision_input import build_vision_text_prompt, resolve_vision_image
from .memory_results import (
    map_memory_search_result,
    map_memory_store_result,
    map_memory_update_result,
    map_memory_delete_result,
    map_memory_summarize_result,
    map_memory_promote_result,
)

__all__ = [
    # Gemma-facing result mappers
    "map_rag_result",
    "map_exact_result",
    "map_artifact_fetch_result",
    "map_list_artifacts_result",
    "map_vision_result",
    "map_coder_result",
    "map_math_result",
    "map_error_for_gemma",
    "map_tool_result_for_gemma",
    "build_tool_results_packet",
    "strip_internal_fields",
    "CANONICAL_GEMMA_MAPPERS",
    # Memory result mappers
    "map_memory_search_result",
    "map_memory_store_result",
    "map_memory_update_result",
    "map_memory_delete_result",
    "map_memory_summarize_result",
    "map_memory_promote_result",
    # Specialist input plug builders
    "build_coder_input",
    "build_vision_text_prompt",
    "resolve_vision_image",
]
