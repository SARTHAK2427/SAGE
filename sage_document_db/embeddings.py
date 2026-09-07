"""
sage_document_db/embeddings.py
EmbeddingService — one reusable SentenceTransformer instance per process.

Rules:
- EMBEDDING_DEVICE = "auto" → detect CUDA/CPU at init
- normalize_embeddings=True for cosine-compatible vectors
- Do not load the model per chunk — keep one instance alive
- count_tokens() uses the model's HuggingFace tokenizer directly
  so the Chunker never needs to import SentenceTransformer itself
- Preserve 384-dimension vector compatibility (all-MiniLM-L6-v2)
"""

from __future__ import annotations
import os
import logging
from typing import TYPE_CHECKING

from .config import EMBEDDING_MODEL, EMBEDDING_DEVICE, EMBED_BATCH_SIZE

logger = logging.getLogger(__name__)


def _resolve_device(configured: str) -> str:
    """Resolve the embedding device string.

    'auto' → CUDA if torch.cuda.is_available(), else CPU.
    Any other value is used verbatim.
    """
    if configured.lower() == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = configured

    logger.info("Embedding device: configured=%s, resolved=%s", configured, device)
    return device


class EmbeddingService:
    """Wraps SentenceTransformer with a stable, reusable interface.

    Instantiate once and share across the pipeline.
    Do NOT instantiate inside the Chunker; inject this instance instead.
    """

    model_name: str = EMBEDDING_MODEL

    def __init__(self, batch_size: int = EMBED_BATCH_SIZE) -> None:
        self._is_mock = os.environ.get("SAGE_MOCK_MODE", "0") == "1"
        self._batch_size = batch_size

        if self._is_mock:
            self._device = "cpu"
            self._model = None
            logger.info("EmbeddingService initialized in MOCK mode (fast local testing).")
            return

        # Import here so the module can be imported without sentence-transformers
        # installed (useful for unit-testing models.py / utils.py in isolation).
        from sentence_transformers import SentenceTransformer

        self._device = _resolve_device(EMBEDDING_DEVICE)
        self._model = SentenceTransformer(EMBEDDING_MODEL, device=self._device)

        logger.info(
            "EmbeddingService initialized: model=%s, device=%s, max_seq_length=%d",
            EMBEDDING_MODEL,
            self._device,
            self._model.max_seq_length,
        )

    @property
    def device(self) -> str:
        """The resolved device string ('cuda' or 'cpu')."""
        return self._device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return normalized embeddings for a list of document strings."""
        if not texts:
            return []
        if getattr(self, "_is_mock", False):
            return [[0.0] * 384 for _ in texts]
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Return normalized embedding for a single query string."""
        if getattr(self, "_is_mock", False):
            return [0.0] * 384
        vector = self._model.encode(
            [text],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector[0].tolist()

    def count_tokens(self, text: str) -> int:
        """Count tokens using tokenizer or whitespace fallback in mock mode."""
        if getattr(self, "_is_mock", False):
            return max(1, len(text.split()))
        tokenizer = self._model.tokenizer
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        return len(token_ids)

    @property
    def max_seq_length(self) -> int:
        """The embedding model's maximum token input length."""
        if getattr(self, "_is_mock", False):
            return 256
        return self._model.max_seq_length

    # ------------------------------------------------------------------
    # Rich Output Sockets (Phase F)
    # ------------------------------------------------------------------

    def embed_documents_socket(self, texts: list[str]) -> dict:
        """Embed document texts and return complete Component F socket."""
        import time
        from core.sockets import build_embedding_socket, build_timing_payload, build_error_payload

        t0 = time.perf_counter()
        try:
            vectors = self.embed_documents(texts)
            dim = len(vectors[0]) if vectors else 384
            approx_tokens = sum(self.count_tokens(t) for t in texts)
            total_ms = (time.perf_counter() - t0) * 1000.0
            timing = build_timing_payload(duration_ms=total_ms)

            return build_embedding_socket(
                model_name=self.model_name,
                device=self.device,
                dimension=dim,
                input_count=len(texts),
                vectors=vectors,
                batch_size=self._batch_size,
                is_mock=getattr(self, "_is_mock", False),
                approx_tokens=approx_tokens,
                timing=timing,
            )
        except Exception as e:
            total_ms = (time.perf_counter() - t0) * 1000.0
            timing = build_timing_payload(duration_ms=total_ms)
            error_payload = build_error_payload(
                code="EMBEDDING_ERROR",
                type_name=type(e).__name__,
                message=str(e),
                exc=e,
            )
            return build_embedding_socket(
                model_name=self.model_name,
                device=self.device,
                dimension=384,
                input_count=len(texts),
                vectors=[],
                batch_size=self._batch_size,
                is_mock=getattr(self, "_is_mock", False),
                timing=timing,
                error=error_payload,
            )

    def embed_query_socket(self, text: str) -> dict:
        """Embed a query text and return complete Component F socket."""
        import time
        from core.sockets import build_embedding_socket, build_timing_payload, build_error_payload

        t0 = time.perf_counter()
        try:
            vector = self.embed_query(text)
            dim = len(vector)
            approx_tokens = self.count_tokens(text)
            total_ms = (time.perf_counter() - t0) * 1000.0
            timing = build_timing_payload(duration_ms=total_ms)

            return build_embedding_socket(
                model_name=self.model_name,
                device=self.device,
                dimension=dim,
                input_count=1,
                vectors=[vector],
                batch_size=1,
                is_mock=getattr(self, "_is_mock", False),
                approx_tokens=approx_tokens,
                timing=timing,
            )
        except Exception as e:
            total_ms = (time.perf_counter() - t0) * 1000.0
            timing = build_timing_payload(duration_ms=total_ms)
            error_payload = build_error_payload(
                code="EMBEDDING_QUERY_ERROR",
                type_name=type(e).__name__,
                message=str(e),
                exc=e,
            )
            return build_embedding_socket(
                model_name=self.model_name,
                device=self.device,
                dimension=384,
                input_count=1,
                vectors=[],
                batch_size=1,
                is_mock=getattr(self, "_is_mock", False),
                timing=timing,
                error=error_payload,
            )
