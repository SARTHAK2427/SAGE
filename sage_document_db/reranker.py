"""Optional, lazy CPU-safe cross-encoder reranking for document RAG."""
from __future__ import annotations

import logging
import os

from .config import DEFAULT_RAG_TOP_K, RERANK_BATCH_SIZE, RERANKER_DEVICE, RERANKER_ENABLED, RERANKER_MODEL
from .models import RagResult

logger = logging.getLogger(__name__)


class RerankService:
    """Loads a CrossEncoder only when explicitly enabled."""

    def __init__(self, model_name: str = RERANKER_MODEL, device: str = RERANKER_DEVICE,
                 batch_size: int = RERANK_BATCH_SIZE, enabled: bool = RERANKER_ENABLED) -> None:
        self.model_name, self._device, self._batch_size = model_name, device, batch_size
        self._enabled = enabled
        self._mock = os.environ.get("SAGE_MOCK_MODE", "0") == "1"
        self._model = None
        if enabled and not self._mock:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(model_name, device=device)
        logger.info("Reranker: enabled=%s model=%s device=%s", enabled, model_name, device)

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def device(self) -> str:
        return self._device

    def rerank(self, query: str, results: list[RagResult], top_k: int = DEFAULT_RAG_TOP_K) -> list[RagResult]:
        if not self._enabled or not results:
            return results[:top_k]
        if self._mock:
            for i, result in enumerate(results):
                result.reranker_score = round(1.0 - i * 0.01, 4)
            return results[:top_k]
        try:
            scores = self._model.predict([[query, result.text] for result in results], batch_size=self._batch_size, show_progress_bar=False)
            scored = []
            for score, result in zip(scores, results):
                result.reranker_score = float(score)
                scored.append(result)
            return sorted(scored, key=lambda result: result.reranker_score, reverse=True)[:top_k]
        except Exception as exc:
            logger.warning("Reranking failed; retaining vector rank: %s", exc)
            return results[:top_k]
