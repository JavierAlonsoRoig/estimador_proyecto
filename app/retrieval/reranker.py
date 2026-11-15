import time
from dataclasses import replace
from functools import lru_cache

import structlog
from sentence_transformers import CrossEncoder

from app.config import get_settings
from app.retrieval.types import RetrievedChunk

logger = structlog.get_logger(__name__)


class CrossEncoderReranker:
    """Scores (query, document) pairs jointly with a cross-encoder and reorders candidates.

    Unlike the bi-encoder (embeddings), the cross-encoder reads query and document together,
    so it can judge relevance far better, but it must run once per pair: only viable on a
    short list of candidates, never on the whole corpus.
    """

    def __init__(self, model_name: str, max_length: int = 512):
        started_at = time.perf_counter()
        self.model_name = model_name
        self._model = CrossEncoder(model_name, max_length=max_length, device="cpu")
        logger.info(
            "reranker_loaded",
            model=model_name,
            load_time_ms=round((time.perf_counter() - started_at) * 1000),
        )

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Relevance logit for each document (higher is more relevant; not a probability)."""
        if not documents:
            return []
        scores = self._model.predict(
            [(query, document) for document in documents],
            batch_size=32,
            show_progress_bar=False,
        )
        return [float(score) for score in scores]

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        """Reorder chunks by cross-encoder score and keep the best top_k."""
        scores = self.score(query, [chunk.content for chunk in chunks])
        rescored = [replace(chunk, score=score) for chunk, score in zip(chunks, scores)]
        # Tie-break by chunk_id so the order is deterministic
        rescored.sort(key=lambda chunk: (-chunk.score, chunk.chunk_id))
        return rescored[:top_k]


@lru_cache
def get_reranker() -> CrossEncoderReranker:
    # Loading the model takes seconds: do it once per process and reuse it
    return CrossEncoderReranker(get_settings().RERANKER_MODEL)
