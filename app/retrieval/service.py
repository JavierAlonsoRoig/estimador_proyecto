import time
from dataclasses import dataclass, field
from typing import Literal

import structlog
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession

from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from app.retrieval.lexical import lexical_search
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.types import RetrievedChunk
from app.retrieval.vector import vector_search

logger = structlog.get_logger(__name__)

SearchMode = Literal["vector", "hybrid"]


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    # Per-stage latency: tells embedding (network) apart from search (DB) and rerank (CPU)
    timings_ms: dict[str, float] = field(default_factory=dict)


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 1)


async def retrieve(
    session: AsyncSession,
    embedder: OpenAIEmbedder,
    query: str,
    *,
    mode: SearchMode,
    top_k: int,
    recall_k: int,
    rrf_k: int = DEFAULT_RRF_K,
    reranker: CrossEncoderReranker | None = None,
) -> RetrievalResult:
    """Single entry point for retrieval: returns the top_k chunks for the query.

    Recall-then-rerank: with a reranker, the first stage retrieves recall_k candidates and
    the cross-encoder keeps the best top_k. Without one, the first stage returns top_k directly.
    """
    timings: dict[str, float] = {}
    started_total = time.perf_counter()

    started_at = time.perf_counter()
    # Same embedder (and model) as ingestion: vectors from different models aren't comparable
    query_vector = await run_in_threadpool(embedder.embed_one, query)
    timings["embed"] = _elapsed_ms(started_at)

    # Without a reranker nothing reorders the list, so fetching more than top_k is wasted work
    candidates_k = recall_k if reranker is not None else top_k

    started_at = time.perf_counter()
    if mode == "vector":
        candidates = await vector_search(session, query_vector, limit=candidates_k)
    else:
        # Each branch always fetches recall_k: RRF needs deep lists to find agreement.
        # Sequential on purpose: an AsyncSession can't run two queries at the same time.
        vector_hits = await vector_search(session, query_vector, limit=recall_k)
        lexical_hits = await lexical_search(session, query, limit=recall_k)
        candidates = reciprocal_rank_fusion([vector_hits, lexical_hits], k=rrf_k)[:candidates_k]
    timings["search"] = _elapsed_ms(started_at)

    if reranker is not None:
        started_at = time.perf_counter()
        # CPU-bound inference: run it in a thread so it doesn't block the event loop
        results = await run_in_threadpool(reranker.rerank, query, candidates, top_k)
        timings["rerank"] = _elapsed_ms(started_at)
    else:
        results = candidates[:top_k]

    timings["total"] = _elapsed_ms(started_total)
    logger.info(
        "retrieval_done",
        mode=mode,
        reranked=reranker is not None,
        top_k=top_k,
        candidates=len(candidates),
        results=len(results),
        **{f"{stage}_ms": value for stage, value in timings.items()},
    )
    return RetrievalResult(chunks=results, timings_ms=timings)
