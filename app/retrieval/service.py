import time
from typing import Literal

import structlog
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession

from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from app.retrieval.lexical import lexical_search
from app.retrieval.types import RetrievedChunk
from app.retrieval.vector import vector_search

logger = structlog.get_logger(__name__)

SearchMode = Literal["vector", "hybrid"]


async def retrieve(
    session: AsyncSession,
    embedder: OpenAIEmbedder,
    query: str,
    *,
    mode: SearchMode,
    top_k: int,
    recall_k: int,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[RetrievedChunk]:
    """Single entry point for retrieval: returns the top_k chunks for the query.

    recall_k is how many candidates each branch retrieves before fusing (and, in step 3,
    before reranking). For plain vector search it has no effect beyond top_k.
    """
    started_at = time.perf_counter()

    # Same embedder (and model) as ingestion: vectors from different models aren't comparable
    query_vector = await run_in_threadpool(embedder.embed_one, query)

    if mode == "vector":
        results = await vector_search(session, query_vector, limit=top_k)
    else:
        # Sequential on purpose: an AsyncSession can't run two queries at the same time
        vector_hits = await vector_search(session, query_vector, limit=recall_k)
        lexical_hits = await lexical_search(session, query, limit=recall_k)
        results = reciprocal_rank_fusion([vector_hits, lexical_hits], k=rrf_k)[:top_k]
        logger.debug("hybrid_branches", vector_hits=len(vector_hits), lexical_hits=len(lexical_hits))

    logger.info(
        "retrieval_done",
        mode=mode,
        top_k=top_k,
        recall_k=recall_k,
        results=len(results),
        retrieval_time_ms=round((time.perf_counter() - started_at) * 1000),
    )
    return results
