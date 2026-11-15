import time

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from openai import OpenAIError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_session
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.embedding_pipeline.router import get_embedder
from app.embedding_pipeline.schemas import SearchRequest, SearchResponse, SearchResult
from app.retrieval.reranker import get_reranker
from app.retrieval.service import retrieve

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    session: AsyncSession = Depends(get_session),
    embedder: OpenAIEmbedder = Depends(get_embedder),
):
    settings = get_settings()
    # Request values win over server defaults: the 4 configurations are reachable per call
    mode = request.mode or settings.SEARCH_MODE
    rerank = request.rerank if request.rerank is not None else settings.RERANK_ENABLED
    started_at = time.perf_counter()

    # The model is only loaded the first time reranking is requested (it takes seconds);
    # in a thread so that first load doesn't block the event loop
    reranker = await run_in_threadpool(get_reranker) if rerank else None

    try:
        result = await retrieve(
            session,
            embedder,
            request.query,
            mode=mode,
            top_k=request.k,
            recall_k=settings.RECALL_K,
            rrf_k=settings.RRF_K,
            reranker=reranker,
        )
    except OpenAIError:
        logger.exception("search_embedding_failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating query embedding",
        )

    search_time_ms = round((time.perf_counter() - started_at) * 1000)
    logger.info(
        "search_done",
        mode=mode,
        reranked=rerank,
        k=request.k,
        results=len(result.chunks),
        search_time_ms=search_time_ms,
    )
    return SearchResponse(
        query=request.query,
        k=request.k,
        mode=mode,
        reranked=rerank,
        search_time_ms=search_time_ms,
        timings_ms=result.timings_ms,
        results=[
            SearchResult(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                chunk_type=chunk.chunk_type,
                content=chunk.content,
                score=chunk.score,
                metadata=chunk.metadata,
            )
            for chunk in result.chunks
        ],
    )
