import time

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from openai import OpenAIError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_session
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.embedding_pipeline.router import get_embedder
from app.embedding_pipeline.schemas import SearchRequest, SearchResponse, SearchResult
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
    mode = request.mode or settings.SEARCH_MODE
    started_at = time.perf_counter()

    try:
        chunks = await retrieve(
            session,
            embedder,
            request.query,
            mode=mode,
            top_k=request.k,
            recall_k=settings.RECALL_K,
            rrf_k=settings.RRF_K,
        )
    except OpenAIError:
        logger.exception("search_embedding_failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating query embedding",
        )

    search_time_ms = round((time.perf_counter() - started_at) * 1000)
    logger.info("search_done", mode=mode, k=request.k, results=len(chunks), search_time_ms=search_time_ms)
    return SearchResponse(
        query=request.query,
        k=request.k,
        mode=mode,
        search_time_ms=search_time_ms,
        results=[
            SearchResult(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                chunk_type=chunk.chunk_type,
                content=chunk.content,
                score=chunk.score,
                metadata=chunk.metadata,
            )
            for chunk in chunks
        ],
    )
