import time

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from openai import OpenAIError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.embedding_pipeline.router import get_embedder
from app.embedding_pipeline.schemas import SearchRequest, SearchResponse, SearchResult
from app.models import Chunk

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    session: AsyncSession = Depends(get_session),
    embedder: OpenAIEmbedder = Depends(get_embedder),
):
    started_at = time.perf_counter()

    # Same embedder (and model) as ingestion: vectors from different models aren't comparable
    try:
        query_vector = await run_in_threadpool(embedder.embed_one, request.query)
    except OpenAIError:
        logger.exception("search_embedding_failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating query embedding",
        )

    distance = Chunk.embedding.cosine_distance(query_vector)
    stmt = (
        select(
            Chunk.id,
            Chunk.document_id,
            Chunk.chunk_type,
            Chunk.content,
            # Chunk.metadata would be SQLAlchemy's MetaData, not the column
            Chunk.metadata_,
            distance.label("distance"),
        )
        # embedding is nullable: chunks not embedded yet can't be ranked
        .where(Chunk.embedding.is_not(None))
        .order_by(distance)
        .limit(request.k)
    )
    rows = (await session.execute(stmt)).all()

    search_time_ms = round((time.perf_counter() - started_at) * 1000)
    logger.info("search_done", k=request.k, results=len(rows), search_time_ms=search_time_ms)
    return SearchResponse(
        query=request.query,
        k=request.k,
        search_time_ms=search_time_ms,
        results=[
            SearchResult(
                chunk_id=row.id,
                document_id=row.document_id,
                chunk_type=row.chunk_type,
                content=row.content,
                distance=row.distance,
                metadata=row.metadata_,
            )
            for row in rows
        ],
    )
