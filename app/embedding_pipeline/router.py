import time
from functools import lru_cache

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from openai import OpenAIError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import OpenAIEmbedder, estimate_cost_usd
from app.embedding_pipeline.schemas import IngestRequest, IngestResponse
# Aliased: schemas also has a Chunk (the pydantic one produced by the chunker)
from app.models import Chunk as ChunkRow
from app.models import Document

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["embeddings"])


@lru_cache
def get_chunker() -> JSONStructuralChunker:
    return JSONStructuralChunker()


@lru_cache
def get_embedder() -> OpenAIEmbedder:
    return OpenAIEmbedder()


def _already_ingested(document_id: int) -> JSONResponse:
    # JSONResponse instead of HTTPException so the body is flat: {"detail": ..., "document_id": ...}
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={"detail": "Document already ingested", "document_id": document_id},
    )


@router.post(
    "/ingest",
    response_model=IngestResponse,
    responses={409: {"description": "Document already ingested"}},
)
async def ingest(
    request: IngestRequest,
    session: AsyncSession = Depends(get_session),
    chunker: JSONStructuralChunker = Depends(get_chunker),
    embedder: OpenAIEmbedder = Depends(get_embedder),
):
    started_at = time.perf_counter()

    # 1. Check duplicates first so we don't pay for embeddings we won't store
    existing_id = await session.scalar(
        select(Document.id).where(Document.source_path == request.source_path)
    )
    if existing_id is not None:
        return _already_ingested(existing_id)

    # 2. Only added to the session: nothing reaches the database until commit
    document = Document(
        source_path=request.source_path,
        document_type=request.document_type,
        metadata_={"budget_id": request.content.budget_id},
    )
    session.add(document)

    # 3. The course chunker works with plain dicts
    chunks = chunker.chunk([request.content.model_dump()])

    # 4. The OpenAI client is sync: run it in a thread so it doesn't block the event loop
    try:
        embedded = await run_in_threadpool(embedder.embed_many, chunks)
    except OpenAIError:
        # No commit happened, so the session closes with a rollback: no orphan document
        logger.exception("embedding_ingest_failed", total_chunks=len(chunks))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating embeddings",
        )

    # 5. document=document fills document_id on flush, no need to know the id beforehand
    session.add_all(
        [
            ChunkRow(
                document=document,
                chunk_type="component",
                content=chunk.text,
                token_count=chunk.token_count,
                embedding=chunk.embedding,
                metadata_=chunk.metadata.model_dump(),
            )
            for chunk in embedded
        ]
    )

    # Model and price change over time: record what this ingestion actually used
    total_tokens = sum(chunk.token_count for chunk in embedded)
    document.metadata_ = {
        **document.metadata_,
        "embedding_model": embedder.model_name,
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimate_cost_usd(total_tokens),
    }

    # 6. Document and chunks are written together, or nothing is
    try:
        await session.commit()
    except IntegrityError:
        # Another request ingested the same source_path between steps 1 and 6
        await session.rollback()
        existing_id = await session.scalar(
            select(Document.id).where(Document.source_path == request.source_path)
        )
        return _already_ingested(existing_id)

    total_tokens = sum(chunk.token_count for chunk in embedded)
    ingestion_time_ms = round((time.perf_counter() - started_at) * 1000)
    logger.info(
        "document_ingested",
        document_id=document.id,
        chunks=len(embedded),
        total_tokens=total_tokens,
        estimated_cost_usd=estimate_cost_usd(total_tokens),
        ingestion_time_ms=ingestion_time_ms,
    )
    return IngestResponse(
        document_id=document.id,
        chunks_created=len(embedded),
        embedding_dimension=len(embedded[0].embedding),
        ingestion_time_ms=ingestion_time_ms,
    )
