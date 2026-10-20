from functools import lru_cache

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from openai import OpenAIError

from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import OpenAIEmbedder, estimate_cost_usd
from app.embedding_pipeline.schemas import IngestRequest, IngestResponse, IngestStats

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["embeddings"])


@lru_cache
def get_chunker() -> JSONStructuralChunker:
    return JSONStructuralChunker()


@lru_cache
def get_embedder() -> OpenAIEmbedder:
    return OpenAIEmbedder()


@router.post("/ingest", response_model=IngestResponse)
def ingest(
    request: IngestRequest,
    chunker: JSONStructuralChunker = Depends(get_chunker),
    embedder: OpenAIEmbedder = Depends(get_embedder),
):
    # The course chunker works with plain dicts; budgets are already validated by IngestRequest
    chunks = chunker.chunk([budget.model_dump() for budget in request.budgets])

    try:
        embedded = embedder.embed_many(chunks)
    except OpenAIError:
        logger.exception("embedding_ingest_failed", total_chunks=len(chunks))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating embeddings",
        )

    total_tokens = sum(chunk.token_count for chunk in chunks)
    return IngestResponse(
        chunks=embedded,
        stats=IngestStats(
            total_budgets=len(request.budgets),
            total_chunks=len(embedded),
            total_tokens=total_tokens,
            estimated_cost_usd=estimate_cost_usd(total_tokens),
        ),
    )
