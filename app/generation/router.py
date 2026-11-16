import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from openai import OpenAIError
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_session
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.embedding_pipeline.router import get_embedder
from app.generation.citations import CitationReport, verify_citations
from app.generation.context import assemble_context
from app.generation.generator import EstimateGenerationError, get_generator
from app.generation.schemas import Estimate
from app.retrieval.service import retrieve

logger = structlog.get_logger(__name__)

router = APIRouter()


class RagEstimateRequest(BaseModel):
    query: str = Field(min_length=10, description="Estimation request (already reformulated)")


class RagEstimateResponse(BaseModel):
    request_id: str
    estimate: Estimate
    # Hours of lines that passed citation verification; rejected lines don't count
    verified_hours: float
    citation_report: CitationReport


@router.post("/estimate", response_model=RagEstimateResponse)
async def rag_estimate(
    request: RagEstimateRequest,
    session: AsyncSession = Depends(get_session),
    embedder: OpenAIEmbedder = Depends(get_embedder),
):
    settings = get_settings()
    try:
        retrieval = await retrieve(
            session,
            embedder,
            request.query,
            mode=settings.SEARCH_MODE,
            top_k=settings.CONTEXT_TOP_K,
            recall_k=settings.RECALL_K,
            rrf_k=settings.RRF_K,
        )
        context = assemble_context(retrieval.chunks)
        estimate = await run_in_threadpool(get_generator().generate, request.query, context)
    except (OpenAIError, EstimateGenerationError):
        logger.exception("rag_estimate_failed")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Estimate generation failed")

    report = verify_citations(estimate, context)
    return RagEstimateResponse(
        request_id=structlog.contextvars.get_contextvars().get("request_id", ""),
        estimate=estimate,
        verified_hours=report.verified_hours,
        citation_report=report,
    )
