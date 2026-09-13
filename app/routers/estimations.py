from fastapi import APIRouter, Depends, HTTPException
from app.schemas.estimation import EstimationRequest, EstimationResponse
from app.services.llm_service import generate_estimation

router = APIRouter(prefix="/api/v1", tags=["estimations"])

@router.post("/estimate", response_model=EstimationResponse)
async def estimate(request: EstimationRequest):
    print(f"Received estimation request: {request.transcription}")
    try:
        result = await generate_estimation(request.transcription)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error interno al generar la estimación: {exc}") from exc