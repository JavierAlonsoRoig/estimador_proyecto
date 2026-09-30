import json
import time
from app.prompts.loader import PROMPT_VERSION, render_estimation_prompt
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from app.config import get_settings
from app.schemas.schemas import EstimationRequest, EstimationResponse
from app.schemas.schemas import EstimationRequest as FormEstimationRequest
from app.services.llm_service import agregador_llm, build_system_prompt, build_user_prompt, generate_estimation

router = APIRouter(prefix="/api/v1", tags=["estimations"])

@router.post("/estimate", response_model=EstimationResponse)
async def estimate(request: EstimationRequest):
    system_prompt, user_prompt = render_estimation_prompt(request)
    try:
        text = await generate_estimation(system_prompt, user_prompt)
        return EstimationResponse(text=text, prompt_version=PROMPT_VERSION)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error interno al generar la estimación: {exc}") from exc


def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

@router.post("/estimate/stream")
async def estimate_stream(request: FormEstimationRequest):
    if not get_settings().ANTHROPIC_API_KEY.strip():
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY no configurada")

    def event_stream():
        client = agregador_llm()
        start_time = time.time()
        try:
            response = client.create_stream(
                system_prompt=build_system_prompt(),
                transcripcion=build_user_prompt(request),
            )
            for text in client.stream_to_text(response):
                yield sse_event("token", {"text": text})
        except Exception as exc:
            yield sse_event("error", {"detail": f"Error al consultar el LLM: {exc}"})
            return

        yield sse_event("metrics", {
            "model": client.last_model,
            "input_tokens": getattr(client, "prompt_tokens", None),
            "output_tokens": getattr(client, "completion_tokens", None),
            "elapsed_seconds": time.time() - start_time,
            "chunks": client.chunks,
        })
        yield sse_event("done", {})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
