from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.prompts.loader import PROMPT_VERSION, render_session_prompts
from app.schemas.schemas import (
    DetailLevel,
    OutputFormat,
    ProjectType,
    SessionEstimateResponse,
    SessionResponse,
)
from app.services.attachments import build_transcript_with_attachments
from app.services.facts import extract_project_facts
from app.services.llm_service import agregador_llm
from app.services.sessions import create_session, get_session

router = APIRouter(tags=["sessions"])


@router.post("/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def new_session():
    session = create_session()
    return SessionResponse(session_id=session.session_id)


@router.post("/sessions/{session_id}/estimate", response_model=SessionEstimateResponse)
def session_estimate(
    session_id: str,
    transcript: str = Form(min_length=20),
    project_type: ProjectType = Form(ProjectType.WEB_SAAS),
    detail_level: DetailLevel = Form(DetailLevel.MEDIUM),
    output_format: OutputFormat = Form(OutputFormat.PHASES_TABLE),
    attachments: list[UploadFile] | None = File(None),
):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")

    full_transcript = build_transcript_with_attachments(transcript, attachments or [])

    system_prompt, user_prompt = render_session_prompts(
        session.project_facts,
        full_transcript,
        project_type.value,
        detail_level.value,
        output_format.value,
    )
    messages = session.history.to_messages_list(system_prompt) + [
        {"role": "user", "content": user_prompt}
    ]

    try:
        answer = agregador_llm().create_from_messages(messages)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Error al consultar el LLM: {exc}") from exc

    session.history.add_turn(user_prompt, answer)
    session.project_facts = extract_project_facts(session.project_facts, user_prompt, answer)

    return SessionEstimateResponse(
        text=answer,
        prompt_version=PROMPT_VERSION,
        project_facts=session.project_facts,
    )
