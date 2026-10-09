import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import streamlit as st
from app.config import get_settings
from app.schemas.schemas import DetailLevel, OutputFormat, ProjectType

settings = get_settings()


class ApiError(Exception):
    pass


def format_api_error(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        return f"HTTP {response.status_code}: {response.text}"
    if isinstance(detail, list):
        return "; ".join(f"{error['loc'][-1]}: {error['msg']}" for error in detail)
    return f"HTTP {response.status_code}: {detail}"


def create_session() -> str:
    """POST /sessions → devuelve el session_id nuevo."""
    response = httpx.post(f"{settings.API_URL}/sessions", timeout=10.0)
    if response.status_code != 201:
        raise ApiError(format_api_error(response))
    return response.json()["session_id"]


def estimate(session_id: str, transcript: str, project_type, detail_level, output_format, files) -> dict:
    """POST /sessions/{id}/estimate como multipart/form-data."""
    response = httpx.post(
        f"{settings.API_URL}/sessions/{session_id}/estimate",
        data={
            "transcript": transcript,
            "project_type": project_type.value,
            "detail_level": detail_level.value,
            "output_format": output_format.value,
        },
        files=[("attachments", (f.name, f.getvalue(), f.type)) for f in files],
        timeout=httpx.Timeout(10.0, read=180.0),
    )
    if response.status_code == 404:
        raise ApiError("La sesión ya no existe (¿se reinició la API?). Pulsa «Nueva conversación».")
    if response.status_code != 200:
        raise ApiError(format_api_error(response))
    return response.json()


def reset_conversation():
    st.session_state.session_id = create_session()
    st.session_state.messages = []
    st.session_state.project_facts = None


st.title("Estimador de tiempos Javier Alonso")

# Crear la sesión una sola vez, al cargar la página
if "session_id" not in st.session_state:
    try:
        reset_conversation()
    except httpx.ConnectError:
        st.error(f"No se puede conectar con la API en {settings.API_URL}. ¿Está arrancado uvicorn?")
        st.stop()

# Formulario
with st.form("estimation_form", clear_on_submit=True):
    transcript = st.text_area("Transcripción o nuevo mensaje sobre el proyecto", max_chars=2000)
    files = st.file_uploader(
        "Adjuntos (PDF o Word)", type=["pdf", "docx"], accept_multiple_files=True
    )
    project_type = st.selectbox("Tipo de proyecto", list(ProjectType), format_func=lambda e: e.value)
    detail_level = st.selectbox("Nivel de detalle", list(DetailLevel), format_func=lambda e: e.value)
    output_format = st.selectbox("Formato de salida", list(OutputFormat), format_func=lambda e: e.value)
    submitted = st.form_submit_button("Estimar")

# Conversación hasta ahora
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Turno nuevo
if submitted and transcript.strip():
    shown = transcript
    if files:
        shown += "\n\n" + "\n".join(f"📎 {f.name}" for f in files)
    st.session_state.messages.append({"role": "user", "content": shown})
    with st.chat_message("user"):
        st.markdown(shown)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Estimando..."):
                result = estimate(
                    st.session_state.session_id, transcript, project_type, detail_level, output_format, files
                )
        except httpx.ConnectError:
            st.error(f"No se puede conectar con la API en {settings.API_URL}. ¿Está arrancado uvicorn?")
        except (ApiError, httpx.HTTPError) as exc:
            st.error(f"Error de la API: {exc}")
        else:
            st.markdown(result["text"])
            st.session_state.messages.append({"role": "assistant", "content": result["text"]})
            st.session_state.project_facts = result["project_facts"]

# Panel lateral
with st.sidebar:
    st.title("Panel del sistema")
    st.caption(f"Sesión: `{st.session_state.session_id}`")

    if st.button("Nueva conversación"):
        reset_conversation()
        st.rerun()

    st.subheader("Project facts (memoria)")
    if st.session_state.project_facts:
        st.json(st.session_state.project_facts)
    else:
        st.caption("Vacío: se rellena tras la primera estimación.")

    st.subheader("Historial")
    turns = len(st.session_state.messages) // 2
    st.caption(
        f"{turns} turnos en pantalla. La API solo envía al LLM "
        f"los últimos {settings.MAX_TURNS}, pero los facts se conservan siempre."
    )
