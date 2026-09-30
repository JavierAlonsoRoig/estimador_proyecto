import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import streamlit as st
from app.config import get_settings
from app.services.llm_service import build_system_prompt, build_user_prompt
from app.context.examples import ESTIMATION_EXAMPLES
from app.schemas.schemas import DetailLevel, EstimationRequest, OutputFormat, ProjectType
from pydantic import ValidationError

settings = get_settings()
system_prompt = build_system_prompt()


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


def stream_estimation(request: EstimationRequest):
    """Llama al endpoint SSE de la API y va devolviendo el texto según llega."""
    with httpx.stream(
        "POST",
        f"{settings.API_URL}/api/v1/estimate/stream",
        json=request.model_dump(mode="json"),
        timeout=httpx.Timeout(10.0, read=120.0),
    ) as response:
        if response.status_code != 200:
            response.read()
            raise ApiError(format_api_error(response))

        event = None
        for line in response.iter_lines():
            if line.startswith("event:"):
                event = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data = json.loads(line.removeprefix("data:").strip())
                if event == "token":
                    yield data["text"]
                elif event == "metrics":
                    st.session_state.last_call_metrics = data
                elif event == "error":
                    raise ApiError(data["detail"])


st.write ("Estimador de tiempos Javier Alonso")
#inicio historial
if "messages" not in st.session_state:
    st.session_state.messages = []

#Renderizar mensajes
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#Input usuario
with st.form("estimation_form"):
    description = st.text_area("Descripción del proyecto", max_chars=2000)
    project_type = st.selectbox("Tipo de proyecto", list(ProjectType), format_func=lambda e: e.value)
    detail_level = st.selectbox("Nivel de detalle", list(DetailLevel), format_func=lambda e: e.value)
    output_format = st.selectbox("Formato de salida", list(OutputFormat), format_func=lambda e: e.value)
    submitted = st.form_submit_button("Estimar")

request = None
if submitted:
    try:
        request = EstimationRequest(
            description=description,
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
        )
    except ValidationError as exc:
        for error in exc.errors():
            st.error(f"{error['loc'][0]}: {error['msg']}")

if request:
    prompt = build_user_prompt(request)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    #Generar respuesta vía API (SSE):
    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(stream_estimation(request))
        except httpx.ConnectError:
            answer = None
            st.error(f"No se puede conectar con la API en {settings.API_URL}. ¿Está arrancado uvicorn?")
        except (ApiError, httpx.HTTPError) as exc:
            answer = None
            st.error(f"Error de la API: {exc}")

    if answer:
        st.session_state.messages.append({"role": "assistant", "content": answer})

#Panel lateral
with st.sidebar:
    st.title("Panel del sistema")

    st.subheader("System prompt activo")
    st.text_area(
        "System prompt",
        value=system_prompt,
        height=200,
        disabled=True,
        label_visibility="collapsed",
    )

    st.subheader("Contexto estático (ejemplos CAG)")
    for i, example in enumerate(ESTIMATION_EXAMPLES, start=1):
        with st.expander(f"Ejemplo {i}: {example['meeting_summary'][:40]}..."):
            st.markdown(f"**Resumen:** {example['meeting_summary']}")
            st.markdown(f"**Estimación:**\n{example['estimation']}")

    st.subheader("Métricas de la última llamada")
    metrics = st.session_state.get("last_call_metrics")
    if metrics:
        st.metric("Modelo", metrics["model"])
        col1, col2 = st.columns(2)
        col1.metric("Tokens entrada", metrics["input_tokens"])
        col2.metric("Tokens salida", metrics["output_tokens"])
        st.metric("Tiempo de respuesta", f"{metrics['elapsed_seconds']:.2f} s")
    else:
        st.caption("Todavía no se ha realizado ninguna llamada.")
