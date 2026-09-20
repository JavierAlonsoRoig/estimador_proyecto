import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from app.config import get_settings
from app.services.llm_service import agregador_llm, build_system_prompt
from app.context.examples import ESTIMATION_EXAMPLES

settings = get_settings()
api_key = settings.ANTHROPIC_API_KEY.strip()
if not api_key:
    st.error("ANTHROPIC_API_KEY no configurada")
    st.stop()

client = agregador_llm()
system_prompt = build_system_prompt()
st.write ("Estimador de tiempos Javier Alonso")
#inicio historial
if "messages" not in st.session_state:
    st.session_state.messages = []

#Renderizar mensajes
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#Input usuario
if prompt := st.chat_input("Escribe tu mensaje"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    #Generar respuesta:
    with st.chat_message("assistant"):
        start_time = time.time()
        # response = client.create(system_prompt=system_prompt, transcripcion=prompt)
        response = client.create_stream(system_prompt=system_prompt, transcripcion=prompt)
        elapsed_seconds = time.time() - start_time
        # answer = response.choices[0].message.content
        answer = client.stream_to_text(response)
        st.write_stream(answer)

        st.session_state.last_call_metrics = {
            "model": client.last_model,
            "input_tokens": client.prompt_tokens,
            "output_tokens": client.completion_tokens,
            "elapsed_seconds": elapsed_seconds,
            "chunks":client.chunks
        }

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
