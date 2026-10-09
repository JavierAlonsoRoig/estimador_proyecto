"""Extracción local de texto de los adjuntos (camino B).

Extraemos el texto en el servicio en lugar de enviar el archivo al LLM:
funciona con cualquier proveedor (usamos LiteLLM) y prepara el terreno
para el chunking de RAG del módulo 3.
"""
import io

from docx import Document
from fastapi import HTTPException, UploadFile
from pypdf import PdfReader

# Límite por adjunto para no disparar los tokens con documentos enormes
MAX_ATTACHMENT_CHARS = 20_000


def extract_text(filename: str, content: bytes) -> str:
    name = filename.lower()
    try:
        if name.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(content))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        if name.endswith(".docx"):
            document = Document(io.BytesIO(content))
            return "\n".join(paragraph.text for paragraph in document.paragraphs)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"No se pudo leer {filename}: {exc}") from exc
    raise HTTPException(status_code=415, detail=f"Formato no soportado: {filename}. Solo PDF y DOCX.")


def build_transcript_with_attachments(transcript: str, attachments: list[UploadFile]) -> str:
    """Concatena la transcripción con el texto de cada adjunto, con un separador claro."""
    parts = [transcript]
    for file in attachments:
        text = extract_text(file.filename, file.file.read()).strip()
        parts.append(f"--- attachment: {file.filename} ---\n{text[:MAX_ATTACHMENT_CHARS]}")
    return "\n\n".join(parts)
