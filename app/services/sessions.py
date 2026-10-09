"""Estado de las sesiones conversacionales del estimador.

Las sesiones viven en un diccionario en memoria del proceso, sin BBDD ni Redis.
Aceptamos esa volatilidad en esta fase: si el servicio se reinicia se pierden
las sesiones y el cliente simplemente crea una nueva. Basta para validar el
patrón de memoria; la persistencia hará falta cuando haya varias instancias
del servicio o las conversaciones tengan que sobrevivir a un reinicio.
"""
import uuid
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from app.config import get_settings


class ProjectFacts(BaseModel):
    """Hechos conocidos del proyecto en curso (la "memoria").

    Vive aparte del historial: el historial se recorta con la ventana
    deslizante, pero estos hechos se conservan y se inyectan en el
    system prompt en cada turno.
    """
    project_name: str | None = None
    assumed_team_size: int | None = None
    mentioned_technologies: list[str] = Field(default_factory=list)
    agreed_scope: str | None = None

    def is_empty(self) -> bool:
        return not any([
            self.project_name,
            self.assumed_team_size,
            self.mentioned_technologies,
            self.agreed_scope,
        ])


class ConversationHistory:
    """Historial de mensajes con ventana deslizante.

    Guarda solo los últimos `max_turns` turnos (un turno = mensaje user +
    respuesta assistant). El system prompt NO se guarda aquí: se regenera
    en cada llamada a partir del project_facts actual, así que siempre
    está presente y nunca se descarta.
    """

    def __init__(self, max_turns: int | None = None):
        self.max_turns = max_turns or get_settings().MAX_TURNS
        self.messages: list[dict] = []

    def add_turn(self, user_content: str, assistant_content: str) -> None:
        self.messages.append({"role": "user", "content": user_content})
        self.messages.append({"role": "assistant", "content": assistant_content})
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            # Descartamos los pares más antiguos
            self.messages = self.messages[-max_messages:]

    def to_messages_list(self, system_prompt: str) -> list[dict]:
        """Array `messages` listo para la API: system prompt + historial recortado."""
        return [{"role": "system", "content": system_prompt}, *self.messages]


@dataclass
class Session:
    session_id: str
    history: ConversationHistory = field(default_factory=ConversationHistory)
    project_facts: ProjectFacts = field(default_factory=ProjectFacts)


# Almacén en memoria del proceso: session_id -> Session
_SESSIONS: dict[str, Session] = {}


def create_session() -> Session:
    session = Session(session_id=str(uuid.uuid4()))
    _SESSIONS[session.session_id] = session
    return session


def get_session(session_id: str) -> Session | None:
    return _SESSIONS.get(session_id)
