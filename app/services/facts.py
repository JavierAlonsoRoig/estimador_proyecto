import re

from pydantic import ValidationError

from app.prompts.loader import render_facts_prompt
from app.services.llm_service import agregador_llm
from app.services.sessions import ProjectFacts


def extract_project_facts(
    current: ProjectFacts, user_message: str, assistant_message: str
) -> ProjectFacts:
    """Segunda llamada al LLM que devuelve los project_facts actualizados.

    Si el modelo devuelve algo que no es un JSON válido, conservamos los
    facts anteriores: perder una actualización es mejor que romper el turno.
    """
    prompt = render_facts_prompt(current, user_message, assistant_message)
    raw = agregador_llm().create_from_messages([{"role": "user", "content": prompt}])

    # El modelo a veces envuelve el JSON en ```json ... ```; nos quedamos con el {...}
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not match:
        return current
    try:
        return ProjectFacts.model_validate_json(match.group(0))
    except ValidationError:
        return current
