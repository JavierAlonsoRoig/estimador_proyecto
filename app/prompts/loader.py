from pathlib import Path
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.schemas.schemas import EstimationRequest
from app.services.sessions import ProjectFacts


PROMPTS_DIR = Path(__file__).parent
PROMPT_VERSION = "v1"

_env = Environment(
    loader=FileSystemLoader(PROMPTS_DIR),
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
    undefined=StrictUndefined,
)


def render_estimation_prompt(
    request: EstimationRequest,
    version: str = PROMPT_VERSION,
) -> tuple[str, str]:
    system = _env.get_template(f"estimation/{version}/system.j2")
    user = _env.get_template(f"estimation/{version}/user.j2")

    context = {
        "project_type": request.project_type.value,
        "detail_level": request.detail_level.value,
        "output_format": request.output_format.value,
        "description": request.description,
        "project_facts":  ProjectFacts(),
    }

    return system.render(**context), user.render(**context)

def render_session_prompts(
    project_facts: ProjectFacts,
    transcript: str,
    project_type: str,
    detail_level: str,
    output_format: str,
    version: str = PROMPT_VERSION,
) -> tuple[str, str]:
    system = _env.get_template(f"estimation/{version}/system.j2")
    user = _env.get_template(f"estimation/{version}/user.j2")

    context = {
        "project_type": project_type,
        "detail_level": detail_level,
        "output_format": output_format,
        "description": transcript,
        "project_facts": project_facts,
    }

    return system.render(**context), user.render(**context)


def render_facts_prompt(
    current_facts: ProjectFacts,
    user_message: str,
    assistant_message: str,
    version: str = "v1",
) -> str:
    template = _env.get_template(f"facts/{version}/extract.j2")
    return template.render(
        current_facts=current_facts.model_dump_json(indent=2),
        user_message=user_message,
        assistant_message=assistant_message,
    )

