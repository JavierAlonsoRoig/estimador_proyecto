from fastapi import HTTPException
from openai import OpenAI
from anthropic import Anthropic, AuthenticationError as AnthropicAuthError
from app.config import get_settings
from app.context.examples import ESTIMATION_EXAMPLES

settings = get_settings()


def build_system_prompt() -> str:
    examples_text = ESTIMATION_EXAMPLES
    return f"""Eres un experto en estimación de proyectos de software.
    
Utiliza los siguientes presupuestos históricos como referencia:

{examples_text}

Genera una estimación detallada para el proyecto descrito."""

async def generate_estimation(transcription: str) -> dict:
    system_prompt = build_system_prompt()

    if settings.LLM_PROVIDER == "openai":
        api_key = settings.OPENAI_API_KEY.strip()
        if not api_key:
            raise HTTPException(status_code=503, detail="OPENAI_API_KEY no configurada")

        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": transcription}
                ]
            )
            return {
                "estimation": response.choices[0].message.content,
                "model": settings.LLM_MODEL,
                "provider": settings.LLM_PROVIDER,
            }
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Error al consultar OpenAI: {exc}") from exc

    elif settings.LLM_PROVIDER == "anthropic":
        api_key = settings.ANTHROPIC_API_KEY.strip()
        if not api_key:
            raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY no configurada")

        try:
            client = Anthropic(api_key=api_key)
            response = client.messages.create(
                model=settings.LLM_MODEL.strip(),
                max_tokens=500,
                system=system_prompt,
                messages=[{"role": "user", "content": transcription}],
            )

            content = response.content[0].text if response.content else ""
            return {
                "estimation": content,
                "model": settings.LLM_MODEL.strip(),
                "provider": settings.LLM_PROVIDER,
            }
        except AnthropicAuthError as exc:
            raise HTTPException(status_code=401, detail="API key de Anthropic inválida o caducada") from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Error al consultar Anthropic: {exc}") from exc

    raise HTTPException(status_code=400, detail=f"Unsupported LLM provider: {settings.LLM_PROVIDER}")

class ConversationManager:
    def __init__(self, system_prompt: str):
        self.messages = [
            {"role": "system", "content": system_prompt}
        ]
    
    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})
    
    def add_assistant_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})
    
    def get_messages(self) -> list:
        return self.messages.copy()