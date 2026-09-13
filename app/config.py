from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    OPENAI_API_KEY: str = ''
    ANTHROPIC_API_KEY: str = ''
    LLM_PROVIDER: str = "anthropic" #"openai"
    LLM_MODEL: str = "claude-haiku-4-5" #"gpt-4o-mini"
    APP_ENV: str = "development"
    LOG_LEVEL: str = "DEBUG"

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()
