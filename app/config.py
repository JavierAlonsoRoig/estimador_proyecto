from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    OPENAI_API_KEY: str = ''
    ANTHROPIC_API_KEY: str = ''
    LLM_PROVIDER: str = "anthropic" #"openai"
    LLM_MODEL: str = "claude-haiku-4-5" #"gpt-4o-mini"
    APP_ENV: str = "development"
    LOG_LEVEL: str = "DEBUG"
    API_URL: str = "http://localhost:8000"
    MAX_TURNS: int = 6
    DATABASE_URL: str = "postgresql+asyncpg://estimator:estimator@localhost:5432/estimator"
     # Retrieval (session 10)
    SEARCH_MODE: str = "hybrid"   # "vector" | "hybrid"
    RECALL_K: int = 50            # candidates per branch before fusing/reranking
    RRF_K: int = 60               # RRF smoothing constant
    
    RERANK_ENABLED: bool = False  # cross-encoder second stage on/off
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Generation (session 11)
    GENERATION_MODEL: str = "gpt-4.1-mini"  # OpenAI model with Structured Outputs support
    CONTEXT_TOP_K: int = 8                  # chunks passed to the generator
    
    JUDGE_MODEL: str = "gpt-4.1"  # RAGAS judge: stronger than, and different from, the generator




    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()
