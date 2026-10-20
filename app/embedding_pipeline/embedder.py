import time

import structlog
from openai import OpenAI, RateLimitError

from app.config import get_settings
from app.embedding_pipeline.schemas import Chunk, EmbeddedChunk

logger = structlog.get_logger(__name__)

BATCH_SIZE = 100
RETRY_WAITS_SECONDS = (1, 2, 4)

# Pricing for text-embedding-3-small (USD per 1M input tokens).
# Changes over time: check https://openai.com/api/pricing before trusting it.
PRICE_PER_MILLION_TOKENS_USD = 0.02


def estimate_cost_usd(total_tokens: int) -> float:
    return total_tokens / 1_000_000 * PRICE_PER_MILLION_TOKENS_USD


class OpenAIEmbedder:
    def __init__(self, model_name: str = "text-embedding-3-small", batch_size: int = BATCH_SIZE):
        # max_retries=0: the SDK retries on its own by default; we handle RateLimitError ourselves
        self.client = OpenAI(api_key=get_settings().OPENAI_API_KEY, max_retries=0)
        self.model_name = model_name
        self.batch_size = batch_size

    def embed_one(self, text: str) -> list[float]:
        response = self._create_with_retry([text])
        return response.data[0].embedding

    def embed_many(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        lista: list[EmbeddedChunk] = []
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]

            started_at = time.perf_counter()
            response = self._create_with_retry([chunk.text for chunk in batch])
            latency_ms = (time.perf_counter() - started_at) * 1000

            logger.info(
                "embedding_batch_processed",
                batch_index=start // self.batch_size,
                chunks=len(batch),
                tokens=response.usage.total_tokens,
                latency_ms=round(latency_ms, 1),
            )

            # The API returns one embedding per input, in the same order
            for chunk, item in zip(batch, response.data):
                lista.append(EmbeddedChunk(**chunk.model_dump(), embedding=item.embedding))
        return lista

    def _create_with_retry(self, texts: list[str]):
        # Waits 1s, 2s, 4s between attempts; on the 4th failure (wait=None) re-raises
        for attempt, wait in enumerate((*RETRY_WAITS_SECONDS, None)):
            try:
                return self.client.embeddings.create(model=self.model_name, input=texts)
            except RateLimitError:
                if wait is None:
                    raise
                logger.warning("embedding_rate_limited", attempt=attempt + 1, retry_in_s=wait)
                time.sleep(wait)
