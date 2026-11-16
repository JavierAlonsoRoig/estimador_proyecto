import time
from functools import lru_cache

import structlog
from openai import OpenAI

from app.config import get_settings
from app.generation.context import AssembledContext
from app.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from app.generation.schemas import Estimate

logger = structlog.get_logger(__name__)


class EstimateGenerationError(Exception):
    pass


class EstimateGenerator:
    def __init__(self, model: str):
        self.client = OpenAI(api_key=get_settings().OPENAI_API_KEY)
        self.model = model

    def generate(self, request: str, context: AssembledContext) -> Estimate:
        started_at = time.perf_counter()
        # text_format turns Estimate into a strict JSON Schema (text.format) and parses the reply
        response = self.client.responses.parse(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=build_user_prompt(request, context.text),
            text_format=Estimate,
        )
        estimate = response.output_parsed
        if estimate is None:
            # Strict mode guarantees the shape, but the model can still refuse
            raise EstimateGenerationError("The model returned no parsed estimate (refusal)")

        logger.info(
            "estimate_generated",
            model=self.model,
            line_items=len(estimate.line_items),
            context_chunks=len(context.chunks),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=round((time.perf_counter() - started_at) * 1000),
        )
        return estimate


@lru_cache
def get_generator() -> EstimateGenerator:
    return EstimateGenerator(get_settings().GENERATION_MODEL)
