from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Closed universe of values seen in the historical budgets.
# Extend these if your dataset contains new ones (otherwise the API returns 422).
Sector = Literal["finance", "ecommerce", "healthcare", "industrial"]
Complexity = Literal["low", "medium", "high"]


class ClientMetadata(BaseModel):
    name: str = Field(min_length=1)
    sector: Sector
    country: str = Field(min_length=2, max_length=2, description="ISO 3166-1 alpha-2")


class BudgetComponent(BaseModel):
    component_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    tech_stack: list[str] = Field(default_factory=list)
    estimated_hours: float = Field(gt=0)
    complexity: Complexity
    dependencies: list[str] = Field(default_factory=list)


class Budget(BaseModel):
    budget_id: str = Field(min_length=1)
    client_metadata: ClientMetadata
    project_summary: str = Field(min_length=1)
    main_technology: str = Field(min_length=1)
    year: int = Field(ge=2000, le=2100)
    total_estimated_hours: float = Field(gt=0)
    components: list[BudgetComponent] = Field(min_length=1)

    @field_validator("components")
    @classmethod
    def component_ids_must_be_unique(cls, components: list[BudgetComponent]) -> list[BudgetComponent]:
        # chunk_id is "{budget_id}::{component_id}", so duplicates would collide
        ids = [c.component_id for c in components]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"Duplicate component_id values: {sorted(duplicates)}")
        return components


class ChunkMetadata(BaseModel):
    # Filterable fields: travel with the chunk but are NOT embedded
    model_config = ConfigDict(extra="allow")

    budget_id: str
    component_id: str
    client_sector: Sector
    main_technology: str
    year: int
    complexity: Complexity
    estimated_hours: float


class Chunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    token_count: int = Field(ge=0)


class EmbeddedChunk(Chunk):
    embedding: list[float]


class IngestRequest(BaseModel):
    budgets: list[Budget] = Field(min_length=1)


class IngestStats(BaseModel):
    total_budgets: int
    total_chunks: int
    total_tokens: int
    estimated_cost_usd: float


class IngestResponse(BaseModel):
    chunks: list[EmbeddedChunk]
    stats: IngestStats