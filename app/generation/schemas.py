from pydantic import BaseModel, Field


class SourceReference(BaseModel):
    chunk_id: str = Field(description="chunk_id of a <source> in the context, copied exactly")
    document_id: str = Field(description="document_id of that same <source>, copied exactly")
    evidence: str = Field(description="Verbatim span or figure copied from that source that backs the line")


class EstimateLineItem(BaseModel):
    component: str = Field(description="Component of the requested project, e.g. 'payments module'")
    # Nullable on purpose: an ungrounded line must not carry invented hours
    hours: float | None = Field(description="Estimated hours; null when grounded is false")
    rationale: str = Field(description="How the hours derive from the sources, including scope adjustments")
    grounded: bool = Field(description="False when the context has no sufficient data for this component")
    sources: list[SourceReference] = Field(description="Non-empty if and only if grounded is true")


class Estimate(BaseModel):
    summary: str = Field(description="One or two sentences describing the estimated project")
    insufficient_context: bool = Field(
        description="True when the context does not support estimating the project as a whole"
    )
    line_items: list[EstimateLineItem]

    @property
    def grounded_hours(self) -> float:
        # Computed in code, not by the LLM: models are unreliable at arithmetic
        return sum(item.hours for item in self.line_items if item.grounded and item.hours is not None)
